"""Check reference selection and required Apple campaign verification."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import prepare


class PreparationTests(unittest.TestCase):
    def test_session_argument_is_required(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            result = subprocess.run([
                sys.executable, str(Path(prepare.__file__).resolve()), "--output", str(output),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn("required: --session", result.stderr)
            self.assertFalse(output.exists())

    def test_output_argument_is_required(self):
        before = {path.relative_to(self.root): path.read_bytes()
                  for path in self.root.rglob("*") if path.is_file()}
        result = subprocess.run([
            sys.executable, str(Path(prepare.__file__).resolve()), "--session", str(self.root),
        ], cwd=self.root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("required: --output", result.stderr)
        after = {path.relative_to(self.root): path.read_bytes()
                 for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(after, before)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.base = self.root / "base"
        self.base.mkdir()
        self.original = {
            "dispatches_per_sample": 32,
            "cases": [{"name": "histogram_uniform", "size": 7,
                       "expected_path": "histogram-7.expected.bin",
                       "single_dispatch_expected_path": "histogram-7.single-dispatch.expected.bin"}],
        }
        self.config_bytes = (json.dumps(self.original) + "\n").encode()
        (self.base / "config.json").write_bytes(self.config_bytes)
        prepare.write_json(self.base / "artifacts.json", {
            "config.json": prepare.digest(self.config_bytes),
        })
        self.record_bytes = b'{"base/config.json": "fixture"}\n'
        (self.root / "record-manifest.json").write_bytes(self.record_bytes)

    def test_selects_single_dispatch_reference_without_changing_apple_config(self):
        with mock.patch("prepare.run", return_value="verified") as command:
            base, original, normalized, _manifest, metadata = prepare.measured_session(self.root)
        self.assertEqual(base, self.base)
        self.assertEqual(original, self.config_bytes)
        self.assertEqual((self.base / "config.json").read_bytes(), self.config_bytes)
        self.assertEqual(normalized["dispatches_per_sample"], 1)
        self.assertEqual(normalized["cases"][0]["expected_path"],
                         self.original["cases"][0]["single_dispatch_expected_path"])
        self.assertEqual(metadata["apple_campaign_record_manifest_sha256"],
                         prepare.digest(self.record_bytes))
        arguments = command.call_args.args[0]
        self.assertEqual(arguments[-2:], ["--verify", str(self.root)])
        self.assertTrue(arguments[1].endswith("run_comparison.py"))

    def test_failed_campaign_verification_stops_preparation(self):
        error = subprocess.CalledProcessError(1, ["verify"])
        with mock.patch("prepare.run", side_effect=error) as command:
            with self.assertRaises(subprocess.CalledProcessError):
                prepare.prepare(self.root, self.root / "output")
        self.assertEqual(command.call_count, 1)
        self.assertFalse((self.root / "output").exists())

    def test_changed_original_configuration_is_rejected(self):
        with (self.base / "config.json").open("a") as config:
            config.write("\n")
        with mock.patch("prepare.run", return_value="verified"):
            with self.assertRaisesRegex(ValueError, "configuration checksum differs"):
                prepare.measured_session(self.root)


if __name__ == "__main__":
    unittest.main()
