"""Check notebook reruns and invalid timing records without a CUDA device."""

import contextlib
import ctypes as ct
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock

import prepare
import rounding
import run


class EventDriver:
    def __init__(self, elapsed):
        self.elapsed = elapsed

    def call(self, name, *arguments):
        if name == "cuEventElapsedTime":
            arguments[0]._obj.value = self.elapsed


class RunnerTests(unittest.TestCase):
    def test_invalid_event_duration_fails(self):
        for elapsed in (float("nan"), float("inf"), float("-inf"), -1.0):
            with self.subTest(elapsed=elapsed):
                allocation = run.Allocation(EventDriver(elapsed), bytes(4))
                with self.assertRaisesRegex(ValueError, "invalid CUDA event duration"):
                    allocation.execute(ct.c_void_p(1), {"groups": 1, "threads": 128}, 1)

    def test_zero_event_duration_is_retained(self):
        allocation = run.Allocation(EventDriver(0.0), bytes(4))
        memory, timing = allocation.execute(
            ct.c_void_p(1), {"groups": 1, "threads": 128}, 1
        )
        self.assertEqual(memory, bytes(4))
        self.assertEqual(timing["event_ms"], 0.0)

    def test_json_rejects_nonfinite_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(ValueError):
                run.write_json(Path(temporary) / "record.json", {"event_ms": float("nan")})

    def test_literal_rounding_contract_distinguishes_contraction(self):
        self.assertEqual(rounding.EXPECTED_WORDS[4:7], [0, 0, 0xA8800000])
        self.assertEqual(rounding.INITIAL_WORDS[:3], [0x3F800001, 0x3F7FFFFE, 0xBF800000])
        contracted = bytearray(rounding.expected_bytes())
        contracted[16:20] = contracted[24:28]
        with self.assertRaisesRegex(ValueError, "word 4"):
            run.check_memory(bytes(contracted), rounding.expected_bytes(), "contracted multiply/add")

    def test_disassembler_is_required_before_compilation(self):
        with mock.patch.object(run.shutil, "which", side_effect=lambda name: "/absent/nvcc" if name == "nvcc" else None):
            with tempfile.TemporaryDirectory() as temporary:
                with self.assertRaisesRegex(RuntimeError, "nvdisasm is required"):
                    run.compile_kernels(Path(temporary), Path(temporary), object(), [])

    def test_failed_rounding_witness_prevents_timing_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "output"
            output.mkdir()
            run.write_json(root / "config.json", {"cases": [{"name": "vector_add", "size": 1}]})
            run.write_json(root / "manifest.json", {})
            driver = mock.Mock(info={"name": "mock"})
            with mock.patch.object(run, "Driver", return_value=driver), \
                    mock.patch.object(run, "verify_payload", return_value={}), \
                    mock.patch.object(run.shutil, "which", return_value=None), \
                    mock.patch.object(run, "compile_kernels", return_value={"rounding": {}}), \
                    mock.patch.object(run, "verify_disassembly"), \
                    mock.patch.object(run, "check_rounding", side_effect=ValueError("incorrect rounding")):
                with self.assertRaisesRegex(ValueError, "incorrect rounding"):
                    run.run(root, output, 2, 1, 1, 42)
            self.assertFalse((output / "samples.jsonl").exists())
            self.assertFalse((output / "summary.json").exists())
            driver.close.assert_called_once()

    def test_notebook_preserves_attempts_and_reports_startup_failure(self):
        notebook = prepare.notebook(b"unused test payload")
        source = "".join(notebook["cells"][3]["source"])
        attempts = []

        def execute(arguments):
            destination = Path(arguments[arguments.index("--output") + 1])
            self.assertFalse(destination.exists())
            attempts.append(destination)
            if len(attempts) == 3:
                return subprocess.CompletedProcess(arguments, 2)
            destination.mkdir()
            (destination / "status.json").write_text(json.dumps({"state": "completed"}))
            (destination / "summary.json").write_text("[]")
            return subprocess.CompletedProcess(arguments, 0)

        with tempfile.TemporaryDirectory() as temporary:
            workspace = Path(temporary)
            namespace = {"workspace": workspace, "payload": workspace / "payload"}
            output = io.StringIO()
            with mock.patch("subprocess.run", side_effect=execute), contextlib.redirect_stdout(output):
                exec(source, namespace)
                exec(source, namespace)
                output.truncate(0)
                output.seek(0)
                exec(source, namespace)
            self.assertEqual(len(set(attempts)), 3)
            self.assertTrue(all(path.is_dir() for path in attempts[:2]))
            self.assertFalse(attempts[2].exists())
            self.assertIn("Runner exit code: 2", output.getvalue())
            self.assertIn("failed before writing an error record", output.getvalue())
            self.assertNotIn('"state": "completed"', output.getvalue())


if __name__ == "__main__":
    unittest.main()
