"""Exercise campaign completion and failure handling without a CUDA device."""

import contextlib
import io
import json
from pathlib import Path
import random
import subprocess
import tempfile
import unittest
from unittest import mock

import campaign
import rounding
import run


def seal(directory):
    run.write_json(directory / "artifacts.json", {
        str(path.relative_to(directory)): run.digest(path.read_bytes())
        for path in directory.rglob("*") if path.is_file() and path.name != "artifacts.json"
    })


class CampaignTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.payload = self.root / "payload"
        (self.payload / "data").mkdir(parents=True)
        expected = bytes.fromhex("0000803f")
        (self.payload / "data/expected.bin").write_bytes(expected)
        (self.payload / "run.py").write_text("# mocked process runner\n")
        witness_data = {"initial": rounding.initial_bytes(), "expected": rounding.expected_bytes(),
                        "wbin": b"mock witness binary", "ptx": b"mock witness PTX"}
        for field, contents in witness_data.items():
            (self.payload / "data" / rounding.CONTRACT[field + "_path"]).write_bytes(contents)
        run.write_json(self.payload / "rounding-check.json", {
            **rounding.CONTRACT,
            **{field + "_sha256": run.digest(data) for field, data in witness_data.items()},
            "emulator": {"correct": True, "dispatches": 1,
                         "memory_sha256": run.digest(rounding.expected_bytes())},
        })
        self.reference = run.digest(expected)
        run.write_json(self.payload / "config.json", {"cases": [{
            "name": "vector_add", "size": 1, "comparison": "bit_exact",
            "expected_path": "expected.bin",
            "emulator": {"correct": True, "memory_sha256": self.reference},
        }]})
        self.manifest = {str(path.relative_to(self.payload)): run.digest(path.read_bytes())
                         for path in self.payload.rglob("*") if path.is_file()}
        run.write_json(self.payload / "manifest.json", self.manifest)
        self.parameters = {"repeats": 2, "warmup": 1, "dispatches_per_sample": 1}
        self.planned = {"session": "session-01", "seed": 42}

    def record(self, directory, planned, parameters):
        directory.mkdir()
        rng = random.Random(planned["seed"])
        case_order = [["vector_add", 1]]
        rng.shuffle(case_order)
        run.write_json(directory / "status.json", {"state": "completed"})
        run.write_json(directory / "protocol.json", {
            **parameters, "order_seed": planned["seed"], "case_order": case_order,
            "payload_manifest_sha256": run.digest((self.payload / "manifest.json").read_bytes()),
        })
        run.write_json(directory / "payload-manifest.json", self.manifest)
        run.write_json(directory / "environment.json", {"device": {"name": "mock"}})
        fake_allocation = mock.Mock()
        fake_allocation.execute.return_value = (rounding.expected_bytes(), {})
        with mock.patch.object(run, "Allocation", return_value=fake_allocation), contextlib.redirect_stdout(io.StringIO()):
            run.check_rounding(self.payload, directory, object(), {"wave": object(), "native": object()})
        disassembly, commands = [], []
        for variant, names, stem in (("native", ["vector_add"], "native"),
                                     ("wave", ["vector_add"], "vector_add.wave"),
                                     ("wave", [rounding.NAME], rounding.NAME + ".wave"),
                                     ("native", [rounding.NAME], rounding.NAME + ".native")):
            cubin, sass = stem + ".cubin", stem + ".sass"
            (directory / cubin).write_bytes(b"mock cubin")
            (directory / sass).write_text("mock disassembly: " + " ".join(names))
            disassembly.append({"variant": variant, "kernels": names, "cubin": cubin, "sass": sass,
                                "cubin_sha256": run.digest((directory / cubin).read_bytes()),
                                "sass_sha256": run.digest((directory / sass).read_bytes())})
            commands.append({"returncode": 0, "log": sass})
        run.write_json(directory / "disassembly.json", disassembly)
        run.write_json(directory / "compilation.json", commands)
        run.write_json(directory / "correctness.json", [{
            "name": "vector_add", "size": 1, "variant": variant,
            "correct": True, "dispatches": 1, "memory_sha256": self.reference,
        } for variant in ("wave", "native")])
        rows = []
        for pair in range(-parameters["warmup"], parameters["repeats"]):
            variants = ["wave", "native"]
            rng.shuffle(variants)
            for position, variant in enumerate(variants):
                duration = 2.0 if variant == "wave" else 1.0
                rows.append({
                    "name": "vector_add", "size": 1, "pair": pair, "variant": variant,
                    "position": position, "warmup": pair < 0, "correct": True,
                    "dispatches": parameters["dispatches_per_sample"],
                    "memory_sha256": self.reference, "event_ms": duration / 1000,
                    "event_us_per_dispatch": duration / parameters["dispatches_per_sample"],
                    "host_ns": 10000,
                })
        (directory / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
        run.write_json(directory / "summary.json", run.summarize(
            [row for row in rows if not row["warmup"]], planned["seed"]))
        seal(directory)

    def subprocess_record(self, arguments, **_kwargs):
        def value(flag):
            return arguments[arguments.index(flag) + 1]

        directory = Path(value("--output"))
        self.assertFalse(directory.exists())
        parameters = {"repeats": int(value("--repeats")), "warmup": int(value("--warmup")),
                      "dispatches_per_sample": int(value("--dispatches"))}
        self.record(directory, {"session": directory.name, "seed": int(value("--seed"))}, parameters)
        return subprocess.CompletedProcess(arguments, 0)

    def execute(self, output, side_effect, sessions=3):
        output.mkdir()
        with mock.patch("subprocess.run", side_effect=side_effect), contextlib.redirect_stdout(io.StringIO()):
            campaign.execute_campaign(self.payload, output, sessions, 2, 1, 1, 42)

    def test_complete_campaign_uses_distinct_process_seeds(self):
        output = self.root / "campaign"
        self.execute(output, self.subprocess_record, sessions=5)
        plan = campaign.read_json(output / "campaign.json")
        self.assertEqual(len({session["seed"] for session in plan["sessions"]}), 5)
        attempts = campaign.read_json(output / "attempts.json")
        self.assertTrue(all(attempt["state"] == "verified" for attempt in attempts))
        summary = campaign.read_json(output / "summary.json")
        self.assertEqual(summary["state"], "completed")
        self.assertEqual(summary["cases"][0]["wave_over_native_geomean"], 2.0)
        self.assertEqual(summary["cases"][0]["between_process_bootstrap_95"], [2.0, 2.0])

    def test_nonzero_process_cannot_use_completed_status(self):
        def failed_process(arguments, **kwargs):
            self.subprocess_record(arguments, **kwargs)
            return subprocess.CompletedProcess(arguments, 1)

        output = self.root / "failed"
        with self.assertRaisesRegex(ValueError, "campaign is incomplete"):
            self.execute(output, failed_process)
        self.assertEqual(campaign.read_json(output / "summary.json")["verified_sessions"], 0)
        self.assertEqual(campaign.read_json(output / "session-01/status.json")["state"], "completed")
        self.assertEqual(campaign.read_json(output / "attempts.json")[0]["state"], "failed")

    def test_startup_failure_preserves_attempt_and_later_sessions(self):
        calls = 0

        def process(arguments, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("mock process could not start")
            return self.subprocess_record(arguments, **kwargs)

        output = self.root / "partial"
        with self.assertRaisesRegex(ValueError, "campaign is incomplete"):
            self.execute(output, process)
        self.assertEqual(calls, 3)
        summary = campaign.read_json(output / "summary.json")
        self.assertEqual(summary["state"], "incomplete")
        self.assertEqual(summary["verified_sessions"], 2)
        self.assertFalse(summary["cases"][0]["complete"])
        self.assertTrue((output / "session-02-verification-error.txt").is_file())

    def test_missing_sample_or_false_summary_is_rejected(self):
        for mutation in ("missing", "duplicate", "summary", "reference", "nonfinite", "seed"):
            with self.subTest(mutation=mutation):
                directory = self.root / mutation
                self.record(directory, self.planned, self.parameters)
                rows = [json.loads(line) for line in (directory / "samples.jsonl").read_text().splitlines()]
                if mutation == "missing":
                    rows.pop()
                elif mutation == "duplicate":
                    rows[-1] = rows[-2]
                elif mutation == "summary":
                    run.write_json(directory / "summary.json", [{"wave_over_native_geomean": 0.01}])
                elif mutation == "reference":
                    rows[-1]["memory_sha256"] = "0" * 64
                elif mutation == "nonfinite":
                    rows[-1]["event_ms"] = float("nan")
                else:
                    protocol = campaign.read_json(directory / "protocol.json")
                    protocol["order_seed"] += 1
                    run.write_json(directory / "protocol.json", protocol)
                (directory / "samples.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
                seal(directory)
                with self.assertRaises(ValueError):
                    campaign.verify_session(self.payload, directory, self.planned, self.parameters)

    def test_existing_session_cannot_be_reused(self):
        output = self.root / "stale"
        output.mkdir()
        self.record(output / "session-01", self.planned, self.parameters)
        with mock.patch("subprocess.run") as process:
            with self.assertRaisesRegex(ValueError, "already exists"):
                campaign.execute_campaign(self.payload, output, 3, 2, 1, 1, 42)
        process.assert_not_called()

    def test_rounding_and_disassembly_proof_are_required(self):
        for mutation in ("missing_rounding", "false_rounding", "missing_disassembly", "missing_command"):
            with self.subTest(mutation=mutation):
                directory = self.root / mutation
                self.record(directory, self.planned, self.parameters)
                if mutation == "missing_rounding":
                    (directory / "rounding-check.json").unlink()
                elif mutation == "false_rounding":
                    (directory / "rounding-wave.actual.bin").write_bytes(rounding.initial_bytes())
                elif mutation == "missing_disassembly":
                    run.write_json(directory / "disassembly.json", [])
                else:
                    commands = campaign.read_json(directory / "compilation.json")
                    run.write_json(directory / "compilation.json", commands[:-1])
                seal(directory)
                with self.assertRaises((ValueError, FileNotFoundError)):
                    campaign.verify_session(self.payload, directory, self.planned, self.parameters)

    def test_payload_manifest_cannot_change_between_processes(self):
        calls = 0

        def process(arguments, **kwargs):
            nonlocal calls
            calls += 1
            result = self.subprocess_record(arguments, **kwargs)
            with (self.payload / "manifest.json").open("a") as manifest:
                manifest.write("\n")
            return result

        output = self.root / "changed-payload"
        with self.assertRaisesRegex(ValueError, "campaign is incomplete"):
            self.execute(output, process)
        self.assertEqual(calls, 1)
        attempts = campaign.read_json(output / "attempts.json")
        self.assertTrue(all("payload manifest changed" in attempt["error"] for attempt in attempts))


if __name__ == "__main__":
    unittest.main()
