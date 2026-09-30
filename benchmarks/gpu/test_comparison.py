"""Exercise rejection of incomplete, mispaired and invalid timing records."""

import copy
import unittest

from run_comparison import sample_records, session_summary


def fixture():
    config = {"variant_names": ["wave", "native", "slang"],
              "cases": [{"name": "vector_add", "size": 17}],
              "warmup": 1, "repeats": 2, "dispatches_per_sample": 32, "order_seed": 9}
    records = []
    for repeat in (-1, 0, 1):
        for order, (variant, elapsed) in enumerate((("wave", 2e-5), ("native", 1e-5), ("slang", 4e-5))):
            records.append({"kind": "sample", "workload": "vector_add", "size": 17,
                            "variant": variant, "repeat": repeat, "order_in_pair": order,
                            "warmup": repeat < 0, "dispatches": 32, "correct": True,
                            "gpu_status": "completed", "gpu_seconds": elapsed,
                            "gpu_batch_seconds": elapsed * 32,
                            "host_dispatch_seconds": elapsed * 2,
                            "host_batch_seconds": elapsed * 64})
    return config, records


class ComparisonRecords(unittest.TestCase):
    def test_known_pair_ratios(self):
        config, records = fixture()
        summary = session_summary(config, records)[0]
        for pair, wanted in (("wave_over_native", 2), ("wave_over_slang", .5), ("slang_over_native", 4)):
            self.assertAlmostEqual(summary["ratios"][pair]["geomean"], wanted)

    def test_incomplete_and_duplicate_records_fail(self):
        config, records = fixture()
        for invalid in (records[:-1], records + [records[0]]):
            with self.subTest(records=len(invalid)), self.assertRaises(ValueError):
                sample_records(config, invalid)

    def test_invalid_measurements_fail(self):
        config, records = fixture()
        for value in (0, -1, float("nan"), float("inf")):
            changed = copy.deepcopy(records)
            changed[4]["gpu_seconds"] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                sample_records(config, changed)

    def test_false_success_and_protocol_changes_fail(self):
        config, records = fixture()
        for field, value in (("correct", False), ("gpu_status", "failed"),
                             ("warmup", False), ("dispatches", 1),
                             ("order_in_pair", 1), ("gpu_batch_seconds", 1)):
            changed = copy.deepcopy(records)
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                sample_records(config, changed)


if __name__ == "__main__":
    unittest.main()
