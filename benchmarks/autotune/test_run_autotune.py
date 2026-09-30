# SPDX-License-Identifier: Apache-2.0
"""Verify that tuning selection cannot accept incomplete or failed measurements."""

import copy
import unittest

from run_autotune import check_config, valid_result


def record():
    return {
        "status": "ok", "correct": True, "bench_size": 2, "gflops": 100,
        "gpu_seconds": [0.1, 0.2], "measurement": {"timed_iterations": 2},
        "validation": {
            "correct": True, "coverage": "full_output_at_measured_size",
            "elements_checked": 4, "elements_expected": 4,
            "failed_elements": 0, "nonfinite_elements": 0,
        },
    }


class SelectionTests(unittest.TestCase):
    def test_complete_record_is_eligible(self):
        self.assertTrue(valid_result(record()))

    def test_invalid_measurements_are_ineligible(self):
        for field, value in [("correct", False), ("status", "error"), ("bench_size", 4),
                             ("gflops", float("nan")), ("gpu_seconds", [0.1]),
                             ("gpu_seconds", [0.1, float("inf")])]:
            with self.subTest(field=field, value=value):
                changed = copy.deepcopy(record())
                changed[field] = value
                self.assertFalse(valid_result(changed))

    def test_incomplete_validation_is_ineligible(self):
        for field, value in [("correct", False), ("coverage", "sample"),
                             ("elements_checked", 2), ("failed_elements", 1),
                             ("nonfinite_elements", 1)]:
            with self.subTest(field=field):
                changed = record()
                changed["validation"][field] = value
                self.assertFalse(valid_result(changed))

    def test_partial_tiles_are_rejected(self):
        config = dict(tile_m=128, tile_n=128, tile_k=8, block_m=8, block_n=8, prefetch=True)
        check_config(config, 256)
        with self.assertRaises(ValueError):
            check_config(config, 257)
        config["tile_m"] = 0
        with self.assertRaises(ValueError):
            check_config(config, 256)


if __name__ == "__main__":
    unittest.main()
