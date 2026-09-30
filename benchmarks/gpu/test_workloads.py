#!/usr/bin/env python3
"""Check workload references, edge cases, and assembled emulator execution."""

import argparse
from array import array
from collections import Counter
from fractions import Fraction
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from workloads import (
    APPLICATION_WORKLOADS, GUARD, WORKLOADS, assembly, float_word,
    guarded_allocation, make_case,
)


REPOSITORY = Path(__file__).resolve().parents[2]
ASSEMBLER = REPOSITORY / "wave-asm/target/release/wave-asm"
EMULATOR = REPOSITORY / "wave-emu/target/release/wave-emu"


def words(data):
    result = array("I")
    result.frombytes(data)
    return result


def number(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


class WorkloadTests(unittest.TestCase):
    def execute(self, name, initial, expected, groups):
        with tempfile.TemporaryDirectory(prefix="wave-application-test-") as temporary:
            directory = Path(temporary)
            source, binary, allocation = (directory / name).with_suffix(".wave"), directory / "kernel.wbin", directory / "input.bin"
            source.write_text(assembly(name))
            allocation.write_bytes(initial)
            assembled = subprocess.run([ASSEMBLER, source, "-o", binary], capture_output=True, text=True)
            self.assertEqual(assembled.returncode, 0, assembled.stderr)
            result = subprocess.run([
                EMULATOR, binary, "--grid", f"{groups},1,1", "--wave-width", "32",
                "--device-memory", str(len(initial)), "--arg", f"0:{allocation}",
                "--max-instructions", "10000000", "--dump-u32", f"0:{len(initial) // 4}",
            ], check=True, capture_output=True, text=True, timeout=60)
            actual = array("I", map(int, result.stdout.splitlines())).tobytes()
            self.assertEqual(actual, expected, result.stderr)

    def test_generated_applications(self):
        for name in APPLICATION_WORKLOADS:
            for size in (1, 2, 3, 17, 127, 128, 129, 257):
                with self.subTest(name=name, size=size):
                    initial, expected, case = make_case(name, size, 20260930)
                    self.execute(name, initial, expected, case["groups"])
                    start, count = case["output_word_offset"], case["output_words"]
                    before, after = words(initial), words(expected)
                    self.assertEqual(before[:start], after[:start])
                    self.assertEqual(before[start + count:], after[start + count:])
                    self.assertTrue(all(value == GUARD for value in before[start:start + count]))
                    self.assertTrue(all(math.isfinite(number(value)) for value in after[start:start + count]))

    def test_csr_hand_computed_fixture(self):
        # A = [[2, 0, -1], [0, 0, 0], [.5, 3, 0]], x = [4, -2, 8].
        expected_output = list(map(float_word, (0, 0, -4)))
        memory, offsets = guarded_allocation([
            ("row_ptr", [0, 2, 2, 4]), ("col_idx", [0, 2, 0, 1]),
            ("values", list(map(float_word, (2, -1, .5, 3)))),
            ("vector", list(map(float_word, (4, -2, 8)))),
        ], expected_output)
        memory[:8] = [3, offsets["row_ptr"] * 4, offsets["col_idx"] * 4,
                      offsets["output"] * 4, offsets["values"] * 4, offsets["vector"] * 4, 3, 4]
        initial = array("I", memory).tobytes()
        start = offsets["output"]
        memory[start:start + 3] = expected_output
        self.execute("csr_spmv", initial, array("I", memory).tobytes(), 1)

    def test_stencil_hand_computed_fixture(self):
        # The two interior cells are 16 and -8; their results are 9 and .5.
        inputs = [1, 2, 3, 4, 5, 16, -8, 6, 7, 9, 11, 13]
        expected_output = list(map(float_word, [1, 2, 3, 4, 5, 9, .5, 6, 7, 9, 11, 13]))
        memory, offsets = guarded_allocation([("input", list(map(float_word, inputs)))], expected_output)
        memory[:6] = [12, offsets["input"] * 4, 4, offsets["output"] * 4, 3, 2]
        initial = array("I", memory).tobytes()
        start = offsets["output"]
        memory[start:start + 12] = expected_output
        self.execute("stencil_2d", initial, array("I", memory).tobytes(), 1)

    def test_csr_reference_as_dense_matrix(self):
        initial, expected, case = make_case("csr_spmv", 17, 123)
        memory, result = words(initial), words(expected)
        row_ptr, columns, output, values, vector = [memory[index] // 4 for index in (1, 2, 3, 4, 5)]
        dense = [[Fraction(0) for _ in range(17)] for _ in range(17)]
        for row in range(17):
            for entry in range(memory[row_ptr + row], memory[row_ptr + row + 1]):
                dense[row][memory[columns + entry]] += Fraction(number(memory[values + entry]))
        reference = [sum(value * Fraction(number(memory[vector + column]))
                         for column, value in enumerate(row)) for row in dense]
        self.assertEqual(list(result[output:output + 17]), [float_word(float(value)) for value in reference])
        self.assertGreater(len(set(memory[row_ptr + i + 1] - memory[row_ptr + i] for i in range(17))), 4)
        self.assertEqual(case["parameters"]["rows"], 17)

    def test_core_workloads_against_independent_references(self):
        for name in WORKLOADS:
            for size in (1, 31, 32, 33, 127, 128, 129):
                with self.subTest(name=name, size=size):
                    initial, expected, _ = make_case(name, size, 20260930)
                    memory = words(initial)
                    count, a_bytes, b_bytes, output_bytes = memory[:4]
                    self.assertEqual(count, size)
                    a_offset, b_offset, output = a_bytes // 4, b_bytes // 4, output_bytes // 4
                    a_words = memory[a_offset:a_offset + count]
                    b_words = memory[b_offset:b_offset + count]
                    groups = (count + 127) // 128
                    if name.startswith("histogram_"):
                        self.assertTrue(all(value < 64 for value in a_words))
                        frequencies = Counter(a_words)
                        reference = [frequencies[bucket] for bucket in range(64)]
                        self.assertEqual(sum(reference), size)
                    elif name == "gather":
                        self.assertEqual(sorted(b_words), list(range(count)))
                        reference = [a_words[index] for index in b_words]
                    else:
                        # The bounded dyadic inputs make exact rational arithmetic an
                        # independent reference for these binary32 operations.
                        a = [Fraction(number(value)) for value in a_words]
                        if name == "vector_add":
                            b = [Fraction(number(value)) for value in b_words]
                            values = [left + right for left, right in zip(a, b)]
                        elif name == "divergent":
                            values = [value - 1 if value < 0 else value * 2 for value in a]
                        elif name in ("reduce_shuffle", "reduce_shared"):
                            values = [sum(a[start:start + 32])
                                      for start in range(0, groups * 128, 32)]
                        else:
                            self.fail(f"missing independent reference for workload: {name}")
                        reference = [float_word(float(value)) for value in values]
                    output_initial = 0 if name.startswith("histogram_") else GUARD
                    self.assertEqual(list(memory[output:output + len(reference)]),
                                     [output_initial] * len(reference))
                    memory[output:output + len(reference)] = array("I", reference)
                    independent = memory.tobytes()
                    self.assertEqual(expected, independent)
                    self.execute(name, initial, independent, groups)

    def test_nonpositive_size_is_rejected(self):
        for name in APPLICATION_WORKLOADS:
            for size in (0, -1):
                with self.assertRaisesRegex(ValueError, "positive"):
                    make_case(name, size, 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-dir", type=Path,
                        help="directory containing wave-asm and wave-emu release binaries")
    args, remaining = parser.parse_known_args()
    if args.tool_dir is not None:
        ASSEMBLER = args.tool_dir.resolve() / "wave-asm"
        EMULATOR = args.tool_dir.resolve() / "wave-emu"
    unittest.main(argv=[sys.argv[0], *remaining])
