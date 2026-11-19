"""Offline regression for the atom.* / red.* PTX probe (no GPU, no driver).

Checks the case table itself: every kernel contains exactly the atomic
instruction it claims to test, every ``red`` case has an ``atom`` twin that
differs in that one instruction only, the expectations equal literal values
written here independently of the module's CPU model, and the verdict logic
never downgrades a wrong device result to "unsupported".
"""
import importlib.util
import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import ptx_atomic_forms as forms  # noqa: E402

spec = importlib.util.spec_from_file_location("zluda_capabilities_for_atomics_test", SCRIPTS / "capability_probe.py")
cap = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cap)

ATOMIC_LINE = re.compile(r"^\s*((?:atom|red)(?:\.[a-z0-9]+)+)\s+(.*);\s*$")

# Literal expectations, written without using forms.cpu_reference.
LITERAL = {
    ("store", "u32"): [7, 7, 7, 7],
    ("add", "u32"): [64, 64, 64, 64],
    ("add", "u64"): [0x4000000040] * 4,
    ("add", "f32"): [64.0, 64.0, 64.0, 64.0],
    ("max", "s32"): [252, 253, 254, 255],
    ("min", "s32"): [-300, -299, -298, -297],
}


def atomic_lines(case):
    found = []
    for line in case.ptx.splitlines():
        m = ATOMIC_LINE.match(line)
        if m:
            found.append((m.group(1), m.group(2)))
    return found


def mnemonic(case):
    space = "" if case.space == "generic" else "." + case.space
    return f"{case.instruction}{space}.{case.op}.{case.ctype}"


class PtxAtomicFormsTest(unittest.TestCase):
    def test_names_and_kernels_are_unique_and_declared(self):
        names = [c.name for c in forms.CASES]
        kernels = [c.kernel for c in forms.CASES]
        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(len(kernels), len(set(kernels)))
        for case in forms.CASES:
            self.assertIn(f".visible .entry {case.kernel}(", case.ptx)
            self.assertEqual(case.ptx.count(".visible .entry"), 1)
            case.ptx.encode("ascii")  # the driver receives ASCII PTX

    def test_each_case_contains_exactly_the_instruction_it_names(self):
        for case in forms.CASES:
            with self.subTest(case=case.name):
                lines = atomic_lines(case)
                if case.instruction == "control":
                    self.assertEqual(lines, [])
                    continue
                self.assertEqual(len(lines), 1, lines)
                found, operands = lines[0]
                self.assertEqual(found, mnemonic(case))
                parts = [p.strip() for p in operands.split(",")]
                # atom d, [a], b   /   red [a], b
                self.assertEqual(len(parts), 3 if case.instruction == "atom" else 2)
                source = parts[-1]
                if case.operand == "immediate":
                    self.assertRegex(source, r"^\d+$")
                else:
                    self.assertTrue(source.startswith("%"), source)
                self.assertTrue(parts[-2].startswith("[") and parts[-2].endswith("]"))

    def test_every_red_case_has_an_atom_twin_differing_in_one_line(self):
        by_name = {c.name: c for c in forms.CASES}
        reds = [c for c in forms.CASES if c.instruction == "red"]
        self.assertGreaterEqual(len(reds), 11)
        for red in reds:
            with self.subTest(case=red.name):
                twin = by_name["atom" + red.name[len("red"):]]
                self.assertEqual((twin.space, twin.op, twin.ctype, twin.operand),
                                 (red.space, red.op, red.ctype, red.operand))
                a = twin.ptx.replace(twin.kernel, "K").splitlines()
                r = red.ptx.replace(red.kernel, "K").splitlines()
                self.assertEqual(len(a), len(r))
                differing = [(x, y) for x, y in zip(a, r) if x != y]
                self.assertEqual(len(differing), 1, differing)
                self.assertTrue(ATOMIC_LINE.match(differing[0][0]) and ATOMIC_LINE.match(differing[0][1]))
                self.assertEqual(forms.cpu_reference(twin), forms.cpu_reference(red))

    def test_cpu_reference_matches_literal_expectations(self):
        seen = set()
        for case in forms.CASES:
            with self.subTest(case=case.name):
                self.assertEqual(forms.cpu_reference(case), LITERAL[(case.op, case.ctype)])
                seen.add((case.op, case.ctype))
        self.assertEqual(seen, set(LITERAL))

    def test_matrix_covers_spaces_operands_and_both_instructions(self):
        cells = {(c.instruction, c.space, c.operand) for c in forms.CASES if c.op == "add" and c.ctype == "u32"}
        for instruction in ("atom", "red"):
            for space in ("global", "shared", "generic"):
                for operand in ("register", "immediate"):
                    self.assertIn((instruction, space, operand), cells)

    def test_dump_ptx_writes_every_module_without_a_driver(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(forms.dump_ptx(tmp), 0)
            written = sorted(p.name for p in Path(tmp).iterdir())
            self.assertEqual(written, sorted(c.name + ".ptx" for c in forms.CASES))
            for case in forms.CASES:
                self.assertEqual((Path(tmp) / (case.name + ".ptx")).read_bytes(), case.ptx.encode("ascii"))

    def test_verdict_never_hides_a_wrong_result(self):
        ok = {"status": "pass"}
        self.assertEqual(forms.summarize({"a": ok, "b": ok})["verdict"], "pass")
        self.assertEqual(forms.summarize({"a": ok, "b": {"status": "not_exported"}})["verdict"], "unsupported")
        self.assertEqual(forms.summarize({"a": ok, "b": {"status": "rejected"}})["verdict"], "unsupported")
        mixed = forms.summarize({"a": ok, "b": {"status": "not_exported"}, "c": {"status": "incorrect"}})
        self.assertEqual(mixed["verdict"], "incorrect")
        self.assertEqual(mixed["incorrect"], ["c"])
        self.assertEqual(forms.summarize({"a": {"status": "something_new"}})["verdict"], "incorrect")
        self.assertEqual(forms.summarize({"a": {}})["verdict"], "incorrect")

    def test_probe_is_registered_and_maps_verdicts_to_harness_statuses(self):
        self.assertIn("driver_ptx_atomics", cap.TESTS)
        self.assertIs(cap.RUNNERS["driver_ptx_atomics"], cap.run_driver_ptx_atomics)

        class FakeCuda:
            class _Fn:
                def __call__(self, pointer):
                    pointer._obj.value = 1
                    return 0

            cuCtxGetCurrent = _Fn()

        class FakeTorch:
            class cuda:  # noqa: N801
                @staticmethod
                def synchronize():
                    return None

            @staticmethod
            def empty(*args, **kwargs):
                return object()

        def report(statuses):
            results = {name: {"status": status} for name, status in statuses.items()}
            return {"cases": results, "summary": forms.summarize(results)}

        with patch.object(cap, "_load_win_dll", return_value=FakeCuda()):
            with patch.object(forms, "run_cases", return_value=report({"a": "pass", "b": "pass"})):
                self.assertTrue(cap.run_driver_ptx_atomics(FakeTorch())["ok"])
            with patch.object(forms, "run_cases", return_value=report({"a": "pass", "b": "incorrect", "c": "not_exported"})):
                self.assertFalse(cap.run_driver_ptx_atomics(FakeTorch())["ok"])
            with patch.object(forms, "run_cases", return_value=report({"a": "pass", "b": "not_exported"})):
                # "not exported" is one of the harness's unsupported markers
                with self.assertRaisesRegex(RuntimeError, r"not exported by this driver \(1 of 2\).*details=\{"):
                    cap.run_driver_ptx_atomics(FakeTorch())


if __name__ == "__main__":
    unittest.main()
