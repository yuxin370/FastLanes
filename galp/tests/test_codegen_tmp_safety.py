from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "codegen" / "check_generated_reproducible.py"
SPEC = importlib.util.spec_from_file_location("galp_codegen_reproducibility_check", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CHECKER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKER)


class CodegenTemporaryDirectorySafetyTest(unittest.TestCase):
    def test_existing_directory_is_rejected_without_modification(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            caller_dir = Path(parent) / "caller-owned"
            caller_dir.mkdir()
            sentinel = caller_dir / "sentinel.txt"
            sentinel.write_text("preserve me")

            with self.assertRaises(SystemExit) as raised:
                CHECKER.main(["--tmp-dir", str(caller_dir)])

            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(sentinel.read_text(), "preserve me")

    def test_owned_explicit_directory_is_cleaned(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            target = Path(parent) / "checker-owned"
            with mock.patch.object(CHECKER, "check", return_value=0):
                self.assertEqual(CHECKER.main(["--tmp-dir", str(target)]), 0)
            self.assertFalse(target.exists())

    def test_exception_cleans_only_the_owned_directory(self) -> None:
        with tempfile.TemporaryDirectory() as parent:
            parent_path = Path(parent)
            sentinel = parent_path / "sentinel.txt"
            sentinel.write_text("preserve me")
            target = parent_path / "checker-owned"

            with mock.patch.object(CHECKER, "check", side_effect=RuntimeError("injected failure")):
                with self.assertRaisesRegex(RuntimeError, "injected failure"):
                    CHECKER.main(["--tmp-dir", str(target)])

            self.assertFalse(target.exists())
            self.assertEqual(sentinel.read_text(), "preserve me")


if __name__ == "__main__":
    unittest.main()
