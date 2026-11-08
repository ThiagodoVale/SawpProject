from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sawp.reproducibility import read_checksum_file, sha256, verify_checksums


class ReproducibilityTests(unittest.TestCase):
    def test_checksum_parser_and_verifier(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "artifact.bin"
            artifact.write_bytes(b"SAWP")
            checksums = root / "checksums.sha256"
            checksums.write_text(
                f"{sha256(artifact)}  artifact.bin\n",
                encoding="utf-8",
            )
            self.assertEqual(read_checksum_file(checksums)["artifact.bin"], sha256(artifact))
            results = verify_checksums(root, checksums)
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].identical)

    def test_capsule_contains_all_frozen_reference_figures(self) -> None:
        root = Path(__file__).resolve().parents[1]
        results = verify_checksums(
            root,
            root / "checksums" / "reference_figures.sha256",
        )
        self.assertEqual(len(results), 18)
        self.assertTrue(all(result.identical for result in results))

    def test_repository_contains_reproduction_entrypoints(self) -> None:
        root = Path(__file__).resolve().parents[1]
        required = (
            ".python-version",
            "uv.lock",
            "reproduce.ps1",
            "reproduce.sh",
            "CITATION.cff",
            "scripts/reproduce_capsule.py",
            "scripts/verify_capsule.py",
            "docs/REPRODUCIBILITY.md",
            ".github/workflows/reproduce.yml",
        )
        self.assertEqual([name for name in required if not (root / name).is_file()], [])


if __name__ == "__main__":
    unittest.main()
