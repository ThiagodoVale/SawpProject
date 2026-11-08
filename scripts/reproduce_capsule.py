from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sawp.reproducibility import (  # noqa: E402
    require_identical,
    verify_checksums,
    write_verification_report,
)


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "MPLBACKEND": "Agg",
            "PYTHONHASHSEED": "0",
            "PYTHONUTF8": "1",
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (str(SOURCE_ROOT), environment.get("PYTHONPATH", "")) if value
    )
    return environment


def _run(command: list[str]) -> None:
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=_environment(),
        check=False,
    )
    if completed.returncode:
        raise RuntimeError(f"command failed with code {completed.returncode}: {command}")


def main() -> int:
    argparse.ArgumentParser(
        description="Reproduce and verify all 18 SAWP publication figures"
    ).parse_args()

    reference_results = verify_checksums(
        PROJECT_ROOT,
        PROJECT_ROOT / "checksums" / "reference_figures.sha256",
    )
    require_identical(reference_results, label="reference figures")

    _run([sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"])

    _run(
        [
            sys.executable,
            "scripts/generate_all_figures.py",
            "--config",
            "config/default.toml",
            "--manifest",
            "config/publication.toml",
        ]
    )

    output_results = verify_checksums(
        PROJECT_ROOT,
        PROJECT_ROOT / "checksums" / "publication_outputs.sha256",
    )
    require_identical(output_results, label="publication outputs")
    write_verification_report(
        PROJECT_ROOT / "results" / "reproduction_report.json",
        reference_results=reference_results,
        output_results=output_results,
        tests_passed=True,
    )
    print("SAWP capsule reproduced successfully: 18/18 byte-identical figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
