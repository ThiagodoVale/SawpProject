from __future__ import annotations

import hashlib
import json
import platform
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ChecksumResult:
    relative_path: str
    expected_sha256: str
    actual_sha256: str | None
    exists: bool
    identical: bool


def sha256(path: Path, *, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def read_checksum_file(path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        digest, separator, relative = stripped.partition("  ")
        if not separator or len(digest) != 64:
            raise ValueError(f"invalid checksum entry at {path}:{line_number}")
        entries[relative.replace("\\", "/")] = digest.lower()
    return entries


def verify_checksums(root: Path, checksum_file: Path) -> list[ChecksumResult]:
    results = []
    for relative, expected in read_checksum_file(checksum_file).items():
        target = root / Path(relative)
        actual = sha256(target) if target.is_file() else None
        results.append(
            ChecksumResult(
                relative_path=relative,
                expected_sha256=expected,
                actual_sha256=actual,
                exists=target.is_file(),
                identical=actual == expected,
            )
        )
    return results


def require_identical(results: Iterable[ChecksumResult], *, label: str) -> None:
    failed = [result for result in results if not result.identical]
    if failed:
        paths = ", ".join(result.relative_path for result in failed)
        raise RuntimeError(f"{label} checksum verification failed: {paths}")


def runtime_identity() -> dict[str, str]:
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
    }


def write_verification_report(
    path: Path,
    *,
    reference_results: Iterable[ChecksumResult],
    output_results: Iterable[ChecksumResult],
    tests_passed: bool,
) -> None:
    references = list(reference_results)
    outputs = list(output_results)
    report = {
        "schema_version": 1,
        "reproducibility_level": "artifact-byte-identical",
        "runtime": runtime_identity(),
        "tests_passed": tests_passed,
        "reference_figures": [asdict(result) for result in references],
        "output_figures": [asdict(result) for result in outputs],
        "complete": (
            tests_passed
            and all(result.identical for result in references)
            and all(result.identical for result in outputs)
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
