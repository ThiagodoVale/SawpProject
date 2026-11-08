from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sawp.reproducibility import require_identical, verify_checksums  # noqa: E402


def main() -> int:
    checksum_file = PROJECT_ROOT / "checksums" / "raw_inputs.sha256"
    if not checksum_file.is_file():
        print(f"Missing raw-input checksum manifest: {checksum_file}", file=sys.stderr)
        return 1
    results = verify_checksums(PROJECT_ROOT, checksum_file)
    require_identical(results, label="raw scientific inputs")
    print(f"Raw scientific inputs: {len(results)}/{len(results)} identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
