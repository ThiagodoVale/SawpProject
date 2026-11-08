from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sawp.reproducibility import require_identical, verify_checksums  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the byte-identical SAWP capsule")
    parser.add_argument(
        "--outputs",
        action="store_true",
        help="verify generated outputs in addition to the frozen reference figures",
    )
    args = parser.parse_args()
    references = verify_checksums(
        PROJECT_ROOT,
        PROJECT_ROOT / "checksums" / "reference_figures.sha256",
    )
    require_identical(references, label="reference figures")
    print(f"Reference figures: {len(references)}/{len(references)} identical")
    if args.outputs:
        outputs = verify_checksums(
            PROJECT_ROOT,
            PROJECT_ROOT / "checksums" / "publication_outputs.sha256",
        )
        require_identical(outputs, label="publication outputs")
        print(f"Publication outputs: {len(outputs)}/{len(outputs)} identical")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
