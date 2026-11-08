from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sawp.publication.reproduce import reproduce_publication  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate all 18 accepted figures from raw inputs and vendored original scripts"
    )
    parser.add_argument("--clean", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-intermediates", action="store_true")
    args = parser.parse_args()
    reports = reproduce_publication(
        PROJECT_ROOT,
        clean=args.clean,
        force=args.force,
        keep_intermediates=args.keep_intermediates,
    )
    print(f"Publication reproduction complete: {len(reports)}/18 figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
