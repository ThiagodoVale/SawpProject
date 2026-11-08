from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sawp.config import load_config  # noqa: E402
from sawp.scientific import ScientificPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recalculate all 18 SAWP figures from NetCDF and measured rainfall data"
    )
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "default.toml")
    parser.add_argument("--clean", action="store_true", help="remove reproducible outputs first")
    parser.add_argument("--force", action="store_true", help="recalculate processed products")
    args = parser.parse_args()
    pipeline = ScientificPipeline(load_config(args.config))
    reports = pipeline.run(force=args.force, clean=args.clean)
    print(json.dumps(reports, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
