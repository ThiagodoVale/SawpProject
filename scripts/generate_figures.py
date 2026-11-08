"""Generate all 18 SAWP publication figures from raw scientific inputs."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "default.toml")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse valid processed files instead of rebuilding from raw inputs",
    )
    args = parser.parse_args()
    pipeline = ScientificPipeline(load_config(args.config))
    reports = pipeline.run(force=not args.resume, clean=not args.resume)
    print(
        json.dumps(
            {
                "version": "1.1.0",
                "figure_count": len(reports),
                "outputs": [report["output"] for report in reports],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
