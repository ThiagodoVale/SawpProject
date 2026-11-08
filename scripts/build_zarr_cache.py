from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sawp.config import load_config  # noqa: E402
from sawp.scientific.cache import write_zarr_cache  # noqa: E402
from sawp.scientific.common import open_canonical  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the optional compressed Zarr cache")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "default.toml")
    args = parser.parse_args()
    config = load_config(args.config)
    for source in sorted(config.paths.raw_era5.glob("*.nc")):
        print(f"Caching {source.name}...")
        dataset = open_canonical(source, chunks={"time": config.cache.time_chunk})
        destination = write_zarr_cache(dataset, source, config, force=args.force)
        dataset.close()
        print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
