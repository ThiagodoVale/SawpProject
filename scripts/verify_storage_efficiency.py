from __future__ import annotations

import json
import sys
from pathlib import Path

import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sawp.config import load_config  # noqa: E402

GIB = 1024**3
MAX_COMPACT_BYTES = 3 * GIB


def _tree_size(directory: Path, *, exclude: Path | None = None) -> int:
    if not directory.exists():
        return 0
    return sum(
        path.stat().st_size
        for path in directory.rglob("*")
        if path.is_file() and (exclude is None or exclude not in path.parents)
    )


def main() -> int:
    config = load_config(PROJECT_ROOT / "config" / "default.toml")
    zarr_root = config.paths.workspace / config.cache.zarr_directory
    processed_bytes = _tree_size(config.paths.processed)
    interim_bytes = _tree_size(config.paths.interim, exclude=zarr_root.resolve())
    zarr_bytes = _tree_size(zarr_root.resolve())
    compact_bytes = processed_bytes + interim_bytes
    failures: list[str] = []

    legacy_npz = [
        path
        for root in (config.paths.processed, config.paths.interim)
        if root.exists()
        for path in root.rglob("*.npz")
    ]
    if legacy_npz:
        failures.append(f"legacy NPZ products found: {len(legacy_npz)}")
    if compact_bytes > MAX_COMPACT_BYTES:
        failures.append(f"compact products exceed 3 GiB: {compact_bytes / GIB:.2f} GiB")

    coefficient_files = (
        config.paths.processed / "ocean" / "sst_trend_coefficients.nc",
        config.paths.processed / "atmosphere" / "atmospheric_trend_coefficients.nc",
    )
    for path in coefficient_files:
        if not path.is_file():
            failures.append(f"missing compact trend coefficients: {path}")
            continue
        with xr.open_dataset(path) as dataset:
            forbidden = [name for name in dataset.data_vars if "time" in dataset[name].dims]
            if forbidden:
                failures.append(
                    f"coefficient file stores full time cubes ({path.name}): {forbidden}"
                )

    report = {
        "schema_version": 1,
        "compact_bytes": compact_bytes,
        "compact_gib": compact_bytes / GIB,
        "processed_bytes": processed_bytes,
        "interim_bytes_excluding_zarr": interim_bytes,
        "optional_zarr_bytes": zarr_bytes,
        "legacy_npz_count": len(legacy_npz),
        "limit_gib": MAX_COMPACT_BYTES / GIB,
        "passed": not failures,
        "failures": failures,
    }
    destination = config.paths.results / "storage_efficiency_report.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
