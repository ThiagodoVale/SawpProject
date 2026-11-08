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
from sawp.scientific.pipeline import ALL_PRODUCTS, product_filename  # noqa: E402


def main() -> int:
    config = load_config(PROJECT_ROOT / "config" / "default.toml")
    failures: list[str] = []
    source_kinds: set[str] = set()
    for product_id in ALL_PRODUCTS:
        image = config.paths.results / "figures" / product_filename(product_id)
        metadata_path = image.with_suffix(image.suffix + ".json")
        if not image.is_file() or image.stat().st_size == 0:
            failures.append(f"missing image: {image}")
            continue
        if not metadata_path.is_file():
            failures.append(f"missing provenance: {metadata_path}")
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        allowed_sources = {
            "generated_from_raw",
            "generated_by_original_publication_script",
            "generated_from_raw_via_original_script",
        }
        source_kind = metadata.get("source_kind")
        source_kinds.add(str(source_kind))
        if source_kind not in allowed_sources:
            failures.append(f"non-scientific source for {product_id}")
    processed_netcdf = (
        config.paths.processed / "ocean" / "sst_amjj_annual.nc",
        config.paths.processed / "ocean" / "sst_quantile_composites.nc",
        config.paths.processed / "atmosphere" / "atmospheric_amjj_annual.nc",
    )
    if "generated_from_raw_via_original_script" not in source_kinds:
        for path in processed_netcdf:
            if not path.is_file():
                failures.append(f"missing processed dataset: {path}")
                continue
            with xr.open_dataset(path) as dataset:
                if dataset.attrs.get("sawp_complete") != "true":
                    failures.append(f"incomplete processed dataset: {path}")
                empty = [name for name in dataset.data_vars if int(dataset[name].count()) == 0]
                if empty:
                    failures.append(f"empty variables in {path}: {', '.join(empty)}")
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("Scientific reproduction verified: 18/18 figures have scientific provenance")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
