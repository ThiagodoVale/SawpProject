from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import xarray as xr

from sawp.config import load_config
from sawp.scientific.cache import open_zarr_cache, write_zarr_cache


def test_zarr_cache_roundtrip_and_source_invalidation(tmp_path: Path) -> None:
    source = tmp_path / "sample.nc"
    dataset = xr.Dataset(
        {"value": (("time", "lat", "lon"), np.arange(24).reshape(2, 3, 4))},
        coords={"time": range(2), "lat": range(3), "lon": range(4)},
    )
    dataset.to_netcdf(source)
    config = load_config()
    config = replace(
        config,
        cache=replace(
            config.cache,
            use_zarr=True,
            zarr_directory=str(tmp_path / "zarr"),
            time_chunk=1,
            latitude_chunk=2,
            longitude_chunk=2,
        ),
    )

    write_zarr_cache(dataset, source, config)
    cached = open_zarr_cache(config, source)
    assert cached is not None
    xr.testing.assert_equal(cached.compute(), dataset)
    cached.close()

    source.touch()
    assert open_zarr_cache(config, source) is None
