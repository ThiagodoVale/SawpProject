from __future__ import annotations

import argparse
from pathlib import Path

import dask.array as da
import xarray as xr

from sawp.io import inspect_npz, load_npz, open_npz_memmap


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate a legacy uncompressed SAWP NPZ cube to chunked Zarr"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--variable", default="data")
    parser.add_argument("--time-chunk", type=int, default=12)
    parser.add_argument("--level-chunk", type=int, default=1)
    parser.add_argument("--spatial-chunk", type=int, default=90)
    args = parser.parse_args()

    data = open_npz_memmap(args.source, args.variable)
    member_names = set(inspect_npz(args.source))
    time_key = "time" if "time" in member_names else "valid_time"
    metadata_members = [time_key, "lat", "lon"]
    if data.ndim == 4:
        metadata_members.append("level")
    metadata = load_npz(
        args.source,
        required=tuple(metadata_members),
        allow_pickle=True,
        members=tuple(metadata_members),
    )
    if data.ndim == 3:
        dims = ("time", "latitude", "longitude")
        chunks = (args.time_chunk, args.spatial_chunk, args.spatial_chunk)
    elif data.ndim == 4:
        dims = ("time", "pressure_level", "latitude", "longitude")
        chunks = (
            args.time_chunk,
            args.level_chunk,
            args.spatial_chunk,
            args.spatial_chunk,
        )
    else:
        raise ValueError(f"unsupported data rank: {data.ndim}")
    coordinates = {
        "time": metadata[time_key],
        "latitude": metadata["lat"],
        "longitude": metadata["lon"],
    }
    if data.ndim == 4:
        coordinates["pressure_level"] = metadata["level"]
    array = xr.DataArray(
        da.from_array(data, chunks=chunks, asarray=False),
        dims=dims,
        coords=coordinates,
        name=args.variable,
    )
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    array.to_dataset().to_zarr(args.destination, mode="w", consolidated=True)
    print(args.destination)


if __name__ == "__main__":
    main()
