from __future__ import annotations

import json
from pathlib import Path

import xarray as xr
from numcodecs import Blosc

from sawp.config import ProjectConfig


def cache_directory(config: ProjectConfig) -> Path:
    configured = Path(config.cache.zarr_directory)
    if configured.is_absolute():
        return configured
    return (config.paths.workspace / configured).resolve()


def _chunks(config: ProjectConfig, dataset: xr.Dataset) -> dict[str, int]:
    requested = {
        "time": config.cache.time_chunk,
        "level": config.cache.level_chunk,
        "lat": config.cache.latitude_chunk,
        "lon": config.cache.longitude_chunk,
    }
    return {
        name: min(size, dataset.sizes[name])
        for name, size in requested.items()
        if name in dataset.dims
    }


def zarr_path(config: ProjectConfig, source: Path) -> Path:
    return cache_directory(config) / f"{source.stem}.zarr"


def write_zarr_cache(
    dataset: xr.Dataset,
    source: Path,
    config: ProjectConfig,
    *,
    force: bool = False,
) -> Path:
    destination = zarr_path(config, source)
    marker = destination / ".sawp-complete.json"
    if marker.is_file() and not force and _cache_is_current(marker, source):
        return destination
    if destination.exists():
        import shutil

        shutil.rmtree(destination)
    chunks = _chunks(config, dataset)
    chunked = dataset.chunk(chunks)
    compressor = Blosc(
        cname="zstd",
        clevel=config.cache.compression_level,
        shuffle=Blosc.BITSHUFFLE,
    )
    encoding = {
        name: {
            "compressor": compressor,
            "chunks": tuple(
                chunks.get(dim, size)
                for dim, size in zip(variable.dims, variable.shape, strict=True)
            ),
        }
        for name, variable in chunked.data_vars.items()
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    chunked.to_zarr(
        destination,
        mode="w",
        encoding=encoding,
        consolidated=True,
        zarr_version=2,
    )
    source_stat = source.stat()
    marker.write_text(
        json.dumps(
            {
                "source": str(source.resolve()),
                "source_size": source_stat.st_size,
                "source_mtime_ns": source_stat.st_mtime_ns,
                "chunks": chunks,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return destination


def _cache_is_current(marker: Path, source: Path) -> bool:
    try:
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        source_stat = source.stat()
    except (OSError, ValueError, TypeError):
        return False
    return (
        metadata.get("source") == str(source.resolve())
        and metadata.get("source_size") == source_stat.st_size
        and metadata.get("source_mtime_ns") == source_stat.st_mtime_ns
    )


def open_zarr_cache(
    config: ProjectConfig,
    source: Path,
    *,
    chunks: dict[str, int] | None = None,
) -> xr.Dataset | None:
    destination = zarr_path(config, source)
    marker = destination / ".sawp-complete.json"
    if not marker.is_file() or not _cache_is_current(marker, source):
        return None
    return xr.open_zarr(destination, consolidated=True, chunks=chunks or {})
