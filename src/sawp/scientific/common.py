from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import xarray as xr

from sawp.config import ProjectConfig

from .cache import open_zarr_cache, write_zarr_cache

GRAVITY = 9.80665


def open_canonical(
    path: Path,
    *,
    chunks: dict[str, int] | None = None,
    decode_times: bool = True,
    config: ProjectConfig | None = None,
) -> xr.Dataset:
    """Open a NetCDF file and normalize coordinate names and ordering."""
    if config is not None and config.cache.use_zarr:
        cached = open_zarr_cache(config, path, chunks=chunks)
        if cached is not None:
            return cached
    # ERA5 files use more than one naming convention. Chunk only after the
    # coordinates have canonical names so one request works for every file.
    dataset = xr.open_dataset(path, decode_times=decode_times)
    rename: dict[str, str] = {}
    if "valid_time" in dataset.dims or "valid_time" in dataset.coords:
        rename["valid_time"] = "time"
    if "latitude" in dataset.dims or "latitude" in dataset.coords:
        rename["latitude"] = "lat"
    if "longitude" in dataset.dims or "longitude" in dataset.coords:
        rename["longitude"] = "lon"
    if "pressure_level" in dataset.dims or "pressure_level" in dataset.coords:
        rename["pressure_level"] = "level"
    dataset = dataset.rename(rename)
    if "lat" in dataset.coords and dataset.lat.size > 1 and dataset.lat[0] > dataset.lat[-1]:
        dataset = dataset.sortby("lat")
    if "lon" in dataset.coords:
        longitude = ((dataset.lon + 180.0) % 360.0) - 180.0
        dataset = dataset.assign_coords(lon=longitude).sortby("lon")
    if config is not None and config.cache.use_zarr:
        destination = write_zarr_cache(dataset, path, config)
        dataset.close()
        return xr.open_zarr(destination, consolidated=True, chunks=chunks or {})
    if chunks:
        available = {name: size for name, size in chunks.items() if name in dataset.dims}
        if available:
            dataset = dataset.chunk(available)
    return dataset


def subset_domain(
    value: xr.Dataset | xr.DataArray,
    *,
    west: float,
    east: float,
    south: float,
    north: float,
) -> xr.Dataset | xr.DataArray:
    return value.sel(lon=slice(west, east), lat=slice(south, north))


def seasonal_annual(
    value: xr.DataArray,
    months: Iterable[int] = (4, 5, 6, 7),
    *,
    reducer: str = "mean",
) -> xr.DataArray:
    selected = value.where(value.time.dt.month.isin(tuple(months)), drop=True)
    grouped = selected.groupby("time.year")
    if reducer == "sum":
        return grouped.sum("time", skipna=True, min_count=1)
    if reducer == "mean":
        return grouped.mean("time", skipna=True)
    raise ValueError(f"unsupported reducer: {reducer}")


def anomaly_from_climatology(
    annual: xr.DataArray,
    *,
    start: int = 1980,
    end: int = 2010,
) -> tuple[xr.DataArray, xr.DataArray]:
    climatology = annual.sel(year=slice(start, end)).mean("year", skipna=True)
    return annual - climatology, climatology


def linear_detrend_annual(
    annual: xr.DataArray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Return mean-preserving detrended values, slope per year, and fitted trend."""
    years = annual.year.astype(float)
    centered = years - years.mean()
    valid = annual.notnull()
    weights = centered.where(valid)
    centered_values = annual - annual.mean("year", skipna=True)
    denominator = (weights**2).sum("year", skipna=True)
    slope = (weights * centered_values).sum("year", skipna=True) / denominator
    trend = annual.mean("year", skipna=True) + slope * centered
    detrended = annual - slope * centered
    return detrended, slope, trend


def linear_detrend_time(
    value: xr.DataArray,
    *,
    assume_complete: bool = False,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Remove a monthly linear trend lazily and return detrended data, slope, and mean.

    Only the compact slope and temporal mean need to be persisted. The full detrended
    cube remains a lazy expression backed by NetCDF or the optional Zarr cache.
    """
    index = xr.DataArray(
        np.arange(value.sizes["time"], dtype=np.float64),
        dims="time",
        coords={"time": value.time},
    )
    centered = index - index.mean("time")
    mean = value.mean("time", skipna=True)
    if assume_complete:
        denominator = (centered**2).sum("time")
        slope = (centered * value).sum("time") / denominator
    else:
        valid = value.notnull()
        denominator = (centered.where(valid) ** 2).sum("time", skipna=True)
        slope = (centered.where(valid) * (value - mean)).sum("time", skipna=True) / denominator
    detrended = value - slope * centered
    detrended.attrs.update(value.attrs)
    slope.attrs.update(units=f"{value.attrs.get('units', '')} month-1".strip())
    return detrended, slope, mean


def regional_mean(
    value: xr.DataArray,
    *,
    west: float,
    east: float,
    south: float,
    north: float,
) -> xr.DataArray:
    """Return the unweighted box mean used by the legacy SAWP analyses.

    The publication scripts used ``numpy.nanmean`` over the selected grid cells.
    Keeping the arithmetic mean here is therefore intentional; cosine weighting
    would change the event series and some years close to the quantile limits.
    """
    region = subset_domain(value, west=west, east=east, south=south, north=north)
    return region.mean(("lat", "lon"), skipna=True)


def legacy_indexed_box_mean(
    value: xr.DataArray,
    *,
    west: float,
    east: float,
    south: float,
    north: float,
) -> xr.DataArray:
    """Reproduce the half-open grid slices used by ``ProcessarTsmv2``.

    The legacy coordinate indices were sliced as ``[north:south, west:east]``.
    On the canonical ascending grid this includes west/north and excludes
    east/south: ``west <= lon < east`` and ``south < lat <= north``.
    """
    region = value.where(
        (value.lon >= west)
        & (value.lon < east)
        & (value.lat > south)
        & (value.lat <= north),
        drop=True,
    )
    return region.mean(("lat", "lon"), skipna=True)


def theil_sen_slope(
    value: xr.DataArray,
    *,
    dim: str = "year",
    minimum_count: int = 20,
    chunk_size: int = 512,
) -> xr.DataArray:
    """Calculate the exact Sen median slope along one dimension.

    This is equivalent to the slope returned by ``scipy.stats.theilslopes`` but
    evaluates spatial fields in bounded chunks so Figure S8 does not require a
    large pairwise-slope allocation.
    """
    if dim not in value.dims:
        raise ValueError(f"dimension {dim!r} is not present")
    coordinate = value[dim].to_numpy().astype(float)
    axis = value.get_axis_num(dim)
    moved = np.moveaxis(value.to_numpy(), axis, 0)
    flat = moved.reshape(moved.shape[0], -1)
    first, second = np.triu_indices(coordinate.size, k=1)
    delta = coordinate[second] - coordinate[first]
    result = np.full(flat.shape[1], np.nan, dtype=float)
    for start in range(0, flat.shape[1], chunk_size):
        stop = min(start + chunk_size, flat.shape[1])
        block = flat[:, start:stop]
        valid_count = np.isfinite(block).sum(axis=0)
        slopes = (block[second] - block[first]) / delta[:, None]
        medians = np.nanmedian(slopes, axis=0)
        medians[valid_count < minimum_count] = np.nan
        result[start:stop] = medians
    output_dims = tuple(name for name in value.dims if name != dim)
    output_coords = {name: value.coords[name] for name in output_dims}
    output = xr.DataArray(
        result.reshape(tuple(value.sizes[name] for name in output_dims)),
        dims=output_dims,
        coords=output_coords,
        name=f"{value.name or 'value'}_sen_slope",
    )
    output.attrs.update(value.attrs)
    output.attrs["trend_method"] = "Theil-Sen median pairwise slope"
    return output


def save_dataset(dataset: xr.Dataset, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    completed = dataset.copy(deep=False)
    completed.attrs = {**dataset.attrs, "sawp_complete": "true"}
    encoding = {
        name: {"zlib": True, "complevel": 3, "dtype": "float32"}
        for name, variable in completed.data_vars.items()
        if np.issubdtype(variable.dtype, np.floating)
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    completed.to_netcdf(temporary, encoding=encoding)
    temporary.replace(path)
    return path
