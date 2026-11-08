from __future__ import annotations

from pathlib import Path


def detrend_netcdf_variable(
    source: Path,
    destination: Path,
    variable: str,
    *,
    time_dimension: str | None = None,
    chunks: dict[str, int] | None = None,
) -> Path:
    """Detrend one NetCDF variable lazily and write exactly one NetCDF product."""
    import xarray as xr

    with xr.open_dataset(source, chunks=chunks or {}) as dataset:
        data = dataset[variable]
        time_name = time_dimension or ("time" if "time" in data.dims else "valid_time")
        coefficients = data.polyfit(dim=time_name, deg=1, skipna=True)
        fitted = xr.polyval(data[time_name], coefficients.polyfit_coefficients)
        detrended = data - fitted + fitted.mean(time_name)
        detrended.attrs.update(data.attrs)
        detrended.attrs["processing"] = (
            "least-squares linear trend removed; temporal mean preserved"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        detrended.to_dataset(name=variable).to_netcdf(destination)
    return destination
