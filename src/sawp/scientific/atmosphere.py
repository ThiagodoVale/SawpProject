from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from dask import compute as dask_compute

from sawp.config import ProjectConfig
from sawp.physics.moisture import moisture_flux_convergence

from .common import (
    GRAVITY,
    anomaly_from_climatology,
    linear_detrend_time,
    open_canonical,
    regional_mean,
    save_dataset,
    seasonal_annual,
    subset_domain,
)

METHOD_VERSION = "legacy-parity-2026-11-07-vector-domain-v4"
CORRELATION_LOW_LEVELS = (850, 875, 900, 925, 1000)
CORRELATION_MID_LEVELS = (400, 500, 600, 700, 850)


@dataclass(frozen=True)
class AtmosphereProducts:
    annual_fields: Path
    regional_series: Path
    trend_coefficients: Path


def _write_regional_series(config: ProjectConfig, annual_fields: Path, destination: Path) -> None:
    domain = config.domains["sawp_index"]
    regional_names = (
        "latent_heat_flux",
        "sensible_heat_flux",
        "wind850_speed",
        "wind1000_speed",
        "omega500",
        "precipitable_water",
        "moisture_flux_convergence",
    )
    with xr.open_dataset(annual_fields) as saved:
        frame = pd.DataFrame(index=saved.year.to_numpy().astype(int))
        frame.index.name = "year"
        for name in regional_names:
            series = regional_mean(
                saved[name],
                west=domain.west,
                east=domain.east,
                south=domain.south,
                north=domain.north,
            )
            frame[name] = series.to_numpy()
    frame.to_csv(destination)


def _annual_cache_is_complete(
    path: Path,
    *,
    required_variables: tuple[str, ...] = (),
) -> bool:
    if not path.is_file():
        return False
    try:
        with xr.open_dataset(path) as dataset:
            return (
                dataset.attrs.get("sawp_complete") == "true"
                and dataset.attrs.get("method_version") == METHOD_VERSION
                and all(name in dataset for name in required_variables)
            )
    except (OSError, ValueError):
        return False


def _annual_anomaly(value: xr.DataArray, config: ProjectConfig) -> xr.DataArray:
    annual = seasonal_annual(value, config.period.season_months)
    anomaly, _ = anomaly_from_climatology(
        annual,
        start=config.period.climatology_start,
        end=config.period.climatology_end,
    )
    return anomaly


def _legacy_cross_latitude_mean(
    value: xr.DataArray,
    *,
    southern_boundary: float,
) -> xr.DataArray:
    """Match the legacy half-open SAWP latitude slice after canonical sorting."""
    selected = value.where(value.lat > southern_boundary, drop=True)
    if selected.sizes.get("lat", 0) == 0:
        raise ValueError("the legacy cross-section latitude slice is empty")
    return selected.mean("lat", skipna=True)


def _apply_monthly_detrend(
    value: xr.DataArray,
    slope: xr.DataArray,
) -> xr.DataArray:
    """Apply a previously calculated compact trend coefficient lazily."""
    slope_chunks = {
        dimension: value.chunksizes[dimension]
        for dimension in slope.dims
        if dimension in value.chunksizes
    }
    # ``dask_compute`` returns the compact coefficient as an in-memory array.
    # Rechunk it before broadcasting over time; otherwise xarray eagerly
    # allocates the complete time-level-lat-lon trend cube (about 3.8 GiB).
    slope = slope.chunk(slope_chunks)
    centered = xr.DataArray(
        np.arange(value.sizes["time"], dtype=np.float64),
        dims="time",
        coords={"time": value.time},
    ).chunk({"time": value.chunksizes["time"]})
    centered = centered - centered.mean("time")
    detrended = value - slope * centered
    detrended.attrs.update(value.attrs)
    return detrended


def _compute(config: ProjectConfig, *values: xr.DataArray) -> tuple[xr.DataArray, ...]:
    """Compute one bounded product group with an explicit worker limit."""
    return dask_compute(
        *values,
        scheduler="threads",
        num_workers=max(1, config.runtime.workers),
    )


def _theta_e_xarray(temperature: xr.DataArray, humidity: xr.DataArray) -> xr.DataArray:
    """Reproduce the legacy dew-point/LCL equivalent-potential-temperature formula."""
    pressure = temperature.level
    partial_pressure = humidity / (1.0 - humidity)
    saturation_pressure = 6.112 * np.exp((17.67 * (temperature - 273.15)) / (temperature - 29.65))
    saturation_mixing_ratio = 0.622 * saturation_pressure / (pressure - saturation_pressure)
    relative_humidity = partial_pressure / saturation_mixing_ratio
    vapor_pressure = relative_humidity * saturation_pressure
    logarithm = np.log(vapor_pressure / 6.112)
    dew_point = 243.5 * logarithm / (17.67 - logarithm) + 273.15
    dew_point_pressure = 6.112 * np.exp((17.67 * (dew_point - 273.15)) / (dew_point - 29.65))
    mixing_ratio = 0.622 * dew_point_pressure / (pressure - dew_point_pressure)
    lcl_temperature = 56.0 + 1.0 / (
        1.0 / (dew_point - 56.0) + np.log(temperature / dew_point) / 800.0
    )
    lcl_potential_temperature = (
        temperature
        * (1000.0 / (pressure - dew_point_pressure)) ** 0.286
        * (temperature / lcl_temperature) ** (0.28 * mixing_ratio)
    )
    theta_e = lcl_potential_temperature * np.exp(
        mixing_ratio * (1.0 + 0.448 * mixing_ratio) * (3036.0 / lcl_temperature - 1.78)
    )
    theta_e.name = "theta_e"
    theta_e.attrs.update(units="K", long_name="equivalent potential temperature")
    return theta_e


def prepare_atmosphere(config: ProjectConfig, *, force: bool = False) -> AtmosphereProducts:
    output = config.paths.processed / "atmosphere"
    output.mkdir(parents=True, exist_ok=True)
    products = AtmosphereProducts(
        annual_fields=output / "atmospheric_amjj_annual.nc",
        regional_series=output / "atmospheric_amjj_series.csv",
        trend_coefficients=output / "atmospheric_trend_coefficients.nc",
    )
    if not force:
        annual_complete = _annual_cache_is_complete(
            products.annual_fields,
            required_variables=(
                "wind1000_speed",
                "correlation_precipitable_water_low",
                "correlation_precipitable_water_mid",
                "correlation_moisture_flux_convergence_low",
            ),
        )
        coefficients_complete = _annual_cache_is_complete(products.trend_coefficients)
        if annual_complete and coefficients_complete and products.regional_series.is_file():
            return products
        if annual_complete and coefficients_complete:
            _write_regional_series(config, products.annual_fields, products.regional_series)
            return products

    chunks_3d = {"time": 60, "lat": 181, "lon": 201}
    # Match the physical ERA5 u/v/w layout. Small time chunks repeatedly decode
    # the same compressed NetCDF blocks and make the first reproduction much slower.
    chunks_4d = {"time": 195, "level": 4, "lat": 46, "lon": 51}
    analysis_bounds = {"west": -60.0, "east": -10.0, "south": -30.0, "north": 15.0}

    heat = open_canonical(
        config.paths.raw_era5 / "Era5_Dataset_meanHF.nc",
        chunks=chunks_3d,
        config=config,
    )
    heat = subset_domain(heat, **analysis_bounds)
    latent_raw = -heat["avg_slhtf"]
    sensible_raw = -heat["avg_ishf"]
    _latent_lazy, latent_slope, latent_mean = linear_detrend_time(latent_raw, assume_complete=True)
    _sensible_lazy, sensible_slope, sensible_mean = linear_detrend_time(
        sensible_raw, assume_complete=True
    )

    uvw = open_canonical(
        config.paths.raw_era5 / "Era5_Dataset_u_v_w.nc",
        chunks=chunks_4d,
        config=config,
    ).sortby("level")
    tq = open_canonical(
        config.paths.raw_era5 / "Era5_Dataset_t_q.nc",
        chunks=chunks_4d,
        config=config,
    ).sortby("level")
    uvw = subset_domain(uvw, **analysis_bounds)
    tq = subset_domain(tq, **analysis_bounds)
    q_raw, u_raw, v_raw, w_raw, t_raw = xr.align(tq.q, uvw.u, uvw.v, uvw.w, tq.t, join="inner")
    _q_lazy, q_slope, q_mean = linear_detrend_time(q_raw, assume_complete=True)
    _u_lazy, u_slope, u_mean = linear_detrend_time(u_raw, assume_complete=True)
    _v_lazy, v_slope, v_mean = linear_detrend_time(v_raw, assume_complete=True)
    _w_lazy, w_slope, w_mean = linear_detrend_time(w_raw, assume_complete=True)
    _temperature_lazy, temperature_slope, temperature_mean = linear_detrend_time(
        t_raw, assume_complete=True
    )

    # Resolve the small coefficient fields first. Downstream products then depend
    # on in-memory coefficients instead of embedding global time reductions in
    # every chunk of a very large Dask graph.
    latent_slope, latent_mean, sensible_slope, sensible_mean = _compute(
        config,
        latent_slope,
        latent_mean,
        sensible_slope,
        sensible_mean,
    )
    q_slope, q_mean, temperature_slope, temperature_mean = _compute(
        config,
        q_slope,
        q_mean,
        temperature_slope,
        temperature_mean,
    )
    (
        u_slope,
        u_mean,
        v_slope,
        v_mean,
        w_slope,
        w_mean,
    ) = _compute(
        config,
        u_slope,
        u_mean,
        v_slope,
        v_mean,
        w_slope,
        w_mean,
    )

    save_dataset(
        xr.Dataset(
            {
                "latent_heat_flux_slope": latent_slope,
                "latent_heat_flux_mean": latent_mean,
                "sensible_heat_flux_slope": sensible_slope,
                "sensible_heat_flux_mean": sensible_mean,
                "specific_humidity_slope": q_slope,
                "specific_humidity_mean": q_mean,
                "zonal_wind_slope": u_slope,
                "zonal_wind_mean": u_mean,
                "meridional_wind_slope": v_slope,
                "meridional_wind_mean": v_mean,
                "vertical_velocity_slope": w_slope,
                "vertical_velocity_mean": w_mean,
                "temperature_slope": temperature_slope,
                "temperature_mean": temperature_mean,
            },
            attrs={
                "method_version": METHOD_VERSION,
                "reconstruction": (
                    "detrended(time) = original(time) - slope * (time_index - mean_time_index)"
                ),
            },
        ),
        products.trend_coefficients,
    )

    latent_monthly = _apply_monthly_detrend(latent_raw, latent_slope)
    sensible_monthly = _apply_monthly_detrend(sensible_raw, sensible_slope)
    q = _apply_monthly_detrend(q_raw, q_slope)
    u = _apply_monthly_detrend(u_raw, u_slope)
    v = _apply_monthly_detrend(v_raw, v_slope)
    w = _apply_monthly_detrend(w_raw, w_slope)
    temperature = _apply_monthly_detrend(t_raw, temperature_slope)
    latent = _annual_anomaly(latent_monthly, config)
    sensible = _annual_anomaly(sensible_monthly, config)

    wind850 = np.hypot(u.sel(level=850), v.sel(level=850))
    wind1000 = np.hypot(u.sel(level=1000), v.sel(level=1000))
    wind850_anomaly = _annual_anomaly(wind850, config)
    wind1000_anomaly = _annual_anomaly(wind1000, config)
    omega500_anomaly = _annual_anomaly(w.sel(level=500), config)

    q_column = q.sel(level=slice(500, 1000))
    precipitable_water = q_column.integrate("level") * 100.0 / GRAVITY
    pw_anomaly = _annual_anomaly(precipitable_water, config)
    pw_low = q.sel(level=slice(850, 1000)).integrate("level") * 100.0 / GRAVITY
    pw_mid = q.sel(level=slice(400, 850)).integrate("level") * 100.0 / GRAVITY
    pw_low_anomaly = _annual_anomaly(pw_low, config)
    pw_mid_anomaly = _annual_anomaly(pw_mid, config)
    correlation_q_low = q.sel(level=list(CORRELATION_LOW_LEVELS)).sortby("level")
    correlation_q_mid = q.sel(level=list(CORRELATION_MID_LEVELS)).sortby("level")
    correlation_pw_low_anomaly = _annual_anomaly(
        correlation_q_low.integrate("level") * 100.0 / GRAVITY, config
    )
    correlation_pw_mid_anomaly = _annual_anomaly(
        correlation_q_mid.integrate("level") * 100.0 / GRAVITY, config
    )
    raw_pw_mid = q_raw.sel(level=slice(400, 850)).integrate("level") * 100.0 / GRAVITY
    raw_pw_mid_anomaly = _annual_anomaly(raw_pw_mid, config)

    q_low = q.sel(level=slice(700, 1000))
    u_low = u.sel(level=slice(700, 1000))
    v_low = v.sel(level=slice(700, 1000))
    transport_u = (q_low * u_low).integrate("level") * 100.0 / GRAVITY
    transport_v = (q_low * v_low).integrate("level") * 100.0 / GRAVITY
    transport_u_anomaly = _annual_anomaly(transport_u, config)
    transport_v_anomaly = _annual_anomaly(transport_v, config)

    q_mid = q.sel(level=slice(400, 850))
    u_mid = u.sel(level=slice(400, 850))
    v_mid = v.sel(level=slice(400, 850))
    transport_u_mid_anomaly = _annual_anomaly(
        (q_mid * u_mid).integrate("level") * 100.0 / GRAVITY, config
    )
    transport_v_mid_anomaly = _annual_anomaly(
        (q_mid * v_mid).integrate("level") * 100.0 / GRAVITY, config
    )
    raw_q_mid = q_raw.sel(level=slice(400, 850))
    raw_u_mid = u_raw.sel(level=slice(400, 850))
    raw_v_mid = v_raw.sel(level=slice(400, 850))
    raw_transport_u_mid_anomaly = _annual_anomaly(
        (raw_q_mid * raw_u_mid).integrate("level") * 100.0 / GRAVITY, config
    )
    raw_transport_v_mid_anomaly = _annual_anomaly(
        (raw_q_mid * raw_v_mid).integrate("level") * 100.0 / GRAVITY, config
    )

    q_boundary = q.sel(level=slice(850, 1000))
    u_boundary = u.sel(level=slice(850, 1000))
    v_boundary = v.sel(level=slice(850, 1000))
    transport_u_boundary_anomaly = _annual_anomaly(
        (q_boundary * u_boundary).integrate("level") * 100.0 / GRAVITY, config
    )
    transport_v_boundary_anomaly = _annual_anomaly(
        (q_boundary * v_boundary).integrate("level") * 100.0 / GRAVITY, config
    )
    raw_q_boundary = q_raw.sel(level=slice(850, 1000))
    raw_u_boundary = u_raw.sel(level=slice(850, 1000))
    raw_v_boundary = v_raw.sel(level=slice(850, 1000))
    raw_transport_u_boundary_anomaly = _annual_anomaly(
        (raw_q_boundary * raw_u_boundary).integrate("level") * 100.0 / GRAVITY,
        config,
    )
    raw_transport_v_boundary_anomaly = _annual_anomaly(
        (raw_q_boundary * raw_v_boundary).integrate("level") * 100.0 / GRAVITY,
        config,
    )
    correlation_u_low = u.sel(level=list(CORRELATION_LOW_LEVELS)).sortby("level")
    correlation_v_low = v.sel(level=list(CORRELATION_LOW_LEVELS)).sortby("level")
    correlation_transport_u_low_anomaly = _annual_anomaly(
        (correlation_q_low * correlation_u_low).integrate("level") * 100.0 / GRAVITY,
        config,
    )
    correlation_transport_v_low_anomaly = _annual_anomaly(
        (correlation_q_low * correlation_v_low).integrate("level") * 100.0 / GRAVITY,
        config,
    )

    cross_domain = config.domains["sawp_index"]
    theta_e = _theta_e_xarray(
        subset_domain(
            temperature,
            west=analysis_bounds["west"],
            east=analysis_bounds["east"],
            south=cross_domain.south,
            north=cross_domain.north,
        ),
        subset_domain(
            q,
            west=analysis_bounds["west"],
            east=analysis_bounds["east"],
            south=cross_domain.south,
            north=cross_domain.north,
        ),
    )
    theta_region = _legacy_cross_latitude_mean(
        subset_domain(
            theta_e,
            west=analysis_bounds["west"],
            east=analysis_bounds["east"],
            south=cross_domain.south,
            north=cross_domain.north,
        ),
        southern_boundary=cross_domain.south,
    )
    u_region = _legacy_cross_latitude_mean(
        subset_domain(
            u,
            west=analysis_bounds["west"],
            east=analysis_bounds["east"],
            south=cross_domain.south,
            north=cross_domain.north,
        ),
        southern_boundary=cross_domain.south,
    )
    w_region = _legacy_cross_latitude_mean(
        subset_domain(
            w,
            west=analysis_bounds["west"],
            east=analysis_bounds["east"],
            south=cross_domain.south,
            north=cross_domain.north,
        ),
        southern_boundary=cross_domain.south,
    )
    theta_cross = _annual_anomaly(theta_region, config)
    u_cross = _annual_anomaly(u_region, config)
    w_cross = _annual_anomaly(w_region, config)

    raw_theta_e = _theta_e_xarray(
        subset_domain(
            t_raw,
            west=analysis_bounds["west"],
            east=analysis_bounds["east"],
            south=cross_domain.south,
            north=cross_domain.north,
        ),
        subset_domain(
            q_raw,
            west=analysis_bounds["west"],
            east=analysis_bounds["east"],
            south=cross_domain.south,
            north=cross_domain.north,
        ),
    )
    raw_theta_region = _legacy_cross_latitude_mean(
        raw_theta_e,
        southern_boundary=cross_domain.south,
    )
    raw_u_region = _legacy_cross_latitude_mean(
        subset_domain(
            u_raw,
            west=analysis_bounds["west"],
            east=analysis_bounds["east"],
            south=cross_domain.south,
            north=cross_domain.north,
        ),
        southern_boundary=cross_domain.south,
    )
    raw_w_region = _legacy_cross_latitude_mean(
        subset_domain(
            w_raw,
            west=analysis_bounds["west"],
            east=analysis_bounds["east"],
            south=cross_domain.south,
            north=cross_domain.north,
        ),
        southern_boundary=cross_domain.south,
    )
    raw_theta_cross = _annual_anomaly(raw_theta_region, config)
    raw_u_cross = _annual_anomaly(raw_u_region, config)
    raw_w_cross = _annual_anomaly(raw_w_region, config)

    latent, sensible = _compute(config, latent, sensible)
    wind850_anomaly, wind1000_anomaly, omega500_anomaly = _compute(
        config, wind850_anomaly, wind1000_anomaly, omega500_anomaly
    )
    pw_anomaly, pw_low_anomaly, pw_mid_anomaly, raw_pw_mid_anomaly = _compute(
        config, pw_anomaly, pw_low_anomaly, pw_mid_anomaly, raw_pw_mid_anomaly
    )
    correlation_pw_low, correlation_pw_mid = _compute(
        config, correlation_pw_low_anomaly, correlation_pw_mid_anomaly
    )
    tu, tv = _compute(config, transport_u_anomaly, transport_v_anomaly)
    tu_mid, tv_mid, raw_tu_mid, raw_tv_mid = _compute(
        config,
        transport_u_mid_anomaly,
        transport_v_mid_anomaly,
        raw_transport_u_mid_anomaly,
        raw_transport_v_mid_anomaly,
    )
    tu_boundary, tv_boundary, raw_tu_boundary, raw_tv_boundary = _compute(
        config,
        transport_u_boundary_anomaly,
        transport_v_boundary_anomaly,
        raw_transport_u_boundary_anomaly,
        raw_transport_v_boundary_anomaly,
    )
    correlation_tu_low, correlation_tv_low = _compute(
        config,
        correlation_transport_u_low_anomaly,
        correlation_transport_v_low_anomaly,
    )
    theta_cross, u_cross, w_cross = _compute(config, theta_cross, u_cross, w_cross)
    raw_theta_cross, raw_u_cross, raw_w_cross = _compute(
        config, raw_theta_cross, raw_u_cross, raw_w_cross
    )

    mfc_values = moisture_flux_convergence(
        tu.to_numpy(),
        tv.to_numpy(),
        tu.lon.to_numpy(),
        tu.lat.to_numpy(),
    )
    mfc = xr.DataArray(
        mfc_values,
        dims=tu.dims,
        coords=tu.coords,
        name="moisture_flux_convergence",
        attrs={"units": "kg m-2 s-1", "positive": "convergence"},
    )
    boundary_mfc_values = moisture_flux_convergence(
        tu_boundary.to_numpy(),
        tv_boundary.to_numpy(),
        tu_boundary.lon.to_numpy(),
        tu_boundary.lat.to_numpy(),
    )
    boundary_mfc = xr.DataArray(
        boundary_mfc_values,
        dims=tu_boundary.dims,
        coords=tu_boundary.coords,
        attrs={"units": "kg m-2 s-1", "positive": "convergence"},
    )
    raw_boundary_mfc_values = moisture_flux_convergence(
        raw_tu_boundary.to_numpy(),
        raw_tv_boundary.to_numpy(),
        raw_tu_boundary.lon.to_numpy(),
        raw_tu_boundary.lat.to_numpy(),
    )
    raw_boundary_mfc = xr.DataArray(
        raw_boundary_mfc_values,
        dims=raw_tu_boundary.dims,
        coords=raw_tu_boundary.coords,
        attrs={"units": "kg m-2 s-1", "positive": "convergence"},
    )
    correlation_mfc_values = moisture_flux_convergence(
        correlation_tu_low.to_numpy(),
        correlation_tv_low.to_numpy(),
        correlation_tu_low.lon.to_numpy(),
        correlation_tu_low.lat.to_numpy(),
    )
    correlation_mfc_low = xr.DataArray(
        correlation_mfc_values,
        dims=correlation_tu_low.dims,
        coords=correlation_tu_low.coords,
        attrs={"units": "kg m-2 s-1", "positive": "convergence"},
    )

    dataset = xr.Dataset(
        {
            "latent_heat_flux": latent,
            "sensible_heat_flux": sensible,
            "wind850_speed": wind850_anomaly,
            "wind1000_speed": wind1000_anomaly,
            "omega500": omega500_anomaly,
            "precipitable_water": pw_anomaly,
            "precipitable_water_low": pw_low_anomaly,
            "precipitable_water_mid": pw_mid_anomaly,
            "raw_precipitable_water_mid": raw_pw_mid_anomaly,
            "correlation_precipitable_water_low": correlation_pw_low,
            "correlation_precipitable_water_mid": correlation_pw_mid,
            "transport_u": tu,
            "transport_v": tv,
            "transport_u_mid": tu_mid,
            "transport_v_mid": tv_mid,
            "raw_transport_u_mid": raw_tu_mid,
            "raw_transport_v_mid": raw_tv_mid,
            "moisture_flux_convergence": mfc,
            "moisture_flux_convergence_low": boundary_mfc,
            "raw_moisture_flux_convergence_low": raw_boundary_mfc,
            "correlation_moisture_flux_convergence_low": correlation_mfc_low,
            "theta_e_cross": theta_cross,
            "u_cross": u_cross,
            "omega_cross": w_cross,
            "raw_theta_e_cross": raw_theta_cross,
            "raw_u_cross": raw_u_cross,
            "raw_omega_cross": raw_w_cross,
        },
        attrs={"method_version": METHOD_VERSION},
    )
    save_dataset(dataset, products.annual_fields)

    _write_regional_series(config, products.annual_fields, products.regional_series)
    heat.close()
    uvw.close()
    tq.close()
    return products
