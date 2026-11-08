from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import xarray as xr

from sawp.analysis.anomalies import classify_quantiles
from sawp.config import ProjectConfig

from .common import (
    anomaly_from_climatology,
    legacy_indexed_box_mean,
    linear_detrend_annual,
    linear_detrend_time,
    open_canonical,
    save_dataset,
    seasonal_annual,
)

METHOD_VERSION = "legacy-parity-2026-11-07"


@dataclass(frozen=True)
class OceanProducts:
    annual_fields: Path
    composites: Path
    series: Path
    monthly_series: Path
    trend_coefficients: Path


def prepare_ocean(config: ProjectConfig, *, force: bool = False) -> OceanProducts:
    output = config.paths.processed / "ocean"
    output.mkdir(parents=True, exist_ok=True)
    products = OceanProducts(
        annual_fields=output / "sst_amjj_annual.nc",
        composites=output / "sst_quantile_composites.nc",
        series=output / "sawp_amjj_series.csv",
        monthly_series=output / "sawp_monthly_series.csv",
        trend_coefficients=output / "sst_trend_coefficients.nc",
    )
    if not force and all(path.is_file() for path in products.__dict__.values()):
        try:
            with xr.open_dataset(products.annual_fields) as cached:
                if cached.attrs.get("method_version") == METHOD_VERSION:
                    return products
        except (OSError, ValueError):
            pass

    source = config.paths.raw_era5 / "Era5_Dataset_sst.nc"
    dataset = open_canonical(source, chunks={"time": 60}, config=config)
    sst_c = dataset["sst"] - 273.15
    detrended_monthly, monthly_slope, monthly_mean = linear_detrend_time(sst_c)
    reference = sst_c.sel(
        time=slice(
            f"{config.period.climatology_start}-01-01",
            f"{config.period.climatology_end}-12-31",
        )
    )
    detrended_reference = detrended_monthly.sel(time=reference.time)
    monthly_climatology = reference.groupby("time.month").mean("time", skipna=True)
    detrended_climatology = detrended_reference.groupby("time.month").mean("time", skipna=True)
    monthly_anomaly = sst_c.groupby("time.month") - monthly_climatology
    detrended_monthly_anomaly = detrended_monthly.groupby("time.month") - detrended_climatology
    anomaly = seasonal_annual(monthly_anomaly, config.period.season_months)
    detrended = seasonal_annual(detrended_monthly_anomaly, config.period.season_months)
    annual = seasonal_annual(sst_c, config.period.season_months)
    _unused, annual_slope, _annual_trend = linear_detrend_annual(anomaly)
    _annual_anomaly, climatology = anomaly_from_climatology(
        annual,
        start=config.period.climatology_start,
        end=config.period.climatology_end,
    )
    domain = config.domains["sawp_index"]
    regional = legacy_indexed_box_mean(
        anomaly,
        west=domain.west,
        east=domain.east,
        south=domain.south,
        north=domain.north,
    ).compute()
    regional_detrended = legacy_indexed_box_mean(
        detrended,
        west=domain.west,
        east=domain.east,
        south=domain.south,
        north=domain.north,
    ).compute()
    reference_values = regional_detrended.sel(
        year=slice(config.period.climatology_start, config.period.climatology_end)
    ).to_numpy()
    labels, thresholds = classify_quantiles(
        regional_detrended.to_numpy(), reference=reference_values
    )
    raw_reference_values = regional.sel(
        year=slice(config.period.climatology_start, config.period.climatology_end)
    ).to_numpy()
    raw_labels, _raw_thresholds = classify_quantiles(
        regional.to_numpy(), reference=raw_reference_values
    )
    years = annual.year.to_numpy().astype(int)
    series = pd.DataFrame(
        {
            "year": years,
            "sst_anomaly_c": regional.to_numpy(),
            "sst_detrended_c": regional_detrended.to_numpy(),
            "sst_class": labels,
            "sst_raw_class": raw_labels,
        }
    ).set_index("year")
    for name, value in thresholds.items():
        series.attrs[name] = value
    series.to_csv(products.series)

    regional_monthly = legacy_indexed_box_mean(
        sst_c,
        west=domain.west,
        east=domain.east,
        south=domain.south,
        north=domain.north,
    ).compute()
    regional_monthly_anomaly = legacy_indexed_box_mean(
        monthly_anomaly,
        west=domain.west,
        east=domain.east,
        south=domain.south,
        north=domain.north,
    ).compute()
    regional_monthly_detrended_absolute = legacy_indexed_box_mean(
        detrended_monthly,
        west=domain.west,
        east=domain.east,
        south=domain.south,
        north=domain.north,
    ).compute()
    regional_monthly_detrended = legacy_indexed_box_mean(
        detrended_monthly_anomaly,
        west=domain.west,
        east=domain.east,
        south=domain.south,
        north=domain.north,
    ).compute()
    pd.DataFrame(
        {
            "time": pd.to_datetime(regional_monthly.time.to_numpy()),
            "sst_c": regional_monthly.to_numpy(),
            "sst_detrended_c": regional_monthly_detrended_absolute.to_numpy(),
            "sst_anomaly_c": regional_monthly_anomaly.to_numpy(),
            "sst_detrended_anomaly_c": regional_monthly_detrended.to_numpy(),
        }
    ).set_index("time").to_csv(products.monthly_series)

    composite_variables: dict[str, xr.DataArray] = {}
    for event_class in sorted(set(labels) - {"missing"}):
        event_years = years[labels == event_class]
        if event_years.size:
            composite_variables[f"raw_{event_class}"] = anomaly.sel(year=event_years).mean("year")
            composite_variables[f"detrended_{event_class}"] = detrended.sel(year=event_years).mean(
                "year"
            )
    save_dataset(
        xr.Dataset(
            composite_variables,
            attrs={
                **{key: float(value) for key, value in thresholds.items()},
                "method_version": METHOD_VERSION,
                "sawp_box_convention": "west/north inclusive; east/south exclusive",
            },
        ),
        products.composites,
    )
    save_dataset(
        xr.Dataset(
            {
                "sst_anomaly": anomaly,
                "sst_detrended": detrended,
                "sst_slope_per_year": annual_slope,
                "sst_climatology": climatology,
            },
            attrs={"method_version": METHOD_VERSION},
        ),
        products.annual_fields,
    )
    save_dataset(
        xr.Dataset(
            {
                "sst_slope_per_month": monthly_slope,
                "sst_temporal_mean": monthly_mean,
                "sst_monthly_climatology": monthly_climatology,
                "sst_detrended_monthly_climatology": detrended_climatology,
            },
            attrs={
                "reconstruction": (
                    "detrended(time) = original(time) - slope_per_month * "
                    "(time_index - mean_time_index)"
                )
            },
        ),
        products.trend_coefficients,
    )
    dataset.close()
    return products
