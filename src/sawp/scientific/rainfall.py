from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from sawp.config import ProjectConfig
from sawp.rainfall import (
    basic_quality_control,
    build_station_matrix,
    regional_monthly_series,
    spatial_outlier_control,
    station_completeness,
    station_homogeneity,
)


@dataclass(frozen=True)
class RainfallProducts:
    matrix: Path
    metadata: Path
    regional: Path
    homogeneity: Path
    robustness: Path
    annual: Path


def _first_valid_year(series: pd.Series) -> float:
    valid = series.dropna()
    return float(valid.index.year.min()) if not valid.empty else np.nan


def prepare_rainfall(config: ProjectConfig, *, force: bool = False) -> RainfallProducts:
    output = config.paths.processed / "rainfall"
    output.mkdir(parents=True, exist_ok=True)
    products = RainfallProducts(
        matrix=output / "rain_gauge_monthly_matrix_qc.csv",
        metadata=output / "rain_gauge_metadata_qc.csv",
        regional=output / "ENEB_monthly_precipitation_qc.csv",
        homogeneity=output / "rain_gauge_homogeneity.csv",
        robustness=output / "rainfall_robustness_series.csv",
        annual=output / "ENEB_AMJJ_annual.csv",
    )
    if not force and all(path.is_file() for path in products.__dict__.values()):
        return products

    domain = config.domains["eneb"]
    matrix, metadata = build_station_matrix(
        config.paths.raw_rainfall / "monitor",
        west=domain.west,
        east=domain.east,
        south=domain.south,
        north=domain.north,
        states=config.rainfall_qc.states,
    )
    matrix = matrix.loc["1959-01-01":"2023-12-01"]
    cleaned = basic_quality_control(
        matrix,
        minimum_mm=config.rainfall_qc.minimum_mm,
        maximum_mm=config.rainfall_qc.maximum_mm,
    )
    cleaned, removed = spatial_outlier_control(
        cleaned,
        absolute_threshold_mm=config.rainfall_qc.spatial_outlier_mm,
        median_factor=config.rainfall_qc.spatial_outlier_median_factor,
    )
    useful = cleaned.notna().any(axis=0)
    cleaned = cleaned.loc[:, useful]
    metadata = metadata.loc[metadata.index.intersection(cleaned.columns)].copy()

    homogeneity = station_homogeneity(
        cleaned,
        metadata,
        nearest_neighbors=config.rainfall_qc.nearest_neighbors,
        minimum_common_months=config.rainfall_qc.minimum_common_months,
    )
    homogeneity["completeness"] = station_completeness(cleaned)
    homogeneity["first_year"] = cleaned.apply(_first_valid_year)
    homogeneity["flag_unusual_peak_month"] = ~homogeneity["peak_month"].isin(
        config.period.season_months
    )
    homogeneity["flag_potentially_inconsistent"] = (
        homogeneity["flag_annual_extreme"]
        | homogeneity["flag_low_neighbor_corr"]
        | homogeneity["flag_unusual_peak_month"]
    )

    # The published regional series excludes only stations whose climatological
    # rainfall maximum falls outside AMJJ. Other diagnostics remain in the audit
    # table but do not remove a station.
    accepted = homogeneity.index[~homogeneity["flag_unusual_peak_month"]]
    cleaned = cleaned.loc[:, accepted]
    metadata = metadata.loc[accepted].copy()

    regional = regional_monthly_series(
        cleaned,
        minimum_valid_stations=config.rainfall_qc.minimum_valid_stations,
        minimum_valid_fraction=config.rainfall_qc.minimum_valid_fraction,
    )
    completeness = homogeneity.loc[accepted, "completeness"]
    first_year = homogeneity.loc[accepted, "first_year"]
    robustness = pd.DataFrame(index=cleaned.index)
    robustness["all"] = cleaned.mean(axis=1, skipna=True)
    for threshold in (0.50, 0.70, 0.90):
        stations = completeness.index[completeness >= threshold]
        robustness[f"completeness_{int(threshold * 100)}"] = cleaned[stations].mean(
            axis=1, skipna=True
        )
    for year in (1970, 1965, 1960):
        stations = first_year.index[first_year <= year]
        robustness[f"start_{year}"] = cleaned[stations].mean(axis=1, skipna=True)

    amjj = regional.loc[regional.index.month.isin(config.period.season_months)]
    annual = amjj["precipitation_mm"].groupby(amjj.index.year).sum(min_count=4)
    years = annual.index.to_numpy(float)
    valid = np.isfinite(annual.to_numpy(float))
    slope, intercept = np.polyfit(years[valid], annual.to_numpy(float)[valid], 1)
    annual_frame = pd.DataFrame(
        {
            "rainfall_mm": annual,
            "rainfall_detrended_mm": annual - slope * (annual.index - years.mean()),
            "linear_trend_mm": intercept + slope * annual.index,
        }
    )

    cleaned.to_csv(products.matrix)
    metadata.join(homogeneity.drop(columns=metadata.columns, errors="ignore")).to_csv(
        products.metadata
    )
    regional.to_csv(products.regional)
    homogeneity.to_csv(products.homogeneity)
    robustness.to_csv(products.robustness)
    annual_frame.to_csv(products.annual, index_label="year")
    removed.to_csv(output / "removed_outliers_qc.csv", index=False)
    return products
