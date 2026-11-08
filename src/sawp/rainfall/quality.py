from __future__ import annotations

import numpy as np
import pandas as pd


def basic_quality_control(
    monthly: pd.DataFrame,
    *,
    minimum_mm: float = 0.0,
    maximum_mm: float = 2000.0,
) -> pd.DataFrame:
    values = monthly.apply(pd.to_numeric, errors="coerce").astype(float)
    return values.mask((values < minimum_mm) | (values > maximum_mm))


def spatial_outlier_control(
    monthly: pd.DataFrame,
    *,
    absolute_threshold_mm: float = 500.0,
    median_factor: float = 5.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cleaned = monthly.copy()
    regional_median = cleaned.median(axis=1, skipna=True)
    threshold = np.maximum(absolute_threshold_mm, median_factor * regional_median)
    flagged = (
        cleaned.gt(threshold, axis=0)
        & cleaned.gt(absolute_threshold_mm)
        & regional_median.gt(0).to_numpy()[:, None]
    )
    rows, columns = np.where(flagged.to_numpy())
    removed = pd.DataFrame(
        {
            "time": cleaned.index.take(rows),
            "station": cleaned.columns.take(columns),
            "precipitation_mm": cleaned.to_numpy()[rows, columns],
            "regional_median_mm": regional_median.to_numpy()[rows],
        }
    )
    return cleaned.mask(flagged), removed


def regional_monthly_series(
    monthly: pd.DataFrame,
    *,
    minimum_valid_stations: int = 30,
    minimum_valid_fraction: float = 0.10,
) -> pd.DataFrame:
    n_total = monthly.shape[1]
    required = max(minimum_valid_stations, int(np.ceil(minimum_valid_fraction * n_total)))
    n_valid = monthly.notna().sum(axis=1)
    regional = monthly.mean(axis=1, skipna=True).where(n_valid >= required)
    return pd.DataFrame(
        {
            "precipitation_mm": regional,
            "n_valid_stations": n_valid,
            "n_total_stations": n_total,
            "coverage_fraction": n_valid / n_total,
            "required_stations": required,
        },
        index=monthly.index,
    )


def station_completeness(monthly: pd.DataFrame) -> pd.Series:
    return monthly.notna().mean(axis=0).rename("completeness")


def station_homogeneity(
    monthly: pd.DataFrame,
    metadata: pd.DataFrame,
    *,
    latitude_column: str = "latitude",
    longitude_column: str = "longitude",
    nearest_neighbors: int = 10,
    minimum_common_months: int = 60,
) -> pd.DataFrame:
    from scipy.spatial import cKDTree
    from scipy.stats import pearsonr

    stations = [station for station in monthly.columns if station in metadata.index]
    if len(stations) <= nearest_neighbors:
        raise ValueError("not enough stations for the requested neighborhood")
    coordinates = metadata.loc[stations, [longitude_column, latitude_column]].to_numpy(float)
    tree = cKDTree(coordinates)
    rows = []
    for position, station in enumerate(stations):
        _, neighbor_indices = tree.query(coordinates[position], k=nearest_neighbors + 1)
        neighbors = [stations[index] for index in np.atleast_1d(neighbor_indices)[1:]]
        station_series = monthly[station]
        neighbor_mean = monthly[neighbors].mean(axis=1, skipna=True)
        pair = pd.concat([station_series, neighbor_mean], axis=1).dropna()
        if (
            len(pair) >= minimum_common_months
            and pair.iloc[:, 0].std() > 0
            and pair.iloc[:, 1].std() > 0
        ):
            correlation, p_value = pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
        else:
            correlation, p_value = np.nan, np.nan
        annual_mean = station_series.groupby(station_series.index.year).sum(min_count=1).mean()
        monthly_cycle = station_series.groupby(station_series.index.month).mean()
        cycle_mean = monthly_cycle.mean()
        rows.append(
            {
                "station": station,
                "latitude": metadata.loc[station, latitude_column],
                "longitude": metadata.loc[station, longitude_column],
                "neighbor_correlation": correlation,
                "p_value": p_value,
                "n_common_months": len(pair),
                "annual_mean_mm": annual_mean,
                "peak_month": monthly_cycle.idxmax(),
                "seasonality_ratio": (monthly_cycle.max() / cycle_mean if cycle_mean else np.nan),
                "neighbors": ",".join(neighbors),
            }
        )
    result = pd.DataFrame(rows).set_index("station")
    lower = result["annual_mean_mm"].quantile(0.01)
    upper = result["annual_mean_mm"].quantile(0.99)
    result["flag_annual_extreme"] = ~result["annual_mean_mm"].between(lower, upper)
    result["flag_low_neighbor_corr"] = result["neighbor_correlation"] < 0.30
    return result
