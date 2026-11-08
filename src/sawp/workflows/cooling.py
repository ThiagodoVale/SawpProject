from __future__ import annotations

import numpy as np
import pandas as pd

from sawp.analysis.anomalies import classify_quantiles, standardize


def monthly_anomaly(series: pd.Series, start: int = 1980, end: int = 2010) -> pd.Series:
    series = series.sort_index().astype(float)
    reference = series[(series.index.year >= start) & (series.index.year <= end)]
    climatology = reference.groupby(reference.index.month).mean()
    return series - series.index.month.map(climatology).to_numpy()


def seasonal_series(
    series: pd.Series,
    *,
    months: tuple[int, ...] = (4, 5, 6, 7),
    reducer: str = "mean",
) -> pd.Series:
    selected = series[series.index.month.isin(months)]
    grouped = selected.groupby(selected.index.year)
    complete = grouped.count() == len(months)
    values = grouped.mean() if reducer == "mean" else grouped.sum(min_count=len(months))
    return values[complete].rename(series.name)


def build_event_table(
    sst_monthly: pd.Series,
    rainfall_monthly: pd.Series,
    *,
    months: tuple[int, ...] = (4, 5, 6, 7),
    climatology: tuple[int, int] = (1980, 2010),
) -> pd.DataFrame:
    sst_anomaly = monthly_anomaly(sst_monthly, *climatology)
    rain_anomaly = monthly_anomaly(rainfall_monthly, *climatology)
    sst_amjj = seasonal_series(sst_anomaly, months=months, reducer="mean")
    rain_amjj = seasonal_series(rain_anomaly, months=months, reducer="sum")
    table = pd.concat(
        [sst_amjj.rename("sst_anomaly"), rain_amjj.rename("rainfall_anomaly")], axis=1
    ).dropna()
    table["sst_standardized"] = standardize(table["sst_anomaly"].to_numpy())
    table["rainfall_standardized"] = standardize(table["rainfall_anomaly"].to_numpy())
    labels, thresholds = classify_quantiles(table["sst_anomaly"].to_numpy())
    table["sst_class"] = labels
    table.attrs["thresholds"] = thresholds
    return table


def cooling_composite(
    monthly_sst_anomaly: pd.Series,
    event_years: list[int] | np.ndarray,
    *,
    anchor_months: pd.Series | None = None,
    lags: tuple[int, ...] = (-2, -1, 0, 1, 2, 3),
) -> pd.DataFrame:
    """Composite monthly SST cooling around a per-year anchor month.

    Cooling is previous-month anomaly minus current-month anomaly, so positive values
    denote ocean cooling, matching the supplementary material convention.
    """
    anomaly = monthly_sst_anomaly.sort_index().astype(float)
    cooling = anomaly.shift(1) - anomaly
    rows = []
    for year in map(int, event_years):
        if anchor_months is None:
            seasonal = anomaly[
                (anomaly.index.year == year) & anomaly.index.month.isin((4, 5, 6, 7))
            ]
            if seasonal.empty:
                continue
            anchor = seasonal.idxmax()
        else:
            month = int(anchor_months.loc[year])
            anchor = pd.Timestamp(year=year, month=month, day=1)
        for lag in lags:
            timestamp = anchor + pd.DateOffset(months=lag)
            rows.append(
                {
                    "year": year,
                    "anchor_month": anchor.month,
                    "lag": lag,
                    "sst_anomaly": anomaly.get(timestamp, np.nan),
                    "sst_cooling": cooling.get(timestamp, np.nan),
                }
            )
    return pd.DataFrame(rows)
