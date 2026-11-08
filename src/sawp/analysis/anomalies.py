from __future__ import annotations

from collections.abc import Iterable
from enum import StrEnum

import numpy as np


class EventClass(StrEnum):
    EXTREME_COLD = "extreme_cold"
    MODERATE_COLD = "moderate_cold"
    NEUTRAL = "neutral"
    MODERATE_WARM = "moderate_warm"
    EXTREME_WARM = "extreme_warm"


def standardize(values: np.ndarray, axis: int | None = None, ddof: int = 1) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    mean = np.nanmean(data, axis=axis, keepdims=axis is not None)
    std = np.nanstd(data, axis=axis, ddof=ddof, keepdims=axis is not None)
    return np.divide(data - mean, std, out=np.full_like(data, np.nan), where=std > 0)


def linear_detrend(
    values: np.ndarray,
    *,
    axis: int = 0,
    x: np.ndarray | None = None,
    preserve_mean: bool = True,
    chunk_size: int = 65_536,
    out: np.ndarray | None = None,
) -> np.ndarray:
    """Remove a least-squares linear trend with independent NaN handling per series.

    The selected time axis is moved to the front and the remaining dimensions are
    processed in chunks. Supplying ``out`` avoids allocating the final result twice.
    """
    source = np.asarray(values)
    moved = np.moveaxis(source, axis, 0)
    n_time = moved.shape[0]
    if n_time < 2:
        raise ValueError("at least two time steps are required")
    coordinate = np.arange(n_time, dtype=float) if x is None else np.asarray(x, dtype=float)
    if coordinate.shape != (n_time,):
        raise ValueError("x must be one-dimensional and match the time axis")
    if not np.all(np.isfinite(coordinate)):
        raise ValueError("x must contain only finite values")

    if out is not None and out.shape != source.shape:
        raise ValueError("out must have the same shape as values")
    result_moved = np.empty(moved.shape, dtype=float)
    flat_source = moved.reshape(n_time, -1)
    flat_result = result_moved.reshape(n_time, -1)
    x_column = coordinate[:, None]

    for start in range(0, flat_source.shape[1], chunk_size):
        stop = min(start + chunk_size, flat_source.shape[1])
        block = flat_source[:, start:stop].astype(float, copy=False)
        valid = np.isfinite(block)
        count = valid.sum(axis=0)
        safe_count = np.where(count > 0, count, 1)
        x_mean = np.where(valid, x_column, 0.0).sum(axis=0) / safe_count
        y_mean = np.where(valid, block, 0.0).sum(axis=0) / safe_count
        x_centered = x_column - x_mean
        y_centered = block - y_mean
        numerator = np.where(valid, x_centered * y_centered, 0.0).sum(axis=0)
        denominator = np.where(valid, x_centered**2, 0.0).sum(axis=0)
        slope = np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=(count >= 2) & (denominator > 0),
        )
        if preserve_mean:
            trend = x_centered * slope
        else:
            trend = x_column * slope + (y_mean - slope * x_mean)
        detrended = block - trend
        detrended[~valid] = np.nan
        flat_result[:, start:stop] = detrended
    restored = np.moveaxis(result_moved, 0, axis)
    if out is None:
        return restored
    out[...] = restored
    return out


def _year_month(times: Iterable[object]) -> tuple[np.ndarray, np.ndarray]:
    dates = np.asarray(list(times), dtype="datetime64[M]")
    months = dates.astype(int) % 12 + 1
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    return years, months


def monthly_anomalies(
    values: np.ndarray,
    times: Iterable[object],
    *,
    climatology_start: int = 1980,
    climatology_end: int = 2010,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.moveaxis(np.asarray(values, dtype=float), axis, 0)
    years, months = _year_month(times)
    if data.shape[0] != years.size:
        raise ValueError("times must match the selected time axis")
    reference = (years >= climatology_start) & (years <= climatology_end)
    if not reference.any():
        raise ValueError("no observations fall inside the climatology period")
    climatology = np.full((12, *data.shape[1:]), np.nan, dtype=float)
    anomalies = np.full_like(data, np.nan, dtype=float)
    for month in range(1, 13):
        month_reference = reference & (months == month)
        if month_reference.any():
            climatology[month - 1] = np.nanmean(data[month_reference], axis=0)
            anomalies[months == month] = data[months == month] - climatology[month - 1]
    return np.moveaxis(anomalies, 0, axis), climatology


def classify_quantiles(
    values: np.ndarray,
    *,
    reference: np.ndarray | None = None,
    quantiles: tuple[float, float, float, float] = (0.10, 0.25, 0.75, 0.90),
) -> tuple[np.ndarray, dict[str, float]]:
    data = np.asarray(values, dtype=float)
    baseline = data if reference is None else np.asarray(reference, dtype=float)
    q10, q25, q75, q90 = np.nanquantile(baseline, quantiles)
    labels = np.full(data.shape, EventClass.NEUTRAL.value, dtype="U16")
    labels[data <= q10] = EventClass.EXTREME_COLD.value
    labels[(data > q10) & (data <= q25)] = EventClass.MODERATE_COLD.value
    labels[(data >= q75) & (data < q90)] = EventClass.MODERATE_WARM.value
    labels[data >= q90] = EventClass.EXTREME_WARM.value
    labels[~np.isfinite(data)] = "missing"
    thresholds = {"q10": float(q10), "q25": float(q25), "q75": float(q75), "q90": float(q90)}
    return labels, thresholds
