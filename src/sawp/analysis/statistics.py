from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegressionResult:
    slope: float
    intercept: float
    r: float
    p: float
    stderr: float
    n: int


def linear_regression(x: np.ndarray, y: np.ndarray) -> RegressionResult:
    from scipy.stats import linregress

    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    if valid.sum() < 3:
        raise ValueError("at least three paired finite observations are required")
    result = linregress(x_values[valid], y_values[valid])
    return RegressionResult(
        slope=float(result.slope),
        intercept=float(result.intercept),
        r=float(result.rvalue),
        p=float(result.pvalue),
        stderr=float(result.stderr),
        n=int(valid.sum()),
    )


def pearson_map(
    field: np.ndarray,
    series: np.ndarray,
    *,
    axis: int = 0,
    minimum_pairs: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pearson correlation and p-value between a time series and every grid point."""
    from scipy.stats import t as student_t

    data = np.moveaxis(np.asarray(field, dtype=float), axis, 0)
    target = np.asarray(series, dtype=float)
    if data.shape[0] != target.size:
        raise ValueError("series must match the selected field axis")
    flat = data.reshape(data.shape[0], -1)
    valid = np.isfinite(flat) & np.isfinite(target[:, None])
    n = valid.sum(axis=0)
    safe_n = np.where(n > 0, n, 1)
    x_mean = np.where(valid, flat, 0.0).sum(axis=0) / safe_n
    y_mean = np.where(valid, target[:, None], 0.0).sum(axis=0) / safe_n
    x_centered = flat - x_mean
    y_centered = target[:, None] - y_mean
    covariance = np.where(valid, x_centered * y_centered, 0.0).sum(axis=0)
    x_ss = np.where(valid, x_centered**2, 0.0).sum(axis=0)
    y_ss = np.where(valid, y_centered**2, 0.0).sum(axis=0)
    denominator = np.sqrt(x_ss * y_ss)
    r = np.divide(
        covariance, denominator, out=np.full_like(covariance, np.nan), where=denominator > 0
    )
    r = np.clip(r, -1.0, 1.0)
    degrees = n - 2
    t_stat = np.abs(r) * np.sqrt(
        np.divide(degrees, 1.0 - r**2, out=np.full_like(r, np.inf), where=np.abs(r) < 1)
    )
    p = 2.0 * student_t.sf(t_stat, degrees)
    invalid = n < minimum_pairs
    r[invalid] = np.nan
    p[invalid] = np.nan
    shape = data.shape[1:]
    return r.reshape(shape), p.reshape(shape), n.reshape(shape)
