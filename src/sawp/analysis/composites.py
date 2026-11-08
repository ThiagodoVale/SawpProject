from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


def _calendar(times: Iterable[object]) -> tuple[np.ndarray, np.ndarray]:
    dates = np.asarray(list(times), dtype="datetime64[M]")
    return dates.astype("datetime64[Y]").astype(int) + 1970, dates.astype(int) % 12 + 1


def seasonal_reduce(
    values: np.ndarray,
    times: Iterable[object],
    *,
    months: tuple[int, ...] = (4, 5, 6, 7),
    reducer: str = "mean",
    require_complete: bool = True,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    data = np.moveaxis(np.asarray(values, dtype=float), axis, 0)
    years, calendar_months = _calendar(times)
    if data.shape[0] != years.size:
        raise ValueError("times must match the selected time axis")
    selected = np.isin(calendar_months, months)
    output_years = np.unique(years[selected])
    reduced = []
    valid_years = []
    for year in output_years:
        mask = selected & (years == year)
        present = set(calendar_months[mask].tolist())
        if require_complete and present != set(months):
            continue
        block = data[mask]
        if reducer == "mean":
            result = np.nanmean(block, axis=0)
        elif reducer == "sum":
            result = np.nansum(block, axis=0)
        elif reducer == "max":
            result = np.nanmax(block, axis=0)
        else:
            raise ValueError("reducer must be mean, sum, or max")
        reduced.append(result)
        valid_years.append(year)
    return np.asarray(reduced), np.asarray(valid_years, dtype=int)


def composite_by_class(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    axis: int = 0,
    classes: Iterable[str] | None = None,
) -> Mapping[str, np.ndarray]:
    data = np.moveaxis(np.asarray(values, dtype=float), axis, 0)
    labels = np.asarray(labels)
    if data.shape[0] != labels.size:
        raise ValueError("labels must match the composite axis")
    selected_classes = list(classes) if classes is not None else sorted(set(labels) - {"missing"})
    return {
        name: np.nanmean(data[labels == name], axis=0)
        for name in selected_classes
        if np.any(labels == name)
    }
