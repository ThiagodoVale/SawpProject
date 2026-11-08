from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

import numpy as np
import pandas as pd

from sawp.analysis.anomalies import EventClass, classify_quantiles
from sawp.analysis.composites import seasonal_reduce
from sawp.domains import Domain
from sawp.io import inspect_npz, load_npz, open_npz_memmap


class FluxKind(StrEnum):
    LATENT = "latent"
    SENSIBLE = "sensible"


@dataclass(frozen=True)
class FluxCompositeSpec:
    kind: FluxKind
    variable: str
    lag_months: int = 0
    positive_upward: bool = True
    months: tuple[int, ...] = (4, 5, 6, 7)
    climatology: tuple[int, int] = (1980, 2010)


def _regional_mean(
    data: np.ndarray, lon: np.ndarray, lat: np.ndarray, domain: Domain
) -> np.ndarray:
    mask = domain.mask(lon, lat)
    return np.nanmean(data[..., mask], axis=-1)


def classify_sst_years(
    sst_npz: Path,
    domain: Domain,
    *,
    months: tuple[int, ...] = (4, 5, 6, 7),
) -> pd.DataFrame:
    members = inspect_npz(sst_npz)
    metadata = load_npz(
        sst_npz,
        required=("lat", "lon"),
        allow_pickle=True,
        members=tuple(key for key in ("lat", "lon", "time", "valid_time") if key in members),
    )
    time_key = "time" if "time" in metadata else "valid_time"
    data = open_npz_memmap(sst_npz, "data")
    regional = _regional_mean(data, metadata["lon"], metadata["lat"], domain)
    seasonal, years = seasonal_reduce(regional, metadata[time_key], months=months)
    labels, thresholds = classify_quantiles(seasonal)
    frame = pd.DataFrame({"year": years, "sst_amjj": seasonal, "sst_class": labels}).set_index(
        "year"
    )
    frame.attrs["thresholds"] = thresholds
    data._mmap.close()
    return frame


def compute_flux_composites(
    flux_dataset: Path,
    classification: pd.DataFrame,
    spec: FluxCompositeSpec,
    *,
    output_npz: Path | None = None,
) -> dict[str, np.ndarray]:
    """Compute one flux composite family for a variable and one explicit lag."""
    import xarray as xr

    with xr.open_dataset(flux_dataset, chunks={"time": 12}) as dataset:
        variable = dataset[spec.variable]
        time_name = "time" if "time" in variable.dims else "valid_time"
        sign = -1.0 if spec.positive_upward else 1.0
        variable = sign * variable
        shifted = variable.assign_coords(
            {time_name: variable[time_name].to_index() - pd.DateOffset(months=spec.lag_months)}
        )
        selected = shifted.where(shifted[time_name].dt.month.isin(spec.months), drop=True)
        annual = selected.groupby(f"{time_name}.year").mean(time_name)
        composites = {}
        for event_class in EventClass:
            years = classification.index[
                classification["sst_class"] == event_class.value
            ].to_numpy()
            if years.size:
                composites[event_class.value] = (
                    annual.sel(year=years).mean("year").compute().to_numpy()
                )
        composites["latitude"] = dataset["latitude"].to_numpy()
        composites["longitude"] = dataset["longitude"].to_numpy()
    if output_npz is not None:
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output_npz, **composites)
    return composites
