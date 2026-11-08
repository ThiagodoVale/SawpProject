from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StationRecord:
    station_id: str
    name: str
    longitude: float
    latitude: float
    precipitation: pd.Series
    source: Path


def _header(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    with path.open("r", encoding="utf-8", errors="replace") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            key, _, value = line[1:].partition(":")
            result[key.strip().lower()] = value.strip()
    return result


def read_station(path: str | Path) -> StationRecord:
    source = Path(path)
    header = _header(source)
    table = pd.read_csv(
        source,
        comment="#",
        sep=r"\s+",
        names=("year", "month", "day", "precipitation_mm"),
        usecols=(0, 1, 2, 3),
        na_values=("nan", "NaN", "-999", "-999.0"),
    )
    dates = pd.to_datetime(
        {"year": table["year"], "month": table["month"], "day": 1}, errors="coerce"
    )
    values = pd.to_numeric(table["precipitation_mm"], errors="coerce")
    series = pd.Series(values.to_numpy(), index=dates, name=source.stem)
    series = series[~series.index.isna()].groupby(level=0).mean().sort_index()
    station_id = header.get("code", source.stem)
    return StationRecord(
        station_id=str(station_id),
        name=header.get("name", source.stem),
        longitude=float(header.get("x", np.nan)),
        latitude=float(header.get("y", np.nan)),
        precipitation=series,
        source=source,
    )


def build_station_matrix(
    directory: str | Path,
    *,
    west: float = -38.0,
    east: float = -34.5,
    south: float = -15.0,
    north: float = 0.0,
    states: tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    root = Path(directory)
    paths = (
        (path for state in states for path in sorted((root / state).glob("*.txt")))
        if states is not None
        else root.rglob("*.txt")
    )
    for path in sorted(paths):
        try:
            record = read_station(path)
        except (OSError, ValueError, pd.errors.ParserError):
            continue
        if not np.isfinite(record.longitude) or not np.isfinite(record.latitude):
            continue
        if west <= record.longitude <= east and south <= record.latitude <= north:
            records.append(record)
    if not records:
        raise ValueError(f"no valid stations found in {directory}")
    columns: dict[str, pd.Series] = {}
    metadata_rows = []
    seen: dict[str, int] = {}
    for record in records:
        base = re.sub(r"\s+", "_", record.station_id.strip()) or record.source.stem
        seen[base] = seen.get(base, 0) + 1
        station = base if seen[base] == 1 else f"{base}_{seen[base]}"
        columns[station] = record.precipitation
        metadata_rows.append(
            {
                "station": station,
                "code": record.station_id,
                "name": record.name,
                "longitude": record.longitude,
                "latitude": record.latitude,
                "source": str(record.source),
            }
        )
    matrix = pd.DataFrame(columns).sort_index()
    matrix.index.name = "time"
    metadata = pd.DataFrame(metadata_rows).set_index("station")
    return matrix, metadata
