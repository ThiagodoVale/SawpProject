from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .domains import Domain


@dataclass(frozen=True)
class PeriodConfig:
    start_year: int
    end_year: int
    climatology_start: int
    climatology_end: int
    season_months: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.start_year > self.end_year:
            raise ValueError("start_year must not exceed end_year")
        if not self.start_year <= self.climatology_start <= self.climatology_end <= self.end_year:
            raise ValueError("climatology must be inside the analysis period")
        if not self.season_months or any(month < 1 or month > 12 for month in self.season_months):
            raise ValueError("season_months must contain calendar months from 1 to 12")


@dataclass(frozen=True)
class ThresholdConfig:
    extreme_cold: float = 0.10
    moderate_cold: float = 0.25
    moderate_warm: float = 0.75
    extreme_warm: float = 0.90
    significance: float = 0.05

    def __post_init__(self) -> None:
        values = (
            self.extreme_cold,
            self.moderate_cold,
            self.moderate_warm,
            self.extreme_warm,
        )
        if tuple(sorted(values)) != values or values[0] <= 0 or values[-1] >= 1:
            raise ValueError("quantile thresholds must be strictly ordered inside (0, 1)")


@dataclass(frozen=True)
class RainfallQCConfig:
    minimum_mm: float = 0.0
    maximum_mm: float = 2000.0
    spatial_outlier_mm: float = 500.0
    spatial_outlier_median_factor: float = 5.0
    minimum_valid_stations: int = 30
    minimum_valid_fraction: float = 0.10
    nearest_neighbors: int = 10
    minimum_common_months: int = 60
    maximum_gap_months: int = 36
    states: tuple[str, ...] = ("AL", "PB", "PE", "RN", "CE", "BA", "SE", "PI", "MA")


@dataclass(frozen=True)
class RuntimeConfig:
    python: str = "auto"
    chunk_size: int = 32
    workers: int = 1
    strict: bool = True


@dataclass(frozen=True)
class ProjectPaths:
    workspace: Path
    results: Path
    raw_data: Path
    raw_era5: Path
    raw_rainfall: Path
    raw_domain: Path
    interim: Path
    processed: Path

    @property
    def figures(self) -> Path:
        return self.results / "figures"


@dataclass(frozen=True)
class ProjectConfig:
    source: Path
    name: str
    version: str
    paths: ProjectPaths
    period: PeriodConfig
    thresholds: ThresholdConfig
    rainfall_qc: RainfallQCConfig
    domains: dict[str, Domain] = field(default_factory=dict)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def default_config_path() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "default.toml"


def load_config(path: str | Path | None = None) -> ProjectConfig:
    source = Path(path) if path is not None else default_config_path()
    source = source.expanduser().resolve()
    with source.open("rb") as stream:
        raw = tomllib.load(stream)
    base = source.parent

    paths_raw = raw["paths"]
    paths = ProjectPaths(
        workspace=_resolve(base, paths_raw["workspace"]),
        results=_resolve(base, paths_raw["results"]),
        raw_data=_resolve(base, paths_raw["raw_data"]),
        raw_era5=_resolve(base, paths_raw["raw_era5"]),
        raw_rainfall=_resolve(base, paths_raw["raw_rainfall"]),
        raw_domain=_resolve(base, paths_raw["raw_domain"]),
        interim=_resolve(base, paths_raw["interim"]),
        processed=_resolve(base, paths_raw["processed"]),
    )
    period_raw = raw["period"]
    period = PeriodConfig(
        start_year=int(period_raw["start_year"]),
        end_year=int(period_raw["end_year"]),
        climatology_start=int(period_raw["climatology_start"]),
        climatology_end=int(period_raw["climatology_end"]),
        season_months=tuple(int(value) for value in period_raw["season_months"]),
    )
    domains = {
        name: Domain(name=name, **{key: float(value) for key, value in values.items()})
        for name, values in raw.get("domains", {}).items()
    }
    rainfall_raw = dict(raw.get("rainfall_qc", {}))
    if "states" in rainfall_raw:
        rainfall_raw["states"] = tuple(str(value) for value in rainfall_raw["states"])
    return ProjectConfig(
        source=source,
        name=str(raw["project"]["name"]),
        version=str(raw["project"]["version"]),
        paths=paths,
        period=period,
        thresholds=ThresholdConfig(**raw.get("thresholds", {})),
        rainfall_qc=RainfallQCConfig(**rainfall_raw),
        domains=domains,
        runtime=RuntimeConfig(**raw.get("runtime", {})),
        raw=raw,
    )
