from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Domain:
    name: str
    west: float
    east: float
    south: float
    north: float

    def __post_init__(self) -> None:
        if self.south >= self.north:
            raise ValueError(f"invalid latitude bounds for {self.name}")
        if self.west == self.east:
            raise ValueError(f"empty longitude range for {self.name}")

    def contains(self, longitude: float, latitude: float) -> bool:
        lon = float(normalize_longitude(longitude))
        west = float(normalize_longitude(self.west))
        east = float(normalize_longitude(self.east))
        longitude_ok = west <= lon <= east if west <= east else lon >= west or lon <= east
        return longitude_ok and self.south <= latitude <= self.north

    def mask(self, longitude: np.ndarray, latitude: np.ndarray) -> np.ndarray:
        lon = normalize_longitude(np.asarray(longitude, dtype=float))
        lat = np.asarray(latitude, dtype=float)
        if lon.ndim == 1 and lat.ndim == 1:
            lon, lat = np.meshgrid(lon, lat)
        elif lon.shape != lat.shape:
            lon, lat = np.broadcast_arrays(lon, lat)
        west = float(normalize_longitude(self.west))
        east = float(normalize_longitude(self.east))
        longitude_ok = (
            (lon >= west) & (lon <= east) if west <= east else (lon >= west) | (lon <= east)
        )
        return longitude_ok & (lat >= self.south) & (lat <= self.north)


def normalize_longitude(longitude: np.ndarray | float) -> np.ndarray | float:
    values = np.asarray(longitude, dtype=float)
    normalized = (values + 180.0) % 360.0 - 180.0
    if np.isscalar(longitude):
        return float(normalized)
    return normalized
