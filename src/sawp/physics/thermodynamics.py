from __future__ import annotations

import numpy as np


def equivalent_potential_temperature(
    temperature_k: np.ndarray,
    specific_humidity: np.ndarray,
    pressure_hpa: np.ndarray | float,
) -> np.ndarray:
    """Bolton (1980) equivalent potential temperature in kelvin.

    Parameters are broadcast following NumPy rules. Specific humidity is kg kg-1 and
    pressure is hPa. Invalid or nonphysical inputs produce NaN.
    """
    temperature = np.asarray(temperature_k, dtype=float)
    humidity = np.asarray(specific_humidity, dtype=float)
    pressure = np.asarray(pressure_hpa, dtype=float)
    temperature, humidity, pressure = np.broadcast_arrays(temperature, humidity, pressure)
    valid = (
        np.isfinite(temperature)
        & np.isfinite(humidity)
        & np.isfinite(pressure)
        & (temperature > 0)
        & (humidity >= 0)
        & (humidity < 1)
        & (pressure > 0)
    )
    result = np.full(temperature.shape, np.nan, dtype=float)
    mixing_ratio = np.divide(humidity, 1.0 - humidity, out=np.zeros_like(humidity), where=valid)
    vapor_pressure = np.divide(
        mixing_ratio * pressure,
        0.622 + mixing_ratio,
        out=np.full_like(pressure, np.nan),
        where=valid,
    )
    positive_vapor = valid & (vapor_pressure > 0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        lcl_temperature = (
            2840.0 / (3.5 * np.log(temperature) - np.log(vapor_pressure) - 4.805) + 55.0
        )
        dry_lcl_theta = temperature * (1000.0 / (pressure - vapor_pressure)) ** (
            0.2854 * (1.0 - 0.28 * mixing_ratio)
        )
        theta_e = dry_lcl_theta * np.exp(
            (3376.0 / lcl_temperature - 2.54) * mixing_ratio * (1.0 + 0.81 * mixing_ratio)
        )
    result[positive_vapor] = theta_e[positive_vapor]
    return result
