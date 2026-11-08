from __future__ import annotations

import numpy as np

GRAVITY = 9.80665
EARTH_RADIUS_M = 6_371_000.0


def _descending_pressure(
    field: np.ndarray, pressure_hpa: np.ndarray, axis: int
) -> tuple[np.ndarray, np.ndarray]:
    pressure = np.asarray(pressure_hpa, dtype=float)
    if pressure.ndim != 1 or pressure.size != field.shape[axis]:
        raise ValueError("pressure_hpa must match the pressure axis")
    order = np.argsort(pressure)[::-1]
    return np.take(field, order, axis=axis), pressure[order]


def integrate_pressure(
    field: np.ndarray,
    pressure_hpa: np.ndarray,
    *,
    axis: int = 0,
    gravity: float = GRAVITY,
) -> np.ndarray:
    """Trapezoidal integral ``1/g ∫ field dp`` with pressure in hPa."""
    values, pressure = _descending_pressure(np.asarray(field, dtype=float), pressure_hpa, axis)
    lower = np.take(values, np.arange(pressure.size - 1), axis=axis)
    upper = np.take(values, np.arange(1, pressure.size), axis=axis)
    layer_mean = 0.5 * (lower + upper)
    dp_pa = (pressure[:-1] - pressure[1:]) * 100.0
    shape = [1] * values.ndim
    shape[axis] = dp_pa.size
    return np.nansum(layer_mean * dp_pa.reshape(shape), axis=axis) / gravity


def water_vapor_transport(
    specific_humidity: np.ndarray,
    u_wind: np.ndarray,
    v_wind: np.ndarray,
    pressure_hpa: np.ndarray,
    *,
    axis: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = np.asarray(specific_humidity, dtype=float)
    u = np.asarray(u_wind, dtype=float)
    v = np.asarray(v_wind, dtype=float)
    if q.shape != u.shape or q.shape != v.shape:
        raise ValueError("specific humidity and wind arrays must have identical shapes")
    content = integrate_pressure(q, pressure_hpa, axis=axis)
    transport_u = integrate_pressure(q * u, pressure_hpa, axis=axis)
    transport_v = integrate_pressure(q * v, pressure_hpa, axis=axis)
    return content, transport_u, transport_v


def horizontal_divergence(
    zonal_flux: np.ndarray,
    meridional_flux: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
) -> np.ndarray:
    """Spherical horizontal divergence on a regular latitude/longitude grid."""
    u = np.asarray(zonal_flux, dtype=float)
    v = np.asarray(meridional_flux, dtype=float)
    lon = np.deg2rad(np.asarray(longitude, dtype=float))
    lat = np.deg2rad(np.asarray(latitude, dtype=float))
    if u.shape != v.shape or u.shape[-2:] != (lat.size, lon.size):
        raise ValueError("flux arrays must end in (latitude, longitude)")
    if lon.ndim != 1 or lat.ndim != 1:
        raise ValueError("longitude and latitude must be one-dimensional")
    cos_lat = np.cos(lat)
    reshape = (1,) * (u.ndim - 2) + (lat.size, 1)
    cos_grid = cos_lat.reshape(reshape)
    du_dlon = np.gradient(u, lon, axis=-1, edge_order=2)
    dvcos_dlat = np.gradient(v * cos_grid, lat, axis=-2, edge_order=2)
    denominator = EARTH_RADIUS_M * cos_grid
    return np.divide(
        du_dlon + dvcos_dlat,
        denominator,
        out=np.full_like(u, np.nan),
        where=np.abs(denominator) > 1e-12,
    )


def moisture_flux_convergence(
    zonal_transport: np.ndarray,
    meridional_transport: np.ndarray,
    longitude: np.ndarray,
    latitude: np.ndarray,
) -> np.ndarray:
    """Return ``-∇·Q`` so positive values denote moisture convergence."""
    return -horizontal_divergence(zonal_transport, meridional_transport, longitude, latitude)
