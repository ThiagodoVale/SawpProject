from __future__ import annotations

from pathlib import Path

import numpy as np

from sawp.physics.moisture import moisture_flux_convergence, water_vapor_transport
from sawp.physics.thermodynamics import equivalent_potential_temperature


def build_moisture_fields(
    source: Path,
    output: Path,
    *,
    humidity: str = "q",
    u_wind: str = "u",
    v_wind: str = "v",
) -> Path:
    """Create one self-contained NPZ product with vertically integrated moisture fields."""
    import xarray as xr

    with xr.open_dataset(source) as dataset:
        pressure_name = "pressure_level" if "pressure_level" in dataset.coords else "level"
        latitude_name = "latitude" if "latitude" in dataset.coords else "lat"
        longitude_name = "longitude" if "longitude" in dataset.coords else "lon"
        pressure_axis = dataset[humidity].get_axis_num(pressure_name)
        content, transport_u, transport_v = water_vapor_transport(
            dataset[humidity].to_numpy(),
            dataset[u_wind].to_numpy(),
            dataset[v_wind].to_numpy(),
            dataset[pressure_name].to_numpy(),
            axis=pressure_axis,
        )
        convergence = moisture_flux_convergence(
            transport_u,
            transport_v,
            dataset[longitude_name].to_numpy(),
            dataset[latitude_name].to_numpy(),
        )
        time_name = "valid_time" if "valid_time" in dataset.coords else "time"
        payload = {
            "time": dataset[time_name].to_numpy(),
            "latitude": dataset[latitude_name].to_numpy(),
            "longitude": dataset[longitude_name].to_numpy(),
            "water_vapor_content": content,
            "transport_u": transport_u,
            "transport_v": transport_v,
            "convergence": convergence,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    return output


def build_theta_e(
    source: Path,
    output: Path,
    *,
    temperature: str = "t",
    humidity: str = "q",
) -> Path:
    """Create one NetCDF product containing equivalent potential temperature."""
    import xarray as xr

    with xr.open_dataset(source) as dataset:
        pressure_name = "pressure_level" if "pressure_level" in dataset.coords else "level"
        field = dataset[temperature]
        pressure = dataset[pressure_name].to_numpy()
        pressure_shape = [1] * field.ndim
        pressure_shape[field.get_axis_num(pressure_name)] = pressure.size
        values = equivalent_potential_temperature(
            field.to_numpy(),
            dataset[humidity].to_numpy(),
            pressure.reshape(pressure_shape),
        )
        result = xr.Dataset(
            {"theta_e": (field.dims, values)},
            coords={name: field.coords[name] for name in field.dims},
            attrs={"source": str(source), "method": "Bolton (1980)"},
        )
        result["theta_e"].attrs.update(
            {"long_name": "equivalent potential temperature", "units": "K"}
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_netcdf(output)
    return output
