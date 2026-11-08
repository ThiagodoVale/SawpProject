from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.lib.format import open_memmap

from sawp.io import open_npz_memmap

ERA5_FILES = (
    "Era5_Dataset_decomposit.nc",
    "Era5_Dataset_evp.nc",
    "Era5_Dataset_lh.nc",
    "Era5_Dataset_lhsh.nc",
    "Era5_Dataset_meanHF.nc",
    "Era5_Dataset_sst.nc",
    "Era5_Dataset_t_q.nc",
    "Era5_Dataset_u_v_w.nc",
)

LEGACY_MODULES = (
    "CarregarDados",
    "ConfigurarTempoArea",
    "CorrelacionarChuvaTsm",
    "ExtrairChuvaMensal",
    "ExtrairVariaveisQuantisDataframe",
    "Graficos",
    "MediasVetoriaisPercentil",
    "NormalizarNiveisDados",
    "PlotAnomaliaQuantil",
    "ProcessamentoAguaPrecipitavelTransporteQuantis",
    "ProcessamentoTsmQuantisv2",
    "TeorTransporte",
)


def _link_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _link_tree(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if path.is_file():
            _link_file(path, destination / path.relative_to(source))


def _legacy_init() -> str:
    imports = "\n".join(f"from .{name} import *  # noqa: F403" for name in LEGACY_MODULES)
    return f'"""CPU-only API used by the original SAWP publication scripts."""\n\n{imports}\n'


def prepare_workspace(project_root: Path, workspace: Path, *, clean: bool = False) -> Path:
    if clean and workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    archive = project_root / "archive" / "legacy_sources"
    script_source = archive / "root"
    package_source = archive / "sawp"
    if not script_source.is_dir() or not package_source.is_dir():
        raise FileNotFoundError("vendored original scripts are missing from archive/legacy_sources")
    for script in script_source.glob("*.py"):
        shutil.copy2(script, workspace / script.name)
    runtime_package = workspace / "sawp"
    runtime_package.mkdir(exist_ok=True)
    for module in LEGACY_MODULES:
        shutil.copy2(package_source / f"{module}.py", runtime_package / f"{module}.py")
    (runtime_package / "__init__.py").write_text(_legacy_init(), encoding="utf-8")

    raw = project_root / "data" / "raw"
    data = workspace / "data"
    for filename in ERA5_FILES:
        _link_file(raw / "era5" / filename, data / filename)
    for filename in ("sst.nc", "srtm30plus_v11_land.nc"):
        _link_file(raw / "domain" / filename, data / "dominio" / filename)
    _link_tree(raw / "rainfall" / "monitor", data / "monitor")
    return workspace


def _write_npz(destination: Path, members: dict[str, Path | np.ndarray]) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".npz.tmp")
    temporary.unlink(missing_ok=True)
    small_files: list[Path] = []
    with zipfile.ZipFile(
        temporary,
        "w",
        compression=zipfile.ZIP_STORED,
        allowZip64=True,
    ) as archive:
        for name, value in members.items():
            if isinstance(value, Path):
                member_path = value
            else:
                member_path = destination.parent / f".{destination.stem}.{name}.npy"
                np.save(member_path, value, allow_pickle=False)
                small_files.append(member_path)
            archive.write(member_path, arcname=f"{name}.npy")
    temporary.replace(destination)
    for path in small_files:
        path.unlink(missing_ok=True)


def _coordinate_name(data: xr.DataArray, *names: str) -> str | None:
    return next((name for name in names if name in data.coords), None)


def _build_variable(
    source: Path,
    variable: str,
    destination: Path,
    *,
    detrend: bool,
    output_name: str,
    force: bool,
) -> Path:
    if destination.is_file() and not force:
        return destination
    with xr.open_dataset(source) as dataset:
        data = dataset[variable]
        time_name = "time" if "time" in data.dims else "valid_time"
        level_name = _coordinate_name(data, "level", "pressure_level")
        latitude_name = _coordinate_name(data, "latitude", "lat")
        longitude_name = _coordinate_name(data, "longitude", "lon")
        if latitude_name is None or longitude_name is None:
            raise ValueError(f"coordinates not found in {source}")
        output_dtype = np.float64 if detrend else data.dtype
        scratch = destination.parent / f".{output_name}.data.npy"
        scratch.unlink(missing_ok=True)
        output = open_memmap(scratch, mode="w+", dtype=output_dtype, shape=data.shape)
        t = np.arange(data.shape[0], dtype=np.float64)
        t_mean = t.mean()
        variance = np.mean((t - t_mean) ** 2)
        if level_name:
            blocks = [
                (slice(None), slice(index, index + 1), slice(None), slice(None))
                for index in range(data.shape[1])
            ]
        else:
            blocks = [
                (slice(None), slice(start, min(start + 30, data.shape[1])), slice(None))
                for start in range(0, data.shape[1], 30)
            ]
        for block in blocks:
            values = np.asarray(data.data[block])
            if detrend:
                mean = np.nanmean(values, axis=0)
                expand = (slice(None),) + (None,) * (values.ndim - 1)
                covariance = np.nanmean((t[expand] - t_mean) * (values - mean), axis=0)
                slope = covariance / variance
                trend = slope[None, ...] * t[expand] + (mean - slope * t_mean)[None, ...]
                output[block] = values - trend + mean
            else:
                output[block] = values
            output.flush()
        del output
        time_values = data[time_name].values
        if np.issubdtype(time_values.dtype, np.datetime64):
            time_values = time_values.astype("datetime64[s]")
        members: dict[str, Path | np.ndarray] = {
            "data": scratch,
            "lat": data[latitude_name].values,
            "lon": data[longitude_name].values,
            time_name: time_values,
        }
        if level_name:
            members["level"] = data[level_name].values
        _write_npz(destination, members)
        scratch.unlink(missing_ok=True)
    return destination


def _theta_e(t: np.ndarray, q: np.ndarray, pressure: np.ndarray) -> np.ndarray:
    p = pressure[:, None, None].astype(np.float32)
    temperature = t.astype(np.float32)
    humidity = q.astype(np.float32)
    saturation = 6.112 * np.exp((17.67 * (temperature - 273.15)) / (temperature - 29.65))
    mixing = 0.622 * saturation / (p - saturation)
    vapor = (humidity / (1.0 - humidity) / mixing) * saturation
    value = np.log(vapor / 6.112)
    dewpoint = 243.5 * value / (17.67 - value) + 273.15
    e = 6.112 * np.exp((17.67 * (dewpoint - 273.15)) / (dewpoint - 29.65))
    ratio = 0.622 * e / (p - e)
    tl = 56.0 + 1.0 / ((1.0 / (dewpoint - 56.0)) + np.log(temperature / dewpoint) / 800.0)
    theta_l = temperature * (1000.0 / (p - e)) ** 0.286 * (temperature / tl) ** (0.28 * ratio)
    exponent = ratio * (1.0 + 0.448 * ratio) * ((3036.0 / tl) - 1.78)
    return (theta_l * np.exp(exponent)).astype(np.float32)


def _build_theta_e(directory: Path, prefix: str, *, force: bool) -> Path:
    destination = directory / f"{prefix}_uvq_te.npz"
    if destination.is_file() and not force:
        return destination
    t_path = directory / f"{prefix}_uvq_t.npz"
    q_path = directory / f"{prefix}_uvq_q.npz"
    temperature = open_npz_memmap(t_path)
    humidity = open_npz_memmap(q_path)
    with np.load(t_path, allow_pickle=False) as metadata:
        pressure = metadata["level"]
        latitude = metadata["lat"]
        longitude = metadata["lon"]
        time_values = metadata["time"]
    scratch = directory / f".{prefix}_uvq_te.data.npy"
    output = open_memmap(scratch, mode="w+", dtype=np.float32, shape=temperature.shape)
    for index in range(temperature.shape[0]):
        output[index] = _theta_e(temperature[index], humidity[index], pressure)
        if index % 12 == 0:
            output.flush()
    output.flush()
    del output, temperature, humidity
    _write_npz(
        destination,
        {
            "data": scratch,
            "level": pressure,
            "lat": latitude,
            "lon": longitude,
            "time": time_values,
        },
    )
    scratch.unlink(missing_ok=True)
    return destination


def build_legacy_intermediates(workspace: Path, *, force: bool = False) -> list[Path]:
    data = workspace / "data"
    detrended = workspace / "outputs"
    trended = workspace / "outputs_trended"
    detrended.mkdir(exist_ok=True)
    trended.mkdir(exist_ok=True)
    specifications = (
        ("Era5_Dataset_sst.nc", "sst", "sst"),
        ("Era5_Dataset_u_v_w.nc", "u", "u"),
        ("Era5_Dataset_u_v_w.nc", "v", "v"),
        ("Era5_Dataset_u_v_w.nc", "w", "w"),
        ("Era5_Dataset_t_q.nc", "t", "t"),
        ("Era5_Dataset_t_q.nc", "q", "q"),
        ("Era5_Dataset_lh.nc", "slhf", "slhf"),
        ("Era5_Dataset_lhsh.nc", "slhf", "lh"),
    )
    outputs: list[Path] = []
    for filename, variable, name in specifications:
        detrended_name = "detrended_sst.npz" if name == "sst" else f"detrended_uvq_{name}.npz"
        trended_name = f"trended_uvq_{name}.npz"
        outputs.append(
            _build_variable(
                data / filename,
                variable,
                detrended / detrended_name,
                detrend=True,
                output_name=f"detrended_{name}",
                force=force,
            )
        )
        outputs.append(
            _build_variable(
                data / filename,
                variable,
                trended / trended_name,
                detrend=False,
                output_name=f"trended_{name}",
                force=force,
            )
        )
    outputs.append(_build_theta_e(detrended, "detrended", force=force))
    outputs.append(_build_theta_e(trended, "trended", force=force))
    return outputs
