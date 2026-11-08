from pathlib import Path

import numpy as np
import xarray as xr

from sawp.io import open_npz_memmap
from sawp.publication.legacy_builder import ERA5_FILES, LEGACY_MODULES
from sawp.publication.legacy_builder import _build_variable as build_variable
from sawp.publication.reproduce import CAPTURE_AFTER, RAINFALL_CAPTURE, SCRIPT_SEQUENCE


def test_from_zero_pipeline_captures_all_figures() -> None:
    names = set(RAINFALL_CAPTURE)
    for mapping in CAPTURE_AFTER.values():
        names.update(mapping)
    assert names == {
        *(f"Figure{number}.png" for number in range(1, 10)),
        *(f"FigureS{number}.png" for number in range(1, 10)),
    }


def test_original_sources_are_vendored() -> None:
    root = Path(__file__).resolve().parents[1]
    source = root / "archive" / "legacy_sources" / "root"
    package = root / "archive" / "legacy_sources" / "sawp"
    for script in (
        "RealizarTratamentoDadosMedidosMapaDominio.py",
        "RealizarTratamentoDadosMedidosMapaDominioAuxiliar.py",
        *SCRIPT_SEQUENCE,
    ):
        assert (source / script).is_file(), script
    for module in LEGACY_MODULES:
        assert (package / f"{module}.py").is_file(), module
    assert len(ERA5_FILES) == 8


def test_chunked_legacy_detrend_matches_original_formula(tmp_path: Path) -> None:
    time = np.arange(6)
    values = np.arange(6 * 2 * 3 * 4, dtype=np.float32).reshape(6, 2, 3, 4)
    values[:, 1, 2, 3] = np.nan
    source = tmp_path / "source.nc"
    xr.Dataset(
        {"u": (("time", "level", "latitude", "longitude"), values)},
        coords={
            "time": np.arange("2000-01", "2000-07", dtype="datetime64[M]"),
            "level": [850, 1000],
            "latitude": [-1.0, 0.0, 1.0],
            "longitude": [10.0, 11.0, 12.0, 13.0],
        },
    ).to_netcdf(source)
    destination = tmp_path / "detrended_uvq_u.npz"
    build_variable(
        source,
        "u",
        destination,
        detrend=True,
        output_name="u",
        force=True,
    )
    actual = open_npz_memmap(destination)
    mean = np.nanmean(values, axis=0)
    centered_time = time - time.mean()
    covariance = np.nanmean(centered_time[:, None, None, None] * (values - mean), axis=0)
    slope = covariance / np.mean(centered_time**2)
    trend = slope[None, ...] * time[:, None, None, None] + (mean - slope * time.mean())[None, ...]
    expected = values - trend + mean
    np.testing.assert_allclose(actual, expected, equal_nan=True)
