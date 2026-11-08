"""Download the four ERA5 monthly-mean files required by SAWP."""

from __future__ import annotations

import argparse
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "raw" / "era5"
MONTHS = tuple(f"{month:02d}" for month in range(1, 13))
AREA = (15, -60, -30, -10)
PRESSURE_LEVELS = (
    "100",
    "200",
    "250",
    "300",
    "400",
    "500",
    "600",
    "700",
    "750",
    "800",
    "825",
    "850",
    "875",
    "900",
    "925",
    "950",
    "975",
    "1000",
)


@dataclass(frozen=True)
class Era5Product:
    key: str
    filename: str
    dataset: str
    variables: tuple[str, ...]
    expected_variables: tuple[str, ...]
    pressure_levels: tuple[str, ...] = ()


PRODUCTS = (
    Era5Product(
        key="sst",
        filename="Era5_Dataset_sst.nc",
        dataset="reanalysis-era5-single-levels-monthly-means",
        variables=("sea_surface_temperature",),
        expected_variables=("sst",),
    ),
    Era5Product(
        key="heat-flux",
        filename="Era5_Dataset_meanHF.nc",
        dataset="reanalysis-era5-single-levels-monthly-means",
        variables=("mean_surface_latent_heat_flux", "mean_surface_sensible_heat_flux"),
        expected_variables=("avg_slhtf", "avg_ishf"),
    ),
    Era5Product(
        key="temperature-humidity",
        filename="Era5_Dataset_t_q.nc",
        dataset="reanalysis-era5-pressure-levels-monthly-means",
        variables=("specific_humidity", "temperature"),
        expected_variables=("q", "t"),
        pressure_levels=PRESSURE_LEVELS,
    ),
    Era5Product(
        key="wind-omega",
        filename="Era5_Dataset_u_v_w.nc",
        dataset="reanalysis-era5-pressure-levels-monthly-means",
        variables=("u_component_of_wind", "v_component_of_wind", "vertical_velocity"),
        expected_variables=("u", "v", "w"),
        pressure_levels=PRESSURE_LEVELS,
    ),
)


def build_request(product: Era5Product, year: int) -> dict[str, object]:
    request: dict[str, object] = {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": list(product.variables),
        "year": [str(year)],
        "month": list(MONTHS),
        "time": ["00:00"],
        "area": list(AREA),
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    if product.pressure_levels:
        request["pressure_level"] = list(product.pressure_levels)
    return request


def retrieve_year(client, product: Era5Product, year: int, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        logging.info("Reusing %s", destination)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, 4):
        try:
            destination.unlink(missing_ok=True)
            logging.info("Downloading %s for %s", product.key, year)
            client.retrieve(product.dataset, build_request(product, year), str(destination))
            return
        except Exception:
            destination.unlink(missing_ok=True)
            if attempt == 3:
                raise
            logging.exception("Download failed; retrying in 30 seconds")
            time.sleep(30)


def merge_years(parts: list[Path], product: Era5Product, destination: Path) -> None:
    logging.info("Merging %d yearly files into %s", len(parts), destination)
    dataset = xr.open_mfdataset(
        [str(path) for path in parts],
        combine="by_coords",
        combine_attrs="override",
        parallel=False,
    )
    missing = set(product.expected_variables) - set(dataset.data_vars)
    if missing:
        dataset.close()
        raise RuntimeError(
            f"{product.key} returned unexpected variables; missing {sorted(missing)}"
        )
    time_coordinate = "valid_time" if "valid_time" in dataset.coords else "time"
    if time_coordinate in dataset.coords:
        dataset = dataset.sortby(time_coordinate)
    encoding = {
        name: {"zlib": True, "complevel": 3, "dtype": "float32"}
        for name, variable in dataset.data_vars.items()
        if np.issubdtype(variable.dtype, np.floating)
    }
    temporary = destination.with_suffix(".nc.tmp")
    temporary.unlink(missing_ok=True)
    dataset.to_netcdf(temporary, format="NETCDF4", encoding=encoding)
    dataset.close()
    temporary.replace(destination)


def download_product(
    client,
    product: Era5Product,
    output_directory: Path,
    years: range,
    *,
    overwrite: bool,
    keep_parts: bool,
) -> None:
    destination = output_directory / product.filename
    if destination.is_file() and not overwrite:
        logging.info("Already complete: %s", destination)
        return
    parts_directory = output_directory / ".parts" / product.key
    parts = [parts_directory / f"{year}.nc" for year in years]
    for year, part in zip(years, parts, strict=True):
        retrieve_year(client, product, year, part)
    merge_years(parts, product, destination)
    if not keep_parts:
        shutil.rmtree(parts_directory)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-year", type=int, default=1959)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--only", choices=[item.key for item in PRODUCTS], nargs="+")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-parts", action="store_true")
    args = parser.parse_args()
    if args.start_year > args.end_year:
        parser.error("--start-year must not exceed --end-year")

    import cdsapi

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    selected = [item for item in PRODUCTS if not args.only or item.key in args.only]
    years = range(args.start_year, args.end_year + 1)
    output_directory = args.output_dir.expanduser().resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()
    for product in selected:
        download_product(
            client,
            product,
            output_directory,
            years,
            overwrite=args.overwrite,
            keep_parts=args.keep_parts,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
