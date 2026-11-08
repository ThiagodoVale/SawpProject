from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Iterable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from sawp.config import ProjectConfig, load_config

from .atmosphere import AtmosphereProducts, prepare_atmosphere
from .figures import render_atmosphere_figures, render_ocean_figures, render_rainfall_figures
from .ocean import OceanProducts, prepare_ocean
from .rainfall import RainfallProducts, prepare_rainfall

MAIN_PRODUCTS = tuple(f"figure-{number}" for number in range(1, 10))
SUPPLEMENT_PRODUCTS = tuple(f"figure-s{number}" for number in range(1, 10))
ALL_PRODUCTS = MAIN_PRODUCTS + SUPPLEMENT_PRODUCTS

RAINFALL_PRODUCTS = {"figure-1", "figure-s1", "figure-s2", "figure-s3", "figure-s4"}
OCEAN_PRODUCTS = {"figure-2", "figure-3", "figure-8", "figure-9", "figure-s5", "figure-s7"}
ATMOSPHERE_PRODUCTS = {
    "figure-4",
    "figure-5",
    "figure-6",
    "figure-7",
    "figure-s6",
    "figure-s8",
    "figure-s9",
}


def product_filename(product_id: str) -> str:
    suffix = product_id.removeprefix("figure-")
    if suffix.startswith("s"):
        return f"FigureS{int(suffix[1:])}.png"
    return f"Figure{int(suffix)}.png"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ScientificPipeline:
    def __init__(self, config: ProjectConfig | None = None) -> None:
        self.config = config or load_config()
        self.destination = self.config.paths.results / "figures"

    def required_inputs(self) -> tuple[Path, ...]:
        era5 = self.config.paths.raw_era5
        domain = self.config.paths.raw_domain
        return (
            era5 / "Era5_Dataset_meanHF.nc",
            era5 / "Era5_Dataset_sst.nc",
            era5 / "Era5_Dataset_t_q.nc",
            era5 / "Era5_Dataset_u_v_w.nc",
            domain / "sst.nc",
            domain / "srtm30plus_v11_land.nc",
            self.config.paths.raw_rainfall / "monitor",
        )

    def validate_inputs(self) -> None:
        missing = [path for path in self.required_inputs() if not path.exists()]
        if missing:
            raise FileNotFoundError("missing raw inputs:\n" + "\n".join(map(str, missing)))
        station_count = len(list((self.config.paths.raw_rainfall / "monitor").rglob("*.txt")))
        if station_count == 0:
            raise RuntimeError("no measured rainfall station files were found")

    def clean(self) -> None:
        """Remove only reproducible outputs inside configured processed/results roots."""
        project_root = self.config.source.parent.parent.resolve()
        targets = (self.config.paths.processed.resolve(), self.config.paths.interim.resolve())
        for target in targets:
            if project_root not in target.parents:
                raise RuntimeError(f"refusing to clean outside the project: {target}")
            if target.exists():
                shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
        self.destination.mkdir(parents=True, exist_ok=True)
        for product_id in ALL_PRODUCTS:
            output = self.destination / product_filename(product_id)
            output.unlink(missing_ok=True)
            output.with_suffix(output.suffix + ".json").unlink(missing_ok=True)

    def prepare(
        self, *, force: bool = False
    ) -> tuple[RainfallProducts, OceanProducts, AtmosphereProducts]:
        self.validate_inputs()
        rainfall = prepare_rainfall(self.config, force=force)
        ocean = prepare_ocean(self.config, force=force)
        atmosphere = prepare_atmosphere(self.config, force=force)
        return rainfall, ocean, atmosphere

    def run(
        self,
        products: Iterable[str] | None = None,
        *,
        force: bool = False,
        clean: bool = False,
    ) -> list[dict]:
        requested = tuple(products or ALL_PRODUCTS)
        unknown = set(requested) - set(ALL_PRODUCTS)
        if unknown:
            raise ValueError(f"unknown products: {sorted(unknown)}")
        if clean:
            self.clean()
        rainfall, ocean, atmosphere = self.prepare(force=force)
        self.destination.mkdir(parents=True, exist_ok=True)

        requested_set = set(requested)
        rendered: set[str] = set()
        if requested_set & RAINFALL_PRODUCTS:
            render_rainfall_figures(self.config, rainfall, self.destination)
            rendered |= RAINFALL_PRODUCTS
        if requested_set & OCEAN_PRODUCTS:
            render_ocean_figures(
                self.config,
                rainfall,
                ocean,
                atmosphere,
                self.destination,
            )
            rendered |= OCEAN_PRODUCTS
        if requested_set & ATMOSPHERE_PRODUCTS:
            render_atmosphere_figures(
                self.config,
                rainfall,
                ocean,
                atmosphere,
                self.destination,
            )
            rendered |= ATMOSPHERE_PRODUCTS

        reports = []
        raw_inputs = [str(path) for path in self.required_inputs()]
        for product_id in requested:
            output = self.destination / product_filename(product_id)
            if not output.is_file():
                raise RuntimeError(f"scientific producer did not create {output}")
            metadata = {
                "schema_version": 1,
                "product_id": product_id,
                "source_kind": "generated_from_raw",
                "output": str(output),
                "sha256": _sha256(output),
                "bytes": output.stat().st_size,
                "raw_inputs": raw_inputs,
                "processed_inputs": [
                    str(path)
                    for products_group in (rainfall, ocean, atmosphere)
                    for path in asdict(products_group).values()
                ],
                "created_utc": datetime.now(UTC).isoformat(),
            }
            output.with_suffix(output.suffix + ".json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
            reports.append(metadata)
        report_path = self.config.paths.results / "scientific_reproduction_report.json"
        report_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "source_kind": "generated_from_raw",
                    "complete": len(reports) == len(requested),
                    "requested_products": list(requested),
                    "products": reports,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return reports

    def render_product(self, product_id: str) -> Path:
        self.run([product_id])
        return self.destination / product_filename(product_id)
