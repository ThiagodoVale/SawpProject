from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

from .config import load_config
from .pipeline import PipelineRunner, load_manifest
from .pipeline.verify import verify_products


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sawp", description="SAWP 1.0 reproducible pipeline")
    parser.add_argument("--config", type=Path, help="TOML project configuration")
    parser.add_argument("--manifest", type=Path, help="TOML publication manifest")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("inventory", help="list publication products")
    subparsers.add_parser("doctor", help="check paths, inputs, and dependencies")

    run = subparsers.add_parser("run", help="run a pipeline stage")
    run.add_argument("target", help="stage identifier or publication")
    run.add_argument("--resume", action="store_true", help="skip complete stages")
    run.add_argument("--dry-run", action="store_true", help="print actions without executing")

    verify = subparsers.add_parser("verify", help="verify publication products")
    verify.add_argument("--mode", choices=("reference", "generated"), default="generated")
    verify.add_argument("--json", action="store_true", dest="as_json")

    collect = subparsers.add_parser("collect", help="copy publication products to results")
    collect.add_argument("--reference", action="store_true", help="collect accepted references")
    return parser


def _inventory(manifest) -> int:
    print(f"{'ID':<12} {'STAGE':<24} PRODUCT")
    for product in manifest.products:
        print(f"{product.id:<12} {product.stage:<24} {product.label}")
    return 0


def _doctor(config, manifest) -> int:
    required_paths = {
        "workspace": config.paths.workspace,
        "raw data": config.paths.raw_data,
        "rain gauges": config.paths.raw_rainfall / "monitor",
        "domain SST": config.paths.raw_domain / "sst.nc",
        "domain topography": config.paths.raw_domain / "srtm30plus_v11_land.nc",
        "ERA5 SST": config.paths.raw_era5 / "Era5_Dataset_sst.nc",
        "ERA5 temperature/humidity": config.paths.raw_era5 / "Era5_Dataset_t_q.nc",
        "ERA5 wind/omega": config.paths.raw_era5 / "Era5_Dataset_u_v_w.nc",
        "processed rainfall": (
            config.paths.processed / "rainfall" / "ENEB_monthly_precipitation_qc.csv"
        ),
        "interim data": config.paths.interim,
        "reference figures": config.paths.reference_figures,
    }
    failures = 0
    print("PATHS")
    for label, path in required_paths.items():
        exists = path.exists()
        failures += not exists
        print(f"  {'OK' if exists else 'MISSING':<7} {label:<22} {path}")
    print("DEPENDENCIES")
    for module in (
        "numpy",
        "pandas",
        "scipy",
        "xarray",
        "netCDF4",
        "matplotlib",
        "cartopy",
        "metpy",
    ):
        exists = importlib.util.find_spec(module) is not None
        failures += not exists
        print(f"  {'OK' if exists else 'MISSING':<7} {module}")
    references = verify_products(config, manifest, "reference")
    present = sum(row["exists"] for row in references)
    print(f"REFERENCE PRODUCTS\n  {present}/{len(references)} present")
    failures += present != len(references)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_config(args.config)
    manifest = load_manifest(args.manifest)
    if args.command == "inventory":
        return _inventory(manifest)
    if args.command == "doctor":
        return _doctor(config, manifest)
    runner = PipelineRunner(config, manifest)
    if args.command == "run":
        reports = runner.run(args.target, resume=args.resume, dry_run=args.dry_run)
        print(json.dumps(reports, ensure_ascii=False, indent=2))
        return 0
    if args.command == "verify":
        rows = verify_products(config, manifest, args.mode)
        if args.as_json:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        else:
            for row in rows:
                status = "OK" if row["exists"] else "MISSING"
                comparison = ""
                if "identical" in row:
                    comparison = " identical" if row["identical"] else " changed"
                print(f"{status:<7} {row['label']:<12} {row['path']}{comparison}")
        return 0 if all(row["exists"] for row in rows) else 1
    if args.command == "collect":
        outputs = runner.collect_products(use_reference=args.reference)
        print("\n".join(str(path) for path in outputs))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
