from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .scientific.pipeline import ALL_PRODUCTS, ScientificPipeline, product_filename


def generate_all_figures(
    *,
    config_path: Path | None = None,
    manifest_path: Path | None = None,
    resume: bool = False,
    dry_run: bool = False,
) -> list[dict]:
    """Generate every publication figure directly from the raw observations."""
    config = load_config(config_path)
    del manifest_path  # Retained for command-line compatibility with version 1.0.
    pipeline = ScientificPipeline(config)
    if dry_run:
        return [
            {
                "stage": product_id,
                "product_id": product_id,
                "status": "dry-run",
                "return_code": None,
                "artifacts": [str(pipeline.destination / product_filename(product_id))],
            }
            for product_id in ALL_PRODUCTS
        ]
    if resume and all(
        (pipeline.destination / product_filename(product_id)).is_file()
        for product_id in ALL_PRODUCTS
    ):
        return [
            {
                "stage": product_id,
                "product_id": product_id,
                "status": "skipped",
                "return_code": 0,
                "artifacts": [str(pipeline.destination / product_filename(product_id))],
            }
            for product_id in ALL_PRODUCTS
        ]
    reports = pipeline.run(force=not resume)
    return [
        {
            **report,
            "stage": report["product_id"],
            "status": "completed",
            "return_code": 0,
            "artifacts": [report["output"]],
        }
        for report in reports
    ]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate all 18 main and supplementary publication figures"
    )
    parser.add_argument("--config", type=Path, help="TOML project configuration")
    parser.add_argument("--manifest", type=Path, help="TOML publication manifest")
    parser.add_argument("--resume", action="store_true", help="skip figures that already exist")
    parser.add_argument("--dry-run", action="store_true", help="show commands without running them")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    reports = generate_all_figures(
        config_path=args.config,
        manifest_path=args.manifest,
        resume=args.resume,
        dry_run=args.dry_run,
    )
    summary = [
        {
            "stage": report["stage"],
            "status": report.get("status", "completed"),
            "return_code": report.get("return_code"),
            "artifacts": report.get("artifacts", []),
        }
        for report in reports
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
