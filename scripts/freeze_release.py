from __future__ import annotations

import argparse
import json
import shutil
import sys
import tomllib
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import sawp  # noqa: E402
from sawp.reproducibility import sha256  # noqa: E402
from sawp.scientific.pipeline import ALL_PRODUCTS, product_filename  # noqa: E402


def _checksum_lines(paths: list[Path]) -> str:
    return "".join(
        f"{sha256(path)}  {path.relative_to(PROJECT_ROOT).as_posix()}\n" for path in paths
    )


def _declared_version() -> str:
    with (PROJECT_ROOT / "pyproject.toml").open("rb") as stream:
        return str(tomllib.load(stream)["project"]["version"])


def _validate_version(version: str) -> None:
    declarations = {
        "pyproject.toml": _declared_version(),
        "sawp.__version__": sawp.__version__,
    }
    mismatches = {name: value for name, value in declarations.items() if value != version}
    if mismatches:
        raise RuntimeError(f"version declarations do not match {version}: {mismatches}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze one validated SAWP release")
    parser.add_argument("--version", default="1.0.0")
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()
    _validate_version(args.version)

    figure_names = [product_filename(product) for product in ALL_PRODUCTS]
    generated = [PROJECT_ROOT / "results" / "figures" / name for name in figure_names]
    missing = [path for path in generated if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing generated figures: {missing}")

    reference_directory = PROJECT_ROOT / "data" / "reference_figures"
    reference_directory.mkdir(parents=True, exist_ok=True)
    references = []
    for source in generated:
        destination = reference_directory / source.name
        shutil.copy2(source, destination)
        references.append(destination)

    distribution_directory = PROJECT_ROOT / "dist"
    distributions = sorted(distribution_directory.glob(f"sawp-{args.version}*"))
    if len(distributions) != 2 or {path.suffix for path in distributions} != {".gz", ".whl"}:
        raise RuntimeError(
            "expected one wheel and one source distribution; run `uv build` before freezing"
        )

    checksums = PROJECT_ROOT / "checksums"
    checksums.mkdir(parents=True, exist_ok=True)
    (checksums / "publication_outputs.sha256").write_text(
        _checksum_lines(generated), encoding="utf-8", newline="\n"
    )
    (checksums / "reference_figures.sha256").write_text(
        _checksum_lines(references), encoding="utf-8", newline="\n"
    )
    release_paths = generated + references + distributions
    checksum_name = f"release_v{args.version}.sha256"
    (checksums / checksum_name).write_text(
        _checksum_lines(release_paths), encoding="utf-8", newline="\n"
    )

    manifest = {
        "schema_version": 1,
        "release": f"v{args.version}",
        "release_date": args.date,
        "python_package": f"sawp=={args.version}",
        "figure_count": len(generated),
        "scientific_source": "generated_from_raw",
        "moisture_flux_convention": "-div(Q); positive is convergence",
        "checksums": checksum_name,
        "artifacts": [
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in distributions
        ],
    }
    manifest_path = checksums / f"release_v{args.version}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
