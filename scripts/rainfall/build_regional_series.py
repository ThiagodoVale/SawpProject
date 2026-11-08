from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sawp.config import load_config
from sawp.rainfall import regional_monthly_series


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one ENEB regional monthly rainfall series")
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config()
    matrix = pd.read_csv(args.matrix, index_col=0, parse_dates=True)
    series = regional_monthly_series(
        matrix,
        minimum_valid_stations=config.rainfall_qc.minimum_valid_stations,
        minimum_valid_fraction=config.rainfall_qc.minimum_valid_fraction,
    )
    output = args.output or (
        config.paths.processed / "rainfall" / "ENEB_monthly_precipitation_qc.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
