from __future__ import annotations

import argparse
from pathlib import Path

from sawp.config import load_config
from sawp.rainfall import basic_quality_control, build_station_matrix, spatial_outlier_control


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one quality-controlled station matrix")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config = load_config()
    domain = config.domains["eneb"]
    matrix, _metadata = build_station_matrix(
        config.paths.raw_rainfall / "monitor",
        west=domain.west,
        east=domain.east,
        south=domain.south,
        north=domain.north,
    )
    cleaned = basic_quality_control(
        matrix,
        minimum_mm=config.rainfall_qc.minimum_mm,
        maximum_mm=config.rainfall_qc.maximum_mm,
    )
    cleaned, _removed = spatial_outlier_control(
        cleaned,
        absolute_threshold_mm=config.rainfall_qc.spatial_outlier_mm,
        median_factor=config.rainfall_qc.spatial_outlier_median_factor,
    )
    output = args.output or (
        config.paths.processed / "rainfall" / "rain_gauge_monthly_matrix_qc.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(output)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
