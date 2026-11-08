from __future__ import annotations

import argparse
from pathlib import Path

from sawp.workflows.detrending import detrend_netcdf_variable


def main() -> int:
    parser = argparse.ArgumentParser(description="Detrend one variable into one NetCDF product")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--variable", required=True)
    args = parser.parse_args()
    print(detrend_netcdf_variable(args.source, args.destination, args.variable))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
