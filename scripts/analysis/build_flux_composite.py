from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sawp.workflows.fluxes import FluxCompositeSpec, FluxKind, compute_flux_composites


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one heat-flux composite NPZ")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("classification", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--kind", choices=[kind.value for kind in FluxKind], required=True)
    parser.add_argument("--variable", required=True)
    parser.add_argument("--lag", type=int, default=0)
    args = parser.parse_args()
    classification = pd.read_csv(args.classification, index_col="year")
    spec = FluxCompositeSpec(FluxKind(args.kind), args.variable, lag_months=args.lag)
    compute_flux_composites(args.dataset, classification, spec, output_npz=args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
