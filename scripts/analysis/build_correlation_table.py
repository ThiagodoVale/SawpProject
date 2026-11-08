from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sawp.workflows.correlation import pairwise_statistics


def main() -> int:
    parser = argparse.ArgumentParser(description="Build one pairwise correlation table")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    frame = pd.read_csv(args.input, index_col=0)
    result = pairwise_statistics(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
