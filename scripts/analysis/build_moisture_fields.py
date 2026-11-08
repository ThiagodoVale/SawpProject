from __future__ import annotations

import argparse
from pathlib import Path

from sawp.workflows.fields import build_moisture_fields


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one vertically integrated moisture product")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--humidity", default="q")
    parser.add_argument("--u-wind", default="u")
    parser.add_argument("--v-wind", default="v")
    args = parser.parse_args()
    build_moisture_fields(
        args.source,
        args.output,
        humidity=args.humidity,
        u_wind=args.u_wind,
        v_wind=args.v_wind,
    )


if __name__ == "__main__":
    main()
