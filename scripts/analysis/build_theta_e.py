from __future__ import annotations

import argparse
from pathlib import Path

from sawp.workflows.fields import build_theta_e


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build one equivalent-potential-temperature product"
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--temperature", default="t")
    parser.add_argument("--humidity", default="q")
    args = parser.parse_args()
    build_theta_e(
        args.source,
        args.output,
        temperature=args.temperature,
        humidity=args.humidity,
    )


if __name__ == "__main__":
    main()
