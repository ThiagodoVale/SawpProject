from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download one ERA5 request described by JSON")
    parser.add_argument("dataset", help="CDS dataset identifier")
    parser.add_argument("request", type=Path, help="JSON file containing the CDS request")
    parser.add_argument("output", type=Path, help="single output file")
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import cdsapi

    cdsapi.Client().retrieve(args.dataset, request, str(args.output))


if __name__ == "__main__":
    main()
