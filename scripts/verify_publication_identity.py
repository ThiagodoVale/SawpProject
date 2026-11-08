from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from sawp.publication import PUBLICATION_PRODUCTS  # noqa: E402
from sawp.publication.verification import compare_images  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare the 18 regenerated figures to references")
    parser.add_argument("--generated", type=Path, default=PROJECT_ROOT / "results" / "figures")
    parser.add_argument(
        "--reference", type=Path, default=PROJECT_ROOT / "data" / "reference_figures"
    )
    parser.add_argument("--maximum-mae", type=float, default=20.0)
    args = parser.parse_args()
    reports = []
    failures = []
    for product in PUBLICATION_PRODUCTS:
        generated = args.generated / product.filename
        reference = args.reference / product.filename
        if not generated.is_file() or not reference.is_file():
            failures.append(f"missing pair for {product.product_id}")
            continue
        report = compare_images(generated, reference, maximum_mae=args.maximum_mae).to_dict()
        report["product_id"] = product.product_id
        reports.append(report)
        status = "EXACT" if report["sha256_equal"] else "VISUAL"
        if not report["visually_equivalent"]:
            status = "FAIL"
            failures.append(f"{product.product_id}: thumbnail MAE {report['thumbnail_mae']:.3f}")
        print(f"{product.product_id}: {status} (MAE={report['thumbnail_mae']:.3f})")
    output = PROJECT_ROOT / "results" / "publication_identity_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"products": reports, "failures": failures}, indent=2),
        encoding="utf-8",
    )
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1
    print("Publication identity verified: 18/18 figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
