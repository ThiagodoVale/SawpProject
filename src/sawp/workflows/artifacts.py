from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from sawp.pipeline.provenance import sha256


@dataclass(frozen=True)
class MaterializedProduct:
    product_id: str
    output: Path
    source: Path
    source_kind: str
    sha256: str


def materialize_single_product(
    *,
    product_id: str,
    output: Path,
    generated_source: Path,
    reference_source: Path,
    allow_reference_fallback: bool = True,
) -> MaterializedProduct:
    """Publish exactly one product and write its provenance next to it."""
    if generated_source.is_file():
        source = generated_source
        source_kind = "generated"
    elif allow_reference_fallback and reference_source.is_file():
        source = reference_source
        source_kind = "accepted_reference"
    else:
        raise FileNotFoundError(generated_source)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output)
    digest = sha256(output)
    metadata = {
        "product_id": product_id,
        "output": str(output),
        "source": str(source),
        "source_kind": source_kind,
        "sha256": digest,
        "created_utc": datetime.now(UTC).isoformat(),
    }
    output.with_suffix(output.suffix + ".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return MaterializedProduct(product_id, output, source, source_kind, digest)
