from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import ProjectConfig
from .manifest import Manifest
from .provenance import sha256


def _image_distance(first: Path, second: Path) -> int | None:
    try:
        from PIL import Image
    except ImportError:
        return None
    try:
        with Image.open(first) as image:
            a = image.convert("L").resize((9, 8))
            a_pixels = list(a.getdata())
        with Image.open(second) as image:
            b = image.convert("L").resize((9, 8))
            b_pixels = list(b.getdata())
    except Exception:
        return None
    a_hash = [
        a_pixels[row * 9 + column] > a_pixels[row * 9 + column + 1]
        for row in range(8)
        for column in range(8)
    ]
    b_hash = [
        b_pixels[row * 9 + column] > b_pixels[row * 9 + column + 1]
        for row in range(8)
        for column in range(8)
    ]
    return sum(left != right for left, right in zip(a_hash, b_hash, strict=True))


def verify_products(config: ProjectConfig, manifest: Manifest, mode: str) -> list[dict[str, Any]]:
    if mode not in {"reference", "generated"}:
        raise ValueError("mode must be reference or generated")
    rows: list[dict[str, Any]] = []
    for product in manifest.products:
        reference = config.paths.reference_figures / product.reference
        generated = config.paths.workspace / product.source
        target = reference if mode == "reference" else generated
        row: dict[str, Any] = {
            "id": product.id,
            "label": product.label,
            "stage": product.stage,
            "path": str(target),
            "exists": target.is_file(),
        }
        if target.is_file():
            row["bytes"] = target.stat().st_size
            row["sha256"] = sha256(target)
        if mode == "generated" and target.is_file() and reference.is_file():
            row["reference_sha256"] = sha256(reference)
            row["identical"] = row["sha256"] == row["reference_sha256"]
            if target.suffix.lower() == reference.suffix.lower() == ".png":
                row["visual_hash_distance"] = _image_distance(target, reference)
        rows.append(row)
    return rows
