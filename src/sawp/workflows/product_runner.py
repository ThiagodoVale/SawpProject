from __future__ import annotations

from pathlib import Path

from sawp.config import load_config
from sawp.scientific import ScientificPipeline


def run_product(product_id: str) -> Path:
    config = load_config()
    output = ScientificPipeline(config).render_product(product_id)
    print(f"{product_id}: {output} (generated_from_raw)")
    return output
