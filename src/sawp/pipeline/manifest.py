from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Stage:
    id: str
    description: str
    script: str
    executor: str = "native"
    args: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    expected: tuple[str, ...] = ()


@dataclass(frozen=True)
class Product:
    id: str
    label: str
    stage: str
    reference: str
    source: str


@dataclass(frozen=True)
class Manifest:
    source: Path
    title: str
    stages: tuple[Stage, ...]
    products: tuple[Product, ...]

    def stage(self, identifier: str) -> Stage:
        for stage in self.stages:
            if stage.id == identifier:
                return stage
        raise KeyError(f"unknown stage: {identifier}")

    def product(self, identifier: str) -> Product:
        for product in self.products:
            if product.id == identifier:
                return product
        raise KeyError(f"unknown product: {identifier}")


def default_manifest_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "publication.toml"


def load_manifest(path: str | Path | None = None) -> Manifest:
    source = Path(path) if path is not None else default_manifest_path()
    source = source.expanduser().resolve()
    with source.open("rb") as stream:
        raw = tomllib.load(stream)
    stages = tuple(
        Stage(
            id=item["id"],
            description=item["description"],
            script=item["script"],
            executor=item.get("executor", "native"),
            args=tuple(item.get("args", ())),
            depends_on=tuple(item.get("depends_on", ())),
            expected=tuple(item.get("expected", ())),
        )
        for item in raw.get("stages", ())
    )
    products = tuple(Product(**item) for item in raw.get("products", ()))
    known_stages = {stage.id for stage in stages}
    for stage in stages:
        unknown = set(stage.depends_on) - known_stages
        if unknown:
            raise ValueError(f"stage {stage.id} has unknown dependencies: {sorted(unknown)}")
    for product in products:
        if product.stage not in known_stages:
            raise ValueError(f"product {product.id} uses unknown stage {product.stage}")
    identifiers = [product.id for product in products]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("product identifiers must be unique")
    return Manifest(
        source=source,
        title=raw["publication"]["title"],
        stages=stages,
        products=products,
    )
