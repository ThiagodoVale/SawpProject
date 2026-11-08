from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


@dataclass(frozen=True)
class ImageIdentity:
    generated: str
    reference: str
    generated_size: tuple[int, int]
    reference_size: tuple[int, int]
    sha256_equal: bool
    thumbnail_mae: float
    visually_equivalent: bool

    def to_dict(self) -> dict:
        return asdict(self)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_images(
    generated: Path,
    reference: Path,
    *,
    thumbnail_size: tuple[int, int] = (512, 512),
    maximum_mae: float = 20.0,
) -> ImageIdentity:
    with Image.open(generated) as generated_image, Image.open(reference) as reference_image:
        generated_rgb = generated_image.convert("RGB")
        reference_rgb = reference_image.convert("RGB")
        generated_size = generated_rgb.size
        reference_size = reference_rgb.size
        generated_thumbnail = np.asarray(
            ImageOps.fit(generated_rgb, thumbnail_size, method=Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        reference_thumbnail = np.asarray(
            ImageOps.fit(reference_rgb, thumbnail_size, method=Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
    mae = float(np.mean(np.abs(generated_thumbnail - reference_thumbnail)))
    exact = sha256(generated) == sha256(reference)
    return ImageIdentity(
        generated=str(generated),
        reference=str(reference),
        generated_size=generated_size,
        reference_size=reference_size,
        sha256_equal=exact,
        thumbnail_mae=mae,
        visually_equivalent=exact or mae <= maximum_mae,
    )
