from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from sawp.config import load_config
from sawp.domains import Domain, normalize_longitude
from sawp.generate_all import generate_all_figures
from sawp.io import inspect_npz, open_npz_memmap
from sawp.pipeline import load_manifest


class StructureTests(unittest.TestCase):
    def test_domain_with_360_degree_longitudes(self) -> None:
        domain = Domain("sawp", west=-35.0, east=-25.0, south=-12.5, north=-4.0)
        mask = domain.mask(np.array([320.0, 330.0, 340.0]), np.array([-10.0]))
        np.testing.assert_array_equal(mask, [[False, True, False]])
        self.assertEqual(normalize_longitude(330.0), -30.0)

    def test_uncompressed_npz_member_can_be_memory_mapped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "array.npz"
            expected = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            np.savez(path, data=expected, time=np.arange(2))
            schema = inspect_npz(path)
            self.assertEqual(schema["data"].shape, (2, 3, 4))
            mapped = open_npz_memmap(path)
            np.testing.assert_array_equal(mapped, expected)
            mapped._mmap.close()

    def test_publication_manifest_has_all_products(self) -> None:
        manifest = load_manifest()
        self.assertEqual(len(manifest.products), 18)
        self.assertEqual(len(manifest.stages), 18)
        self.assertEqual(len({product.id for product in manifest.products}), 18)
        self.assertEqual(len({product.reference for product in manifest.products}), 18)
        self.assertTrue(all(stage.executor == "native" for stage in manifest.stages))
        self.assertTrue(all(len(stage.expected) == 1 for stage in manifest.stages))
        self.assertEqual(
            {product.stage for product in manifest.products},
            {stage.id for stage in manifest.stages},
        )

    def test_project_paths_use_the_explicit_raw_data_layout(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = load_config(root / "config" / "default.toml")
        self.assertEqual(config.paths.raw_data, root / "data" / "raw")
        self.assertEqual(config.paths.raw_era5, root / "data" / "raw" / "era5")
        self.assertEqual(config.paths.raw_rainfall, root / "data" / "raw" / "rainfall")
        self.assertEqual(config.paths.raw_domain, root / "data" / "raw" / "domain")
        self.assertEqual(config.paths.interim, root / "data" / "interim")
        self.assertEqual(config.paths.processed, root / "data" / "processed")

    def test_legacy_sources_are_auditable_but_inactive(self) -> None:
        root = Path(__file__).resolve().parents[1]
        archived = list((root / "archive" / "legacy_sources").rglob("*.py"))
        self.assertEqual(len(archived), 64)
        manifest = load_manifest()
        self.assertFalse(any(stage.script.startswith("archive/") for stage in manifest.stages))

    def test_generate_all_figures_covers_the_complete_manifest(self) -> None:
        root = Path(__file__).resolve().parents[1]
        reports = generate_all_figures(
            config_path=root / "config" / "default.toml",
            manifest_path=root / "config" / "publication.toml",
            dry_run=True,
        )
        self.assertEqual(len(reports), 18)
        self.assertTrue(all(report["status"] == "dry-run" for report in reports))


if __name__ == "__main__":
    unittest.main()
