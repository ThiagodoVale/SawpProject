# Changelog

## [1.1.0] - 2026-11-08

- Reduced the repository to the ERA5 download and raw-to-figure scientific path.
- Reduced the required ERA5 inventory from eight files to the four files actually used.
- Removed legacy, NPZ, Zarr, migration, artifact-freezing, and per-figure wrappers.
- Consolidated execution into `download_era5.py` and `generate_figures.py`.
- Preserved all SAWP 1.0 history under the `v1.0.0` and `v1.0.1` tags.
