# Changelog

All notable changes to SAWP are documented here.

## [1.0.1] - 2026-11-08

Patch release for portable continuous integration.

### Fixed

- Changed the Python request from the unavailable exact patch `3.12.14` to the
  portable minor request `3.12`.
- Updated GitHub Actions to run lint, the complete test suite, and checksum
  verification of all 18 versioned reference figures without requiring the large
  raw NetCDF files that are intentionally excluded from Git.

### Versioning

- Every accepted small change now increments the patch version, and published tags
  remain immutable.

## [1.0.0] - 2026-11-07

First definitive reproducible release.

### Added

- Modular raw-to-figure pipeline for all 9 main and 9 supplementary figures.
- Frozen `uv.lock`, raw-input SHA-256 inventory, per-figure provenance, and release
  manifests.
- Compact NetCDF intermediates and optional invalidated Zarr v2 cache.
- Public producer script for each figure and automated scientific verification.

### Scientific parity

- Equivalent potential temperature follows the exact legacy dew-point/LCL formula.
- Figures 7 and 9 retain the complete 60°W–10°W pressure-longitude vector domain.
- Figure 9 standardizes circulation components before vertical interpolation.
- Figures 6 and 9 consistently plot moisture-flux convergence (`-div(Q)`), with
  positive convergence represented by the positive side of the colorbar.
- Legacy half-open SAWP latitude slicing is reproduced for vertical sections.

### Validation

- All 18 figures generated exclusively from raw NetCDF and measured rainfall inputs.
- Scientific provenance verified for 18/18 figures.
- No active dependency on the former workspace, legacy NPZ products, or Docker.
