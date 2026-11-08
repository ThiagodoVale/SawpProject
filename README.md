# SAWP 1.0.0

SAWP 1.0 is a modular raw-to-figure implementation of the South Atlantic Warm Pool
(SAWP) and Eastern Northeast Brazil (ENEB) rainfall analyses. It reads the original
ERA5 NetCDF files and measured rain-gauge text files, performs quality control and
scientific preprocessing, and creates the 9 main and 9 supplementary figures.

The definitive `v1.0.0` release freezes the raw-input inventory, Python dependency
lock, 18 accepted reference figures, distribution artifacts, and SHA-256 manifests.
See `CHANGELOG.md` and `docs/releases/v1.0.0.md` for the validated release record.

## Complete reproduction

Install [uv](https://docs.astral.sh/uv/), place the raw inputs under `data/raw`, and run:

```powershell
./reproduce.ps1
```

On Linux or macOS, run:

```sh
./reproduce.sh
```

The command creates the frozen Python environment, verifies the raw inputs, calculates
monthly trends lazily, persists only compact coefficients and annual/composite fields,
renders all 18 figures, writes provenance JSON beside every PNG, and verifies the
scientific products. It does not create the historical 38-42 GB NPZ duplication and
does not read code or derived products from the former SAWP workspace.

A clean run reads the NetCDF files directly and normally needs less than 3 GB of
additional persistent storage. For repeated experiments, an optional compressed Zarr
cache can be enabled in `config/default.toml` or built explicitly with
`scripts/build_zarr_cache.py`. The cache uses the stable Zarr v2 layout with Zstandard
compression and is automatically invalidated when the size or modification time of
its source NetCDF changes. It is a performance cache, never an authoritative input.

The final validated clean run produced 161,646,110 bytes (0.151 GiB) of compact
scientific products, no legacy NPZ files, and all 18 PNGs. The multilevel atmospheric
calculation is the dominant cost and typically takes 35-45 minutes on the validation
machine. Runtime is hardware-dependent; the storage report is regenerated on every
`reproduce.ps1` run. A normal rerun reuses valid processed products and is much faster.

There are two separate validation levels:

1. `verify_scientific_reproduction.py` checks that all 18 figures were generated from
   raw inputs and complete calculated products.
2. `verify_publication_identity.py` compares the PNG raster and layout with the
   accepted article figures. The final compact pipeline passes this validation for all
   18 figures. Reference PNGs are validation targets only and are never read as data or
   copied into generated output.

The expected raw layout is:

```text
data/raw/
  domain/
    sst.nc
    srtm30plus_v11_land.nc
  era5/
    Era5_Dataset_decomposit.nc
    Era5_Dataset_evp.nc
    Era5_Dataset_lh.nc
    Era5_Dataset_lhsh.nc
    Era5_Dataset_meanHF.nc
    Era5_Dataset_sst.nc
    Era5_Dataset_t_q.nc
    Era5_Dataset_u_v_w.nc
  rainfall/monitor/**/*.txt
```

Raw files are intentionally excluded from Git because of their size. The authoritative
SHA-256 inventory covers all 5,694 local inputs in `checksums/raw_inputs.sha256`; a
GitHub release or data repository must distribute the matching data bundle with the
source repository.

## Commands

```powershell
# Recalculate all scientific products and figures efficiently
uv run --frozen python scripts/reproduce_scientific.py --clean --force

# Optional: build a reusable Zarr cache for repeated experiments
uv run --frozen python scripts/build_zarr_cache.py

# Verify the 18 raw-derived outputs
uv run --frozen python scripts/verify_scientific_reproduction.py

# Verify the visual identity of all 18 figures against the accepted article rasters
uv run --frozen python scripts/verify_publication_identity.py

# Verify that no legacy NPZ duplication was recreated and report disk usage
uv run --frozen python scripts/verify_storage_efficiency.py

# Generate all figures while reusing valid processed products
uv run --frozen python scripts/generate_all_figures.py --resume

# Preview the complete output plan without calculating
uv run --frozen python scripts/generate_all_figures.py --dry-run

# Run one public figure producer
uv run --frozen python scripts/products/figure_1.py

# Build and freeze the v1.0.0 release after a successful clean reproduction
uv build
uv run --frozen python scripts/freeze_release.py --version 1.0.0
```

## Original publication rendering

The accepted article figures were produced by a specific set of historical scripts.
Their product-to-script mapping is recorded in `src/sawp/publication/catalog.py` and
the sources are vendored under `archive/legacy_sources` for audit. The active pipeline
implements their calculations with chunked xarray/Dask operations and compact products;
it does not execute the storage-heavy NPZ workflow.

`use_original_rendering.ps1` remains only as an audit/migration command for collecting
results from an older workspace; it is not used by the self-contained reproduction.

```powershell
./use_original_rendering.ps1 `
  -OriginalWorkspace "C:\Users\silva\Documents\SAWP\SAWP"
```

This command does not use the accepted PNG files as generated output. It copies only
the PNGs created by the mapped producer scripts, records each producer script and its
SHA-256 hash in provenance JSON, and then performs both exact-hash and normalized
visual comparisons against `data/reference_figures`.

The validation report is written to `results/publication_identity_report.json`.
`EXACT` means byte-for-byte equality. `VISUAL` means that the scientific content and
layout match but the raster differs slightly because of font/FreeType/Pillow rendering.
The default renderer is pinned to Matplotlib 3.10.7; Figure 6 records its historical
Matplotlib 3.11.2 exception in the publication catalog.

## Project layout

```text
src/sawp/scientific/  raw-to-processed calculations and figure rendering
src/sawp/rainfall/    station ingestion, QC, homogeneity, and regional series
src/sawp/physics/     moisture, convergence, and thermodynamic calculations
scripts/products/     one public entry point for each of the 18 figures
data/raw/             immutable scientific inputs
data/processed/       regenerated scientific datasets
results/figures/      regenerated PNG files and provenance JSON
archive/              inactive historical sources retained for audit only
```

The analysis period is 1959-2023, the season is April-July (AMJJ), and the
climatological reference is 1980-2010. All paths and domains are explicit in
`config/default.toml`. Active figure producers never import archived code and never use
accepted PNG files as a fallback.

See `docs/REPRODUCIBILITY.md` for the reproducibility contract and validation details.
