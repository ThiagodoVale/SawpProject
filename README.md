# SAWP 1.1

SAWP 1.1 is the minimal raw-to-figure implementation used to create the 9 main and
9 supplementary publication figures. It contains only the ERA5 downloader, the
scientific processing code, the figure renderer, configuration, and essential tests.

The legacy scripts, NPZ workflow, Zarr cache, frozen reference figures, migration
utilities, Docker files, and release accessories from SAWP 1.0 are intentionally not
part of this version. They remain preserved in the `v1.0.0` and `v1.0.1` Git tags and
in the separate `SAWP_v1.0` folder.

## 1. Install

Install [uv](https://docs.astral.sh/uv/) and create the locked environment:

```powershell
uv sync --frozen
```

## 2. Prepare non-ERA5 observations

Place the measured rain-gauge and domain files in this layout:

```text
data/raw/
  domain/
    sst.nc
    srtm30plus_v11_land.nc
  rainfall/
    monitor/**/*.txt
```

These files are not provided by the Climate Data Store and are therefore not created
by the ERA5 downloader.

## 3. Download ERA5

Create the CDS API credentials file, accept the dataset licences on the CDS website,
and run:

```powershell
uv run python scripts/download_era5.py
```

The downloader retrieves yearly parts and merges them into the only four ERA5 files
used by the scientific pipeline:

```text
data/raw/era5/Era5_Dataset_sst.nc
data/raw/era5/Era5_Dataset_meanHF.nc
data/raw/era5/Era5_Dataset_t_q.nc
data/raw/era5/Era5_Dataset_u_v_w.nc
```

Interrupted downloads resume from completed yearly parts. Use `--keep-parts` to retain
those parts after a successful merge.

## 4. Generate every figure from zero

```powershell
uv run python scripts/generate_figures.py
```

This removes reproducible intermediates, recalculates all scientific products from
the raw inputs, and writes the 18 figures to `results/figures`. A later exploratory
rerun can reuse completed products with `--resume`.

## Minimal structure

```text
config/default.toml          Scientific configuration
scripts/download_era5.py     Download the four required ERA5 datasets
scripts/generate_figures.py  Recalculate and render all 18 figures
src/sawp/scientific/         Scientific calculations and rendering
src/sawp/rainfall/           Measured rainfall ingestion and quality control
src/sawp/physics/            Moisture transport and convergence
tests/test_core.py           Essential pipeline tests
```
