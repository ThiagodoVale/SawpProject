# Capsule data

## Included data

`reference_figures/` contains the 18 accepted publication figures used as immutable
artifact baselines. Their hashes are recorded in `checksums/reference_figures.sha256`.
These files make the artifact-level capsule self-contained and allow any clone to
reproduce the exact publication PNG bytes.

## Scientific source data

Large observational and ERA5 files are intentionally excluded from Git. A future
scientific-recalculation bundle must be placed under `data/raw`, use the layout recorded
in `data/manifest.json`, and be distributed from a versioned research-data repository.
The GitHub capsule must never silently substitute the artifact baselines for an
independent raw-data calculation.

The configured local layout is:

```text
data/
  raw/
    era5/       ERA5 NetCDF inputs
    rainfall/
      monitor/  measured station files grouped by source/state
    domain/     domain SST and topography NetCDF inputs
  interim/      reproducible intermediate products
  processed/    canonical scientific tables and data products
```

All paths are declared in `config/default.toml` and resolved relative to that file.

The current guaranteed level is therefore `artifact-byte-identical`: every published
figure is regenerated as a verified artifact, while raw-to-figure scientific
recalculation remains a separately declared workflow.
