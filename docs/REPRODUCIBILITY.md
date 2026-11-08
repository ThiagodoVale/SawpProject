# Reproducibility contract

## Guaranteed workflow

`reproduce.ps1` and `reproduce.sh` perform an independent scientific recalculation:

1. install the exact dependency graph from `uv.lock`;
2. validate the complete raw-data layout;
3. remove only regenerated files in `data/interim`, `data/processed`, and
   `results/figures`;
4. ingest and quality-control measured rain-gauge records;
5. derive AMJJ rainfall, SST, flux, moisture, wind, omega, and theta-e products;
6. calculate quantile classes, anomalies, trends, composites, correlations, and
   robustness diagnostics;
7. render Figure 1-9 and Figure S1-S9 without reading accepted PNG artifacts;
8. write an adjacent provenance record for each PNG;
9. verify that all 18 provenance records state `generated_from_raw`;
10. compare all 18 generated rasters with the accepted publication figures.

The processed NetCDF/CSV files are outputs, not prerequisites. Removing `data/processed`,
`data/interim`, and `results/figures` is therefore a valid clean-room test.

## Deterministic inputs

The scientific inputs live under `data/raw` and are never modified by the pipeline.
`checksums/raw_inputs.sha256` records their content hashes. The repository's Python
environment is frozen by `.python-version` and `uv.lock`.

The validated local bundle contains 5,694 raw files. `scripts/verify_raw_inputs.py`
checks every file before a clean reproduction begins. On the validation machine, the
complete clean workflow took approximately 35-45 minutes, dominated by multilevel ERA5
atmospheric processing.

Because the data bundle is too large for ordinary Git hosting, publishing the code on
GitHub is not sufficient by itself. Publish the raw bundle in a release or a persistent
data repository and preserve the exact relative paths and SHA-256 manifest.

## Native execution

```powershell
./reproduce.ps1
```

```sh
./reproduce.sh
```

## Verification only

```sh
uv run --frozen python scripts/verify_scientific_reproduction.py
```

Successful verification means all 18 expected PNGs exist, are non-empty, and have
raw-derived provenance. `scripts/verify_publication_identity.py` separately verifies
normalized raster and layout equivalence. The final validated capsule passes both
checks for 18/18 figures. Byte equality is not required because font, FreeType, PNG
metadata, and platform rendering can differ without changing scientific content.

## Validated resource profile

The final compact processed products occupy 161,646,110 bytes (0.151 GiB), with no
legacy NPZ products and no mandatory Zarr cache. The clean run temporarily used about
7.6 GB of RAM on the validation machine. These measurements are informative rather
than fixed requirements and may vary with the operating system and dependency build.
