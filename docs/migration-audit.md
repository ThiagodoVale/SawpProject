# SAWP 1.0 migration audit

## Problems found in the previous code

- The former package initializer imported every module automatically, making a simple
  import load plotting, GPU, and geospatial dependencies.
- Several scripts executed complete analyses during import and had no `main` function.
- Relative paths such as `./data`, `./outputs`, and `./outputs_trended` were duplicated.
- Multi-gigabyte NPZ cubes were loaded completely into memory.
- V2, V3, V7, V8, V9, backup, and copied variants had no authoritative registry.
- Scientific calculations were duplicated with slightly different limits and sign
  conventions.
- Debugging code, undefined variables, and mandatory GPU dependencies remained in some
  workflows.
- Previous tests required complete datasets instead of isolating numerical functions.

## Version 1.0 decisions

- A small explicit root API with no automatic submodule loading.
- One TOML configuration for periods, climatology, quantiles, QC, domains, and paths.
- Pure functions for detrending, anomalies, seasonal aggregation, quantiles, composites,
  pressure integration, moisture transport, convergence, and theta-e.
- Chunked detrending and memory mapping for uncompressed legacy NPZ members.
- Zarr or chunked NetCDF as the recommended format for new four-dimensional products.
- A manifest linking each of the 18 figures to one stage, producer, and source artifact.
- One CLI for inventory, diagnostics, execution, resume, collection, and verification.
- JSON reports and SHA-256 hashes for every stage.
- Small unit tests that do not download ERA5 or load complete data cubes.

## Explicit scientific settings

- Analysis period: 1959–2023.
- Climatology: 1980–2010.
- AMJJ season: April through July.
- Quantile thresholds: 10%, 25%, 75%, and 90%.
- Moisture convergence is published as `-div(Q)`, positive for convergence.
- Latent heat flux is published as positive upward.
- Station gaps are never filled automatically in time.
- Regional coverage requires the greater of 30 stations or 10% of the valid network.

## Configuration-preserved inconsistencies

The publication uses 35°W–25°W for the SAWP index in the methods and several captions,
while the Figure 1 caption and some historical scripts use 34°W–25°W. Version 1.0 keeps
`sawp_index` and `sawp_figure1` as distinct named domains.

Some historical products used a 1990–2020 climatology, although the revised manuscript
specifies 1980–2010. The default configuration follows the revised manuscript and the
provenance report records any override.
