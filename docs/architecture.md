# SAWP 1.0 scientific architecture

## Data flow

1. Rainfall ingestion reads station files, applies physical limits, removes spatial
   outliers, evaluates homogeneity, and produces the regional series.
2. Preprocessing reads ERA5 variables with xarray and removes the linear trend at every
   grid point.
3. The analysis layer calculates the AMJJ SAWP index, Q10/Q25/Q75/Q90 thresholds,
   composites, correlations, and scientific tables.
4. Moisture, stability, trend, correlation, cooling, and heat-flux workflows consume
   canonical products before any figure is rendered.
5. The publication pipeline processes all 18 products, calculates hashes, and records
   execution reports.

## Data contracts

- Monthly time coordinates with no automatic interpolation of observational gaps.
- Coordinates identified by name instead of fixed file positions.
- Pressure explicitly converted from hPa to Pa before vertical integration.
- Published latent heat flux uses a positive-upward convention.
- `omega < 0` denotes ascending motion in pressure coordinates.
- Every composite records the years, months, quantiles, and climatology used.
- Scientific tables or data cubes are retained alongside rendered figures.

## Memory model

Detrending supports spatial chunks and caller-provided output arrays. Zarr or chunked
NetCDF is recommended for new products. The NPZ reader remains available for current
legacy results and validates keys, dimensions, and coordinates before returning data.

## Provenance

Each execution report includes the Python version, platform, effective configuration,
stage, command, duration, return code, artifacts, and SHA-256 hashes. The publication
manifest links every figure to its producer and expected output.
