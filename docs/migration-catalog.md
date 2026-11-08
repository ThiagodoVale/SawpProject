# Complete migration catalog

All 63 historical Python files were inventoried.

| Status | Count |
|---|---:|
| consolidated | 21 |
| duplicate | 1 |
| replaced | 25 |
| split_by_product | 16 |

`migration_catalog.json` records the normalized source SHA-256, migration status, and
replacement for every file. Complete historical sources are stored in
`archive/legacy_sources` and are never imported by the active package.

## Status definitions

- `replaced`: superseded by a clean equivalent API.
- `consolidated`: multiple variants became one parameterized workflow.
- `split_by_product`: a monolithic script was separated into `scripts/products`.
- `duplicate`: a byte-equivalent copy retained only in the historical archive.
