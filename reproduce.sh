#!/usr/bin/env sh
set -eu
project_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
cd "$project_root"
uv sync --frozen --extra dev
uv run --frozen python scripts/verify_raw_inputs.py
uv run --frozen python scripts/reproduce_scientific.py --clean --force
uv run --frozen python scripts/verify_scientific_reproduction.py
uv run --frozen python scripts/verify_storage_efficiency.py
uv run --frozen python scripts/verify_publication_identity.py
