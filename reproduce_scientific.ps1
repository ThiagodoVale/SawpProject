$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot
uv sync --frozen --extra dev
uv run --frozen python scripts/verify_raw_inputs.py
uv run --frozen python scripts/reproduce_scientific.py --clean --force
uv run --frozen python scripts/verify_scientific_reproduction.py
uv run --frozen python scripts/verify_storage_efficiency.py
