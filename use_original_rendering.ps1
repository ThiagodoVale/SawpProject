param(
    [Parameter(Mandatory = $true)]
    [string]$OriginalWorkspace
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $projectRoot

if (-not (Test-Path -LiteralPath $OriginalWorkspace -PathType Container)) {
    throw "Original SAWP workspace not found: $OriginalWorkspace"
}

uv sync --frozen --extra dev
uv run --frozen python scripts/verify_raw_inputs.py
uv run --frozen python scripts/collect_publication_outputs.py --workspace $OriginalWorkspace
uv run --frozen python scripts/verify_publication_identity.py
uv run --frozen python scripts/verify_scientific_reproduction.py
