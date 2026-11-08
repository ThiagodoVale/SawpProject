from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from .legacy_builder import build_legacy_intermediates, prepare_workspace
from .verification import compare_images

SCRIPT_SEQUENCE = (
    "RealizarComparacaoClimatologias_v2.py",
    "RealizarAnalises.py",
    "RealizarAnalisesMudancasResumo.py",
    "RealizarAnalises_CTST_QuiverReferencia.py",
    "RealizarGradienteEstabilidadeVertical.py",
    "RealizarTransporteVaporConvergencia.py",
    "RealizarTratamentoCalorLatenteMedioComLag0.py",
    "RealizarGraficosAuxiliarSerieTemporalTSMxChuvaV7.py",
    "RealizarAnaliseMatrizCorrelacao.py",
    "RealizarAnalisesCoolingSSTV2.py",
    "RealizarAnalises_dev_tendencias_extremos.py",
)

CAPTURE_AFTER = {
    "RealizarComparacaoClimatologias_v2.py": {
        "Figure2.png": "Series_TSM_Com_Sem_tendencia.png",
    },
    "RealizarAnalises.py": {
        "Figure3.png": "TSM_AnomaliasQuantis_Subplot.png",
        "Figure4.png": "Combined_Prec_TS_Subplot.png",
        "Figure5.png": "ACPC_AnomaliasQuantis_Subplot.png",
    },
    "RealizarAnalisesMudancasResumo.py": {"Figure8.png": "SAWP_subplot_unificado.png"},
    "RealizarAnalises_CTST_QuiverReferencia.py": {
        "Figure9.png": (
            "figuras/Subplot_CT_ST_quiver_referencia/"
            "Subplot_CT-ST_quiver_referencia_w_corrigido_unit.png"
        ),
    },
    "RealizarGradienteEstabilidadeVertical.py": {
        "Figure7.png": "figuras/GradienteEstabilidadeVertical/ThetaE_Quantis_original.png",
    },
    "RealizarTransporteVaporConvergencia.py": {
        "Figure6.png": "figuras/TransporteVaporConvergencia/TRSP_AnomaliasQuantis_Subplot.png",
    },
    "RealizarTratamentoCalorLatenteMedioComLag0.py": {
        "FigureS9.png": "figuras/CalorLatenteMedioComLag0/LH_AnomaliasQuantis_Lag0.png",
    },
    "RealizarGraficosAuxiliarSerieTemporalTSMxChuvaV7.py": {
        "FigureS5.png": (
            "outputs/reviewer_event_based_april/"
            "Panel_max_rainfall_vs_max_SST_with_without_trend_independent_maxima.png"
        ),
    },
    "RealizarAnaliseMatrizCorrelacao.py": {
        "FigureS6.png": (
            "figuras_revisor/S5C_matriz_pearson_metodo_original_AMJJ_anual_mascara_quantis_TSM.png"
        ),
    },
    "RealizarAnalisesCoolingSSTV2.py": {
        "FigureS7.png": "Fig_Sx_lag_peak_SST_monthly_cooling_rainfall.png",
    },
    "RealizarAnalises_dev_tendencias_extremos.py": {
        "FigureS8.png": "Subplot_Tendencias_AMJJ.png",
    },
}

RAINFALL_CAPTURE = {
    "Figure1.png": "data/processed/figures/Mapa_dominio_SAWP_ENEB_estacoes.png",
    "FigureS1.png": "data/processed/robustness/valid_stations_per_month.png",
    "FigureS2.png": "data/processed/robustness/robustness_completeness_series.png",
    "FigureS3.png": "data/processed/robustness/robustness_long_records_series.png",
    "FigureS4.png": "data/processed/stations_unusual_regime_map.png",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _execute(project_root: Path, workspace: Path, script: str) -> None:
    command = [
        sys.executable,
        "-m",
        "sawp.publication.legacy_exec",
        str(workspace),
        script,
    ]
    environment = os.environ.copy()
    environment.update({"MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"})
    source_root = project_root / "src"
    environment["PYTHONPATH"] = str(source_root) + os.pathsep + environment.get("PYTHONPATH", "")
    subprocess.run(command, cwd=project_root, env=environment, check=True)


def _capture(workspace: Path, captured: Path, mapping: dict[str, str]) -> None:
    captured.mkdir(parents=True, exist_ok=True)
    for filename, relative in mapping.items():
        source = workspace / relative
        if not source.is_file():
            raise FileNotFoundError(f"original producer did not create {source}")
        shutil.copy2(source, captured / filename)


def reproduce_publication(
    project_root: Path,
    *,
    clean: bool = False,
    force: bool = False,
    keep_intermediates: bool = False,
) -> list[dict]:
    project_root = project_root.resolve()
    workspace = project_root / "data" / "interim" / "publication_workspace"
    prepare_workspace(project_root, workspace, clean=clean)
    captured = workspace / "captured"
    if clean and captured.exists():
        shutil.rmtree(captured)

    _execute(project_root, workspace, "RealizarTratamentoDadosMedidosMapaDominio.py")
    build_legacy_intermediates(workspace, force=force)
    for script in SCRIPT_SEQUENCE:
        if not (workspace / script).is_file():
            raise FileNotFoundError(f"vendored producer is missing: {script}")
        _execute(project_root, workspace, script)
        mapping = CAPTURE_AFTER.get(script)
        if mapping:
            _capture(workspace, captured, mapping)

    # The auxiliary script is the final accepted English renderer for S1-S4.
    _execute(project_root, workspace, "RealizarTratamentoDadosMedidosMapaDominioAuxiliar.py")
    _capture(workspace, captured, RAINFALL_CAPTURE)

    destination = project_root / "results" / "figures"
    destination.mkdir(parents=True, exist_ok=True)
    reference = project_root / "data" / "reference_figures"
    reports: list[dict] = []
    for source in sorted(captured.glob("Figure*.png")):
        output = destination / source.name
        shutil.copy2(source, output)
        comparison = compare_images(output, reference / source.name).to_dict()
        metadata = {
            "schema_version": 3,
            "product_id": source.stem.lower().replace("figure", "figure-"),
            "source_kind": "generated_from_raw_via_original_script",
            "output": str(output),
            "sha256": _sha256(output),
            "created_utc": datetime.now(UTC).isoformat(),
            "publication_identity": comparison,
        }
        output.with_suffix(".png.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        reports.append(metadata)
    if len(reports) != 18:
        raise RuntimeError(f"expected 18 captured figures, found {len(reports)}")
    failures = [
        report["output"]
        for report in reports
        if not report["publication_identity"]["visually_equivalent"]
    ]
    report_path = project_root / "results" / "publication_from_zero_report.json"
    report_path.write_text(
        json.dumps({"complete": not failures, "failures": failures, "products": reports}, indent=2),
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError("publication identity failed:\n" + "\n".join(failures))
    if not keep_intermediates:
        shutil.rmtree(workspace / "outputs", ignore_errors=True)
        shutil.rmtree(workspace / "outputs_trended", ignore_errors=True)
    return reports
