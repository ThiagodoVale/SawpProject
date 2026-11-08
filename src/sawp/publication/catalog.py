from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PublicationProduct:
    product_id: str
    filename: str
    producer_script: str
    source_output: str
    matplotlib_version: str = "3.10.7"


def _product(
    product_id: str,
    producer_script: str,
    source_output: str,
    *,
    matplotlib_version: str = "3.10.7",
) -> PublicationProduct:
    suffix = product_id.removeprefix("figure-")
    if suffix.startswith("s"):
        filename = f"FigureS{int(suffix[1:])}.png"
    else:
        filename = f"Figure{int(suffix)}.png"
    return PublicationProduct(
        product_id=product_id,
        filename=filename,
        producer_script=producer_script,
        source_output=source_output,
        matplotlib_version=matplotlib_version,
    )


PUBLICATION_PRODUCTS = (
    _product(
        "figure-1",
        "RealizarTratamentoDadosMedidosMapaDominio.py",
        "data/processed/figures/Mapa_dominio_SAWP_ENEB_estacoes.png",
    ),
    _product(
        "figure-2",
        "RealizarComparacaoClimatologias_v2.py",
        "Series_TSM_Com_Sem_tendencia.png",
    ),
    _product("figure-3", "RealizarAnalises.py", "TSM_AnomaliasQuantis_Subplot.png"),
    _product(
        "figure-4",
        "RealizarAnalises_dev.py",
        "figuras/Quiver_Alterados_20260925/03_Tendencias_com_referencia/Combined_Prec_TS_Subplot.png",
    ),
    _product("figure-5", "RealizarAnalises.py", "ACPC_AnomaliasQuantis_Subplot.png"),
    _product(
        "figure-6",
        "RealizarTransporteVaporConvergencia.py",
        "figuras/TransporteVaporConvergencia/TRSP_AnomaliasQuantis_Subplot.png",
        matplotlib_version="3.11.2",
    ),
    _product(
        "figure-7",
        "RealizarGradienteEstabilidadeVertical.py",
        "figuras/GradienteEstabilidadeVertical/ThetaE_Quantis_original.png",
    ),
    _product("figure-8", "RealizarAnalisesMudancasResumo.py", "SAWP_subplot_unificado.png"),
    _product(
        "figure-9",
        "RealizarAnalises_CTST_QuiverReferencia.py",
        "figuras/Subplot_CT_ST_quiver_referencia/Subplot_CT-ST_quiver_referencia_w_corrigido.png",
    ),
    _product(
        "figure-s1",
        "RealizarTratamentoDadosMedidosMapaDominioAuxiliar.py",
        "data/processed/robustness/valid_stations_per_month.png",
    ),
    _product(
        "figure-s2",
        "RealizarTratamentoDadosMedidosMapaDominioAuxiliar.py",
        "data/processed/robustness/robustness_completeness_series.png",
    ),
    _product(
        "figure-s3",
        "RealizarTratamentoDadosMedidosMapaDominioAuxiliar.py",
        "data/processed/robustness/robustness_long_records_series.png",
    ),
    _product(
        "figure-s4",
        "RealizarTratamentoDadosMedidosMapaDominioAuxiliar.py",
        "data/processed/stations_unusual_regime_map.png",
    ),
    _product(
        "figure-s5",
        "RealizarGraficosAuxiliarSerieTemporalTSMxChuvaV7.py",
        "outputs/reviewer_event_based_april/Panel_max_rainfall_vs_max_SST_with_without_trend_independent_maxima.png",
    ),
    _product(
        "figure-s6",
        "RealizarAnaliseMatrizCorrelacao.py",
        "figuras_revisor/S5C_matriz_pearson_metodo_original_AMJJ_anual_mascara_quantis_TSM.png",
    ),
    _product(
        "figure-s7",
        "RealizarAnalisesCoolingSSTV2.py",
        "Fig_Sx_lag_peak_SST_monthly_cooling_rainfall.png",
    ),
    _product(
        "figure-s8",
        "RealizarAnalises_dev.py",
        "figuras/Quiver_Alterados_20260925/02_CTST_e_circulacao/Subplot_Tendencias_AMJJ.png",
    ),
    _product(
        "figure-s9",
        "RealizarTratamentoCalorLatenteMedioComLag0.py",
        "figuras/CalorLatenteMedioComLag0/LH_AnomaliasQuantis_Lag0.png",
    ),
)

PRODUCT_BY_ID = {product.product_id: product for product in PUBLICATION_PRODUCTS}

if len(PRODUCT_BY_ID) != 18:
    raise RuntimeError("the publication catalog must contain 18 unique products")
