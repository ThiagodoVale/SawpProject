#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Painel comparativo: máximo anual da chuva ENEB × máximo anual da TSM SAWP
com e sem tendência.

Figura gerada:
    a) Dados com tendência
    b) Dados sem tendência
    c) Tabela com estatísticas e testes

Definição principal:
    - Calcula anomalias mensais em relação à climatologia mensal 1980-2010.
    - Seleciona AMJJ: abril, maio, junho e julho.
    - Para cada ano, calcula:
        x = máximo da anomalia mensal de chuva ENEB durante AMJJ
        y = máximo da anomalia mensal de TSM SAWP durante AMJJ

Observação:
    Por padrão, o máximo de chuva e o máximo de TSM são independentes dentro de AMJJ,
    reproduzindo a lógica do gráfico mostrado. Caso queira usar a TSM do mesmo mês
    em que ocorreu o máximo de chuva, mude USE_SST_SAME_MONTH_AS_RAIN_MAX = True.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import detrend
from scipy.stats import pearsonr, spearmanr, kendalltau, linregress

import sawp

# ============================================================
# CONFIGURAÇÕES
# ============================================================

METODO_SAWP = "quadrado"      # "quadrado" ou "isoterma"
CLIM_INI = 1980
CLIM_FIM = 2010
MESES_ANALISE = [4, 5, 6, 7]  # AMJJ
USE_SST_SAME_MONTH_AS_RAIN_MAX = False

OUT_DIR = "./outputs/reviewer_event_based_april"
os.makedirs(OUT_DIR, exist_ok=True)

AREA_SAWP = [-34.0, -25.0, -12.5, -4.0]  # lon_min, lon_max, lat_min, lat_max

FONTSIZE = 18
plt.rcParams.update({
    "font.size": FONTSIZE,
    "axes.titlesize": FONTSIZE,
    "axes.labelsize": FONTSIZE,
    "xtick.labelsize": FONTSIZE,
    "ytick.labelsize": FONTSIZE,
    "legend.fontsize": 16,
})

# ============================================================
# FUNÇÕES BÁSICAS
# ============================================================

def get_sst_file(tendencias):
    if tendencias == "não":
        return "./outputs/detrended_sst.npz"
    return "./outputs_trended/trended_uvq_sst.npz"


def carregar_tsm(tendencias):
    path = get_sst_file(tendencias)
    if not os.path.exists(path):
        raise FileNotFoundError(f"SST file not found: {path}")

    return sawp.CarregarDados(
        path,
        tendencias=tendencias,
        nome_var_nc="sst",
        substituir_nan=-999,
    )

# ============================================================
# TOP 10 MAIORES EVENTOS DE CHUVA
# ============================================================

def plot_top10_rainfall(ax,
                        annual_rain_without_trend,
                        annual_rain_with_trend):
    """
    Compara os 10 maiores eventos de chuva.

    A ordenação é feita utilizando a série SEM TENDÊNCIA.
    Para os mesmos anos é mostrado o valor COM TENDÊNCIA.
    """

    # remove NaN
    sem = annual_rain_without_trend.dropna()
    com = annual_rain_with_trend.dropna()

    anos_comuns = sem.index.intersection(com.index)

    sem = sem.loc[anos_comuns]
    com = com.loc[anos_comuns]

    # pega os 10 maiores da série sem tendência
    top10 = sem.nlargest(10)

    # ordena em ordem crescente
    top10 = top10.sort_values()

    anos = top10.index

    sem_plot = top10.values
    com_plot = com.loc[anos].values

    x = np.arange(len(anos))
    largura = 0.38

    ax.bar(
        x-largura/2,
        sem_plot,
        largura,
        color="royalblue",
        label="Without trend"
    )

    ax.bar(
        x+largura/2,
        com_plot,
        largura,
        color="darkorange",
        label="With trend"
    )

    ax.set_xticks(x)

    ax.set_xticklabels(
        [f"{i+1}\n{ano}" for i, ano in enumerate(anos)],
        fontsize=14
    )

    ax.set_xlabel(
        "Ranking (based on detrended rainfall)",
        fontsize=18
    )

    ax.set_ylabel(
        "Maximum rainfall (mm)",
        fontsize=18
    )

    ax.set_title(
        "c) Ten largest annual rainfall events",
        fontsize=18,
        fontweight="bold"
    )

    ax.grid(axis="y", alpha=0.3)

    ax.legend()

    # escreve os valores
    for i,(v1,v2) in enumerate(zip(sem_plot,com_plot)):

        ax.text(
            i-largura/2,
            v1+2,
            f"{v1:.0f}",
            ha="center",
            fontsize=14,
            rotation=90
        )

        ax.text(
            i+largura/2,
            v2+2,
            f"{v2:.0f}",
            ha="center",
            fontsize=14,
            rotation=90
        )
        
def carregar_chuva_mensal():
    path_csv = "./data/processed/ENEB_monthly_precipitation_qc.csv"
    path_txt = "./data/EstacoesDados.txt"

    if os.path.exists(path_csv):
        df = pd.read_csv(
            path_csv,
            index_col=0,
            parse_dates=True,
            na_values=[-999, "-999", "-999."],
        )
        if "precipitation_mm" in df.columns:
            chuva = df["precipitation_mm"].copy()
        else:
            chuva = df.iloc[:, 0].copy()

    elif os.path.exists(path_txt):
        chuva = pd.read_csv(
            path_txt,
            header=None,
            sep=r"\s+",
            na_values=[-999, "-999", "-999."],
        ).iloc[:, 0]
        chuva.index = pd.date_range("1959-01-31", periods=len(chuva), freq="ME")

    else:
        raise FileNotFoundError("Rainfall file not found.")

    chuva = pd.to_numeric(chuva, errors="coerce")
    chuva = chuva.mask(chuva < 0, np.nan)
    chuva.name = "eneb_rain_mm"
    return chuva


def padronizar_indice_mensal(serie, nome=None, agg="max"):
    serie = serie.copy()
    serie.index = pd.to_datetime(serie.index).to_period("M").to_timestamp("M")

    if agg == "max":
        serie = serie.groupby(serie.index).max()
    elif agg == "mean":
        serie = serie.groupby(serie.index).mean()
    elif agg == "sum":
        serie = serie.groupby(serie.index).sum()
    else:
        raise ValueError("agg must be 'max', 'mean', or 'sum'.")

    if nome is not None:
        serie.name = nome
    return serie


def convert_sst_to_celsius_if_needed(series):
    m = np.nanmean(series.values)
    if np.isfinite(m) and m > 100:
        out = series - 273.15
        out.name = series.name
        return out
    return series


def ensure_2d_lon_lat(lon, lat):
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    if lon.ndim == 1 and lat.ndim == 1:
        lon2d, lat2d = np.meshgrid(lon, lat)
    elif lon.ndim == 2 and lat.ndim == 2:
        lon2d, lat2d = lon, lat
    else:
        raise ValueError("Unexpected lon/lat dimensions.")

    return lon2d, lat2d

# ============================================================
# SAWP
# ============================================================

def create_sawp_mask_quadrado(lon2d, lat2d, area):
    lon_min, lon_max, lat_min, lat_max = area
    return (
        (lon2d >= lon_min) & (lon2d <= lon_max) &
        (lat2d >= lat_min) & (lat2d <= lat_max)
    )


def create_sawp_mask_isoterma(tsm, tempo, lon2d, lat2d):
    tempo_pd = pd.to_datetime(tempo)
    mask_abril_clim = (
        (tempo_pd.month == 4) &
        (tempo_pd.year >= CLIM_INI) &
        (tempo_pd.year <= CLIM_FIM)
    )

    if mask_abril_clim.sum() == 0:
        raise RuntimeError("No April months found for SAWP isotherm climatology.")

    clim_abril_2d = np.nanmean(tsm[mask_abril_clim, :, :], axis=0)
    clim_median = pd.Series(clim_abril_2d.ravel()).median()
    threshold = 28.5 + 273.15 if clim_median > 100 else 28.5

    mask_iso = clim_abril_2d >= threshold
    mask_iso = mask_iso & (lon2d >= -40.0) & (lon2d <= -25.0)

    print(f"SAWP isotherm threshold: {threshold:.2f}")
    print(f"SAWP isotherm points: {int(mask_iso.sum())}")
    return mask_iso


def serie_maxima_area_3d(data, tempo, mask_area, name="sawp_sst_max"):
    data = np.asarray(data, dtype=float)
    data = np.where(data <= -900, np.nan, data)

    if data.ndim != 3:
        raise ValueError(f"Expected 3D SST data. Got {data.shape}")

    if data.shape[1:] != mask_area.shape:
        raise ValueError(f"SST grid and mask differ: {data.shape[1:]} vs {mask_area.shape}")

    if mask_area.sum() == 0:
        raise RuntimeError("SAWP mask has zero points.")

    masked = np.where(mask_area[None, :, :], data, np.nan)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        values = np.nanmax(masked, axis=(1, 2))

    serie = pd.Series(values, index=pd.to_datetime(tempo), name=name)
    serie = convert_sst_to_celsius_if_needed(serie)
    return serie


def definir_serie_sawp(tsm, tempo, lon, lat, metodo="quadrado"):
    lon2d, lat2d = ensure_2d_lon_lat(lon, lat)

    if metodo == "quadrado":
        mask = create_sawp_mask_quadrado(lon2d, lat2d, AREA_SAWP)
    elif metodo == "isoterma":
        mask = create_sawp_mask_isoterma(tsm, tempo, lon2d, lat2d)
    else:
        raise ValueError("METODO_SAWP must be 'quadrado' or 'isoterma'.")

    serie = serie_maxima_area_3d(
        tsm,
        tempo,
        mask,
        name="sawp_sst_max_c",
    )
    return serie

# ============================================================
# ANOMALIA E REMOÇÃO DE TENDÊNCIA
# ============================================================

def detrend_mensal_por_mes(serie, manter_media=True):
    """
    Remove tendência linear separadamente para cada mês do calendário.
    Isso evita misturar sazonalidade com tendência.
    """
    s = serie.copy().astype(float)
    s.index = pd.to_datetime(s.index)
    out = pd.Series(index=s.index, dtype=float, name=s.name)

    for mes in range(1, 13):
        idx = s.index.month == mes
        sub = s.loc[idx].dropna()
        if len(sub) < 3:
            continue

        y = sub.values.astype(float)
        x = np.arange(len(y), dtype=float)
        y_dt = detrend(y, type="linear")

        if manter_media:
            y_dt = y_dt + np.nanmean(y)

        out.loc[sub.index] = y_dt

    return out


def anomalia_mensal_climatologica(serie, clim_ini=1980, clim_fim=2010):
    serie = serie.copy().astype(float)
    serie.index = pd.to_datetime(serie.index)

    clim = serie[
        (serie.index.year >= clim_ini) &
        (serie.index.year <= clim_fim)
    ]

    if clim.notna().sum() == 0:
        raise RuntimeError(f"No valid climatology for {serie.name}")

    clim_mean = clim.groupby(clim.index.month).mean()

    anom = pd.Series(index=serie.index, dtype=float, name=f"{serie.name}_anom")
    for m in sorted(serie.index.month.unique()):
        idx = serie.index.month == m
        if m in clim_mean.index:
            anom.loc[idx] = serie.loc[idx] - clim_mean.loc[m]

    return anom

# ============================================================
# TABELA ANUAL E ESTATÍSTICAS
# ============================================================

def construir_tabela_anual_maximos(sst_anom, rain_anom):
    df = pd.DataFrame({
        "sst_anom": sst_anom,
        "rain_anom": rain_anom,
    }).dropna()

    df = df[df.index.month.isin(MESES_ANALISE)]
    rows = []

    for year, g in df.groupby(df.index.year):
        if len(g) == 0:
            continue

        if USE_SST_SAME_MONTH_AS_RAIN_MAX:
            idxmax_rain = g["rain_anom"].idxmax()
            rain_max = g.loc[idxmax_rain, "rain_anom"]
            sst_value = g.loc[idxmax_rain, "sst_anom"]
            month_used = idxmax_rain.month
        else:
            rain_max = g["rain_anom"].max()
            sst_value = g["sst_anom"].max()
            month_used = np.nan

        rows.append({
            "year": int(year),
            "rain_max_anom_amjj": rain_max,
            "sst_max_anom_amjj": sst_value,
            "month_used": month_used,
        })

    out = pd.DataFrame(rows).set_index("year").dropna(subset=[
        "rain_max_anom_amjj", "sst_max_anom_amjj"
    ])
    return out


def calcular_estatisticas(tabela, label):
    dados = tabela[["rain_max_anom_amjj", "sst_max_anom_amjj"]].dropna()
    x = dados["rain_max_anom_amjj"].values.astype(float)
    y = dados["sst_max_anom_amjj"].values.astype(float)

    if len(dados) < 3:
        raise RuntimeError(f"Too few valid values for {label}: n={len(dados)}")

    pear_r, pear_p = pearsonr(x, y)
    spear_r, spear_p = spearmanr(x, y)
    kend_r, kend_p = kendalltau(x, y)
    reg = linregress(x, y)

    return {
        "case": label,
        "n": len(dados),
        "Pearson r": pear_r,
        "Pearson p": pear_p,
        "R²": pear_r ** 2,
        "Slope (°C/mm)": reg.slope,
        "Slope p": reg.pvalue,
        "Spearman rho": spear_r,
        "Spearman p": spear_p,
        "Kendall tau": kend_r,
        "Kendall p": kend_p,
    }


def processar_cenario(tendencias, tempo, lon, lat, rain_raw):
    """
    tendencias="sim": usa TSM com tendência e chuva original.
    tendencias="não": usa TSM sem tendência e chuva sem tendência mensal.
    """
    print(f"\nProcessing case: tendencias={tendencias}")

    tsm = carregar_tsm(tendencias)
    sst = definir_serie_sawp(tsm, tempo, lon, lat, metodo=METODO_SAWP)

    sst = padronizar_indice_mensal(sst, nome="sst", agg="max")
    rain = padronizar_indice_mensal(rain_raw, nome="rain", agg="max")

    common_dates = sst.index.intersection(rain.index)
    sst = sst.loc[common_dates]
    rain = rain.loc[common_dates]

    monthly = pd.concat([sst.rename("sst"), rain.rename("rain")], axis=1).dropna()
    sst = monthly["sst"]
    rain = monthly["rain"]

    if tendencias == "não":
        # A TSM já vem do arquivo detrended_sst.npz.
        # A chuva é removida da tendência aqui para manter coerência no painel sem tendência.
        rain = detrend_mensal_por_mes(rain, manter_media=True)
        common_dates = sst.index.intersection(rain.index)
        monthly = pd.concat([
            sst.loc[common_dates].rename("sst"),
            rain.loc[common_dates].rename("rain"),
        ], axis=1).dropna()
        sst = monthly["sst"]
        rain = monthly["rain"]

    sst_anom = anomalia_mensal_climatologica(sst, CLIM_INI, CLIM_FIM)
    rain_anom = anomalia_mensal_climatologica(rain, CLIM_INI, CLIM_FIM)

    tabela = construir_tabela_anual_maximos(sst_anom, rain_anom)

    label = "Original" if tendencias == "sim" else "Detrended"
    stats = calcular_estatisticas(tabela, label)

    return tabela, stats

# ============================================================
# PLOT PAINEL
# ============================================================

def format_p(p):
    if not np.isfinite(p):
        return "nan"
    if p < 0.001:
        return "<0.001"
    return f"{p:.3f}"

def plot_scatter_panel(ax, tabela, stats, panel_label):

    dados = tabela[["rain_max_anom_amjj", "sst_max_anom_amjj"]].dropna().copy()

    # Normalização pelo desvio padrão
    dados["rain_norm"] = (
        dados["rain_max_anom_amjj"] /
        dados["rain_max_anom_amjj"].std(ddof=1)
    )

    dados["sst_norm"] = (
        dados["sst_max_anom_amjj"] /
        dados["sst_max_anom_amjj"].std(ddof=1)
    )

    # INVERTIDO
    x = dados["sst_norm"].values.astype(float)
    y = dados["rain_norm"].values.astype(float)

    reg = linregress(x, y)

    xx = np.linspace(np.nanmin(x), np.nanmax(x), 100)
    yy = reg.intercept + reg.slope * xx

    ax.scatter(x, y, s=70)
    ax.plot(xx, yy, linewidth=2.5)

    ax.axhline(0, linestyle="--", linewidth=1.2)
    ax.axvline(0, linestyle="--", linewidth=1.2)

    ax.grid(alpha=0.3)

    title = (
        f"{panel_label}) {stats['case']}\n"
        f"r = {stats['Pearson r']:.2f}, "
        f"p = {format_p(stats['Pearson p'])}, "
        f"n = {stats['n']}"
    )

    ax.set_title(title)

    ax.set_xlabel(
        "Normalized SST anomaly (σ)",
        fontsize=FONTSIZE,
    )

    ax.set_ylabel(
        "Normalized rainfall anomaly (σ)",
        fontsize=FONTSIZE,
    )

    lim = np.ceil(max(np.max(np.abs(x)), np.max(np.abs(y))))

    ax.set_xlim(-3, 3)
    ax.set_ylim(-2, lim)
    ax.set_aspect("equal")

def plot_painel_com_tabela(tab_with, stats_with, tab_without, stats_without):
    fig = plt.figure(figsize=(18, 20))

    gs = fig.add_gridspec(
        nrows=3,
        ncols=2,
        height_ratios=[3.2, 1.8, 1.15],
        hspace=0.36,
        wspace=0.22,
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_bar = fig.add_subplot(gs[1, :])
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis("off")
    
    # ============================================================
    # Normalização pelo desvio padrão
    # ============================================================
    
    tab_with_plot = tab_with.copy()
    tab_without_plot = tab_without.copy()
    
    for df in [tab_with_plot, tab_without_plot]:
    
        df["rain_norm"] = (
            df["rain_max_anom_amjj"] /
            df["rain_max_anom_amjj"].std(ddof=1)
        )
    
        df["sst_norm"] = (
            df["sst_max_anom_amjj"] /
            df["sst_max_anom_amjj"].std(ddof=1)
        )
    
    plot_scatter_panel(ax1, tab_with_plot, stats_with, "a")
    plot_scatter_panel(ax2, tab_without_plot, stats_without, "b")

    # ========================================================
    # c) Barras comparativas dos 10 maiores eventos
    # Base: chuva SEM tendência
    # ========================================================
    
    tab_without_bar = tab_without.copy()
    tab_with_bar = tab_with.copy()
    
    # Se o ano estiver no índice, transforma em coluna
    if "year" not in tab_without_bar.columns:
        tab_without_bar = tab_without_bar.reset_index()
    
    if "year" not in tab_with_bar.columns:
        tab_with_bar = tab_with_bar.reset_index()
    
    # Detecta coluna de ano
    if "year" not in tab_without_bar.columns:
        tab_without_bar = tab_without_bar.rename(columns={tab_without_bar.columns[0]: "year"})
    
    if "year" not in tab_with_bar.columns:
        tab_with_bar = tab_with_bar.rename(columns={tab_with_bar.columns[0]: "year"})
    
    # Detecta coluna de chuva
    rain_candidates = [
    "rain_max_anom_amjj",
    "rain_max",
    "rain_anom_max",
    "rainfall_max",
    "max_rain",
    "max_rainfall",
    "annual_max_rain",
    "annual_max_rainfall",
   ]
    
    rain_col_without = None
    rain_col_with = None
    
    for c in rain_candidates:
        if c in tab_without_bar.columns:
            rain_col_without = c
            break
    
    for c in rain_candidates:
        if c in tab_with_bar.columns:
            rain_col_with = c
            break
    
    if rain_col_without is None:
        raise KeyError(
            "Não encontrei a coluna de chuva em tab_without. "
            f"Colunas disponíveis: {list(tab_without_bar.columns)}"
        )
    
    if rain_col_with is None:
        raise KeyError(
            "Não encontrei a coluna de chuva em tab_with. "
            f"Colunas disponíveis: {list(tab_with_bar.columns)}"
        )
    
    # Apenas eventos a partir de 1990
    top10 = (
        tab_without_bar.loc[
            tab_without_bar["year"] >= 1958,
            ["year", rain_col_without]
        ]
        .dropna()
        .sort_values(rain_col_without, ascending=False)
        .head(10)
        .sort_values(rain_col_without, ascending=True)
    )
    
    anos = top10["year"].astype(int).values
    chuva_without = top10[rain_col_without].values
    
    tab_with_sel = (
    tab_with_bar.loc[
        tab_with_bar["year"] >= 1958,
        ["year", rain_col_with]
    ]
    .dropna()
    .assign(year=lambda d: d["year"].astype(int))
    .set_index("year")
)
    
    chuva_with = np.array([
        tab_with_sel.loc[ano, rain_col_with] if ano in tab_with_sel.index else np.nan
        for ano in anos
    ])
    
    x = np.arange(len(anos))
    largura = 0.38
    
    ax_bar.bar(
        x - largura / 2,
        chuva_without,
        width=largura,
        color="tab:blue",
        label="Detrended",
    )
    
    ax_bar.bar(
        x + largura / 2,
        chuva_with,
        width=largura,
        color="tab:orange",
        label="Original",
    )
    
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(
        [f"{i+1}\n{ano}" for i, ano in enumerate(anos)],
        fontsize=16,
    )
    
    ax_bar.set_ylabel("Maximum rainfall anomaly (mm)", fontsize=FONTSIZE)
    ax_bar.set_xlabel(
        "Ranking based on detrended rainfall anomaly\n(year shown below rank)",
        fontsize=FONTSIZE,
    )
    
    ax_bar.set_title(
        "c) Ten largest annual rainfall maxima: original versus detrended",
        fontsize=FONTSIZE,
        pad=8,
    )
    
    ax_bar.grid(axis="y", alpha=0.3)
    ax_bar.legend(fontsize=16)
    
    for i, (v_without, v_with) in enumerate(zip(chuva_without, chuva_with)):
        if np.isfinite(v_without):
            ax_bar.text(
                i - largura / 2,
                v_without,
                f"{v_without:.0f}",
                ha="center",
                va="bottom",
                fontsize=14,
                rotation=0,
            )
    
        if np.isfinite(v_with):
            ax_bar.text(
                i + largura / 2,
                v_with,
                f"{v_with:.0f}",
                ha="center",
                va="bottom",
                fontsize=14,
                rotation=0,
            )
    # ========================================================
    # d) Tabela estatística
    # ========================================================

    stats_df = pd.DataFrame([stats_with, stats_without])

    display_cols = [
        "case", "n", "Pearson r", "Pearson p", "R²",
        "Slope (°C/mm)", "Slope p", "Spearman rho", "Spearman p",
        "Kendall tau", "Kendall p",
    ]

    stats_df = stats_df[display_cols].copy()

    for col in stats_df.columns:
        if col == "case":
            continue
        if col == "n":
            stats_df[col] = stats_df[col].astype(int).astype(str)
        elif "p" in col.lower():
            stats_df[col] = stats_df[col].map(format_p)
        else:
            stats_df[col] = stats_df[col].map(lambda v: f"{v:.3f}")

    table = ax_table.table(
        cellText=stats_df.values,
        colLabels=stats_df.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1, 1.65)

    ax_table.set_title(
        "d) Statistical summary and correlation tests",
        fontsize=FONTSIZE,
        pad=8,
    )

    if USE_SST_SAME_MONTH_AS_RAIN_MAX:
        subtitle = "SST anomaly taken from the same month as the annual maximum rainfall anomaly"
        fname_suffix = "same_month_as_rain_max"
    else:
        subtitle = "Independent annual maxima within AMJJ for rainfall and SST anomalies"
        fname_suffix = "independent_maxima"

    fig.suptitle(
        "Annual maximum ENEB rainfall anomaly versus annual maximum SAWP SST anomaly\n"
        f"{subtitle}",
        fontsize=24,
        y=0.98,
    )

    out_png = os.path.join(
        OUT_DIR,
        f"Panel_max_rainfall_vs_max_SST_with_without_trend_{fname_suffix}.png",
    )

    out_pdf = out_png.replace(".png", ".pdf")

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)

    stats_df.to_csv(
        os.path.join(
            OUT_DIR,
            f"Panel_max_rainfall_vs_max_SST_statistics_{fname_suffix}.csv",
        ),
        index=False,
    )

    return out_png, out_pdf, stats_df

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Configuring time and area...")
    tempo, lon, lat = sawp.ConfigurarTempoArea("./data/Era5_Dataset_sst.nc")

    print("Loading ENEB rainfall...")
    rain_raw = carregar_chuva_mensal()

    tab_with, stats_with = processar_cenario("sim", tempo, lon, lat, rain_raw)
    tab_without, stats_without = processar_cenario("não", tempo, lon, lat, rain_raw)

    tab_with.to_csv(os.path.join(OUT_DIR, "annual_max_rain_sst_anomaly_WITH_trend.csv"))
    tab_without.to_csv(os.path.join(OUT_DIR, "annual_max_rain_sst_anomaly_WITHOUT_trend.csv"))

    out_png, out_pdf, stats_df = plot_painel_com_tabela(
        tab_with,
        stats_with,
        tab_without,
        stats_without,
    )

    print("\nStatistics:")
    print(stats_df)
    print("\nSaved files:")
    print(out_png)
    print(out_pdf)
