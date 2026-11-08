#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste de eficiência de dissipação/resfriamento da anomalia de TSM da SAWP
condicionada à intensidade da chuva do ENEB durante AMJJ.

Objetivo físico
---------------
Testar se meses com maior anomalia de precipitação no ENEB removem uma fração
maior da anomalia positiva de TSM da SAWP no mês seguinte.

Métrica principal
-----------------
Cooling efficiency = [SST_anom(m) - SST_anom(m+1)] / SST_anom(m)

onde:
    SST_anom(m)     = anomalia de TSM no mês atual
    SST_anom(m+1)   = anomalia de TSM no mês seguinte

Interpretação:
    efficiency > 0  -> parte da anomalia positiva foi removida/resfriada
    efficiency = 0  -> sem mudança
    efficiency < 0  -> a anomalia aumentou/aquecimento relativo
    efficiency = 1  -> toda a anomalia positiva foi removida

Importante:
    A eficiência só é calculada quando SST_anom(m) >= SST_POSITIVE_MIN.
    Isso evita dividir por valores próximos de zero.

Saídas principais
-----------------
    efficiency_month_pairs_AMJJ.csv
    efficiency_rain_bins_05_summary.csv
    efficiency_rain_intensity_class_summary.csv
    efficiency_statistical_tests.csv
    Fig_efficiency_scatter_bins05.png
    Fig_efficiency_boxplots_rain_classes.png
    Fig_efficiency_conditioned_initial_sst.png

Requisitos esperados no ambiente
--------------------------------
    import sawp
    sawp.calcular_anomalia_tsm_mensal_todo_periodo
    sawp.gerar_serie_chuva_mensal_acumulada

Autor: adaptado para análise SAWP-ENEB
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr, spearmanr, kendalltau, kruskal, mannwhitneyu
import sawp


# ============================================================
# CONFIGURAÇÕES
# ============================================================

TENDENCIAS = "não"  # "sim" ou "não"
ANOS_CLIMA = (1980, 2010)

# Só calcula eficiência quando a anomalia inicial da TSM é positiva e suficientemente grande.
SST_POSITIVE_MIN = 0.30  # °C; ajuste para 0.25 se quiser ser menos restritivo

# Pares AMJJ -> mês seguinte
MONTH_PAIRS = [
    (4, 5, "Apr→May"),
    (5, 6, "May→Jun"),
    (6, 7, "Jun→Jul"),
    (7, 8, "Jul→Aug"),
]

# Bins contínuos de chuva
BIN_WIDTH = 0.5
MIN_N_BIN = 8

# Área SAWP
DICIONARIO_AREA_ESTUDO = {
    "SAWP": [-34, -25, -12.5, -4],
    "NEB": [-50, -34, -15, 0],
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def padronizar_indice_mensal(obj, nome_coluna="valor"):
    """Aceita Series/DataFrame e retorna DataFrame com índice mensal no fim do mês."""
    if isinstance(obj, pd.Series):
        df = obj.to_frame(name=nome_coluna)
    elif isinstance(obj, pd.DataFrame):
        df = obj.copy()
    else:
        raise TypeError("Entrada deve ser pandas Series ou pandas DataFrame.")

    df.index = pd.DatetimeIndex(pd.to_datetime(df.index, errors="coerce"))

    if df.index.isna().any():
        raise ValueError("Há datas inválidas no índice.")

    df.index = df.index.to_period("M").to_timestamp("M")
    df = df.sort_index()
    return df


def ano_decimal(index):
    index = pd.DatetimeIndex(index)
    return index.year + (index.dayofyear - 1) / 365.25


def detrend_keep_mean(series, x_values=None):
    """Remove tendência linear preservando a média da série."""
    y = series.astype(float).copy()
    valid = y.notna()

    if valid.sum() < 3:
        return y, None

    if x_values is None:
        x_all = np.arange(len(y), dtype=float)
    else:
        x_all = np.asarray(x_values, dtype=float)

    reg = linregress(x_all[valid.values], y.loc[valid].values)
    trend = reg.slope * x_all + reg.intercept

    y_dt = y.copy()
    y_dt.loc[valid] = y.loc[valid].values - trend[valid.values] + y.loc[valid].mean()

    return y_dt, reg


# ============================================================
# CHUVA
# ============================================================

def normalizar_anomalia_chuva_mensal(
    chuva_mensal,
    tendencias="não",
    coluna="precipitation_mm",
    anos_clima=(1980, 2010),
):
    """
    Calcula anomalia mensal de chuva e normaliza pelo desvio padrão da anomalia
    no período climatológico.
    """
    chuva_df = padronizar_indice_mensal(chuva_mensal, nome_coluna=coluna)

    if coluna not in chuva_df.columns:
        raise KeyError(f"Coluna '{coluna}' não encontrada em chuva_mensal.")

    df = pd.DataFrame({"chuva_original": chuva_df[coluna].astype(float)})
    df["Ano"] = df.index.year
    df["Mes"] = df.index.month

    remover_tendencia = str(tendencias).lower() in ["sim", "s", "yes", "y"]

    if remover_tendencia:
        x = ano_decimal(df.index)
        df["chuva_processada"], reg = detrend_keep_mean(df["chuva_original"], x_values=x)
        if reg is not None:
            print(f"Tendência removida da chuva mensal: {reg.slope:.4f} mm/ano | p={reg.pvalue:.4f}")
    else:
        df["chuva_processada"] = df["chuva_original"]

    mascara_clima = (df["Ano"] >= anos_clima[0]) & (df["Ano"] <= anos_clima[1])

    climatologia_mensal = (
        df.loc[mascara_clima]
        .groupby("Mes")["chuva_processada"]
        .mean()
    )

    df["climatologia_mensal"] = df["Mes"].map(climatologia_mensal)
    df["anomalia"] = df["chuva_processada"] - df["climatologia_mensal"]

    std_anom = df.loc[mascara_clima, "anomalia"].std(skipna=True, ddof=1)
    if not np.isfinite(std_anom) or std_anom == 0:
        raise RuntimeError("Desvio padrão da anomalia de chuva inválido ou zero.")

    df["anomalia_normalizada"] = df["anomalia"] / std_anom

    return df


# ============================================================
# CARREGAMENTO E ANOMALIA DA TSM
# ============================================================

def carregar_serie_tsm_sawp_anomalia(tendencias="não"):
    """Carrega TSM, recorta SAWP e calcula anomalia mensal da TSM."""
    caminho_ = "./outputs/" if tendencias == "não" else "./outputs_trended/"
    tsm_arquivo = "detrended_sst.npz" if tendencias == "não" else "trended_uvq_sst.npz"

    periodo_tempo_convertido, lon, lat = sawp.ConfigurarTempoArea("./data/Era5_Dataset_sst.nc")

    tsm = sawp.CarregarDados(
        caminho_ + tsm_arquivo,
        tendencias=tendencias,
        nome_var_nc="sst",
        substituir_nan=-999,
    )

    limites = DICIONARIO_AREA_ESTUDO["SAWP"]

    lloW = np.argmin(np.abs(lon[0, :] - limites[0]))
    lloE = np.argmin(np.abs(lon[0, :] - limites[1]))
    llaS = np.argmin(np.abs(lat[:, 0] - limites[2]))
    llaN = np.argmin(np.abs(lat[:, 0] - limites[3]))

    resultado, climatologia = sawp.calcular_anomalia_tsm_mensal_todo_periodo(
        tsm,
        periodo_tempo_convertido,
        lloW,
        lloE,
        llaS,
        llaN,
    )

    serie_tsm = padronizar_indice_mensal(resultado, nome_coluna="Anomalia")

    if "Anomalia" not in serie_tsm.columns:
        raise KeyError("A série de TSM não contém a coluna 'Anomalia'.")

    return serie_tsm


# ============================================================
# PARES MÊS -> MÊS SEGUINTE E EFICIÊNCIA
# ============================================================

def construir_pares_eficiencia(
    serie_tsm,
    chuva_norm,
    coluna_sst="Anomalia",
    coluna_chuva="anomalia_normalizada",
    month_pairs=MONTH_PAIRS,
    sst_positive_min=SST_POSITIVE_MIN,
):
    """
    Cria pares AMJJ de chuva mensal e eficiência de dissipação da anomalia de TSM.

    Para cada par m -> m+1:
        cooling = SST_anom(m) - SST_anom(m+1)
        efficiency = cooling / SST_anom(m), somente se SST_anom(m) >= sst_positive_min
    """
    sst = padronizar_indice_mensal(serie_tsm, nome_coluna=coluna_sst)
    rain = padronizar_indice_mensal(chuva_norm)

    if coluna_sst not in sst.columns:
        raise KeyError(f"Coluna '{coluna_sst}' não encontrada na série de TSM.")
    if coluna_chuva not in rain.columns:
        raise KeyError(f"Coluna '{coluna_chuva}' não encontrada na chuva.")

    dados = pd.concat(
        [
            sst[coluna_sst].rename("sst_anomaly"),
            rain[coluna_chuva].rename("rain_anomaly"),
        ],
        axis=1,
    )

    dados["Ano"] = dados.index.year
    dados["Mes"] = dados.index.month

    registros = []

    for year in sorted(dados["Ano"].dropna().unique()):
        for mes_atual, mes_prox, label in month_pairs:
            data_atual = pd.Timestamp(year=int(year), month=mes_atual, day=1).to_period("M").to_timestamp("M")
            data_prox = pd.Timestamp(year=int(year), month=mes_prox, day=1).to_period("M").to_timestamp("M")

            if data_atual not in dados.index or data_prox not in dados.index:
                continue

            rain_current = dados.loc[data_atual, "rain_anomaly"]
            sst_current = dados.loc[data_atual, "sst_anomaly"]
            sst_next = dados.loc[data_prox, "sst_anomaly"]

            if not (np.isfinite(rain_current) and np.isfinite(sst_current) and np.isfinite(sst_next)):
                continue

            cooling = sst_current - sst_next

            if sst_current >= sst_positive_min:
                efficiency = cooling / sst_current
            else:
                efficiency = np.nan

            registros.append(
                {
                    "year": int(year),
                    "month_current": int(mes_atual),
                    "month_next": int(mes_prox),
                    "month_pair": label,
                    "date_current": data_atual,
                    "date_next": data_prox,
                    "rain_anomaly": float(rain_current),
                    "sst_anomaly_current": float(sst_current),
                    "sst_anomaly_next": float(sst_next),
                    "sst_cooling_next_month": float(cooling),
                    "cooling_efficiency": float(efficiency) if np.isfinite(efficiency) else np.nan,
                }
            )

    pairs = pd.DataFrame(registros)

    if pairs.empty:
        raise RuntimeError("Nenhum par AMJJ válido foi criado.")

    # Normaliza cooling bruto apenas para diagnóstico comparável.
    std_cooling = pairs["sst_cooling_next_month"].std(skipna=True, ddof=1)
    if np.isfinite(std_cooling) and std_cooling > 0:
        pairs["sst_cooling_next_month_norm"] = pairs["sst_cooling_next_month"] / std_cooling
    else:
        pairs["sst_cooling_next_month_norm"] = np.nan

    # Estado inicial da TSM
    pairs["initial_sst_state"] = pd.cut(
        pairs["sst_anomaly_current"],
        bins=[-np.inf, 0, sst_positive_min, 1.0, np.inf],
        labels=["SST < 0", f"0 ≤ SST < {sst_positive_min:.2f}", f"{sst_positive_min:.2f} ≤ SST < +1", "SST ≥ +1"],
        include_lowest=True,
    )

    # Classes fixas de chuva
    pairs["rain_intensity_class"] = pd.cut(
        pairs["rain_anomaly"],
        bins=[-np.inf, -1, 0, 1, np.inf],
        labels=["Dry < -1σ", "-1σ to 0", "0 to +1σ", "Wet > +1σ"],
        include_lowest=True,
    )

    return pairs


# ============================================================
# RESUMOS, BINS E TESTES
# ============================================================

def resumo_por_classe_chuva(pairs):
    df = pairs.dropna(subset=["rain_intensity_class", "cooling_efficiency"]).copy()

    summary = (
        df.groupby("rain_intensity_class", observed=True)
        .agg(
            n=("cooling_efficiency", "count"),
            rain_mean=("rain_anomaly", "mean"),
            sst_initial_mean=("sst_anomaly_current", "mean"),
            cooling_mean=("sst_cooling_next_month", "mean"),
            efficiency_mean=("cooling_efficiency", "mean"),
            efficiency_median=("cooling_efficiency", "median"),
            efficiency_std=("cooling_efficiency", "std"),
            frac_positive_efficiency=("cooling_efficiency", lambda x: np.mean(np.asarray(x) > 0)),
        )
        .reset_index()
    )

    summary["efficiency_se"] = summary["efficiency_std"] / np.sqrt(summary["n"])
    return summary


def resumo_bins_chuva_continuos(pairs, bin_width=0.5, min_n=8):
    """Bins contínuos de chuva com largura 0.5σ."""
    df = pairs.dropna(subset=["rain_anomaly", "cooling_efficiency"]).copy()

    rain_min = np.floor(df["rain_anomaly"].min() / bin_width) * bin_width
    rain_max = np.ceil(df["rain_anomaly"].max() / bin_width) * bin_width
    bins = np.arange(rain_min, rain_max + bin_width, bin_width)

    df["rain_bin"] = pd.cut(df["rain_anomaly"], bins=bins, include_lowest=True)

    # Corrige dtype: centro do bin como float, não categoria.
    centers = {interval: float((interval.left + interval.right) / 2) for interval in df["rain_bin"].cat.categories}
    df["rain_bin_center"] = df["rain_bin"].map(centers).astype(float)

    summary = (
        df.groupby("rain_bin", observed=True)
        .agg(
            n=("cooling_efficiency", "count"),
            rain_mean=("rain_anomaly", "mean"),
            rain_bin_center=("rain_bin_center", "first"),
            efficiency_mean=("cooling_efficiency", "mean"),
            efficiency_median=("cooling_efficiency", "median"),
            efficiency_std=("cooling_efficiency", "std"),
            frac_positive_efficiency=("cooling_efficiency", lambda x: np.mean(np.asarray(x) > 0)),
        )
        .reset_index()
    )

    summary["efficiency_se"] = summary["efficiency_std"] / np.sqrt(summary["n"])
    summary["valid_for_plot"] = summary["n"] >= min_n

    return df, summary


def testes_estatisticos_eficiencia(pairs):
    df = pairs.dropna(subset=["rain_anomaly", "cooling_efficiency", "rain_intensity_class"]).copy()

    rows = []

    # Correlações não paramétricas
    if len(df) >= 3:
        pr, pp = pearsonr(df["rain_anomaly"], df["cooling_efficiency"])
        sr, sp = spearmanr(df["rain_anomaly"], df["cooling_efficiency"])
        kr, kp = kendalltau(df["rain_anomaly"], df["cooling_efficiency"])
    else:
        pr = pp = sr = sp = kr = kp = np.nan

    rows.extend([
        {"test": "Pearson rain vs efficiency", "statistic": pr, "p_value": pp, "n": len(df)},
        {"test": "Spearman rain vs efficiency", "statistic": sr, "p_value": sp, "n": len(df)},
        {"test": "Kendall rain vs efficiency", "statistic": kr, "p_value": kp, "n": len(df)},
    ])

    # Kruskal-Wallis entre classes de chuva
    groups = [g["cooling_efficiency"].dropna().values for _, g in df.groupby("rain_intensity_class", observed=True)]
    groups = [g for g in groups if len(g) >= 2]

    if len(groups) >= 2:
        h, p = kruskal(*groups)
    else:
        h, p = np.nan, np.nan

    rows.append({"test": "Kruskal-Wallis rain intensity classes", "statistic": h, "p_value": p, "n": len(df)})

    return pd.DataFrame(rows)


def posthoc_mannwhitney_holm(pairs):
    df = pairs.dropna(subset=["rain_intensity_class", "cooling_efficiency"]).copy()
    classes = list(df["rain_intensity_class"].cat.categories)
    rows = []

    for i in range(len(classes)):
        for j in range(i + 1, len(classes)):
            c1, c2 = classes[i], classes[j]
            x = df.loc[df["rain_intensity_class"] == c1, "cooling_efficiency"].dropna().values
            y = df.loc[df["rain_intensity_class"] == c2, "cooling_efficiency"].dropna().values

            if len(x) < 2 or len(y) < 2:
                u, p = np.nan, np.nan
            else:
                u, p = mannwhitneyu(x, y, alternative="two-sided")

            rows.append({"group_1": str(c1), "group_2": str(c2), "U": u, "p_raw": p, "n1": len(x), "n2": len(y)})

    out = pd.DataFrame(rows)

    # Correção de Holm manual
    valid = out["p_raw"].notna()
    pvals = out.loc[valid, "p_raw"].values
    order = np.argsort(pvals)
    m = len(pvals)
    p_adj = np.full(m, np.nan)

    # Holm: p_i_adj = max_{j<=i} (m-j+1) p_j ordenado, truncado a 1
    running_max = 0
    for rank, idx in enumerate(order):
        adj = (m - rank) * pvals[idx]
        running_max = max(running_max, adj)
        p_adj[idx] = min(running_max, 1.0)

    out.loc[valid, "p_holm"] = p_adj

    return out


# ============================================================
# FIGURAS
# ============================================================

def plot_eficiencia_principal(pairs, bin_summary, class_summary, stats_tests, output="Fig_cooling_efficiency_rain_bins05.png"):
    df = pairs.dropna(subset=["rain_anomaly", "cooling_efficiency"]).copy()
    bins_valid = bin_summary[bin_summary["valid_for_plot"]].copy()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --------------------------------------------------------
    # (a) Scatter
    # --------------------------------------------------------
    ax = axes[0, 0]

    month_markers = {"Apr→May": "o", "May→Jun": "s", "Jun→Jul": "^", "Jul→Aug": "D"}

    for label, sub in df.groupby("month_pair"):
        ax.scatter(
            sub["rain_anomaly"],
            sub["cooling_efficiency"],
            s=35,
            alpha=0.65,
            marker=month_markers.get(label, "o"),
            label=label,
        )

    if len(df) >= 3:
        reg = linregress(df["rain_anomaly"], df["cooling_efficiency"])
        xx = np.linspace(df["rain_anomaly"].min(), df["rain_anomaly"].max(), 100)
        yy = reg.intercept + reg.slope * xx
        ax.plot(xx, yy, linewidth=2)

        sr, sp = spearmanr(df["rain_anomaly"], df["cooling_efficiency"])
        txt = f"Pearson r={reg.rvalue:.2f}, p={reg.pvalue:.3f}\nSpearman ρ={sr:.2f}, p={sp:.3f}\nn={len(df)}"
    else:
        txt = f"n={len(df)}"

    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title("(a) Rainfall anomaly vs cooling efficiency")
    ax.set_xlabel("ENEB rainfall anomaly in current month (σ)")
    ax.set_ylabel("Cooling efficiency\nΔSST anomaly / initial SST anomaly")
    ax.text(0.02, 0.96, txt, transform=ax.transAxes, va="top", ha="left", bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # --------------------------------------------------------
    # (b) Boxplot por classe fixa
    # --------------------------------------------------------
    ax = axes[0, 1]
    order = ["Dry < -1σ", "-1σ to 0", "0 to +1σ", "Wet > +1σ"]
    data = [
        df.loc[df["rain_intensity_class"] == c, "cooling_efficiency"].dropna().values
        for c in order
    ]

    ax.boxplot(data, labels=order, showmeans=True)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("(b) Efficiency conditioned by rainfall intensity")
    ax.set_ylabel("Cooling efficiency")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.3)

    ymax = ax.get_ylim()[1]
    for i, vals in enumerate(data, start=1):
        ax.text(i, ymax * 0.95, f"n={len(vals)}", ha="center", va="top")

    # --------------------------------------------------------
    # (c) Média por classe fixa
    # --------------------------------------------------------
    ax = axes[1, 0]
    cs = class_summary.set_index("rain_intensity_class").reindex(order).reset_index()
    x = np.arange(len(cs))

    ax.errorbar(
        x,
        cs["efficiency_mean"],
        yerr=cs["efficiency_se"],
        marker="o",
        linewidth=2,
        capsize=4,
    )
    ax.axhline(0, color="black", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=25)
    ax.set_title("(c) Mean cooling efficiency by rainfall-intensity class")
    ax.set_ylabel("Mean cooling efficiency")
    ax.grid(axis="y", alpha=0.3)

    for xi, row in cs.iterrows():
        if pd.notna(row.get("n", np.nan)):
            ax.text(xi, row["efficiency_mean"], f"n={int(row['n'])}", ha="left", va="bottom", fontsize=9)

    # --------------------------------------------------------
    # (d) Bins 0.5σ
    # --------------------------------------------------------
    ax = axes[1, 1]

    ax.errorbar(
        bins_valid["rain_bin_center"],
        bins_valid["efficiency_mean"],
        yerr=bins_valid["efficiency_se"],
        marker="o",
        linewidth=2,
        capsize=4,
    )

    ax.axhline(0, color="black", linewidth=1)
    ax.axvline(0, color="black", linewidth=1)
    ax.set_title(f"(d) Mean efficiency in rainfall bins ({BIN_WIDTH:.1f}σ)")
    ax.set_xlabel("Rainfall anomaly bin center (σ)")
    ax.set_ylabel("Mean cooling efficiency")
    ax.grid(alpha=0.3)

    for _, row in bins_valid.iterrows():
        ax.text(row["rain_bin_center"], row["efficiency_mean"], f"n={int(row['n'])}", fontsize=8, ha="center", va="bottom")

    fig.suptitle(
        "Does stronger ENEB rainfall remove a larger fraction of positive SAWP SST anomalies?\n"
        f"Cooling efficiency = [SST'(m) - SST'(m+1)] / SST'(m), only when SST'(m) ≥ {SST_POSITIVE_MIN:.2f} °C",
        fontsize=15,
    )

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figura salva em: {output}")


def plot_eficiencia_condicionada_sst(pairs, output="Fig_cooling_efficiency_conditioned_initial_sst.png"):
    df = pairs.dropna(subset=["initial_sst_state", "rain_intensity_class", "cooling_efficiency"]).copy()

    sst_order = ["SST < 0", f"0 ≤ SST < {SST_POSITIVE_MIN:.2f}", f"{SST_POSITIVE_MIN:.2f} ≤ SST < +1", "SST ≥ +1"]
    rain_order = ["Dry < -1σ", "-1σ to 0", "0 to +1σ", "Wet > +1σ"]

    # Como a eficiência só existe para SST >= SST_POSITIVE_MIN, painéis frios podem ficar vazios.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)

    plot_states = [f"{SST_POSITIVE_MIN:.2f} ≤ SST < +1", "SST ≥ +1"]
    titles = [f"{SST_POSITIVE_MIN:.2f} ≤ initial SST anomaly < +1°C", "Initial SST anomaly ≥ +1°C"]

    for ax, state, title in zip(axes, plot_states, titles):
        sub = df[df["initial_sst_state"] == state]
        data = [sub.loc[sub["rain_intensity_class"] == c, "cooling_efficiency"].dropna().values for c in rain_order]

        ax.boxplot(data, labels=rain_order, showmeans=True)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.3)

        ymax = ax.get_ylim()[1]
        for i, vals in enumerate(data, start=1):
            ax.text(i, ymax * 0.95, f"n={len(vals)}", ha="center", va="top")

    axes[0].set_ylabel("Cooling efficiency")
    fig.suptitle("Cooling efficiency by rainfall intensity, conditioned on positive initial SAWP SST anomaly", fontsize=14)

    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figura salva em: {output}")


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

if __name__ == "__main__":

    print("Carregando e calculando anomalia de TSM da SAWP...")
    serie_tsm = carregar_serie_tsm_sawp_anomalia(tendencias=TENDENCIAS)

    print("Carregando chuva mensal regional ENEB...")
    chuva_mensal, matriz_estacoes_qc, metadata, removidos, regime_test = sawp.gerar_serie_chuva_mensal_acumulada()

    print("Calculando anomalia normalizada de chuva...")
    chuva_norm = normalizar_anomalia_chuva_mensal(
        chuva_mensal,
        tendencias=TENDENCIAS,
        coluna="precipitation_mm",
        anos_clima=ANOS_CLIMA,
    )

    print("Construindo pares AMJJ e eficiência de resfriamento...")
    pairs = construir_pares_eficiencia(
        serie_tsm=serie_tsm,
        chuva_norm=chuva_norm,
        coluna_sst="Anomalia",
        coluna_chuva="anomalia_normalizada",
        month_pairs=MONTH_PAIRS,
        sst_positive_min=SST_POSITIVE_MIN,
    )

    print(f"Pares totais AMJJ: {len(pairs)}")
    print(f"Pares com eficiência válida: {pairs['cooling_efficiency'].notna().sum()}")

    class_summary = resumo_por_classe_chuva(pairs)
    binned_pairs, bin_summary = resumo_bins_chuva_continuos(pairs, bin_width=BIN_WIDTH, min_n=MIN_N_BIN)
    stats_tests = testes_estatisticos_eficiencia(pairs)
    posthoc = posthoc_mannwhitney_holm(pairs)

    print("\nResumo por classe de chuva:")
    print(class_summary)

    print("\nResumo por bins de chuva de 0.5σ:")
    print(bin_summary)

    print("\nTestes estatísticos:")
    print(stats_tests)

    print("\nPós-hoc Mann-Whitney com Holm:")
    print(posthoc)

    plot_eficiencia_principal(
        pairs=pairs,
        bin_summary=bin_summary,
        class_summary=class_summary,
        stats_tests=stats_tests,
        output="Fig_cooling_efficiency_rain_bins05.png",
    )

    plot_eficiencia_condicionada_sst(
        pairs=pairs,
        output="Fig_cooling_efficiency_conditioned_initial_sst.png",
    )

    # Exportações
    pairs.to_csv("cooling_efficiency_AMJJ_month_pairs.csv", index=False)
    class_summary.to_csv("cooling_efficiency_rain_intensity_class_summary.csv", index=False)
    bin_summary.to_csv("cooling_efficiency_rain_bins05_summary.csv", index=False)
    stats_tests.to_csv("cooling_efficiency_statistical_tests.csv", index=False)
    posthoc.to_csv("cooling_efficiency_posthoc_mannwhitney_holm.csv", index=False)

    print("\nArquivos exportados:")
    print("- cooling_efficiency_AMJJ_month_pairs.csv")
    print("- cooling_efficiency_rain_intensity_class_summary.csv")
    print("- cooling_efficiency_rain_bins05_summary.csv")
    print("- cooling_efficiency_statistical_tests.csv")
    print("- cooling_efficiency_posthoc_mannwhitney_holm.csv")
    print("- Fig_cooling_efficiency_rain_bins05.png")
    print("- Fig_cooling_efficiency_conditioned_initial_sst.png")
