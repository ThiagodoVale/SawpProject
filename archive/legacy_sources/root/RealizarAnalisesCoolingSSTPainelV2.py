#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Teste suplementar: o resfriamento da anomalia de TSM da SAWP enfraquece a chuva
regional do ENEB ao longo do restante da estação chuvosa?

Objetivo científico
-------------------
Este script foi desenhado para testar, sem impor a resposta, a afirmativa:

    "SAWP SST cooling weakens its influence on regional precipitation throughout
     the rainy season."

A ideia é separar três etapas por ano/evento:

1. Identificar o mês de máxima anomalia de chuva do ENEB dentro de AMJJ.
2. Calcular o resfriamento posterior da SAWP usando ANOMALIA de TSM:

       SST_cooling_after_peak = Anomalia_TSM(pico_chuva)
                                - média[Anomalia_TSM(meses após pico)]

   Portanto:
       valor positivo = redução da anomalia de TSM após o pico de chuva.

3. Verificar se esse resfriamento está associado a menor chuva posterior:

       Rain_weakening_after_peak = Chuva_anom(pico_chuva)
                                   - média[Chuva_anom(meses após pico)]

   Portanto:
       valor positivo = chuva posterior mais fraca do que no mês de pico.

A hipótese "weakening" ganha suporte quando:
    - a chuva posterior é menor que a chuva no pico;
    - a anomalia de TSM diminui após o pico;
    - anos com maior resfriamento apresentam menor chuva posterior ou maior
      enfraquecimento da chuva.

Observação importante
---------------------
Como o pico de chuva é escolhido por construção, a chuva no lag 0 tende a ser
maior. Por isso o teste principal não é apenas pico vs pós-pico, mas sim a
associação entre resfriamento da TSM e chuva posterior/enfraquecimento da chuva.

Entradas esperadas
------------------
O script usa as mesmas funções do pacote sawp já utilizadas no seu fluxo:
    - sawp.ConfigurarTempoArea
    - sawp.CarregarDados
    - sawp.calcular_anomalia_tsm_mensal_todo_periodo
    - sawp.gerar_serie_chuva_mensal_acumulada

Saídas principais
-----------------
CSV:
    Weakening_event_metrics.csv
    Weakening_summary_by_class.csv
    Weakening_lag_table.csv
    SAWP_AMJJ_event_classification_detrended.csv

Figuras:
    Fig_Sz_weakening_test_summary.png
    Fig_Sz_weakening_test_individual_events.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr, spearmanr, ttest_rel, ttest_1samp
import sawp


# ============================================================
# CONFIGURAÇÕES
# ============================================================

TENDENCIAS = "não"  # 'sim' ou 'não'
ANOS_CLIMA = (1980, 2010)

# Mês de pico será procurado em AMJJ.
PEAK_MONTHS = (4, 5, 6, 7)

# Chuva posterior será avaliada até o fim da janela abaixo.
# Use (4,5,6,7,8) se quiser incluir agosto como fim da estação ampliada.
ANALYSIS_MONTHS = (4, 5, 6, 7, 8)

LAGS = (-2, -1, 0, 1, 2)

FRAC_EXTREMO = 0.10
FRAC_MODERADO = 0.15

OUT_PREFIX = "Weakening"

DICIONARIO_AREA_ESTUDO = {
    "SAWP": [-34, -25, -12.5, -4],
    "NEB": [-50, -34, -15, 0]
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def _padronizar_indice_mensal(obj, nome_coluna="valor"):
    """Aceita Series ou DataFrame e devolve DataFrame com índice no fim do mês."""

    if isinstance(obj, pd.Series):
        df = obj.to_frame(name=nome_coluna)
    elif isinstance(obj, pd.DataFrame):
        df = obj.copy()
    else:
        raise TypeError("Entrada deve ser pandas Series ou DataFrame.")

    df.index = pd.DatetimeIndex(pd.to_datetime(df.index, errors="coerce"))

    if df.index.isna().any():
        raise ValueError("Há datas inválidas no índice.")

    df.index = df.index.to_period("M").to_timestamp("M")
    return df.sort_index()


def _ano_decimal(index):
    index = pd.DatetimeIndex(index)
    return index.year + (index.dayofyear - 1) / 365.25


def _safe_pearson(x, y):
    dados = pd.concat([pd.Series(x).rename("x"), pd.Series(y).rename("y")], axis=1).dropna()
    if len(dados) < 3:
        return np.nan, np.nan, len(dados)
    r, p = pearsonr(dados["x"], dados["y"])
    return r, p, len(dados)


def _safe_spearman(x, y):
    dados = pd.concat([pd.Series(x).rename("x"), pd.Series(y).rename("y")], axis=1).dropna()
    if len(dados) < 3:
        return np.nan, np.nan, len(dados)
    r, p = spearmanr(dados["x"], dados["y"])
    return r, p, len(dados)


# ============================================================
# CHUVA: ANOMALIA MENSAL NORMALIZADA
# ============================================================

def normalizar_anomalia_chuva_mensal(
    chuva_mensal,
    tendencias="não",
    coluna="precipitation_mm",
    anos_clima=(1980, 2010)
):
    """
    Calcula anomalia mensal de chuva e anomalia normalizada pelo desvio padrão.

    Se tendencias == 'sim', remove tendência linear mensal antes da anomalia.
    """

    chuva_df = _padronizar_indice_mensal(chuva_mensal, nome_coluna=coluna)

    if coluna not in chuva_df.columns:
        raise KeyError(f"Coluna '{coluna}' não encontrada em chuva_mensal.")

    df = pd.DataFrame({"chuva_original": chuva_df[coluna].astype(float)})
    df["Ano"] = df.index.year
    df["Mes"] = df.index.month

    remover_tendencia = str(tendencias).lower() in ["sim", "s", "yes", "y"]

    if remover_tendencia:
        valid = df["chuva_original"].notna()
        x = _ano_decimal(df.index)

        if valid.sum() >= 3:
            reg = linregress(x[valid], df.loc[valid, "chuva_original"].values)
            tendencia = reg.slope * x + reg.intercept
            df["chuva_processada"] = (
                df["chuva_original"] - tendencia + df.loc[valid, "chuva_original"].mean()
            )
            print(f"Tendência removida da chuva mensal: {reg.slope:.4f} mm/ano | p={reg.pvalue:.4f}")
        else:
            df["chuva_processada"] = df["chuva_original"]
            print("Chuva: dados insuficientes para remover tendência.")
    else:
        df["chuva_processada"] = df["chuva_original"]

    mascara_clima = (
        (df["Ano"] >= anos_clima[0]) &
        (df["Ano"] <= anos_clima[1])
    )

    climatologia_mensal = (
        df.loc[mascara_clima]
        .groupby("Mes")["chuva_processada"]
        .mean()
    )

    df["climatologia_mensal"] = df["Mes"].map(climatologia_mensal)
    df["anomalia"] = df["chuva_processada"] - df["climatologia_mensal"]

    std_anomalia = df.loc[mascara_clima, "anomalia"].std(skipna=True, ddof=1)

    if not np.isfinite(std_anomalia) or std_anomalia == 0:
        raise RuntimeError("Desvio padrão da anomalia de chuva inválido ou zero.")

    df["anomalia_normalizada"] = df["anomalia"] / std_anomalia

    return df


# ============================================================
# TSM: DETREND MENSAL PARA CLASSIFICAÇÃO DOS EVENTOS
# ============================================================

def remover_tendencia_mensal_tsm(
    serie_tsm,
    coluna="Anomalia",
    nova_coluna="Anomalia_detrended"
):
    """Remove tendência linear separadamente para cada mês do calendário."""

    df = _padronizar_indice_mensal(serie_tsm, nome_coluna=coluna)

    if coluna not in df.columns:
        raise KeyError(f"Coluna '{coluna}' não encontrada na série de TSM.")

    df["Ano"] = df.index.year
    df["Mes"] = df.index.month
    df[nova_coluna] = np.nan

    trend_rows = []

    for mes in range(1, 13):
        mask_mes = df["Mes"] == mes
        y = df.loc[mask_mes, coluna].astype(float)
        valid = y.notna()

        if valid.sum() < 3:
            df.loc[mask_mes, nova_coluna] = y.values
            trend_rows.append({"month": mes, "slope_per_year": np.nan, "p_value": np.nan, "n": int(valid.sum())})
            continue

        x = _ano_decimal(y.index)
        reg = linregress(x[valid], y.loc[valid].values)
        tendencia = reg.slope * x + reg.intercept

        y_dt = y.copy()
        y_dt.loc[valid] = y.loc[valid].values - tendencia[valid] + y.loc[valid].mean()
        df.loc[mask_mes, nova_coluna] = y_dt.values

        trend_rows.append({
            "month": mes,
            "slope_per_year": reg.slope,
            "p_value": reg.pvalue,
            "n": int(valid.sum())
        })

    return df, pd.DataFrame(trend_rows)


def calcular_percentis_amjj_ranking_detrended(
    serie_tsm,
    coluna="Anomalia",
    frac_extremo=0.10,
    frac_moderado=0.15
):
    """
    Classifica os anos usando ranking da média AMJJ da anomalia de TSM detrendada.

    Retorna:
        media_amjj, anos_moderadamente_quentes, anos_extremamente_quentes,
        serie_tsm_detrended, tendencias_mensais_tsm
    """

    tsm_dt, trends_df = remover_tendencia_mensal_tsm(
        serie_tsm,
        coluna=coluna,
        nova_coluna="Anomalia_detrended"
    )

    amjj = tsm_dt[tsm_dt["Mes"].isin([4, 5, 6, 7])].copy()

    media_amjj = pd.DataFrame({
        "TSM_AMJJ_original": amjj.groupby("Ano")[coluna].mean(),
        "TSM_AMJJ_detrended": amjj.groupby("Ano")["Anomalia_detrended"].mean()
    })

    valid = media_amjj["TSM_AMJJ_detrended"].dropna().sort_values()
    n_total = len(valid)

    if n_total < 10:
        raise RuntimeError("Poucos anos válidos para classificação por ranking.")

    n_extremo = max(1, int(round(frac_extremo * n_total)))
    n_moderado = max(1, int(round(frac_moderado * n_total)))
    n_moderado = min(n_moderado, n_total - n_extremo)

    anos_extremamente_quentes = valid.index[-n_extremo:].astype(int)
    anos_moderadamente_quentes = valid.index[-(n_extremo + n_moderado):-n_extremo].astype(int)

    media_amjj["class"] = "other"
    media_amjj.loc[anos_moderadamente_quentes, "class"] = "Moderately warm"
    media_amjj.loc[anos_extremamente_quentes, "class"] = "Extremely warm"

    print("\nClassificação dos eventos por ranking da TSM AMJJ detrendada")
    print(f"Total de anos válidos: {n_total}")
    print(f"Moderately warm ({len(anos_moderadamente_quentes)}): {list(anos_moderadamente_quentes)}")
    print(f"Extremely warm ({len(anos_extremamente_quentes)}): {list(anos_extremamente_quentes)}")

    return (
        media_amjj,
        anos_moderadamente_quentes,
        anos_extremamente_quentes,
        tsm_dt,
        trends_df
    )


# ============================================================
# TESTE DA AFIRMATIVA: WEAKENING OF RAINFALL LATER IN SEASON
# ============================================================

def construir_lag_table_centrado_no_pico_chuva(
    serie_tsm,
    chuva_norm,
    anos_moderadamente_quentes,
    anos_extremamente_quentes,
    coluna_sst="Anomalia",
    coluna_chuva="anomalia_normalizada",
    peak_months=(4, 5, 6, 7),
    lags=(-2, -1, 0, 1, 2)
):
    """
    Constrói tabela ano-lag centrada no mês de máxima anomalia de chuva.

    Usa a ANOMALIA de TSM para calcular o resfriamento relativo ao pico:
        sst_cooling_relative_to_peak = sst_anomaly_peak - sst_anomaly_lag

    Valor positivo = resfriamento da anomalia de TSM em relação ao mês do pico.
    """

    sst = _padronizar_indice_mensal(serie_tsm, nome_coluna=coluna_sst)
    rain = _padronizar_indice_mensal(chuva_norm)

    if coluna_sst not in sst.columns:
        raise KeyError(f"Coluna '{coluna_sst}' não encontrada em serie_tsm.")
    if coluna_chuva not in rain.columns:
        raise KeyError(f"Coluna '{coluna_chuva}' não encontrada em chuva_norm.")

    dados = pd.concat(
        [
            sst[coluna_sst].rename("sst_anomaly"),
            rain[coluna_chuva].rename("rain_anomaly"),
            rain.get("anomalia", pd.Series(index=rain.index, dtype=float)).rename("rain_anomaly_mm"),
            rain.get("chuva_processada", pd.Series(index=rain.index, dtype=float)).rename("rain_processed_mm"),
        ],
        axis=1
    )

    dados["Ano"] = dados.index.year
    dados["Mes"] = dados.index.month

    grupos = {
        "Moderately warm": pd.Index(anos_moderadamente_quentes).astype(int),
        "Extremely warm": pd.Index(anos_extremamente_quentes).astype(int),
    }

    rows = []

    for classe, anos in grupos.items():
        for ano in anos:
            dados_ano = dados[dados["Ano"] == ano].copy()

            chuva_peak_window = dados_ano.loc[
                dados_ano["Mes"].isin(peak_months),
                "rain_anomaly"
            ].dropna()

            if chuva_peak_window.empty:
                continue

            data_pico = chuva_peak_window.idxmax()
            sst_pico = dados.loc[data_pico, "sst_anomaly"]
            rain_pico = dados.loc[data_pico, "rain_anomaly"]

            if not (np.isfinite(sst_pico) and np.isfinite(rain_pico)):
                continue

            for lag in lags:
                data_lag = data_pico + pd.DateOffset(months=int(lag))
                data_lag = data_lag.to_period("M").to_timestamp("M")

                if data_lag not in dados.index:
                    continue

                sst_lag = dados.loc[data_lag, "sst_anomaly"]
                rain_lag = dados.loc[data_lag, "rain_anomaly"]

                if not (np.isfinite(sst_lag) and np.isfinite(rain_lag)):
                    continue

                rows.append({
                    "class": classe,
                    "year": int(ano),
                    "peak_rain_date": data_pico,
                    "peak_rain_month": int(data_pico.month),
                    "peak_rain_month_name": data_pico.strftime("%b"),
                    "lag": int(lag),
                    "date": data_lag,
                    "month": int(data_lag.month),
                    "sst_anomaly": float(sst_lag),
                    "sst_anomaly_at_peak": float(sst_pico),
                    "sst_cooling_relative_to_peak": float(sst_pico - sst_lag),
                    "rain_anomaly": float(rain_lag),
                    "rain_anomaly_at_peak": float(rain_pico),
                    "rain_weakening_relative_to_peak": float(rain_pico - rain_lag),
                })

    lag_df = pd.DataFrame(rows)

    if lag_df.empty:
        raise RuntimeError("Nenhum dado válido para construir a lag table.")

    return lag_df


def calcular_metricas_weakening_por_evento(
    lag_df,
    analysis_months=(4, 5, 6, 7, 8)
):
    """
    Calcula métricas por evento para testar se o resfriamento da SAWP está
    associado a enfraquecimento da chuva posteriormente na estação.
    """

    event_rows = []

    for (classe, ano), sub in lag_df.groupby(["class", "year"]):
        peak_date = pd.to_datetime(sub["peak_rain_date"].iloc[0])
        peak_month = int(sub["peak_rain_month"].iloc[0])
        peak_name = sub["peak_rain_month_name"].iloc[0]

        peak_row = sub[sub["lag"] == 0]
        if peak_row.empty:
            continue

        rain_peak = float(peak_row["rain_anomaly"].iloc[0])
        sst_peak = float(peak_row["sst_anomaly"].iloc[0])

        # Lags posteriores disponíveis dentro da tabela.
        post = sub[sub["lag"] > 0].copy()

        if post.empty:
            continue

        # Opcionalmente restringe o período posterior aos meses da janela de análise.
        post = post[post["month"].isin(analysis_months)]

        if post.empty:
            continue

        post_rain_mean = post["rain_anomaly"].mean()
        post_sst_mean = post["sst_anomaly"].mean()
        post_cooling_mean = post["sst_cooling_relative_to_peak"].mean()
        post_cooling_lag1 = post.loc[post["lag"] == 1, "sst_cooling_relative_to_peak"].mean()
        post_cooling_lag2 = post.loc[post["lag"] == 2, "sst_cooling_relative_to_peak"].mean()

        rain_weakening = rain_peak - post_rain_mean

        # Métrica alternativa: chuva posterior média em si. Se a hipótese for verdadeira,
        # maior cooling deveria se associar a menor post_rain_mean.
        event_rows.append({
            "class": classe,
            "year": int(ano),
            "peak_rain_month": peak_month,
            "peak_rain_month_name": peak_name,
            "rain_peak_anomaly_sigma": rain_peak,
            "post_rain_mean_sigma": post_rain_mean,
            "rain_weakening_after_peak_sigma": rain_weakening,
            "sst_anomaly_at_peak": sst_peak,
            "post_sst_anomaly_mean": post_sst_mean,
            "sst_cooling_after_peak": sst_peak - post_sst_mean,
            "sst_cooling_after_peak_from_lag_table": post_cooling_mean,
            "sst_cooling_lag1": post_cooling_lag1,
            "sst_cooling_lag2": post_cooling_lag2,
            "n_post_months": int(len(post)),
            "post_rain_negative": bool(post_rain_mean < 0),
            "rain_weakened_after_peak": bool(rain_weakening > 0),
            "sst_cooled_after_peak": bool((sst_peak - post_sst_mean) > 0),
        })

    events = pd.DataFrame(event_rows)

    if events.empty:
        raise RuntimeError("Nenhuma métrica por evento foi calculada.")

    return events


def resumir_teste_weakening(events):
    """Gera tabela de estatísticas por classe e diagnóstico objetivo."""

    rows = []

    for classe, sub in events.groupby("class"):
        n = len(sub)

        r_cool_postrain, p_cool_postrain, n_corr1 = _safe_pearson(
            sub["sst_cooling_after_peak"],
            sub["post_rain_mean_sigma"]
        )
        rho_cool_postrain, p_s_cool_postrain, _ = _safe_spearman(
            sub["sst_cooling_after_peak"],
            sub["post_rain_mean_sigma"]
        )

        r_cool_weak, p_cool_weak, n_corr2 = _safe_pearson(
            sub["sst_cooling_after_peak"],
            sub["rain_weakening_after_peak_sigma"]
        )
        rho_cool_weak, p_s_cool_weak, _ = _safe_spearman(
            sub["sst_cooling_after_peak"],
            sub["rain_weakening_after_peak_sigma"]
        )

        # Testa se chuva posterior é, em média, menor que zero e se o enfraquecimento é > 0.
        if n >= 2:
            t_post, p_post = ttest_1samp(sub["post_rain_mean_sigma"].dropna(), popmean=0.0)
            t_weak, p_weak = ttest_1samp(sub["rain_weakening_after_peak_sigma"].dropna(), popmean=0.0)
            t_cool, p_cool = ttest_1samp(sub["sst_cooling_after_peak"].dropna(), popmean=0.0)
        else:
            t_post = p_post = t_weak = p_weak = t_cool = p_cool = np.nan

        # Critério diagnóstico: não é prova causal, mas apoio estatístico/descritivo.
        # Suporte se: cooling médio positivo, weakening médio positivo e correlação
        # cooling x post-rain negativa OU cooling x weakening positiva.
        support_flag = (
            sub["sst_cooling_after_peak"].mean() > 0 and
            sub["rain_weakening_after_peak_sigma"].mean() > 0 and
            (
                (np.isfinite(r_cool_postrain) and r_cool_postrain < 0) or
                (np.isfinite(r_cool_weak) and r_cool_weak > 0)
            )
        )

        rows.append({
            "class": classe,
            "n_events": n,
            "mean_sst_cooling_after_peak": sub["sst_cooling_after_peak"].mean(),
            "median_sst_cooling_after_peak": sub["sst_cooling_after_peak"].median(),
            "fraction_events_with_sst_cooling": sub["sst_cooled_after_peak"].mean(),
            "mean_peak_rain_sigma": sub["rain_peak_anomaly_sigma"].mean(),
            "mean_post_rain_sigma": sub["post_rain_mean_sigma"].mean(),
            "fraction_events_with_negative_post_rain": sub["post_rain_negative"].mean(),
            "mean_rain_weakening_after_peak_sigma": sub["rain_weakening_after_peak_sigma"].mean(),
            "fraction_events_with_rain_weakening": sub["rain_weakened_after_peak"].mean(),
            "p_one_sample_post_rain_vs_zero": p_post,
            "p_one_sample_weakening_vs_zero": p_weak,
            "p_one_sample_cooling_vs_zero": p_cool,
            "pearson_r_cooling_vs_post_rain": r_cool_postrain,
            "pearson_p_cooling_vs_post_rain": p_cool_postrain,
            "spearman_rho_cooling_vs_post_rain": rho_cool_postrain,
            "spearman_p_cooling_vs_post_rain": p_s_cool_postrain,
            "pearson_r_cooling_vs_rain_weakening": r_cool_weak,
            "pearson_p_cooling_vs_rain_weakening": p_cool_weak,
            "spearman_rho_cooling_vs_rain_weakening": rho_cool_weak,
            "spearman_p_cooling_vs_rain_weakening": p_s_cool_weak,
            "diagnostic_supports_weakening": support_flag,
        })

    summary = pd.DataFrame(rows)

    # Também calcula todos os eventos juntos.
    sub = events.copy()
    r_cool_postrain, p_cool_postrain, _ = _safe_pearson(sub["sst_cooling_after_peak"], sub["post_rain_mean_sigma"])
    r_cool_weak, p_cool_weak, _ = _safe_pearson(sub["sst_cooling_after_peak"], sub["rain_weakening_after_peak_sigma"])

    overall = pd.DataFrame([{
        "class": "All warm events",
        "n_events": len(sub),
        "mean_sst_cooling_after_peak": sub["sst_cooling_after_peak"].mean(),
        "median_sst_cooling_after_peak": sub["sst_cooling_after_peak"].median(),
        "fraction_events_with_sst_cooling": sub["sst_cooled_after_peak"].mean(),
        "mean_peak_rain_sigma": sub["rain_peak_anomaly_sigma"].mean(),
        "mean_post_rain_sigma": sub["post_rain_mean_sigma"].mean(),
        "fraction_events_with_negative_post_rain": sub["post_rain_negative"].mean(),
        "mean_rain_weakening_after_peak_sigma": sub["rain_weakening_after_peak_sigma"].mean(),
        "fraction_events_with_rain_weakening": sub["rain_weakened_after_peak"].mean(),
        "p_one_sample_post_rain_vs_zero": ttest_1samp(sub["post_rain_mean_sigma"].dropna(), 0.0).pvalue if len(sub) >= 2 else np.nan,
        "p_one_sample_weakening_vs_zero": ttest_1samp(sub["rain_weakening_after_peak_sigma"].dropna(), 0.0).pvalue if len(sub) >= 2 else np.nan,
        "p_one_sample_cooling_vs_zero": ttest_1samp(sub["sst_cooling_after_peak"].dropna(), 0.0).pvalue if len(sub) >= 2 else np.nan,
        "pearson_r_cooling_vs_post_rain": r_cool_postrain,
        "pearson_p_cooling_vs_post_rain": p_cool_postrain,
        "spearman_rho_cooling_vs_post_rain": _safe_spearman(sub["sst_cooling_after_peak"], sub["post_rain_mean_sigma"])[0],
        "spearman_p_cooling_vs_post_rain": _safe_spearman(sub["sst_cooling_after_peak"], sub["post_rain_mean_sigma"])[1],
        "pearson_r_cooling_vs_rain_weakening": r_cool_weak,
        "pearson_p_cooling_vs_rain_weakening": p_cool_weak,
        "spearman_rho_cooling_vs_rain_weakening": _safe_spearman(sub["sst_cooling_after_peak"], sub["rain_weakening_after_peak_sigma"])[0],
        "spearman_p_cooling_vs_rain_weakening": _safe_spearman(sub["sst_cooling_after_peak"], sub["rain_weakening_after_peak_sigma"])[1],
        "diagnostic_supports_weakening": (
            sub["sst_cooling_after_peak"].mean() > 0 and
            sub["rain_weakening_after_peak_sigma"].mean() > 0 and
            ((np.isfinite(r_cool_postrain) and r_cool_postrain < 0) or
             (np.isfinite(r_cool_weak) and r_cool_weak > 0))
        )
    }])

    summary = pd.concat([summary, overall], ignore_index=True)

    return summary


# ============================================================
# FIGURAS
# ============================================================

def plotar_figura_teste_weakening(events, summary, nome_figura="Fig_Sz_weakening_test_summary.png"):
    """Figura-resumo para comprovar ou não o enfraquecimento da chuva posterior."""

    classes = ["Moderately warm", "Extremely warm"]
    titles = ["(a) Moderately warm", "(b) Extremely warm"]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # --------------------------------------------------------
    # Painéis superiores: pico vs chuva posterior por classe
    # --------------------------------------------------------
    for ax, classe, title in zip(axes[0], classes, titles):
        sub = events[events["class"] == classe].sort_values("year")
        x = np.arange(len(sub))
        width = 0.38

        ax.bar(x - width/2, sub["rain_peak_anomaly_sigma"], width=width, label="Peak rainfall anomaly")
        ax.bar(x + width/2, sub["post_rain_mean_sigma"], width=width, label="Post-peak rainfall anomaly")
        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["year"].astype(int), rotation=90, fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("Rainfall anomaly (σ)")
        ax.grid(axis="y", alpha=0.3)

        s = summary[summary["class"] == classe].iloc[0]
        ax.text(
            0.02, 0.97,
            f"n={int(s['n_events'])}\n"
            f"mean post rain={s['mean_post_rain_sigma']:.2f} σ\n"
            f"rain weakening={s['mean_rain_weakening_after_peak_sigma']:.2f} σ\n"
            f"years weakened={100*s['fraction_events_with_rain_weakening']:.0f}%",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85)
        )

        ax.legend(fontsize=9, frameon=True)

    # --------------------------------------------------------
    # Painel inferior esquerdo: cooling vs post-rain
    # --------------------------------------------------------
    ax = axes[1, 0]
    markers = {"Moderately warm": "o", "Extremely warm": "s"}

    for classe in classes:
        sub = events[events["class"] == classe]
        ax.scatter(
            sub["sst_cooling_after_peak"],
            sub["post_rain_mean_sigma"],
            s=70,
            marker=markers[classe],
            label=classe,
            alpha=0.85
        )

        for _, row in sub.iterrows():
            ax.text(row["sst_cooling_after_peak"], row["post_rain_mean_sigma"], str(int(row["year"])), fontsize=8)

    ax.axhline(0, color="black", linewidth=1.0)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel("SAWP SST cooling after rainfall peak\nAnomaly at peak - post-peak anomaly")
    ax.set_ylabel("Post-peak ENEB rainfall anomaly (σ)")
    ax.set_title("(c) Does stronger cooling correspond to lower later rainfall?")
    ax.grid(alpha=0.3)
    ax.legend(frameon=True)

    overall = summary[summary["class"] == "All warm events"].iloc[0]
    ax.text(
        0.02, 0.04,
        f"All events: r = {overall['pearson_r_cooling_vs_post_rain']:.2f}, "
        f"p = {overall['pearson_p_cooling_vs_post_rain']:.3f}",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85)
    )

    # --------------------------------------------------------
    # Painel inferior direito: cooling vs rain weakening
    # --------------------------------------------------------
    ax = axes[1, 1]

    for classe in classes:
        sub = events[events["class"] == classe]
        ax.scatter(
            sub["sst_cooling_after_peak"],
            sub["rain_weakening_after_peak_sigma"],
            s=70,
            marker=markers[classe],
            label=classe,
            alpha=0.85
        )

        for _, row in sub.iterrows():
            ax.text(row["sst_cooling_after_peak"], row["rain_weakening_after_peak_sigma"], str(int(row["year"])), fontsize=8)

    ax.axhline(0, color="black", linewidth=1.0)
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_xlabel("SAWP SST cooling after rainfall peak\nAnomaly at peak - post-peak anomaly")
    ax.set_ylabel("Rainfall weakening after peak (σ)")
    ax.set_title("(d) Does stronger cooling correspond to stronger rainfall weakening?")
    ax.grid(alpha=0.3)
    ax.legend(frameon=True)

    ax.text(
        0.02, 0.04,
        f"All events: r = {overall['pearson_r_cooling_vs_rain_weakening']:.2f}, "
        f"p = {overall['pearson_p_cooling_vs_rain_weakening']:.3f}",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=10,
        bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85)
    )

    fig.suptitle(
        "Does SAWP SST cooling weaken ENEB rainfall later in the rainy season?\n"
        "Peak rainfall is identified within AMJJ; post-peak rainfall is averaged over subsequent months",
        fontsize=15
    )

    plt.tight_layout()
    plt.savefig(nome_figura, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figura salva em: {nome_figura}")


def plotar_paineis_individuais_weakening(
    lag_df,
    events,
    classe="Extremely warm",
    nome_figura="Fig_Sz_weakening_test_individual_events.png"
):
    """Painéis individuais por ano para mostrar chuva e resfriamento relativo ao pico."""

    years = sorted(events.loc[events["class"] == classe, "year"].unique())

    if len(years) == 0:
        print(f"Sem eventos para {classe}")
        return

    n = len(years)
    ncols = 3
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows), sharex=True, sharey=True)
    axes = np.atleast_1d(axes).ravel()

    letters = list("abcdefghijklmnopqrstuvwxyz")

    for i, (ax, ano) in enumerate(zip(axes, years)):
        sub = lag_df[(lag_df["class"] == classe) & (lag_df["year"] == ano)].sort_values("lag")
        ev = events[(events["class"] == classe) & (events["year"] == ano)].iloc[0]

        ax.bar(sub["lag"], sub["rain_anomaly"], width=0.55, alpha=0.60, label="Rainfall anomaly")
        ax.plot(sub["lag"], sub["sst_cooling_relative_to_peak"], marker="o", linewidth=2, label="SAWP SST cooling")

        ax.axhline(0, color="black", linewidth=1.0)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.set_xticks(list(LAGS))

        ax.set_title(
            f"({letters[i]}) {int(ano)} | Peak: {ev['peak_rain_month_name']}",
            fontsize=11
        )

        ax.text(
            0.03, 0.92,
            f"post rain={ev['post_rain_mean_sigma']:.2f} σ\n"
            f"cooling={ev['sst_cooling_after_peak']:.2f}",
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85)
        )

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    fig.supxlabel("Months relative to rainfall peak", fontsize=12)
    fig.supylabel("Normalized rainfall anomaly / SAWP SST cooling anomaly\nPositive SST line = cooling", fontsize=12)
    fig.suptitle(
        f"Individual events centered on maximum ENEB rainfall anomaly\n"
        f"{classe}: does post-peak SAWP cooling coincide with weaker later rainfall?",
        fontsize=14
    )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)

    plt.tight_layout(rect=[0, 0.06, 1, 0.92])
    plt.savefig(nome_figura, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Figura salva em: {nome_figura}")


# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

if __name__ == "__main__":

    caminho_ = "./outputs/" if TENDENCIAS == "não" else "./outputs_trended/"

    periodo_tempo_convertido, lon, lat = sawp.ConfigurarTempoArea(
        "./data/Era5_Dataset_sst.nc"
    )

    tsm_arquivo = "detrended_sst.npz" if TENDENCIAS == "não" else "trended_uvq_sst.npz"

    tsm = sawp.CarregarDados(
        caminho_ + tsm_arquivo,
        tendencias=TENDENCIAS,
        nome_var_nc="sst",
        substituir_nan=-999
    )

    print("TSM carregada.")

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
        llaN
    )

    serie_tsm = _padronizar_indice_mensal(resultado, nome_coluna="Anomalia")

    if "Anomalia" not in serie_tsm.columns:
        raise KeyError("A série de TSM não contém a coluna 'Anomalia'.")

    chuva_mensal, matriz_estacoes_qc, metadata, removidos, regime_test = (
        sawp.gerar_serie_chuva_mensal_acumulada()
    )

    chuva_norm = normalizar_anomalia_chuva_mensal(
        chuva_mensal,
        tendencias=TENDENCIAS,
        coluna="precipitation_mm",
        anos_clima=ANOS_CLIMA
    )

    (
        media_amjj,
        anos_moderadamente_quentes,
        anos_extremamente_quentes,
        serie_tsm_detrended,
        tendencias_mensais_tsm
    ) = calcular_percentis_amjj_ranking_detrended(
        serie_tsm,
        coluna="Anomalia",
        frac_extremo=FRAC_EXTREMO,
        frac_moderado=FRAC_MODERADO
    )

    lag_df = construir_lag_table_centrado_no_pico_chuva(
        serie_tsm=serie_tsm,
        chuva_norm=chuva_norm,
        anos_moderadamente_quentes=anos_moderadamente_quentes,
        anos_extremamente_quentes=anos_extremamente_quentes,
        coluna_sst="Anomalia",
        coluna_chuva="anomalia_normalizada",
        peak_months=PEAK_MONTHS,
        lags=LAGS
    )

    events = calcular_metricas_weakening_por_evento(
        lag_df,
        analysis_months=ANALYSIS_MONTHS
    )

    summary = resumir_teste_weakening(events)

    print("\nResumo do teste de weakening:")
    print(summary)

    # Exportações
    lag_df.to_csv(f"{OUT_PREFIX}_lag_table.csv", index=False)
    events.to_csv(f"{OUT_PREFIX}_event_metrics.csv", index=False)
    summary.to_csv(f"{OUT_PREFIX}_summary_by_class.csv", index=False)
    media_amjj.to_csv("SAWP_AMJJ_event_classification_detrended.csv")
    tendencias_mensais_tsm.to_csv("SAWP_monthly_SST_trends_removed_before_quantiles.csv", index=False)

    # Figuras
    plotar_figura_teste_weakening(
        events,
        summary,
        nome_figura="Fig_Sz_weakening_test_summary.png"
    )

    plotar_paineis_individuais_weakening(
        lag_df,
        events,
        classe="Moderately warm",
        nome_figura="Fig_Sz_weakening_test_individual_moderately_warm.png"
    )

    plotar_paineis_individuais_weakening(
        lag_df,
        events,
        classe="Extremely warm",
        nome_figura="Fig_Sz_weakening_test_individual_extremely_warm.png"
    )

    print("\nArquivos gerados:")
    print(f"- {OUT_PREFIX}_lag_table.csv")
    print(f"- {OUT_PREFIX}_event_metrics.csv")
    print(f"- {OUT_PREFIX}_summary_by_class.csv")
    print("- SAWP_AMJJ_event_classification_detrended.csv")
    print("- SAWP_monthly_SST_trends_removed_before_quantiles.csv")
    print("- Fig_Sz_weakening_test_summary.png")
    print("- Fig_Sz_weakening_test_individual_moderately_warm.png")
    print("- Fig_Sz_weakening_test_individual_extremely_warm.png")
