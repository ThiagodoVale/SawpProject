#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise suplementar: mudança da ANOMALIA de TSM da SAWP entre abril e julho e anomalia
regional de chuva ENEB para anos moderadamente quentes e extremamente quentes.

Pontos centrais desta versão:
    1. A TSM usada para CLASSIFICAR os eventos é detrendada mensalmente antes
       de calcular a média AMJJ.
    2. A seleção dos grupos é feita por ranking da TSM_AMJJ detrendada:
           - extremely warm: 10% anos mais quentes;
           - moderately warm: 15% imediatamente abaixo dos 10% mais quentes.
       Isso evita sobreposição e impede que P90 tenha mais anos que P75-P90.
    3. O resfriamento da SAWP é calculado SEMPRE sobre a ANOMALIA de TSM,
       não sobre a TSM bruta:
           SST_cooling_anom = Anomalia_TSM(abril) - Anomalia_TSM(julho).
           - SST_cooling_anom > 0 indica redução da anomalia/resfriamento relativo.
           - SST_cooling_anom < 0 indica aumento da anomalia/aquecimento relativo.
    4. A anomalia de chuva AMJJ é adicionada ao mesmo DataFrame e plotada em
       barras ao lado do ΔSST, em dois painéis:
           (a) moderately warm
           (b) extremely warm

Autor: adaptado para análise SAWP-ENEB
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress, pearsonr, ttest_ind
import sawp


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def _padronizar_indice_mensal(obj, nome_coluna="valor"):
    """
    Aceita Series ou DataFrame e devolve DataFrame com DatetimeIndex mensal
    padronizado para o último dia do mês.
    """

    if isinstance(obj, pd.Series):
        df = obj.to_frame(name=nome_coluna)
    elif isinstance(obj, pd.DataFrame):
        df = obj.copy()
    else:
        raise TypeError("Entrada deve ser pandas Series ou pandas DataFrame.")

    df.index = pd.DatetimeIndex(pd.to_datetime(df.index, errors="coerce"))

    if df.index.isna().any():
        raise ValueError("Há datas inválidas no índice da série.")

    df.index = df.index.to_period("M").to_timestamp("M")
    df = df.sort_index()

    return df


def _ano_decimal(index):
    """Converte DatetimeIndex para ano decimal."""

    index = pd.DatetimeIndex(index)
    return index.year + (index.dayofyear - 1) / 365.25


# ============================================================
# CHUVA: ANOMALIA E NORMALIZAÇÃO
# ============================================================

def normalizar_anomalia_chuva_mensal(
    chuva_mensal,
    tendencias="não",
    coluna="precipitation_mm",
    anos_clima=(1980, 2010)
):
    """
    Calcula anomalia mensal de chuva e anomalia normalizada pelo desvio padrão.

    Se tendencias == 'sim', remove a tendência linear da série mensal antes de
    calcular a climatologia mensal e a anomalia.
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
            df["chuva_processada"] = df["chuva_original"] - tendencia + df.loc[valid, "chuva_original"].mean()

            print(
                f"Tendência removida da chuva mensal: "
                f"{reg.slope:.4f} mm/ano | p = {reg.pvalue:.4f}"
            )
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


def calcular_chuva_amjj_eventos(chuva_norm):
    """
    Calcula chuva AMJJ por ano para ser adicionada aos eventos.

    Retorna:
        rainfall_anomaly_normalized_amjj : média AMJJ da anomalia normalizada
        rainfall_anomaly_mm_amjj         : soma AMJJ da anomalia mensal em mm
        rainfall_processed_mm_amjj       : soma AMJJ da chuva processada em mm
    """

    df = _padronizar_indice_mensal(chuva_norm)

    required = ["anomalia_normalizada", "anomalia", "chuva_processada"]
    missing = [col for col in required if col not in df.columns]

    if missing:
        raise KeyError(f"Colunas ausentes em chuva_norm: {missing}")

    df["Ano"] = df.index.year
    df["Mes"] = df.index.month

    amjj = df[df["Mes"].isin([4, 5, 6, 7])].copy()

    chuva_amjj = pd.DataFrame({
        "rainfall_anomaly_normalized_amjj": amjj.groupby("Ano")["anomalia_normalizada"].mean(),
        "rainfall_anomaly_mm_amjj": amjj.groupby("Ano")["anomalia"].sum(min_count=1),
        "rainfall_processed_mm_amjj": amjj.groupby("Ano")["chuva_processada"].sum(min_count=1),
    })

    return chuva_amjj


# ============================================================
# TSM: DETREND MENSAL, AMJJ E CLASSIFICAÇÃO DOS EVENTOS
# ============================================================

def remover_tendencia_mensal_tsm(
    serie_tsm,
    coluna="Anomalia",
    nova_coluna="Anomalia_detrended"
):
    """
    Remove tendência linear separadamente para cada mês do calendário.

    Isso evita misturar sazonalidade com tendência e preserva a variabilidade
    mensal interna antes de calcular os quantis AMJJ.
    """

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
            trend_rows.append({
                "month": mes,
                "slope_per_year": np.nan,
                "p_value": np.nan,
                "n": int(valid.sum())
            })
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

    trends_df = pd.DataFrame(trend_rows)

    print("\nTendência mensal removida da TSM antes dos quantis:")
    print(trends_df)

    return df, trends_df


def calcular_percentis_amjj_ranking_detrended(
    serie_tsm,
    coluna="Anomalia",
    frac_extremo=0.10,
    frac_moderado=0.15,
    anos_clima=(1980, 2010)
):
    """
    Classifica anos quentes da SAWP usando TSM AMJJ detrendada.

    A seleção é feita por ranking sobre todos os anos válidos:
        - anos_q90: 10% mais quentes;
        - anos_q75: 15% imediatamente abaixo dos q90.

    A climatologia 1980-2010 é mantida apenas como referência diagnóstica,
    não para controlar o tamanho dos grupos. Isso garante grupos mutuamente
    exclusivos e tamanho esperado.
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

    # Garante que não passe do número total de anos.
    n_moderado = min(n_moderado, n_total - n_extremo)

    anos_q90 = valid.index[-n_extremo:].astype(int)
    anos_q75 = valid.index[-(n_extremo + n_moderado):-n_extremo].astype(int)

    # Demais grupos apenas para manter compatibilidade com o fluxo anterior.
    n_frio_extremo = n_extremo
    n_frio_moderado = n_moderado

    anos_q10 = valid.index[:n_frio_extremo].astype(int)
    anos_q25 = valid.index[n_frio_extremo:n_frio_extremo + n_frio_moderado].astype(int)

    anos_especiais = set(anos_q10) | set(anos_q25) | set(anos_q75) | set(anos_q90)
    anos_normal = pd.Index([ano for ano in valid.index.astype(int) if ano not in anos_especiais])

    q_ref = media_amjj.loc[
        (media_amjj.index >= anos_clima[0]) &
        (media_amjj.index <= anos_clima[1]),
        "TSM_AMJJ_detrended"
    ].dropna().quantile([0.10, 0.25, 0.75, 0.90])

    tabela_quantis = pd.DataFrame({
        "metric": ["ranking_thresholds", "climatology_1980_2010_reference"],
        "q10": [valid.loc[anos_q10].max(), q_ref.loc[0.10]],
        "q25": [valid.loc[anos_q25].max(), q_ref.loc[0.25]],
        "q75": [valid.loc[anos_q75].min(), q_ref.loc[0.75]],
        "q90": [valid.loc[anos_q90].min(), q_ref.loc[0.90]],
        "n_total_valid_years": [n_total, len(q_ref)]
    })

    media_amjj["class"] = "normal_or_other"
    media_amjj.loc[anos_q10, "class"] = "extremely_cold"
    media_amjj.loc[anos_q25, "class"] = "moderately_cold"
    media_amjj.loc[anos_q75, "class"] = "moderately_warm"
    media_amjj.loc[anos_q90, "class"] = "extremely_warm"

    print("\nClassificação por ranking da TSM AMJJ detrendada")
    print(f"Total de anos válidos: {n_total}")
    print(f"Moderadamente quente, 15% abaixo do P90: {len(anos_q75)} anos")
    print(list(anos_q75))
    print(f"Extremamente quente, 10% mais quentes: {len(anos_q90)} anos")
    print(list(anos_q90))
    print("\nTabela de limiares/quantis:")
    print(tabela_quantis)

    return (
        media_amjj,
        anos_q10,
        anos_q25,
        anos_normal,
        anos_q75,
        anos_q90,
        tabela_quantis,
        tsm_dt,
        trends_df
    )


# ============================================================
# ΔSST ABRIL-JULHO + CHUVA AMJJ
# ============================================================

def analisar_delta_sst_abril_julho_com_chuva(
    serie_tsm,
    chuva_norm,
    anos_moderadamente_quentes,
    anos_extremamente_quentes,
    coluna_sst="Anomalia",
    salvar_figura=True,
    nome_figura="Fig_Sx_delta_SST_April_July_rainfall.png"
):
    """
    Calcula o RESFRIAMENTO da anomalia de TSM entre abril e julho:

        SST_cooling_anom_April_to_July = Anomalia_TSM(abril) - Anomalia_TSM(julho)

    Assim, valores POSITIVOS significam resfriamento relativo da SAWP.

    A figura mostra, para cada classe, barras lado a lado em um único eixo normalizado:
        - SAWP SST cooling anomaly, calculado sobre a anomalia de TSM;
        - anomalia de chuva AMJJ normalizada.
    """

    df_sst = _padronizar_indice_mensal(serie_tsm, nome_coluna=coluna_sst)

    if coluna_sst not in df_sst.columns:
        raise KeyError(f"Coluna '{coluna_sst}' não encontrada na série de TSM.")

    df_sst["Ano"] = df_sst.index.year
    df_sst["Mes"] = df_sst.index.month

    chuva_amjj = calcular_chuva_amjj_eventos(chuva_norm)

    anos_moderadamente_quentes = pd.Index(anos_moderadamente_quentes).astype(int)
    anos_extremamente_quentes = pd.Index(anos_extremamente_quentes).astype(int)

    resultados = []

    grupos = {
        "Moderately warm": anos_moderadamente_quentes,
        "Extremely warm": anos_extremamente_quentes
    }

    for classe, anos in grupos.items():
        for ano in anos:
            sst_abril = df_sst.loc[
                (df_sst["Ano"] == ano) & (df_sst["Mes"] == 4),
                coluna_sst
            ]

            sst_julho = df_sst.loc[
                (df_sst["Ano"] == ano) & (df_sst["Mes"] == 7),
                coluna_sst
            ]

            if len(sst_abril) == 0 or len(sst_julho) == 0:
                continue

            sst_abril = float(sst_abril.iloc[0])
            sst_julho = float(sst_julho.iloc[0])

            if not (np.isfinite(sst_abril) and np.isfinite(sst_julho)):
                continue

            rain_norm = np.nan
            rain_mm = np.nan
            rain_total = np.nan

            if ano in chuva_amjj.index:
                rain_norm = chuva_amjj.loc[ano, "rainfall_anomaly_normalized_amjj"]
                rain_mm = chuva_amjj.loc[ano, "rainfall_anomaly_mm_amjj"]
                rain_total = chuva_amjj.loc[ano, "rainfall_processed_mm_amjj"]

            resultados.append({
                "year": int(ano),
                "class": classe,
                "sst_anom_april": sst_abril,
                "sst_anom_july": sst_julho,
                "delta_sst_anom_july_minus_april": sst_julho - sst_abril,
                "sst_cooling_anom_april_minus_july": sst_abril - sst_julho,
                "rainfall_anomaly_normalized_amjj": rain_norm,
                "rainfall_anomaly_mm_amjj": rain_mm,
                "rainfall_processed_mm_amjj": rain_total
            })

    delta_df = pd.DataFrame(resultados)

    if delta_df.empty:
        raise RuntimeError(
            "Nenhum par abril-julho válido foi encontrado. "
            "Verifique anos de eventos e coluna de TSM."
        )

    # Ordena por classe e ano para deixar a figura reproduzível.
    order = {"Moderately warm": 0, "Extremely warm": 1}
    delta_df["class_order"] = delta_df["class"].map(order)
    delta_df = delta_df.sort_values(["class_order", "year"]).drop(columns="class_order")

    stats_df = (
        delta_df
        .groupby("class")
        .agg(
            n=("sst_cooling_anom_april_minus_july", "count"),
            mean_sst_cooling=("sst_cooling_anom_april_minus_july", "mean"),
            median_sst_cooling=("sst_cooling_anom_april_minus_july", "median"),
            std_sst_cooling=("sst_cooling_anom_april_minus_july", "std"),
            mean_rain_norm=("rainfall_anomaly_normalized_amjj", "mean"),
            mean_rain_mm=("rainfall_anomaly_mm_amjj", "mean")
        )
        .reset_index()
    )

    mod = delta_df.loc[
        delta_df["class"] == "Moderately warm",
        "sst_cooling_anom_april_minus_july"
    ].dropna()

    ext = delta_df.loc[
        delta_df["class"] == "Extremely warm",
        "sst_cooling_anom_april_minus_july"
    ].dropna()

    if len(mod) >= 2 and len(ext) >= 2:
        tstat, pvalue = ttest_ind(mod, ext, equal_var=False, nan_policy="omit")
    else:
        tstat, pvalue = np.nan, np.nan

    print("\nResumo do resfriamento da anomalia de TSM e chuva AMJJ")
    print(stats_df)
    print(f"\nTeste t Welch para SST_cooling_anom: t = {tstat:.3f}, p = {pvalue:.3f}")

    # ------------------------------------------------------------
    # FIGURA
    # ------------------------------------------------------------
    # Para plotar chuva e resfriamento da TSM no mesmo eixo, o resfriamento
    # é padronizado pelo próprio desvio padrão, preservando o zero:
    #     SST_cooling_norm = SST_cooling_anom / std(SST_cooling_anom)
    # Assim, zero é ausência de resfriamento/aquecimento relativo.
    std_delta = delta_df["sst_cooling_anom_april_minus_july"].std(skipna=True, ddof=1)

    if np.isfinite(std_delta) and std_delta > 0:
        delta_df["sst_cooling_anom_normalized"] = (
            delta_df["sst_cooling_anom_april_minus_july"] / std_delta
        )
    else:
        delta_df["sst_cooling_anom_normalized"] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    classes = ["Moderately warm", "Extremely warm"]
    labels = ["(a) Moderately warm", "(b) Extremely warm"]

    width = 0.38

    for ax, classe, label in zip(axes, classes, labels):
        sub = delta_df[delta_df["class"] == classe].copy()
        sub = sub.sort_values("year")

        x = np.arange(len(sub))

        ax.bar(
            x - width/2,
            sub["sst_cooling_anom_normalized"].values,
            width=width,
            label="SAWP SST cooling anomaly",
            alpha=0.80
        )

        ax.bar(
            x + width/2,
            sub["rainfall_anomaly_normalized_amjj"].values,
            width=width,
            label="Rainfall anomaly AMJJ",
            alpha=0.55
        )

        ax.axhline(0, color="black", linewidth=1.0)

        ax.set_title(label, fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["year"].astype(int), rotation=90, fontsize=8)
        ax.set_xlabel("Year")
        ax.grid(axis="y", alpha=0.3)

        mean_delta = sub["sst_cooling_anom_april_minus_july"].mean()
        mean_delta_norm = sub["sst_cooling_anom_normalized"].mean()
        mean_rain = sub["rainfall_anomaly_normalized_amjj"].mean()

        ax.text(
            0.02,
            0.96,
            f"Mean SST cooling = {mean_delta:.2f} °C\nMean cooling = {mean_delta_norm:.2f} σ\nMean rain = {mean_rain:.2f} σ\nn = {len(sub)}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85)
        )

        ax.legend(loc="lower left", fontsize=9)

    axes[0].set_ylabel("Normalized anomaly (standard deviations)\nPositive SST line = cooling")

    fig.suptitle(
        "SAWP SST cooling anomaly and ENEB rainfall anomaly during warm SAWP events",
        fontsize=15
    )

    plt.tight_layout()

    if salvar_figura:
        plt.savefig(nome_figura, dpi=300, bbox_inches="tight")
        print(f"Figura salva em: {nome_figura}")

    plt.show()

    return delta_df, stats_df



# ============================================================
# COMPOSITE CENTRADO NO PICO DE CHUVA
# ============================================================

def analisar_composite_centrado_no_pico_chuva_delta_sst(
    serie_tsm,
    chuva_norm,
    anos_moderadamente_quentes,
    anos_extremamente_quentes,
    coluna_sst="Anomalia",
    coluna_chuva="anomalia_normalizada",
    meses_estacao=(4, 5, 6, 7),
    lags=(-2, -1, 0, 1, 2),
    salvar_figura=True,
    nome_figura="Fig_Sx_lag_peak_rainfall_deltaSST.png"
):
    """
    Composite centrado no mês de maior anomalia de chuva em AMJJ.

    lag 0 = mês de maior anomalia de chuva dentro de AMJJ para cada ano.

    A linha de TSM NÃO mostra TSM bruta. Ela mostra explicitamente o
    RESFRIAMENTO da ANOMALIA de TSM em relação ao mês do pico de chuva:

        SST_cooling_anomaly_relative_to_peak = Anomalia_TSM(lag 0) - Anomalia_TSM(lag)

    Portanto:
        - valor positivo após lag 0 indica resfriamento em relação ao mês do pico;
        - valor negativo indica aumento da anomalia/aquecimento relativo.

    As barras mostram a anomalia de chuva média em cada lag.

    Retorna:
        lag_df    : dados ano a ano, lag a lag;
        composite : média, desvio padrão, erro padrão e n por classe/lag.
    """

    sst = _padronizar_indice_mensal(serie_tsm, nome_coluna=coluna_sst)
    rain = _padronizar_indice_mensal(chuva_norm)

    if coluna_sst not in sst.columns:
        raise KeyError(f"Coluna '{coluna_sst}' não encontrada na série de TSM.")

    if coluna_chuva not in rain.columns:
        raise KeyError(f"Coluna '{coluna_chuva}' não encontrada em chuva_norm.")

    dados = pd.concat(
        [
            sst[coluna_sst].rename("sst_anomaly"),
            rain[coluna_chuva].rename("rain_anomaly")
        ],
        axis=1
    )

    dados["Ano"] = dados.index.year
    dados["Mes"] = dados.index.month

    grupos = {
        "Moderately warm": pd.Index(anos_moderadamente_quentes).astype(int),
        "Extremely warm": pd.Index(anos_extremamente_quentes).astype(int)
    }

    resultados = []

    for classe, anos in grupos.items():
        for ano in anos:
            dados_ano = dados[dados["Ano"] == ano].copy()

            chuva_amjj = dados_ano.loc[
                dados_ano["Mes"].isin(meses_estacao),
                "rain_anomaly"
            ].dropna()

            if chuva_amjj.empty:
                continue

            data_pico = chuva_amjj.idxmax()
            sst_pico = dados.loc[data_pico, "sst_anomaly"]

            if not np.isfinite(sst_pico):
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

                resultados.append({
                    "year": int(ano),
                    "class": classe,
                    "peak_rain_date": data_pico,
                    "peak_rain_month": int(data_pico.month),
                    "lag": int(lag),
                    "date": data_lag,
                    "sst_anomaly": float(sst_lag),
                    "sst_at_peak": float(sst_pico),
                    "delta_sst_anomaly_relative_to_peak": float(sst_lag - sst_pico),
                    "sst_cooling_anomaly_relative_to_peak": float(sst_pico - sst_lag),
                    "rain_anomaly": float(rain_lag)
                })

    lag_df = pd.DataFrame(resultados)

    if lag_df.empty:
        raise RuntimeError(
            "Nenhum dado válido encontrado para o composite centrado no pico de chuva."
        )

    composite = (
        lag_df
        .groupby(["class", "lag"])
        .agg(
            sst_cooling_mean=("sst_cooling_anomaly_relative_to_peak", "mean"),
            sst_cooling_std=("sst_cooling_anomaly_relative_to_peak", "std"),
            rain_mean=("rain_anomaly", "mean"),
            rain_std=("rain_anomaly", "std"),
            n=("year", "nunique")
        )
        .reset_index()
    )

    composite["sst_cooling_se"] = composite["sst_cooling_std"] / np.sqrt(composite["n"])
    composite["rain_se"] = composite["rain_std"] / np.sqrt(composite["n"])

    # Padronização do resfriamento da anomalia de TSM relativo ao pico para
    # ficar no mesmo eixo da chuva normalizada, preservando zero como
    # referência física. Valor positivo = resfriamento.
    std_delta_lag = lag_df["sst_cooling_anomaly_relative_to_peak"].std(skipna=True, ddof=1)

    if np.isfinite(std_delta_lag) and std_delta_lag > 0:
        lag_df["sst_cooling_anomaly_relative_to_peak_normalized"] = (
            lag_df["sst_cooling_anomaly_relative_to_peak"] / std_delta_lag
        )
    else:
        lag_df["sst_cooling_anomaly_relative_to_peak_normalized"] = np.nan

    composite_norm = (
        lag_df
        .groupby(["class", "lag"])
        .agg(
            sst_cooling_norm_mean=("sst_cooling_anomaly_relative_to_peak_normalized", "mean"),
            sst_cooling_norm_std=("sst_cooling_anomaly_relative_to_peak_normalized", "std")
        )
        .reset_index()
    )

    composite = composite.merge(composite_norm, on=["class", "lag"], how="left")
    composite["sst_cooling_norm_se"] = composite["sst_cooling_norm_std"] / np.sqrt(composite["n"])

    # Fração de anos com resfriamento após o pico de chuva.
    # Aqui o resfriamento pós-pico é definido como SST(lag +1 ou +2) < SST(lag 0).
    cooling_summary = []

    for classe in grupos.keys():
        sub = lag_df[lag_df["class"] == classe]
        years = sorted(sub["year"].unique())

        for lag_pos in [1, 2]:
            count_valid = 0
            count_cooling = 0

            for ano in years:
                val = sub.loc[
                    (sub["year"] == ano) & (sub["lag"] == lag_pos),
                    "sst_cooling_anomaly_relative_to_peak"
                ]

                if len(val) == 0:
                    continue

                count_valid += 1

                if float(val.iloc[0]) > 0:
                    count_cooling += 1

            cooling_summary.append({
                "class": classe,
                "lag": lag_pos,
                "n_valid_years": count_valid,
                "n_cooling_years": count_cooling,
                "fraction_cooling": count_cooling / count_valid if count_valid > 0 else np.nan
            })

    cooling_summary = pd.DataFrame(cooling_summary)

    print("\nComposite centrado no pico de chuva:")
    print(composite)
    print("\nFração de anos com resfriamento após o pico de chuva:")
    print(cooling_summary)

    # ------------------------------------------------------------
    # FIGURA
    # ------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5), sharey=True)

    classes = ["Moderately warm", "Extremely warm"]
    titles = ["(a) Moderately warm", "(b) Extremely warm"]

    for ax, classe, title in zip(axes, classes, titles):
        comp = composite[composite["class"] == classe].sort_values("lag")
        x = comp["lag"].values.astype(float)

        ax.bar(
            x,
            comp["rain_mean"].values,
            width=0.55,
            alpha=0.60,
            label="Rainfall anomaly"
        )

        ax.errorbar(
            x,
            comp["rain_mean"].values,
            yerr=comp["rain_se"].values,
            fmt="none",
            linewidth=1.0,
            capsize=3,
            alpha=0.8
        )

        y = comp["sst_cooling_norm_mean"].values
        se = comp["sst_cooling_norm_se"].values

        ax.plot(
            x,
            y,
            marker="o",
            linewidth=2.5,
            label="SAWP SST cooling anomaly"
        )

        ax.fill_between(
            x,
            y - se,
            y + se,
            alpha=0.20
        )

        ax.axhline(0, color="black", linewidth=1.1)
        ax.axvline(0, color="gray", linestyle="--", linewidth=1.1)

        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Months relative to rainfall peak")
        ax.set_xticks(list(lags))
        ax.grid(axis="y", alpha=0.3)

        n_years = lag_df.loc[lag_df["class"] == classe, "year"].nunique()

        ax.text(
            0.03,
            0.95,
            f"n = {n_years}",
            transform=ax.transAxes,
            va="top",
            ha="left",
            fontsize=11,
            bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85)
        )

    axes[0].set_ylabel("Normalized anomaly (standard deviations)\nPositive SST line = cooling")
    axes[0].legend(frameon=False, loc="best")

    fig.suptitle(
        "Composite centered on maximum ENEB rainfall anomaly\n"
        "SAWP SST cooling anomaly relative to the rainfall-peak month\nPositive values indicate cooling",
        fontsize=15
    )

    plt.tight_layout()

    if salvar_figura:
        plt.savefig(nome_figura, dpi=300, bbox_inches="tight")
        print(f"Figura salva em: {nome_figura}")

    plt.show()

    return lag_df, composite, cooling_summary

# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Configurações iniciais
    # --------------------------------------------------------
    tendencias = "sim"  # 'sim' ou 'não'
    caminho_ = "./outputs/" if tendencias == "não" else "./outputs_trended/"

    DicionarioAreaEstudo = {
        "SAWP": [-34, -25, -12.5, -4],
        "NEB": [-50, -34, -15, 0]
    }

    # --------------------------------------------------------
    # Carregamento de dados de TSM
    # --------------------------------------------------------
    periodo_tempo_convertido, lon, lat = sawp.ConfigurarTempoArea(
        "./data/Era5_Dataset_sst.nc"
    )

    tsm_arquivo = "detrended_sst.npz" if tendencias == "não" else "trended_uvq_sst.npz"

    tsm = sawp.CarregarDados(
        caminho_ + tsm_arquivo,
        tendencias=tendencias,
        nome_var_nc="sst",
        substituir_nan=-999
    )

    print("Final da exportação dos dados de TSM do ERA5")

    # --------------------------------------------------------
    # Recorte SAWP
    # --------------------------------------------------------
    limites = DicionarioAreaEstudo["SAWP"]

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

    # Resultado deve ser DataFrame com coluna 'Anomalia'.
    # Caso venha como Series, a função auxiliar resolverá.
    serie_tsm = _padronizar_indice_mensal(resultado, nome_coluna="Anomalia")

    if "Anomalia" not in serie_tsm.columns:
        raise KeyError(
            "A saída de sawp.calcular_anomalia_tsm_mensal_todo_periodo "
            "não contém a coluna 'Anomalia'."
        )

    # --------------------------------------------------------
    # Chuva mensal regional ENEB
    # --------------------------------------------------------
    chuva_mensal, matriz_estacoes_qc, metadata, removidos, regime_test = (
        sawp.gerar_serie_chuva_mensal_acumulada()
    )

    chuva_norm = normalizar_anomalia_chuva_mensal(
        chuva_mensal,
        tendencias=tendencias,
        coluna="precipitation_mm",
        anos_clima=(1980, 2010)
    )

    # --------------------------------------------------------
    # Alinhamento chuva x TSM para diagnóstico de correlação
    # --------------------------------------------------------
    chuva_aux = _padronizar_indice_mensal(chuva_norm)
    tsm_aux = _padronizar_indice_mensal(serie_tsm)

    dados = pd.concat(
        [
            chuva_aux["anomalia_normalizada"].rename("chuva"),
            tsm_aux["Anomalia"].rename("tsm")
        ],
        axis=1
    ).dropna()

    print("\nPares válidos chuva x TSM:", len(dados))

    if len(dados) >= 2:
        r = np.corrcoef(dados["chuva"], dados["tsm"])[0, 1]
        print(f"Correlação mensal chuva x TSM = {r:.3f}")
    else:
        print("Sem pares suficientes para calcular correlação mensal.")

    # --------------------------------------------------------
    # Classificação dos eventos com TSM mensal detrendada
    # antes dos quantis/ranking
    # --------------------------------------------------------
    (
        media_amjj,
        anos_q10,
        anos_q25,
        anos_normal,
        anos_q75,
        anos_q90,
        quantis,
        serie_tsm_detrended,
        tendencias_mensais_tsm
    ) = calcular_percentis_amjj_ranking_detrended(
        serie_tsm,
        coluna="Anomalia",
        frac_extremo=0.10,
        frac_moderado=0.15,
        anos_clima=(1980, 2010)
    )

    # --------------------------------------------------------
    # ΔSST abril-julho + anomalia de chuva AMJJ
    # --------------------------------------------------------
    delta_sst, stats_delta_sst = analisar_delta_sst_abril_julho_com_chuva(
        serie_tsm=serie_tsm,
        chuva_norm=chuva_norm,
        anos_moderadamente_quentes=anos_q75,
        anos_extremamente_quentes=anos_q90,
        coluna_sst="Anomalia",
        salvar_figura=True,
        nome_figura="Fig_Sx_delta_SST_April_July_rainfall.png"
    )

    # --------------------------------------------------------
    # Composite centrado no mês de chuva máxima.
    # Esta versão mostra RESFRIAMENTO da anomalia de TSM relativo ao lag 0, não TSM absoluta.
    # --------------------------------------------------------
    lag_df, composite_lag, cooling_summary = analisar_composite_centrado_no_pico_chuva_delta_sst(
        serie_tsm=serie_tsm,
        chuva_norm=chuva_norm,
        anos_moderadamente_quentes=anos_q75,
        anos_extremamente_quentes=anos_q90,
        coluna_sst="Anomalia",
        coluna_chuva="anomalia_normalizada",
        meses_estacao=(4, 5, 6, 7),
        lags=(-2, -1, 0, 1, 2),
        salvar_figura=True,
        nome_figura="Fig_Sx_lag_peak_rainfall_SST_cooling.png"
    )

    # --------------------------------------------------------
    # Exportação dos resultados principais
    # --------------------------------------------------------
    delta_sst.to_csv(
        "Delta_SST_anomaly_April_July_with_AMJJ_rainfall.csv",
        index=False
    )

    stats_delta_sst.to_csv(
        "Delta_SST_anomaly_April_July_stats.csv",
        index=False
    )

    media_amjj.to_csv(
        "SAWP_AMJJ_event_classification_detrended.csv"
    )

    tendencias_mensais_tsm.to_csv(
        "SAWP_monthly_SST_trends_removed_before_quantiles.csv",
        index=False
    )

    lag_df.to_csv(
        "Lag_composite_centered_on_peak_rainfall_year_by_year.csv",
        index=False
    )

    composite_lag.to_csv(
        "Lag_composite_centered_on_peak_rainfall_mean.csv",
        index=False
    )

    cooling_summary.to_csv(
        "Lag_composite_post_peak_cooling_fraction.csv",
        index=False
    )

    print("\nArquivos exportados:")
    print("- Delta_SST_anomaly_April_July_with_AMJJ_rainfall.csv")
    print("- Delta_SST_anomaly_April_July_stats.csv")
    print("- SAWP_AMJJ_event_classification_detrended.csv")
    print("- SAWP_monthly_SST_trends_removed_before_quantiles.csv")
    print("- Lag_composite_centered_on_peak_rainfall_year_by_year.csv")
    print("- Lag_composite_centered_on_peak_rainfall_mean.csv")
    print("- Lag_composite_post_peak_cooling_fraction.csv")
