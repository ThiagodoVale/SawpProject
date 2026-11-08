#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 22:38:42 2025

@author: simopec
"""

import numpy as np
import pandas as pd

def ProcessarTsmv2(tsm, periodo_tempo_convertido, lloW, lloE, llaS, llaN, anos_clima=(1980,2010)):
    """
    Processa TSM para área selecionada e calcula:
        - Série diária
        - Série período chuvoso (AMJJ)
        - Climatologia AMJJ
        - Anomalia do período chuvoso
        - Médias anuais
        - Quantis com blocos de 5 índices sequenciais

    tsm : np.array (time, lat, lon)
    periodo_tempo_convertido : lista de datetime
    lloW, lloE, llaS, llaN : índices espaciais
    anos_clima : (ano_i, ano_f)
    """

    # ======================================================
    # 1) MÉDIA ESPACIAL NA ÁREA SELECIONADA
    # ======================================================
    area_tsm = np.nanmean(tsm[:, llaN:llaS, :], axis=1)
    serie_tsm_vals = np.nanmean(area_tsm[:, lloW:lloE], axis=1)

    serie_tsm = pd.DataFrame({
        'TSM': serie_tsm_vals - 273.16
    }, index=periodo_tempo_convertido)

    serie_tsm['Mes'] = serie_tsm.index.month
    serie_tsm['Ano'] = serie_tsm.index.year
    serie_tsm['Indices'] = np.arange(len(serie_tsm))

    # ======================================================
    # 2) CLIMATOLOGIA APENAS PARA O PERÍODO CHUVOSO (ABR–JUL)
    # ======================================================
    mascara_clima = (
        (serie_tsm['Ano'] >= anos_clima[0]) &
        (serie_tsm['Ano'] <= anos_clima[1]) &
        (serie_tsm['Mes'].between(4, 7))
    )

    serie_tsm_clima = serie_tsm[mascara_clima].copy()

    # climatologia AMJJ por mês
    climatologia_amjj = serie_tsm_clima.groupby('Mes')['TSM'].mean()
    print(climatologia_amjj)
    
    # ======================================================
    # 3) CALCULAR ANOMALIA SOMENTE PARA ABR–JUL
    # ======================================================
    serie_tsm['Anomalia'] = serie_tsm.apply(
        lambda row: row['TSM'] - climatologia_amjj[row['Mes']]
        if row['Mes'] in climatologia_amjj.index else np.nan,
        axis=1
    )

    # ======================================================
    # 4) EXTRACT PERÍODO CHUVOSO (AMJJ)
    # ======================================================
    serie_periodo_chuvoso = serie_tsm[serie_tsm['Mes'].between(4, 7)].copy()

    # obter o primeiro índice AMJJ por ano
    primeiros_indices = serie_periodo_chuvoso.groupby('Ano')['Indices'].first()
    print(primeiros_indices)
    # criar sequências de 5 índices: ex: 3→[3..7]
    sequencias_indices = {ano: np.arange(idx, idx+4) for ano, idx in primeiros_indices.items()}

    # ======================================================
    # 5) MÉDIAS E MÁXIMOS ANUAIS (APENAS AMJJ)
    # ======================================================
    media_tsm_anual = serie_periodo_chuvoso.groupby('Ano').mean()
    maximo_tsm_anual = serie_periodo_chuvoso.groupby('Ano').max()

    # ======================================================
    # 6) QUANTIS COM SEQUÊNCIAS DE 5 ÍNDICES
    # ======================================================
    clima_anomalia = media_tsm_anual['Anomalia']
    
    mascara_clima_anual = (media_tsm_anual.index >= anos_clima[0]) & (media_tsm_anual.index <= anos_clima[1])
    anomalia_base = media_tsm_anual.loc[mascara_clima_anual, 'Anomalia']

    anos_q10 = clima_anomalia[clima_anomalia <= anomalia_base.quantile(0.1)].index
    
    anos_q25 = clima_anomalia[(clima_anomalia > anomalia_base.quantile(0.1)) &
                              (clima_anomalia <= anomalia_base.quantile(0.25))].index
    
    anos_q75 = clima_anomalia[(clima_anomalia >= anomalia_base.quantile(0.75)) &
                              (clima_anomalia < anomalia_base.quantile(0.9))].index
    
    anos_q90 = clima_anomalia[clima_anomalia >= anomalia_base.quantile(0.9)].index
    
    anos_normal = clima_anomalia[(clima_anomalia > anomalia_base.quantile(0.25)) &
                                 (clima_anomalia < anomalia_base.quantile(0.75))].index
    
    def juntar_blocos(lista_anos):
        return np.concatenate([sequencias_indices[ano] for ano in lista_anos]) if len(lista_anos) > 0 else np.array([])

    percentis_tsm = {
        '10': juntar_blocos(anos_q10),
        '25': juntar_blocos(anos_q25),
        '75': juntar_blocos(anos_q75),
        '90': juntar_blocos(anos_q90),
        'normal': juntar_blocos(anos_normal)
    }

    # ======================================================
    # 7) RETORNO
    # ======================================================
    return {
        'serie_tsm': serie_tsm,
        'serie_periodo_chuvoso': serie_periodo_chuvoso,
        'media_tsm_anual': media_tsm_anual,
        'maximo_tsm_anual': maximo_tsm_anual,
        'percentis_tsm': percentis_tsm
    }

def calcular_anomalia_tsm_mensal_todo_periodo(
    tsm,
    periodo_tempo_convertido,
    lloW,
    lloE,
    llaS,
    llaN,
    anos_clima=(1980, 2010)
):
    """
    Calcula a série média espacial de TSM e sua anomalia mensal
    para todos os meses do ano, sem restringir ao período chuvoso.
    """

    # ======================================================
    # 1) RECORTE E MÉDIA ESPACIAL DA ÁREA
    # ======================================================
    area_tsm = tsm[:, llaN:llaS, lloW:lloE]

    serie_tsm_vals = np.nanmean(area_tsm, axis=(1, 2))

    serie_tsm = pd.DataFrame({
        "TSM": serie_tsm_vals - 273.16
    }, index=pd.to_datetime(periodo_tempo_convertido))

    serie_tsm["Mes"] = serie_tsm.index.month
    serie_tsm["Ano"] = serie_tsm.index.year
    serie_tsm["Indices"] = np.arange(len(serie_tsm))

    # ======================================================
    # 2) CLIMATOLOGIA MENSAL PARA TODOS OS MESES
    # ======================================================
    mascara_clima = (
        (serie_tsm["Ano"] >= anos_clima[0]) &
        (serie_tsm["Ano"] <= anos_clima[1])
    )

    climatologia_mensal = (
        serie_tsm.loc[mascara_clima]
        .groupby("Mes")["TSM"]
        .mean()
    )

    # ======================================================
    # 3) ANOMALIA MENSAL PARA TODOS OS MESES
    # ======================================================
    serie_tsm["Anomalia"] = serie_tsm.apply(
        lambda row: row["TSM"] - climatologia_mensal.loc[row["Mes"]],
        axis=1
    )

    return serie_tsm, climatologia_mensal