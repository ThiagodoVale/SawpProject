import numpy as np
import pandas as pd

def ProcessarTsm(tsm, periodo_tempo_convertido, lloW, lloE, llaS, llaN, anos_clima=(1980,2010)):
    """
    Processa a TSM para uma área definida pelos índices, calculando anomalias, médias anuais e percentis.
    
    Parâmetros:
        tsm : np.ndarray
            Array 3D (time, lat, lon) da TSM/SST
        periodo_tempo_convertido : list[datetime]
            Lista de datetime correspondente ao tempo
        lloW, lloE, llaS, llaN : int
            Índices de longitude oeste/este e latitude sul/norte
        anos_clima : tuple
            Intervalo de anos para climatologia (default: 1980-2010)
            
    Retorna:
        dict com chaves:
            'serie_tsm', 'serie_periodo_chuvoso', 'media_tsm_anual',
            'maximo_tsm_anual', 'percentis_tsm'
    """
    # Extrair média espacial da área definida pelos índices
    area_tsm = np.nanmean(tsm[:, llaN:llaS, :], axis=1)
    serie_tsm_vals = np.nanmean(area_tsm[:, lloW:lloE], axis=1)
    
    # Criar DataFrame
    serie_tsm = pd.DataFrame(serie_tsm_vals - 273.16, columns=['TSM'])
    serie_tsm.index = periodo_tempo_convertido
    serie_tsm['Mes'] = serie_tsm.index.month
    serie_tsm['Indices'] = np.arange(serie_tsm.shape[0])
    
    # Climatologia e anomalia
    serie_tsm_clima = serie_tsm[(serie_tsm.index.year >= anos_clima[0]) & (serie_tsm.index.year <= anos_clima[1])]
    climatologia_tsm = serie_tsm_clima.groupby('Mes')['TSM'].mean()
    serie_tsm['Anomalia'] = serie_tsm['TSM'] - serie_tsm['Mes'].map(climatologia_tsm)
    
    # Seleciona período chuvoso (abril a julho)
    serie_periodo_chuvoso = serie_tsm[(serie_tsm['Mes'] >= 4) & (serie_tsm['Mes'] <= 7)].copy()
    serie_periodo_chuvoso['Indices'] = serie_periodo_chuvoso.groupby(serie_periodo_chuvoso.index.year)['Indices'].transform('first')
    
    # Estatísticas anuais
    media_tsm_anual = serie_periodo_chuvoso.groupby(serie_periodo_chuvoso.index.year).mean()
    maximo_tsm_anual = serie_periodo_chuvoso.groupby(serie_periodo_chuvoso.index.year).max()
    
    # Percentis
    clima_anomalia = media_tsm_anual['Anomalia']
    percentis_tsm = {
        '10': media_tsm_anual['Indices'][clima_anomalia <= clima_anomalia.quantile(0.1)].to_numpy(),
        '25': media_tsm_anual['Indices'][(clima_anomalia > clima_anomalia.quantile(0.1)) & (clima_anomalia <= clima_anomalia.quantile(0.25))].to_numpy(),
        '75': media_tsm_anual['Indices'][(clima_anomalia >= clima_anomalia.quantile(0.75)) & (clima_anomalia <= clima_anomalia.quantile(0.9))].to_numpy(),
        '90': media_tsm_anual['Indices'][clima_anomalia >= clima_anomalia.quantile(0.9)].to_numpy(),
        'normal': media_tsm_anual['Indices'][(clima_anomalia >= clima_anomalia.quantile(0.25)) & (clima_anomalia <= clima_anomalia.quantile(0.75))].to_numpy()
    }
    
    return {
        'serie_tsm': serie_tsm,
        'serie_periodo_chuvoso': serie_periodo_chuvoso,
        'media_tsm_anual': media_tsm_anual,
        'maximo_tsm_anual': maximo_tsm_anual,
        'percentis_tsm': percentis_tsm
    }
