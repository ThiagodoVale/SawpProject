# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 19:13:50 2026

@author: silva
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from timeit import default_timer as timer
from datetime import datetime, timedelta

import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress


timeini = timer()

# ============================================================
# Leitura dos dados
# ============================================================

arquivo_era5_tsm = nc.Dataset('./data/Era5_Dataset_sst.nc')

vetor_tempo_tsm = arquivo_era5_tsm.variables['time']
periodo_inicial = datetime(1900, 1, 1)


def converter_tempo(vetor_tempo, periodo_inicial):
    return [periodo_inicial + timedelta(hours=int(hh)) for hh in vetor_tempo]


periodo_tempo_convertido = converter_tempo(vetor_tempo_tsm[:], periodo_inicial)

lat_ = arquivo_era5_tsm.variables['latitude']
lon_ = arquivo_era5_tsm.variables['longitude']

print('Carregando TSM com e sem tendência...')

loaded_data = np.load("./outputs/detrended_sst.npz")
tsm_st = loaded_data['data']                      # sem tendência
tsm_ct = arquivo_era5_tsm.variables['sst'][:]     # com tendência

print('Dados carregados.')


# ============================================================
# Área de estudo
# ============================================================

DicionarioAreaEstudo = {
    'SAWP': [-34, -25, -12.5, -4],  # [lonW, lonE, latS, latN]
}


def extrair_indices_area(lon_, lat_, limites):
    lloW = np.where(lon_[:].data == limites[0])[0][0]
    lloE = np.where(lon_[:].data == limites[1])[0][0]
    llaS = np.where(lat_[:].data == limites[2])[0][0]
    llaN = np.where(lat_[:].data == limites[3])[0][0]

    return lloW, lloE, llaS, llaN


lloW, lloE, llaS, llaN = extrair_indices_area(
    lon_,
    lat_,
    DicionarioAreaEstudo['SAWP']
)


# ============================================================
# Extração da série temporal média na SAWP
# ============================================================

def extrair_serie_temporal(tsm_data, llaN, llaS, lloW, lloE):
    area_tsm = np.nanmean(tsm_data[:, llaN:llaS, lloW:lloE], axis=(1, 2))
    return area_tsm


tsm_ct_serie = extrair_serie_temporal(tsm_ct, llaN, llaS, lloW, lloE)
tsm_st_serie = extrair_serie_temporal(tsm_st, llaN, llaS, lloW, lloE)


serie_tsm_ct = pd.DataFrame(tsm_ct_serie - 273.16, columns=['TSM'])
serie_tsm_ct.index = periodo_tempo_convertido
serie_tsm_ct['TSM'] = serie_tsm_ct['TSM'].rolling(window=3, center=False).mean()

serie_tsm_st = pd.DataFrame(tsm_st_serie - 273.16, columns=['TSM'])
serie_tsm_st.index = periodo_tempo_convertido
serie_tsm_st['TSM'] = serie_tsm_st['TSM'].rolling(window=3, center=False).mean()


# ============================================================
# Cálculo da anomalia mensal
# ============================================================

def calcular_anomalia_mensal(serie, periodo_climatologico=(1980, 2010)):
    serie = serie.copy()

    serie['year'] = serie.index.year
    serie['month'] = serie.index.month

    clim = serie[
        (serie['year'] >= periodo_climatologico[0]) &
        (serie['year'] <= periodo_climatologico[1])
    ]

    climatology = clim.groupby('month')['TSM'].mean()

    serie['climatology'] = serie['month'].map(climatology)
    serie['anomaly'] = serie['TSM'] - serie['climatology']

    return serie


serie_tsm_ct = calcular_anomalia_mensal(serie_tsm_ct)
serie_tsm_st = calcular_anomalia_mensal(serie_tsm_st)


# ============================================================
# Tendência da série com tendência
# ============================================================

mask_tendencia = (
    (serie_tsm_ct.index.year >= 1980) &
    (serie_tsm_ct.index.year <= 2023)
)

dados_tendencia = serie_tsm_ct.loc[mask_tendencia].dropna(subset=['anomaly'])

x = np.array([
    date.year + (date.timetuple().tm_yday / 365.25)
    for date in dados_tendencia.index
])

y = dados_tendencia['anomaly'].values

slope, intercept, r_value, p_value, std_err = linregress(x, y)

y_tendencia = intercept + slope * x

print("Teste de Significância da Tendência")
print(f"Inclinação: {slope:.6f} °C/ano")
print(f"Intercepto: {intercept:.6f}")
print(f"r: {r_value:.6f}")
print(f"Erro padrão: {std_err:.6f}")
print(f"p-value: {p_value:.6f}")

if p_value <= 0.05:
    print("A tendência é estatisticamente significativa ao nível de 5%.")
else:
    print("A tendência não é estatisticamente significativa ao nível de 5%.")


# ============================================================
# Percentil 90
# ============================================================

p90_st = np.percentile(serie_tsm_st['anomaly'].dropna(), 90)


# ============================================================
# Figura apenas com séries temporais
# ============================================================

plt.rcParams.update({'font.size': 20})

fig, axs = plt.subplots(
    2, 1,
    figsize=(18, 10),
    sharex=True
)

# ------------------------------------------------------------
# Subplot a) Com tendência
# ------------------------------------------------------------

ax = axs[0]

ax.plot(
    serie_tsm_ct.index,
    serie_tsm_ct['anomaly'],
    color='k',
    linewidth=1.8,
    label='SST'
)

ax.plot(
    dados_tendencia.index,
    y_tendencia,
    color='blue',
    linestyle='--',
    linewidth=3,
    label='Trend 1980–2023'
)

ax.axhline(
    y=p90_st,
    color='lightgrey',
    linestyle='--',
    linewidth=2,
    label='Percentile 90'
)

ax.axhline(
    y=0,
    color='grey',
    alpha=0.5,
    linewidth=1
)

above_90_ct = serie_tsm_ct['anomaly'] > p90_st

ax.fill_between(
    serie_tsm_ct.index,
    p90_st,
    serie_tsm_ct['anomaly'],
    where=above_90_ct,
    color='red',
    alpha=0.8
)

ax.set_title('a)', loc='left', fontsize=24)
ax.set_ylabel('SST anomaly (°C)', fontsize=22)
ax.set_ylim([-1.25, 1.25])
ax.legend(loc='upper left', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


# ------------------------------------------------------------
# Subplot b) Sem tendência
# ------------------------------------------------------------

ax = axs[1]

ax.plot(
    serie_tsm_st.index,
    serie_tsm_st['anomaly'],
    color='k',
    linewidth=1.8,
    label='Detrended SST'
)

ax.axhline(
    y=p90_st,
    color='lightgrey',
    linestyle='--',
    linewidth=2,
    label='Percentile 90'
)

ax.axhline(
    y=0,
    color='grey',
    alpha=0.5,
    linewidth=1
)

above_90_st = serie_tsm_st['anomaly'] > p90_st

ax.fill_between(
    serie_tsm_st.index,
    p90_st,
    serie_tsm_st['anomaly'],
    where=above_90_st,
    color='red',
    alpha=0.8
)

ax.set_title('b)', loc='left', fontsize=24)
ax.set_xlabel('Period', fontsize=22)
ax.set_ylabel('SST anomaly (°C)', fontsize=22)
ax.set_ylim([-1.25, 1.25])
ax.legend(loc='upper left', fontsize=16)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)


plt.tight_layout()

plt.savefig(
    'Series_TSM_Com_Sem_tendencia.png',
    dpi=360,
    bbox_inches='tight',
    pad_inches=0.1
)

plt.show()

timefim = timer()
print(f'Tempo total: {(timefim - timeini):.2f} segundos')