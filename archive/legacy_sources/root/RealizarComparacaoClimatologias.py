#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 22:27:55 2025

@author: simopec
"""
from timeit import default_timer as timer

timeini = timer()

import netCDF4 as nc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# Importando arquivo de TSM com 4 dimensões (time, lat, lon)
arquivo_era5_tsm = nc.Dataset('./data/Era5_Dataset_sst.nc')

# Extraindo dimensão de tempo
vetor_tempo_tsm = arquivo_era5_tsm.variables['time']
tamanho_vetor_tempo_tsm = vetor_tempo_tsm.size

# Corrigindo vetor tempo para datetime (tempo do modelo + datetime inicial em 1900)
periodo_inicial = datetime(1900, 1, 1)

# Convertendo o tempo do modelo para datetime
def converter_tempo(vetor_tempo, periodo_inicial):
    return [periodo_inicial + timedelta(hours=int(hh)) for hh in vetor_tempo]

periodo_tempo_convertido = converter_tempo(vetor_tempo_tsm[:], periodo_inicial)

# Extraindo coordenadas
lat_ = arquivo_era5_tsm.variables['latitude']
lon_ = arquivo_era5_tsm.variables['longitude']

# Criando grade de coordenadas
lon, lat = np.meshgrid(lon_[:], lat_[:])

# Carregando dados detrendidos e TSM atual
print('Removendo tendências da TSM...')
loaded_data = np.load("./outputs/detrended_sst.npz")
tsm_st = loaded_data['data']
tsm_ct = arquivo_era5_tsm.variables['sst'][:]

print('Final da exportação dos dados de TSM do ERA5')

# Definindo limites da área de estudo (SAWP e NEB)
DicionarioAreaEstudo = {
    'SAWP': [-34, -25, -12.5, -4],  # Limites [lonW, lonE, latS, latN]
    'NEB': [-50, -34, -15, 0]
}

def extrair_indices_area(lon_, lat_, limites):
    """Retorna índices dos limites definidos no dicionário."""
    lloW = np.where(lon_[:].data == limites[0])[0][0]
    lloE = np.where(lon_[:].data == limites[1])[0][0]
    llaS = np.where(lat_[:].data == limites[2])[0][0]
    llaN = np.where(lat_[:].data == limites[3])[0][0]
    return lloW, lloE, llaS, llaN

# Extraindo índices para a área de estudo SAWP
indices_sawp = extrair_indices_area(lon_, lat_, DicionarioAreaEstudo['SAWP'])
lloW, lloE, llaS, llaN = indices_sawp

# Extração da série temporal de TSM na área de estudo
def extrair_serie_temporal(tsm_data, llaN, llaS, lloW, lloE):
    """Calcula a série temporal média em uma área definida."""
    area_tsm = np.nanmean(tsm_data[:, llaN:llaS, :], axis=1)
    serie_tsm = np.nanmean(area_tsm[:, lloW:lloE], axis=1)
    return serie_tsm

# Série temporal para TSM detrendida
tsm_st_serie = extrair_serie_temporal(tsm_st, llaN, llaS, lloW, lloE)
serie_tsm_st = pd.DataFrame(tsm_st_serie - 273.16, columns=['TSM'])
serie_tsm_st.index = periodo_tempo_convertido
serie_tsm_st['TSM'] = serie_tsm_st['TSM'].rolling(window=3,center=False).mean()

# Série temporal para TSM corrente
tsm_ct_serie = extrair_serie_temporal(tsm_ct, llaN, llaS, lloW, lloE)
serie_tsm_ct = pd.DataFrame(tsm_ct_serie - 273.16, columns=['TSM'])
serie_tsm_ct.index = periodo_tempo_convertido
serie_tsm_ct['TSM'] = serie_tsm_ct['TSM'].rolling(window=3,center=False).mean()

# Calculando anomalia mensal
def calcular_anomalia_mensal(serie):
    """Calcula a anomalia mensal para uma série temporal considerando apenas 1980-2010 para a climatologia."""
    serie['year'] = serie.index.year
    serie['month'] = serie.index.month
    climatology_period = serie[(serie['year'] >= 1980) & (serie['year'] <= 2010)]
    climatology = climatology_period.groupby('month')['TSM'].mean()
    serie['climatology'] = serie['month'].map(climatology)
    serie['anomaly'] = serie['TSM'] - serie['climatology']
    return serie

serie_tsm_st = calcular_anomalia_mensal(serie_tsm_st)
serie_tsm_ct = calcular_anomalia_mensal(serie_tsm_ct)

# Índices de março (mês 2, pois o índice começa em 0)
indices_marco = np.arange(243, 605, 12)  # Seleciona todos os índices de março (mês 2) ao longo dos anos

# Extraindo dados apenas para os meses de março
data_st = tsm_st[indices_marco, :, :]
data_ct = tsm_ct[indices_marco, :, :]


# Configurando os subplots
fig = plt.figure(figsize=(25, 15))

# Subplot a) Mapa com tendência
ax1 = fig.add_subplot(2, 2, 1, projection=ccrs.PlateCarree())
ax1.set_extent([-60, -10, -30, 15], crs=ccrs.PlateCarree())
ax1.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax1.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
ax1.add_feature(cfeature.STATES, linewidth=0.5)
ax1.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgrey', alpha=0.3)
# Adicionando os pontos de medição
# Adicionando os pontos de medição
#ax1.scatter(
#    longitude_rg, latitude_rg,  # Coordenadas dos pontos
#    color='black',  # Cor dos pontos
#    s=.2,  # Tamanho dos pontos
#    transform=ccrs.PlateCarree(),  # Transformação para coordenadas geográficas
#    label='Rain gauges'
#)

# Adicionando um quadrado para SAWP
sawp_coords = [-34, -25, -12.5, -4]  # [lonW, lonE, latS, latN]
ax1.plot([sawp_coords[0], sawp_coords[1], sawp_coords[1], sawp_coords[0], sawp_coords[0]],
         [sawp_coords[2], sawp_coords[2], sawp_coords[3], sawp_coords[3], sawp_coords[2]],
         transform=ccrs.PlateCarree(), color='k', linewidth=1.5, label='SAWP')
levels = np.arange(-1, 1.25, 0.25)
contour1 = ax1.contourf(lon, lat, (tsm_ct[734, :, :] - np.mean(data_ct, axis=0)), levels=levels, 
                        cmap="coolwarm", extend='both', transform=ccrs.PlateCarree())
ax1.set_title("a) March 2020 with trends", fontsize=28, loc = 'left')
plt.colorbar(contour1, ax=ax1, orientation='vertical', pad=0.05, aspect=30).set_label("SST (°C)", fontsize=18)
ax1.legend(fontsize=18)

# Adicionando latitudes e longitudes
ax1.set_xticks(np.arange(-60, -5, 10), crs=ccrs.PlateCarree())
ax1.set_yticks(np.arange(-30, 20, 10), crs=ccrs.PlateCarree())
ax1.tick_params(axis='both', labelsize=20)  # Tamanho da fonte dos rótulos
# Adicionando títulos e legendas
ax1.set_xlabel("Longitude", fontsize=24)
ax1.set_ylabel("Latitude", fontsize=24)

# Subplot b) Série temporal com tendência
from scipy.stats import linregress

# Filtrar dados de 1990 a 2023 para calcular a linha de tendência
mask_tendencia = (serie_tsm_ct.index.year >= 1980) & (serie_tsm_ct.index.year <= 2023)
dados_tendencia = serie_tsm_ct[mask_tendencia]

# Extraindo valores para regressão
x = np.array([(date.year + (date.timetuple().tm_yday / 365.25)) for date in dados_tendencia.index])
y = dados_tendencia['anomaly'].values

# Regressão linear
slope, intercept, r_value, p_value, std_err = linregress(x, y)

# Criar linha de tendência
x_tendencia = np.linspace(x.min(), x.max(), 528)  # Valores para linha de tendência
y_tendencia = intercept + slope * x_tendencia

# Teste de significância da tendência
print("Teste de Significância da Tendência")
print(f"Inclinação (slope): {slope:.6f}")
print(f"Intercepto: {intercept:.6f}")
print(f"Coeficiente de Correlação (r): {r_value:.6f}")
print(f"Erro padrão da inclinação (std_err): {std_err:.6f}")
print(f"Valor-p (p-value): {p_value:.6f}")

# Verificar se a tendência é significativa
if p_value <= 0.01:
    print("A tendência é estatisticamente significativa ao nível de 5%.")
else:
    print("A tendência não é estatisticamente significativa ao nível de 5%.")


ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(serie_tsm_ct.index, serie_tsm_ct['anomaly'], color='k',label = 'With trend')

# Atualizar a Figura (b) para incluir a linha de tendência
ax2.plot(dados_tendencia.index, y_tendencia, color='blue', linestyle='--', linewidth=4, label='Trend (1980-2023)')
ax2.legend(loc='upper left', fontsize=18)  # Atualizar a legenda para incluir a linha de tendência

y_90 = np.percentile(serie_tsm_st['anomaly'].dropna(), 90)
ax2.axhline(y=y_90, color='lightgrey', linestyle='--', label='Percentile 90')
ax2.axhline(y=0, color='grey', alpha=0.4)
above_90 = serie_tsm_ct['anomaly'] > y_90
ax2.fill_between(serie_tsm_ct.index, y_90, serie_tsm_ct['anomaly'], where=above_90, color='red', alpha=0.8)
ax2.set_title("b)",loc='left', fontsize=28)



ax2.set_xlabel("Period")
ax2.set_ylim([-1.25,1.25])
ax2.yaxis.set_label_position("right")
ax2.yaxis.tick_right()
ax2.set_ylabel("SST Anomaly (°C)")
ax2.legend(loc='upper left',fontsize=18)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(True)
ax2.spines['left'].set_visible(False)


# Subplot c) Mapa sem tendência
ax3 = fig.add_subplot(2, 2, 3, projection=ccrs.PlateCarree())
ax3.set_extent([-60, -10, -30, 15], crs=ccrs.PlateCarree())
ax3.add_feature(cfeature.COASTLINE, linewidth=0.5)
ax3.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
ax3.add_feature(cfeature.STATES, linewidth=0.5)
ax3.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgrey', alpha=0.3)

# Adicionando um quadrado para SAWP
sawp_coords = [-34, -25, -12.5, -4]  # [lonW, lonE, latS, latN]
ax3.plot([sawp_coords[0], sawp_coords[1], sawp_coords[1], sawp_coords[0], sawp_coords[0]],
         [sawp_coords[2], sawp_coords[2], sawp_coords[3], sawp_coords[3], sawp_coords[2]],
         transform=ccrs.PlateCarree(), color='k', linewidth=1.5, label='SAWP')
levels = np.arange(-1, 1.25, 0.25)
contour3 = ax3.contourf(lon, lat, (tsm_st[734, :, :] - np.mean(data_st, axis=0)), levels=levels, 
                        cmap="coolwarm", extend='both', transform=ccrs.PlateCarree())
ax3.set_title("c) March 2020 without trends", fontsize=28, loc = 'left')
plt.colorbar(contour3, ax=ax3, orientation='vertical', pad=0.05, aspect=30).set_label("SST (°C)", fontsize=18)
ax3.legend(fontsize=18)

# Adicionando latitudes e longitudes
ax3.set_xticks(np.arange(-60, -5, 10), crs=ccrs.PlateCarree())
ax3.set_yticks(np.arange(-30, 20, 10), crs=ccrs.PlateCarree())
ax3.tick_params(axis='both', labelsize=20)  # Tamanho da fonte dos rótulos
# Adicionando títulos e legendas
ax3.set_xlabel("Longitude", fontsize=24)
ax3.set_ylabel("Latitude", fontsize=24)

# Subplot d) Série temporal sem tendência
ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(serie_tsm_st.index, serie_tsm_st['anomaly'], color='k', label = 'Without trend')
y_90 = np.percentile(serie_tsm_st['anomaly'].dropna(), 90)
ax4.axhline(y=y_90, color='lightgrey', linestyle='--', label='Percentile 90')
ax4.axhline(y=0, color='grey', alpha=0.4)
above_90 = serie_tsm_st['anomaly'] > y_90
ax4.fill_between(serie_tsm_st.index, y_90, serie_tsm_st['anomaly'], where=above_90, color='red', alpha=0.8)
ax4.set_title("d)" ,loc='left',fontsize=28)
ax4.set_xlabel("Period")
ax4.set_ylim([-1.25,1.25])
ax4.yaxis.set_label_position("right")
ax4.yaxis.tick_right()
ax4.set_ylabel("SST Anomaly (°C)")
ax4.legend(loc='upper left',fontsize=18)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(True)
ax4.spines['left'].set_visible(False)


plt.rcParams.update({'font.size': 24})
plt.tight_layout()
plt.savefig('Subplots_TSM_Com_Sem_tendencia', 
            dpi=360, 
            format=None, 
            metadata=None,
            pad_inches=0.1,
            facecolor='auto', 
            edgecolor='auto',
            backend=None,  
            bbox_inches="tight")
plt.show()