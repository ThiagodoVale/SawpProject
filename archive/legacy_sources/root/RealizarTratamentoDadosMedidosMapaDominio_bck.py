#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 21:39:27 2025

@author: simopec
"""
#script para ler e agrupar dados de estacoes.
# Junior 16/Jun/2022

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import netCDF4 as nc
from datetime import datetime
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle
# ============================================================
# FIGURA DO DOMÍNIO, SAWP E ESTAÇÕES PLUVIOMÉTRICAS
# ============================================================


from pathlib import Path

from glob import glob

def read_monitor_file(filename):
    """

    :param str filename:
    :return: a dataframe for the header and the data separately
    :rtype: pd.DataFrame, pd.DataFrame
    """
    try:
        df_header = pd.read_csv(filename, nrows=6, header=None, sep=':').set_index([0])
        
        # Lendo os dados sem a opção parse_dates
        df_data = pd.read_csv(filename, sep='\s+', skiprows=6, header=None)
        
        # Combinando as colunas de data após a leitura dos dados
        df_data['datetime'] = pd.to_datetime(df_data[[0, 1, 2]].astype(str).agg('-'.join, axis=1))
        
        # Definindo o índice como a coluna datetime
        df_data.set_index('datetime', inplace=True)
        
        # Removendo as colunas de data originais
        df_data.drop(columns=[0, 1, 2], inplace=True)
    
    except (IOError, IndexError, pd.errors.ParserError) as e:
        print(e)
        return pd.DataFrame([]), pd.DataFrame([])

    df_header.columns = ['value']
    code = str(int(df_header.loc["#code"].value))
    df_data.columns = [code]
    df_data[code] = df_data[code].astype(float)

    return df_header, df_data


states = ["AL", "PB", "PE", "RN", "CE","BA","SE","PI","MA" ]

code = []
lon = []
lat = []

pr_data = []

d1 = datetime(1959,1,1)
d2 = datetime(2023,12,31)

date_index=pd.date_range(d1,d2,freq='ME')
date_plot=pd.date_range(d1,d2,freq='YE')

# ------------------------------------------------------------
# Caminhos
# ------------------------------------------------------------

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
DOMAIN_DIR = DATA_DIR / "dominio"
OUT_DIR = DATA_DIR / "processed"
FIG_DIR = OUT_DIR / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)

SST_FILE = DOMAIN_DIR / "sst.nc"
TOPO_FILE = DOMAIN_DIR / "srtm30plus_v11_land.nc"

FIG_OUT = FIG_DIR / "Mapa_dominio_SAWP_ENEB_estacoes.png"


# ------------------------------------------------------------
# Domínios
# ------------------------------------------------------------

# Domínio pluviométrico ENEB
ENEB_LON_MIN = -38.0
ENEB_LON_MAX = -34.5
ENEB_LAT_MIN = -15.0
ENEB_LAT_MAX = 0.0

# Domínio SAWP
SAWP_LON_MIN = -34.0
SAWP_LON_MAX = -25.0
SAWP_LAT_MIN = -12.5
SAWP_LAT_MAX = -4.0

# Extensão do mapa
MAP_LON_MIN = -40.0
MAP_LON_MAX = -20.0
MAP_LAT_MIN = -15.0
MAP_LAT_MAX = 0.0

# Campo de TSM usado para ilustração
SST_TIME_INDEX = 3
SST_CONTOUR = 28.5

for ste in states:
	print(ste)
	files = np.sort(glob(f'./data/monitor/{ste}/*'))
    
	print(len(files))
	c=0
	for f1 in files:
		
		print(f1.split("/")[-1])
		df_h, df_d = read_monitor_file(f1)
		
		x=float(df_h.loc["#x"].value)
		y=float(df_h.loc["#y"].value)

		if ( x > -38. and x < -34.5 ):
			if (y > -15 and y < 0):
				cod = df_h.loc["#code"].value
				lon.append(x)
				lat.append(y)
				code.append(cod)
				pr_data.append(df_d)
				c = c+1

	print(c)

df1 = pr_data[0]
df1 = df1.reindex(date_index)
df = pd.concat(pr_data,axis=1).reindex(df1.index)

print(len(pr_data))

# Substituir valores acima de 999 mm por NaN
df = df.where(df < 500, np.nan)
df = df.where(df > 0, np.nan)
# Transformar em NaN todas as colunas onde o valor máximo da série seja menor que 20 mm
cols_to_nan_max = df.max() < 20
df.loc[:, cols_to_nan_max] = np.nan

# Transformar em NaN todas as colunas onde o valor mínimo seja maior que 200 mm
cols_to_nan_min = df.min() > 200
df.loc[:, cols_to_nan_min] = np.nan

dfmean = df.mean(axis=1)
plt.plot(dfmean.mean())

dfmmean = dfmean.groupby(pd.PeriodIndex(dfmean.index,freq='Y')).max().reset_index()
mm=dfmmean[0].to_numpy()
dfmmean['Anomalia']=(mm - np.mean(mm))/(np.mean(mm))

plt.plot(dfmmean['Anomalia'])

dfmean.to_csv("./data/EstacoesDados.txt",header=None,na_rep=-999.,sep=" ")

pontos_medida = np.transpose(np.asarray([lat,lon]))

ds1 = nc.Dataset('./data/dominio/sst.nc')
lat__ = ds1['lat']
lon__ = ds1['lon']
sst = ds1['sst']

# Gerar um array de longitudes de 0 a 360 (como no arquivo original)
longitudes = np.array(lon__)  # Aqui você já tem as longitudes originais do arquivo

# Ajustar as longitudes para o intervalo -180 a 180
adjusted_longitudes = np.where(longitudes > 180, longitudes - 360, longitudes)

# Gerar a grade de longitudes ajustadas (-180 a 180)
long, lati = np.meshgrid(adjusted_longitudes, lat__)

# Reorganizar a matriz para corresponder ao novo índice de longitudes ajustadas
# Em vez de usar `np.argsort` de forma direta, agora usamos a ordem correta
sorted_indices = np.argsort(longitudes)  # Ordenando as longitudes originais

# Ajustar a matriz de dados para refletir a nova ordem das longitudes
adjusted_data = sst[:, :, sorted_indices]

long,lati = np.meshgrid(longitudes, lat__)


ds2 = nc.Dataset('./data/dominio/srtm30plus_v11_land.nc')
lat_ = ds2['lat']
lon_ = ds2['lon']
long_,lati_ = np.meshgrid(lon_, lat_)
hgt = ds2['elev']


# Criando a figura e o eixo com a projeção apropriada
fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

# Configurando o recorte da área de interesse
ax.set_extent([-40, -20, -15, 0], crs=ccrs.PlateCarree())

# Adicionando o fundo com imagem padrão
ax.stock_img()

# Adicionando recursos do Cartopy
ax.add_feature(cfeature.BORDERS, linestyle='--', edgecolor='black', alpha=0.4)  # Fronteiras nacionais
ax.add_feature(cfeature.COASTLINE, edgecolor='black')  # Linhas costeiras

# Adicionando os estados brasileiros
states = cfeature.NaturalEarthFeature(
    category='cultural',
    name='admin_1_states_provinces',
    scale='10m',
    facecolor='none')

ax.add_feature(states, edgecolor='k', alpha=0.1)

# Adicionando o rótulo "SAWP"
ax.text(-31.6, -7.5, 'SAWP', color='k', fontsize=12, weight='bold', transform=ccrs.PlateCarree())

# Adicionando bordas (graticules) com linhas de latitude e longitude
gl = ax.gridlines(draw_labels=True, crs=ccrs.PlateCarree(), color='gray', linestyle='--', linewidth=0.5)
gl.top_labels = False  # Desativar labels no topo
gl.right_labels = False  # Desativar labels à direita
gl.xlabel_style = {'size': 14, 'color': 'black'}
gl.ylabel_style = {'size': 14, 'color': 'black'}
gl.xlocator = plt.MaxNLocator(6)  # Dividir o eixo X em 6 partes
gl.ylocator = plt.MaxNLocator(6)  # Dividir o eixo Y em 6 partes

# Desative as linhas de grid
gl.xlines = False
gl.ylines = False

# Definindo os níveis (levels) para o intervalo entre 28.5 e 32
levels = np.linspace(24, 32, 32)  # 100 pontos entre 28.5 e 32, ajustável conforme necessário

# Plotando o gráfico de contorno preenchido
ct = plt.contourf(long, lati, adjusted_data[3, :, :], levels=levels, cmap='hot_r')
ct2 = plt.contour(long, lati, adjusted_data[3, :, :], levels=[28.5], colors='k', linewidths=2)
plt.clabel(ct2, inline=True, fontsize=18, fmt='%1.1f', colors='k')
#plt.colorbar()


# # Exibindo a topografia com a variável hgt (altitude/topografia)
sh = ax.pcolormesh(long_, lati_, hgt, cmap='terrain', vmin=0, vmax=1200, alpha=0.6)
 
# Adicionando a barra de cores (colorbar) para a topografia
cbar = plt.colorbar(sh, ax=ax, orientation='vertical', pad=0.01, fraction=0.02)
cbar.set_label('Topography (m)', fontsize=20)
 
# Adicionando os pontos de medição (exemplo de dados fictícios)
dv = ax.scatter(
     lon, lat,  # Coordenadas dos pontos de medição
     color='k',  # Cor dos pontos
     s=1,  # Tamanho dos pontos
     transform=ccrs.PlateCarree(),  # Transformação para coordenadas geográficas
     label='Rain gauges'
)

# Adicionando uma legenda para os pontos
ax.legend(loc='upper right', fontsize=12)

# Adicionando o retângulo para a área SAWP
rect = Rectangle(
    xy=(-34, -12.5),  # Coordenadas do canto inferior esquerdo
    width=9,       # Largura do retângulo
    height=8.5,      # Altura do retângulo
    edgecolor='k',
    facecolor='lightgrey', alpha=0.3,
    linewidth=3
)
ax.add_patch(rect)


# Adicionando rótulos "Longitude" e "Latitude"
fig.text(0.5, 0.12, 'Longitude', ha='center', fontsize=16)
fig.text(0.04, 0.5, 'Latitude', va='center', rotation='vertical', fontsize=16)

# Salvando o mapa gerado
plt.savefig('Mapa_estações_SAWP.png', dpi=360, bbox_inches="tight")

# Mostrando o mapa
plt.show()

# ============================================================
# FIGURA DO DOMÍNIO, SAWP E ESTAÇÕES PLUVIOMÉTRICAS
# ============================================================

import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle


FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Estações efetivamente usadas após QC/homogeneidade
# ------------------------------------------------------------

stations_used_codes = df_qc.columns.astype(str)

stations_used = metadata[
    metadata["code"].astype(str).isin(stations_used_codes)
].copy()

# Caso exista uma lista de estações removidas por homogeneidade
if "stations_remove" in globals():
    stations_removed = regime_test[
        regime_test["station"].astype(str).isin(pd.Index(stations_remove).astype(str))
    ].copy()
else:
    stations_removed = pd.DataFrame(columns=["station", "lat", "lon"])


# ------------------------------------------------------------
# Leitura da TSM usada para ilustrar o domínio oceânico
# ------------------------------------------------------------

ds1 = nc.Dataset("./data/dominio/sst.nc")

lat_sst = np.array(ds1["lat"][:])
lon_sst = np.array(ds1["lon"][:])
sst = np.array(ds1["sst"][:])

# Ajusta longitude para -180 a 180
lon_sst_adj = np.where(lon_sst > 180, lon_sst - 360, lon_sst)

# Ordena longitudes após ajuste
idx_sort = np.argsort(lon_sst_adj)

lon_sst_adj = lon_sst_adj[idx_sort]
sst_adj = sst[:, :, idx_sort]

lon_sst_grid, lat_sst_grid = np.meshgrid(lon_sst_adj, lat_sst)


# ------------------------------------------------------------
# Leitura da topografia
# ------------------------------------------------------------

ds2 = nc.Dataset("./data/dominio/srtm30plus_v11_land.nc")

lat_topo = np.array(ds2["lat"][:])
lon_topo = np.array(ds2["lon"][:])
hgt = np.array(ds2["elev"][:])

lon_topo_grid, lat_topo_grid = np.meshgrid(lon_topo, lat_topo)


# ------------------------------------------------------------
# Figura
# ------------------------------------------------------------

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([-40, -20, -15, 0], crs=ccrs.PlateCarree())

# Mapa base
ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
ax.add_feature(cfeature.OCEAN, facecolor="white", alpha=1.0)
ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=1.0)
ax.add_feature(cfeature.BORDERS, linestyle="--", edgecolor="black", alpha=0.4)

states_feature = cfeature.NaturalEarthFeature(
    category="cultural",
    name="admin_1_states_provinces",
    scale="10m",
    facecolor="none"
)

ax.add_feature(states_feature, edgecolor="black", linewidth=0.5, alpha=0.4)

# ------------------------------------------------------------
# TSM e contorno de 28.5 °C
# ------------------------------------------------------------

levels_sst = np.linspace(24, 32, 33)

ct = ax.contourf(
    lon_sst_grid,
    lat_sst_grid,
    sst_adj[3, :, :],
    levels=levels_sst,
    cmap="hot_r",
    transform=ccrs.PlateCarree(),
    alpha=0.75
)

ct2 = ax.contour(
    lon_sst_grid,
    lat_sst_grid,
    sst_adj[3, :, :],
    levels=[28.5],
    colors="black",
    linewidths=2,
    transform=ccrs.PlateCarree()
)

ax.clabel(
    ct2,
    inline=True,
    fontsize=12,
    fmt="%1.1f"
)

# ------------------------------------------------------------
# Topografia
# ------------------------------------------------------------

sh = ax.pcolormesh(
    lon_topo_grid,
    lat_topo_grid,
    hgt,
    cmap="terrain",
    vmin=0,
    vmax=1200,
    alpha=0.55,
    transform=ccrs.PlateCarree()
)

cbar = plt.colorbar(
    sh,
    ax=ax,
    orientation="vertical",
    pad=0.01,
    fraction=0.025
)

cbar.set_label("Topography (m)", fontsize=13)


# ------------------------------------------------------------
# Estações usadas
# ------------------------------------------------------------

ax.scatter(
    stations_used["lon"],
    stations_used["lat"],
    color="black",
    s=5,
    transform=ccrs.PlateCarree(),
    label=f"Rain gauges used (n={len(stations_used)})",
    zorder=4
)

# ------------------------------------------------------------
# Estações removidas por homogeneidade, se existirem
# ------------------------------------------------------------

if len(stations_removed) > 0:
    ax.scatter(
        stations_removed["lon"],
        stations_removed["lat"],
        color="red",
        s=12,
        transform=ccrs.PlateCarree(),
        label=f"Removed by homogeneity test (n={len(stations_removed)})",
        zorder=5
    )


# ------------------------------------------------------------
# Retângulo da área SAWP
# ------------------------------------------------------------

rect_sawp = Rectangle(
    xy=(-34, -12.5),
    width=9,
    height=8.5,
    edgecolor="black",
    facecolor="lightgray",
    alpha=0.25,
    linewidth=2.5,
    transform=ccrs.PlateCarree(),
    zorder=6
)

ax.add_patch(rect_sawp)

ax.text(
    -31.8,
    -7.5,
    "SAWP",
    color="black",
    fontsize=12,
    weight="bold",
    transform=ccrs.PlateCarree(),
    zorder=7
)


# ------------------------------------------------------------
# Retângulo da área pluviométrica ENEB
# ------------------------------------------------------------

rect_eneb = Rectangle(
    xy=(LON_MIN, LAT_MIN),
    width=LON_MAX - LON_MIN,
    height=LAT_MAX - LAT_MIN,
    edgecolor="blue",
    facecolor="none",
    linewidth=2.0,
    linestyle="--",
    transform=ccrs.PlateCarree(),
    zorder=6
)

ax.add_patch(rect_eneb)

ax.text(
    LON_MIN + 0.2,
    LAT_MAX - 1.0,
    "ENEB rain-gauge domain",
    color="blue",
    fontsize=10,
    weight="bold",
    transform=ccrs.PlateCarree(),
    zorder=7
)


# ------------------------------------------------------------
# Grade geográfica
# ------------------------------------------------------------

gl = ax.gridlines(
    draw_labels=True,
    crs=ccrs.PlateCarree(),
    color="gray",
    linestyle="--",
    linewidth=0.5,
    alpha=0.6
)

gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {"size": 12}
gl.ylabel_style = {"size": 12}
gl.xlines = False
gl.ylines = False


# ------------------------------------------------------------
# Finalização
# ------------------------------------------------------------

ax.legend(
    loc="upper right",
    fontsize=10,
    frameon=True
)

fig.text(0.5, 0.08, "Longitude", ha="center", fontsize=14)
fig.text(0.06, 0.5, "Latitude", va="center", rotation="vertical", fontsize=14)

plt.savefig(
    FIG_DIR / "Mapa_dominio_SAWP_ENEB_estacoes.png",
    dpi=360,
    bbox_inches="tight"
)

plt.show()
plt.close()

print(f"Figura de domínio salva em: {FIG_DIR / 'Mapa_dominio_SAWP_ENEB_estacoes.png'}")