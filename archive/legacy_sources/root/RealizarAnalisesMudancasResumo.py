#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unifica todos os gráficos SAWP em uma única figura com subplots. 
"""

from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import netCDF4 as nc
from timeit import default_timer as timer
timeini = timer()

# =====================================================================
# IMPORTAÇÕES
# =====================================================================

# =====================================================================
# FUNÇÃO PRINCIPAL — TODOS OS GRÁFICOS UNIFICADOS
# =====================================================================


def plot_todos_unidos(serie_periodo_chuvoso, sst_1990, sst_2020, serie_tsm_ct, lon, lat):
    # ================================================================
    # (a) SAWP Temperature PDF — Figura com normalização correta
    # ================================================================
    fig = plt.figure(figsize=(13, 14))

    # Ajuste do GridSpec para garantir que apenas a figura (a) seja estreita e as figuras (b) e (c) fiquem com o tamanho normal
    gs = GridSpec(3, 2, height_ratios=[
                  1, 1.5, 1.2], width_ratios=[1, 1], figure=fig)

    # Subgráfico (a) na nova grade
    ax1 = fig.add_subplot(gs[0, :])

    # Dividindo a série de dados em dois períodos
    tsm_1959_1990 = serie_periodo_chuvoso[serie_periodo_chuvoso.index.year < 1990]
    tsm_1990_2023 = serie_periodo_chuvoso[serie_periodo_chuvoso.index.year > 1990]

    # Estimando a PDF para cada período assumindo uma distribuição normal
    mean_1 = np.mean(tsm_1959_1990['TSM'])
    std_1 = np.std(tsm_1959_1990['TSM'])
    mean_2 = np.mean(tsm_1990_2023['TSM'])
    std_2 = np.std(tsm_1990_2023['TSM'])

    # Gerando a PDF normal para cada período
    x = np.linspace(26.5, 30, 200)  # Definindo o intervalo para o eixo X

    pdf_1 = (1 / (std_1 * np.sqrt(2 * np.pi))) * \
        np.exp(-0.5 * ((x - mean_1) / std_1) ** 2)
    pdf_2 = (1 / (std_2 * np.sqrt(2 * np.pi))) * \
        np.exp(-0.5 * ((x - mean_2) / std_2) ** 2)

    # Normalizando as PDFs para garantir que a área sob a curva seja 1
    dx = x[1] - x[0]  # Calculando a largura dos intervalos no eixo X
    # Soma dos valores da PDF para o intervalo de x
    area_1 = np.sum(pdf_1) * dx
    # Soma dos valores da PDF para o intervalo de x
    area_2 = np.sum(pdf_2) * dx

    # Ajustando as PDFs para garantir que a área seja 1
    pdf_1 /= area_1
    pdf_2 /= area_2

    # Percentil 90 para cada período
    p90_1 = np.percentile(tsm_1959_1990['TSM'], 90)
    p90_2 = np.percentile(tsm_1990_2023['TSM'], 90)

    # Plotando a PDF para os dois períodos
    ax1.plot(x, pdf_1, color='#1f77b4', linewidth=2.5, label='1961–1990')
    ax1.plot(x, pdf_2, color='#ff7f0e', linewidth=2.5, label='1991–2020')

    # Preenchendo as áreas abaixo das curvas
    ax1.fill_between(x, pdf_1, color='#1f77b4', alpha=0.15)
    ax1.fill_between(x, pdf_2, color='#ff7f0e', alpha=0.15)

    # Adicionando as linhas verticais para o percentil 90 (P90)
    ax1.axvline(p90_1, color='#1f77b4', linestyle='--', linewidth=2, alpha=0.8)
    ax1.axvline(p90_2, color='#ff7f0e', linestyle='--', linewidth=2, alpha=0.8)

    # Adicionando texto para os percentis
    ax1.text(p90_1 + 0.02, max(pdf_1)*0.85,
             'P90', color='#1f77b4', fontsize=10)
    ax1.text(p90_2 + 0.02, max(pdf_2)*0.85,
             'P90', color='#ff7f0e', fontsize=10)

    # Ajustando os limites do gráfico
    ax1.set_xlim(26.5, 30)
    ax1.set_ylim(0, 1.25)

    # Título e rótulos
    ax1.set_title('(a) SAWP Temperature PDF', fontsize=13,
                  loc='left', color='dimgrey')
    ax1.set_xlabel('SAWP Temperature (°C)', fontsize=13)
    ax1.set_ylabel('Probability Density', fontsize=13)

    # Legenda
    ax1.legend(frameon=False, fontsize=10)

    # Estilo do gráfico
    ax1.tick_params(labelsize=11)
    ax1.spines[['top', 'right']].set_visible(False)

    # ================================================================
    # (b) ΔSST MAPA — Figura maior
    # ================================================================
    proj = ccrs.PlateCarree()
    extent = [lon.min(), lon.max(), lat.min(), lat.max()]
    ax2 = fig.add_subplot(gs[1, 0], projection=proj)
    clim_1990 = np.nanmean(sst_1990, axis=0) - 273.15
    clim_2020 = np.nanmean(sst_2020, axis=0) - 273.15
    sst_diff = clim_2020 - clim_1990
    levels = np.linspace(-1.5, 1.5, 21)
    cmap = plt.get_cmap('coolwarm')
    cf = ax2.contourf(lon, lat, sst_diff, levels=levels,
                      cmap=cmap, extend='both')
    # --- Map features ---
    ax2.coastlines(resolution='50m', color='black', linewidth=0.8)
    ax2.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor='gray')
    ax2.add_feature(cfeature.LAND, color='lightgrey', zorder=0)
    ax2.set_extent(extent)
    ax2.set_title('(b) ΔSST (1991–2020 minus 1961–1990)',
                  fontsize=13, loc='left', color='dimgrey')
    # --- Colorbar ajustada ---
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    # Coordenadas [esquerda, baixo, largura, altura] em fração da figura
    # ajusta posição conforme layout
    cax = fig.add_axes([0.45, 0.5, 0.02, 0.10])
    cbar = fig.colorbar(cf, cax=cax, orientation='vertical')
    cbar.set_label('ΔSST (°C)', fontsize=11, labelpad=8)
    cbar.ax.tick_params(labelsize=9)
    cbar.outline.set_visible(False)

    # ================================================================
    # (c) Contorno 28.5 °C — Figura maior
    # ================================================================
    ax3 = fig.add_subplot(gs[1, 1], projection=proj)
    ax3.set_extent(extent)
    ax3.add_feature(cfeature.LAND, facecolor='white')
    ax3.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor='grey')
    ax3.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor='lightgrey')
    cont_1990 = ax3.contour(lon, lat, clim_1990, levels=[
                            28.5], colors='royalblue', linewidths=2, transform=proj)
    cont_2020 = ax3.contour(lon, lat, clim_2020, levels=[
                            28.5], colors='darkorange', linewidths=2, transform=proj)
    ax3.clabel(cont_1990, fmt='28.5°C', inline=True,
               fontsize=9, colors='royalblue')
    ax3.clabel(cont_2020, fmt='28.5°C', inline=True,
               fontsize=9, colors='darkorange')
    # --- Legenda corrigida ---
    handles = [Line2D([0], [0], color='royalblue', lw=2),
               Line2D([0], [0], color='darkorange', lw=2)]
    labels = ['1961–1990', '1991–2020']
    ax3.legend(handles=handles, labels=labels,
               loc='lower left', frameon=False, fontsize=9)
    ax3.set_title('(c) 28.5°C Isotherm Extension',
                  fontsize=13, loc='left', color='dimgrey')

    # ================================================================
    # (d) Série mensal SAWP — Sem alterações
    # ================================================================
    ax4 = fig.add_subplot(gs[2, :])
    grouped = serie_tsm_ct.groupby(serie_tsm_ct.index.year)
    anos = sorted(grouped.groups.keys())
    cmap = plt.colormaps['PRGn'].resampled(len(anos))
    for i, ano in enumerate(anos):
        dados = grouped.get_group(ano)
        if len(dados) == 12:
            color = cmap(i)
            x = np.arange(1, 13)
            y = dados['TSM'].values
            f = interp1d(x, y, kind='cubic')
            x_new = np.linspace(1, 12, 1000)
            y_new = f(x_new)
            alpha_values = np.where(y_new <= 28.5, 0.1, 0.8)
            for j in range(len(x_new)-1):
                ax4.plot(x_new[j:j+2], y_new[j:j+2], color=color,
                         alpha=alpha_values[j], linewidth=1.5)
    ax4.axhline(28.5, color='lightgrey', linestyle='--',
                linewidth=1.8, alpha=0.7)
    ax4.set_title('(d) Monthly SST Evolution (SAWP)',
                  fontsize=13, loc='left', color='dimgrey')
    ax4.set_xlabel('Month', fontsize=13)
    ax4.set_ylabel('SST (°C)', fontsize=13)
    ax4.set_xticks(np.arange(1, 13))
    ax4.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May',
                        'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax4.spines[['top', 'right']].set_visible(False)
    ax4.tick_params(labelsize=11)
    norm = mcolors.Normalize(vmin=min(anos), vmax=max(anos))
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    axins = inset_axes(ax4, width="20%", height="3%",
                       loc='upper right', borderpad=1.0)
    cbar = fig.colorbar(sm, cax=axins, orientation='horizontal')
    cbar.set_label('Years', fontsize=10, labelpad=5)
    cbar.set_ticks([anos[0], anos[-1]])
    cbar.set_ticklabels([str(anos[0]), str(anos[-1])])
    cbar.ax.tick_params(labelsize=9)

    # ================================================================
    # FINALIZA FIGURA
    # ================================================================
    plt.subplots_adjust(hspace=0.4, wspace=0.05)
    plt.suptitle('SST Variability and Trends over the SAWP Region',
                 fontsize=16, color='grey', y=0.95, x=0.3)
    plt.savefig('SAWP_subplot_unificado.png', dpi=400, bbox_inches='tight')
    plt.show()


# =====================================================================
# LEITURA E PREPARAÇÃO DOS DADOS
# =====================================================================
arquivo_era5_tsm = nc.Dataset('./data/Era5_Dataset_sst.nc')
vetor_tempo_tsm = arquivo_era5_tsm.variables['time']
tamanho_vetor_tempo_tsm = vetor_tempo_tsm.size
periodo_inicial = datetime(1900, 1, 1)
periodo_tempo_convertido = [periodo_inicial +
                            timedelta(hours=int(v)) for v in vetor_tempo_tsm[:]]
lat_ = arquivo_era5_tsm.variables['latitude']
lon_ = arquivo_era5_tsm.variables['longitude']
lon, lat = np.meshgrid(lon_, lat_)
loaded_data = np.load("./outputs/detrended_sst.npz")
arquivo_era5_tsm_st = loaded_data['data']
arquivo_era5_tsm_ct = arquivo_era5_tsm['sst']

DicionarioAreaEstudo = {'SAWP': [-35, -25, -12.5, -4]}
lloW = np.where(lon_[:] == DicionarioAreaEstudo['SAWP'][0])[0][0]
lloE = np.where(lon_[:] == DicionarioAreaEstudo['SAWP'][1])[0][0]
llaS = np.where(lat_[:] == DicionarioAreaEstudo['SAWP'][2])[0][0]
llaN = np.where(lat_[:] == DicionarioAreaEstudo['SAWP'][3])[0][0]

serie_tsm_ct = pd.DataFrame(
    np.nanmean(np.nanmean(
        arquivo_era5_tsm_ct[:, llaN:llaS, lloW:lloE], axis=2), axis=1) - 273.15,
    columns=['TSM'], index=periodo_tempo_convertido)
serie_tsm_ct['Mes'] = serie_tsm_ct.index.month
serie_tsm_ct['Indices'] = np.arange(len(serie_tsm_ct))

serie_tsm_ct_clima = serie_tsm_ct[(serie_tsm_ct.index.year >= 1980) & (
    serie_tsm_ct.index.year <= 2010)]
serie_tsm_ct_climatologia_tsm = serie_tsm_ct_clima.groupby('Mes')['TSM'].mean()
serie_tsm_ct['Anomalia'] = serie_tsm_ct.apply(
    lambda row: row['TSM'] - serie_tsm_ct_climatologia_tsm.loc[row['Mes']], axis=1)
serie_periodo_chuvoso = serie_tsm_ct[(
    serie_tsm_ct['Mes'] >= 3) & (serie_tsm_ct['Mes'] <= 6)]

sst_1990 = arquivo_era5_tsm_ct[serie_periodo_chuvoso['Indices'][(
    serie_periodo_chuvoso.index.month == 4) & (serie_periodo_chuvoso.index.year < 1990)], :, :]
sst_2020 = arquivo_era5_tsm_ct[serie_periodo_chuvoso['Indices'][(
    serie_periodo_chuvoso.index.month == 4) & (serie_periodo_chuvoso.index.year > 1990)], :, :]

# =====================================================================
# EXECUTA PLOTAGEM FINAL
# =====================================================================
plot_todos_unidos(serie_periodo_chuvoso, sst_1990,
                  sst_2020, serie_tsm_ct, lon, lat)
