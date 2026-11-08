#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo: plot_era5
Funções para plotagem de gráficos vetoriais, circulações e mapas médios a partir de dados ERA5.

@author: simopec
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# ------------------------------
# Funções de plotagem
# ------------------------------

def plotar_grafico_sombreado_vetorial(lon, lat, var_u, var_v, var_div, label, cmap, spd_def, ax, identificador=None):
    """
    Plota gráficos vetoriais com contorno sombreado de divergência.
    """
    ax.set_extent([-60, -10, -30, 15], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgrey', alpha=0.3)
    
    states = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lakes',
        scale='10m',
        facecolor='none'
    )
    ax.add_feature(states, edgecolor='black', linewidth=0.3, alpha=0.3)

    # Configuração de rótulos
    if identificador == 'a':  # Só latitudes
        ax.set_yticks(np.arange(-30, 20, 10), crs=ccrs.PlateCarree())
        ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=16)
    elif identificador == 'b':  # Vazio
        ax.set_xticks([])
        ax.set_yticks([])
    elif identificador == 'c':  # Latitudes e longitudes
        ax.set_xticks(np.arange(-60, -5, 10), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-30, 20, 10), crs=ccrs.PlateCarree())
        ax.tick_params(axis='both', labelsize=16)
    elif identificador == 'd':  # Só longitudes
        ax.set_xticks(np.arange(-60, -5, 10), crs=ccrs.PlateCarree())
        ax.set_yticks([])
        ax.tick_params(axis='x', labelsize=16)

# =============================================================================
#     # Streamplot
#     ax.streamplot(lon, lat, var_u, var_v, density=0.8, color='grey',
#                   arrowsize=1, linewidth=0.7, arrowstyle='->', transform=ccrs.PlateCarree(), cmap=cmap)
# =============================================================================
    step = 10
    q = ax.quiver(
        lon[::step, ::step],
        lat[::step, ::step],
        var_u[::step, ::step],
        var_v[::step, ::step],
        transform=ccrs.PlateCarree(),
        scale=300,
        width=0.0025
    )
    # Contorno sombreado da divergência
    levels_div = np.asarray(range(-30, 31, 10)) / 1e6
    # Mantém a convenção original: valores positivos do campo recebido são
    # desenhados com a paleta ``PuOr_r``. No painel P90 o chamador pode
    # informar explicitamente outra paleta; no painel de quantis ``cmap='k'``
    # especifica a cor das setas, não uma paleta do Matplotlib.
    cmap_div = 'PuOr_r' if cmap == 'k' else cmap
    temp = ax.contourf(lon, lat, var_div, transform=ccrs.PlateCarree(), extend='both',
                       cmap=cmap_div, alpha=0.5, levels=levels_div)

    #ax.set_title(legnda, loc='left', fontsize=18)
    return temp

def plotar_grafico_sombreado(
    lon, lat, var, levels, cmap, ax,
    alpha=0.5,
    identificador=None,
    mascara_sawp=None,
    plotar_sawp=True,
    tipo_sawp='quadrado'  # 'isoterma' ou 'quadrado'
):
    """
    Plota contorno sombreado e destaca a região da SAWP.
    
    tipo_sawp:
        'quadrado'  -> desenha caixa fixa
        'isoterma'  -> desenha área da máscara booleana mascara_sawp
    """

    ax.set_extent([-60, -10, -30, 15], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgrey', alpha=0.3)

    states = cfeature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lakes',
        scale='10m',
        facecolor='none'
    )
    ax.add_feature(states, edgecolor='black', linewidth=0.3, alpha=0.3)

    if identificador == 'a':
        ax.set_yticks(np.arange(-30, 20, 10), crs=ccrs.PlateCarree())
        ax.set_xticks([])
        ax.tick_params(axis='y', labelsize=16)
    elif identificador == 'b':
        ax.set_xticks([])
        ax.set_yticks([])
    elif identificador == 'c':
        ax.set_xticks(np.arange(-60, -5, 10), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-30, 20, 10), crs=ccrs.PlateCarree())
        ax.tick_params(axis='both', labelsize=16)
    elif identificador == 'd':
        ax.set_xticks(np.arange(-60, -5, 10), crs=ccrs.PlateCarree())
        ax.set_yticks([])
        ax.tick_params(axis='x', labelsize=16)

    temp = ax.contourf(
        lon, lat, var,
        levels=levels,
        cmap=cmap,
        extend='both',
        transform=ccrs.PlateCarree(),
        alpha=alpha
    )

    if plotar_sawp:

        if tipo_sawp == 'quadrado':
            sawp_lon_min, sawp_lon_max = -34, -25
            sawp_lat_min, sawp_lat_max = -12.5, -4

            sawp_box = Rectangle(
                (sawp_lon_min, sawp_lat_min),
                sawp_lon_max - sawp_lon_min,
                sawp_lat_max - sawp_lat_min,
                linewidth=1.2,
                edgecolor='black',
                facecolor='lightgray',
                hatch='///',
                alpha=0.18,
                transform=ccrs.PlateCarree(),
                zorder=5
            )

            ax.add_patch(sawp_box)

        elif tipo_sawp == 'isoterma':

            if mascara_sawp is None:
                raise ValueError(
                    "Para tipo_sawp='isoterma', forneça mascara_sawp."
                )

            ax.contourf(
                lon, lat, mascara_sawp.astype(int),
                levels=[0.5, 1.5],
                colors='lightgray',
                hatches=['///'],
                alpha=0.18,
                transform=ccrs.PlateCarree(),
                zorder=5
            )

            ax.contour(
                lon, lat, mascara_sawp.astype(int),
                levels=[0.5],
                colors='black',
                linewidths=1.0,
                transform=ccrs.PlateCarree(),
                zorder=6
            )

        sawp_legend = mpatches.Patch(
            facecolor='lightgray',
            edgecolor='black',
            hatch='///',
            alpha=0.35,
            label='SAWP'
        )

        ax.legend(
            handles=[sawp_legend],
            loc='upper right',
            fontsize=8,
            frameon=True,
            facecolor='white',
            edgecolor='black',
            framealpha=1.0,
            handlelength=1.5,
            borderpad=0.4
        )

    return temp

# =============================================================================
# def plotar_grafico_sombreado(lon, lat, var, levels, cmap, ax, alpha=0.5, identificador=None):
#     """
#     Plota contorno sombreado simples.
#     """
#     
#     ax.set_extent([-60, -10, -30, 15], crs=ccrs.PlateCarree())
#     ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
#     ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
#     ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgrey', alpha=0.3)
#     
#     states = cfeature.NaturalEarthFeature(
#         category='cultural',
#         name='admin_1_states_provinces_lakes',
#         scale='10m',
#         facecolor='none'
#     )
#     ax.add_feature(states, edgecolor='black', linewidth=0.3, alpha=0.3)
# 
#     if identificador == 'a':
#         ax.set_yticks(np.arange(-30, 20, 10), crs=ccrs.PlateCarree())
#         ax.set_xticks([])
#         ax.tick_params(axis='y', labelsize=16)
#     elif identificador == 'b':
#         ax.set_xticks([])
#         ax.set_yticks([])
#     elif identificador == 'c':
#         ax.set_xticks(np.arange(-60, -5, 10), crs=ccrs.PlateCarree())
#         ax.set_yticks(np.arange(-30, 20, 10), crs=ccrs.PlateCarree())
#         ax.tick_params(axis='both', labelsize=16)
#     elif identificador == 'd':
#         ax.set_xticks(np.arange(-60, -5, 10), crs=ccrs.PlateCarree())
#         ax.set_yticks([])
#         ax.tick_params(axis='x', labelsize=16)
# 
#     # Caixa sombreada da SAWP
#     sawp_lon_min, sawp_lon_max = -34, -25
#     sawp_lat_min, sawp_lat_max = -12.5, -4
# 
#     sawp_box = Rectangle(
#         (sawp_lon_min, sawp_lat_min),
#         sawp_lon_max - sawp_lon_min,
#         sawp_lat_max - sawp_lat_min,
#         linewidth=1.2,
#         edgecolor='black',
#         facecolor='lightgray',
#         hatch='///',
#         alpha=0.12,
#         transform=ccrs.PlateCarree(),
#         zorder=2
#     )
# 
#     ax.add_patch(sawp_box)
#     
# 
#     temp = ax.contourf(
#         lon, lat, var,
#         levels=levels,
#         cmap=cmap,
#         extend='both',
#         transform=ccrs.PlateCarree(),
#         alpha=alpha
#     )
#  
#     # Legenda da SAWP
#     sawp_legend = Rectangle(
#         (0, 0), 1, 1,
#         facecolor='gray',
#         edgecolor='black',
#         hatch='///',
#         alpha=0.12
#     )
#     
#     ax.legend(
#         [sawp_legend],
#         ['SAWP'],
#         loc='upper right',
#         fontsize=10,
#         frameon=True,
#         facecolor='white',
#         edgecolor='black',
#         framealpha=1.0,
#         handlelength=1.5,
#         borderpad=0.4,
#         labelspacing=0.3
#     )
#     return temp
# =============================================================================

# =============================================================================
# def plotar_grafico_sombreado(lon, lat, var, levels, cmap, ax, alpha=0.5, identificador=None):
#     """
#     Plota contorno sombreado simples.
#     """
#     
#     ax.set_extent([-60, -10, -30, 15], crs=ccrs.PlateCarree())
#     ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
#     ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.5)
#     ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgrey', alpha=0.3)
#     
#     states = cfeature.NaturalEarthFeature(
#         category='cultural',
#         name='admin_1_states_provinces_lakes',
#         scale='10m',
#         facecolor='none'
#     )
#     ax.add_feature(states, edgecolor='black', linewidth=0.3, alpha=0.3)
# 
#     if identificador == 'a':
#         ax.set_yticks(np.arange(-30, 20, 10), crs=ccrs.PlateCarree())
#         ax.set_xticks([])
#         ax.tick_params(axis='y', labelsize=16)
#     elif identificador == 'b':
#         ax.set_xticks([])
#         ax.set_yticks([])
#     elif identificador == 'c':
#         ax.set_xticks(np.arange(-60, -5, 10), crs=ccrs.PlateCarree())
#         ax.set_yticks(np.arange(-30, 20, 10), crs=ccrs.PlateCarree())
#         ax.tick_params(axis='both', labelsize=16)
#     elif identificador == 'd':
#         ax.set_xticks(np.arange(-60, -5, 10), crs=ccrs.PlateCarree())
#         ax.set_yticks([])
#         ax.tick_params(axis='x', labelsize=16)
# 
#     temp = ax.contourf(lon, lat, var, levels=levels, cmap=cmap, extend='both', transform=ccrs.PlateCarree(), alpha=alpha)
#     return temp
# 
# =============================================================================

# Função adaptada para aceitar um eixo (ax) no subplot
def plotar_grafico_sombreado_correlacao(lon, lat, var, sign, levels, label, cmap, ax):
    """
    Função para plotar gráfico sombreado com mapa de correlação e níveis de significância.
    """
    # Configuração do mapa
    ax.set_extent([-60, -10, -30, 15], crs=ccrs.PlateCarree())
    ax.coastlines(resolution='50m')
    ax.add_feature(cartopy.feature.BORDERS)
    ax.set_xticks(np.arange(-60, -5, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(-30, 20, 10), crs=ccrs.PlateCarree())
    ax.tick_params(axis='both', labelsize=16)
    # Adicionar estados e bordas
    states = cartopy.feature.NaturalEarthFeature(
        category='cultural',
        name='admin_1_states_provinces_lakes',
        scale='10m',
        facecolor='none'
    )
    ax.add_feature(states, edgecolor='black', linewidth=0.3,alpha=0.3)
    ax.set_ylabel('Latitude', fontsize=18)
    ax.set_xlabel('Longitude', fontsize=18)


    # Plotar o mapa sombreado
    temp = ax.contourf(lon, lat, var, levels=levels, transform=ccrs.PlateCarree(), extend='min', cmap=cmap, alpha=0.5)
    # Adicionar contornos de linha apenas para correlações específicas (0.3 e 0.5)
    # Adicionar contornos de linha apenas para 0.3 e 0.5
    contour_lines = ax.contour(
        lon, lat, var, 
        levels=[0.4, 0.6],  # Níveis de contorno específicos
        colors='black', 
        linewidths=0.8, 
        transform=ccrs.PlateCarree()
    )
    
    # Adicionar os valores nas linhas de contorno
    ax.clabel(
        contour_lines, 
        fmt="%.1f",  # Formato do rótulo (uma casa decimal)
        inline=True,  # Rótulos dentro da linha
        fontsize=10   # Tamanho da fonte
    )

    # Adicionar a barra de cores
    cbar = plt.colorbar(temp, ax=ax, shrink=0.5, fraction=0.046, pad=0.04, orientation='vertical')
    cbar.set_label(label,fontsize=14, color='k')
    cbar.ax.tick_params(labelsize=14)  # Define o tamanho da fonte como 14

def plot_mapa_correlacao_scatter(
    lon, lat,
    corremap, sign_map,
    t_q10, t_q25, t_qno, t_q75, t_q90,
    c_q10, c_q25, c_qno, c_q75, c_q90,
    slope, intercept, r_value,
    chuva_norm,
    tsm,
    cmap_levels=np.array(range(0, 11, 1)) / 10,
    cmap_name='CMRmap_r',
    fig_size=(14, 7),
    output_file='Combined_Prec_TS_Subplot.png'
):
    """
    Cria figura combinada:
    (a) mapa de correlação TSM x precipitação
    (b) scatter plot com quantis e regressão linear
    """

    fig = plt.figure(figsize=fig_size)

    # =========================
    # Painel (a): Mapa
    # =========================
    ax0 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    ax0.set_title("a)", loc='left', fontsize=20)

    plotar_grafico_sombreado_correlacao(
        lon, lat,
        corremap,
        sign_map,
        cmap_levels,
        'Correlation',
        cmap_name,
        ax0
    )

    # =========================
    # Painel (b): Scatter
    # =========================
    ax1 = fig.add_subplot(1, 2, 2)
    ax1.set_title("b)", loc='left', fontsize=20)

    # Scatter por quantis
    ax1.scatter(t_q10, c_q10, c="b", alpha=1,   label="La Niña (forte)")
    ax1.scatter(t_q25, c_q25, c="b", alpha=0.2, label="La Niña (moderado)")
    ax1.scatter(t_qno, c_qno, c="lightgrey", alpha=0.2, label="Neutro")
    ax1.scatter(t_q75, c_q75, c="r", alpha=0.2, label="El Niño (moderado)")
    ax1.scatter(t_q90, c_q90, c="r", alpha=1,   label="El Niño (forte)")

    # Linhas de referência
    mean_chuva = chuva_norm.mean()
    ax1.axhline(y=mean_chuva, linestyle="dashed", color="lightgrey", alpha=0.6)
    ax1.axvline(x=mean_chuva, linestyle="dashed", color="lightgrey", alpha=0.6)
    ax1.axvline(x=mean_chuva - 0.7, linestyle="dashed", color="b", alpha=0.6)
    ax1.axvline(x=mean_chuva + 0.7, linestyle="dashed", color="r", alpha=0.3)

    # Regressão linear
    ax1.set_xlim([-3, 3])
    ax1.set_ylim([-3, 3])

    x_vals = np.array(ax1.get_xlim())
    y_vals = intercept + slope * x_vals

    ax1.plot(
        x_vals, y_vals, '--',
        color='k', alpha=0.5,
        label=f'Regressão Linear: y={slope:.2f}x+{intercept:.2f}'
    )

    # Texto com r
    text_x = x_vals[0] + (x_vals[1] - x_vals[0]) * 0.05
    text_y = y_vals[0] + (y_vals[1] - y_vals[0]) * 1.5

    ax1.text(
        text_x, text_y,
        f'r = {r_value:.2f}',
        fontsize=20, color='white', weight='bold',
        bbox=dict(
            facecolor='black',
            edgecolor='none',
            alpha=0.7,
            boxstyle='round,pad=0.3'
        )
    )

    # Eixos
    ax1.set_xlabel("SAWP SST anomaly", fontsize=22)
    ax1.set_ylabel("ENEB Anomaly acumulated Precipitation", fontsize=18)

    ax1.yaxis.set_label_position("right")
    ax1.yaxis.tick_right()

    ax1.tick_params(axis='x', labelsize=16)
    ax1.tick_params(axis='y', labelsize=16)

    # Layout e salvamento
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_file, dpi=300, bbox_inches="tight")

def plotar_circulacao(u_quantil, w_quantil, t_quantil, level, y_equal_spacing, lon, legenda, ax, identificador=None):
    """
    Função para plotar gráficos de circulação longitudinal em um eixo específico (ax).
    O argumento 'identificador' define quais rótulos (latitude/longitude) serão exibidos.
    Padrão adotado:
      a) exibe apenas eixo Y
      b) nenhum eixo
      c) ambos X e Y
      d) exibe apenas eixo X
    """
    # Interpolação para nova grade
    u_interp = np.zeros((60, 201))
    w_interp = np.zeros((60, 201))
    t_interp = np.zeros((60, 201))
    
    for i in range(201):
        t_interp[:, i] = np.interp(y_equal_spacing, level, t_quantil[:, i])
        u_interp[:, i] = np.interp(y_equal_spacing, level, u_quantil[:, i])
        w_interp[:, i] = np.interp(y_equal_spacing, level, w_quantil[:, i])
        
    
    long_int, press_int = np.meshgrid(lon[0,:], y_equal_spacing)
    
    # Configura os níveis para o contorno
    levels = np.asarray(range(-30,31,5))/10 
    #[-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]

    # Contorno sombreado (sem barra de cores aqui)
    contour = ax.contourf(long_int, press_int, t_interp, cmap='RdYlBu_r', extend='both', levels=levels, alpha=0.4)

    # Linhas de corrente para vento zonal (u) e vertical (w)
    # w multiplicado por 1000 para ajustar a escala se necessário
    ax.streamplot(lon, y_equal_spacing, u_interp, w_interp * 1000,
                  density=0.8, linewidth=0.5, arrowstyle='->',
                  integration_direction='backward', color='grey')

    # Inverter o eixo Y (pressão)
    ax.invert_yaxis()
    ax.set_ylim([1000, 200])
    ax.set_xlim([-60, -10])
    
    # Ajuste dos rótulos conforme o identificador
    if identificador == 'a':  # Só Y
        ax.set_yticks(np.arange(1000, 200, -200))
        ax.set_xticks([])
        #ax.set_ylabel('Pressure Levels (hPa)', fontsize=12)
        ax.tick_params(axis='y', labelsize=16)
    elif identificador == 'b':  # Vazio
        ax.set_xticks([])
        ax.set_yticks([])
    elif identificador == 'c':  # Ambos X e Y
        ax.set_xticks(np.linspace(lon.min(), lon.max(), 5))  # Ajuste conforme sua necessidade
        ax.set_yticks(np.arange(1000, 200, -200))
        #ax.set_xlabel('Longitude', fontsize=12)
        #ax.set_ylabel('Pressure Levels (hPa)', fontsize=12)
        ax.tick_params(axis='both', labelsize=16)
    elif identificador == 'd':  # Só X
        ax.set_xticks(np.linspace(lon.min(), lon.max(), 5))  # Ajuste conforme sua necessidade
        ax.set_yticks([])
        #ax.set_xlabel('Longitude', fontsize=12)
        ax.tick_params(axis='x', labelsize=16)
    
    # Título do subplot
    ax.set_title(legenda, loc='left')

    # Preenchimento de área específica
    ax.fill_betweenx([1000, 200], -40, -30, color='gray', alpha=0.2)

    # Retornar o objeto contour para criar a barra de cores externamente
    return contour

def plotar_circulacao_diff(u_quantil, w_quantil, t_quantil, level, y_equal_spacing, lon, legenda, ax, identificador=None):
    """
    Função para plotar gráficos de circulação longitudinal em um eixo específico (ax).
    O argumento 'identificador' define quais rótulos (latitude/longitude) serão exibidos.
    Padrão adotado:
      a) exibe apenas eixo Y
      b) nenhum eixo
      c) ambos X e Y
      d) exibe apenas eixo X
    """
    # Interpolação para nova grade
    u_interp = np.zeros((60, 201))
    w_interp = np.zeros((60, 201))
    t_interp = np.zeros((60, 201))
    
    for i in range(201):
        u_interp[:, i] = np.interp(y_equal_spacing, level, u_quantil[:, i])
        w_interp[:, i] = np.interp(y_equal_spacing, level, w_quantil[:, i])
        t_interp[:, i] = np.interp(y_equal_spacing, level, t_quantil[:, i])
    
    long_int, press_int = np.meshgrid(lon[0,:], y_equal_spacing)
    
    # Configura os níveis para o contorno
    levels = np.asarray(range(-10,11,2))/10 
    #[-3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3]

    # Contorno sombreado (sem barra de cores aqui)
    contour = ax.contourf(long_int, press_int, t_interp, cmap='RdYlBu_r', extend='both', levels=levels, alpha=0.4)

    # Linhas de corrente para vento zonal (u) e vertical (w)
    # w multiplicado por 1000 para ajustar a escala se necessário
    ax.streamplot(lon, y_equal_spacing, u_interp, w_interp * 1000,
                  density=0.8, linewidth=0.5, arrowstyle='->',
                  integration_direction='backward', color='grey')

    # Inverter o eixo Y (pressão)
    ax.invert_yaxis()
    ax.set_ylim([1000, 200])
    ax.set_xlim([-60, -10])
    
    # Ajuste dos rótulos conforme o identificador
    if identificador == 'a':  # Só Y
        ax.set_yticks(np.arange(1000, 200, -200))
        ax.set_xticks([])
        #ax.set_ylabel('Pressure Levels (hPa)', fontsize=12)
        ax.tick_params(axis='y', labelsize=16)
    elif identificador == 'b':  # Vazio
        ax.set_xticks([])
        ax.set_yticks([])
    elif identificador == 'c':  # Ambos X e Y
        ax.set_xticks(np.linspace(lon.min(), lon.max(), 5))  # Ajuste conforme sua necessidade
        ax.set_yticks(np.arange(1000, 200, -200))
        #ax.set_xlabel('Longitude', fontsize=12)
        #ax.set_ylabel('Pressure Levels (hPa)', fontsize=12)
        ax.tick_params(axis='both', labelsize=16)
    elif identificador == 'd':  # Só X
        ax.set_xticks(np.linspace(lon.min(), lon.max(), 5))  # Ajuste conforme sua necessidade
        ax.set_yticks([])
        #ax.set_xlabel('Longitude', fontsize=12)
        ax.tick_params(axis='x', labelsize=16)
    
    # Título do subplot
    ax.set_title(legenda, loc='left')

    # Preenchimento de área específica
    ax.fill_betweenx([1000, 200], -40, -30, color='gray', alpha=0.2)

    # Retornar o objeto contour para criar a barra de cores externamente
    return contour

