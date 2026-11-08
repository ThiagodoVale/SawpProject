#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 15:36:16 2025

@author: simopec
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import sawp

def plotar_anomalias_percentis(var, percentis, lon, lat, levels, cmap, label_cbar, nome_arquivo,
                               func_media_2d, plot_func):
    """
    Plota subplots de anomalias de percentis de uma variável (ex: TSM/SST).
    
    Parâmetros:
        var : np.ndarray
            Array 3D/4D da variável
        percentis : dict
            Dicionário de percentis, ex: {'10':..., '25':..., '75':..., '90':..., 'normal':...}
        lon, lat : np.ndarray
            Matrizes de coordenadas
        levels : list ou np.ndarray
            Níveis para o contourf
        cmap : matplotlib colormap
            Colormap para o plot
        label_cbar : str
            Label da barra de cores (ex: 'SST (ºC)')
        nome_arquivo : str
            Nome do arquivo PNG a ser salvo
        func_media_2d : function
            Função para calcular média 2D, ex: MatrizMedia2dEra5
        plot_func : function
            Função de plotagem de contorno sombreado, ex: plotar_grafico_sombreado
    """
    
    # Criar mapas médios
    mapa_medio_norm = func_media_2d(percentis['percentis_tsm']['normal'], var)
    mapa_ef = func_media_2d(percentis['percentis_tsm']['10'], var)
    mapa_lf = func_media_2d(percentis['percentis_tsm']['25'], var)
    mapa_lq = func_media_2d(percentis['percentis_tsm']['75'], var)
    mapa_eq = func_media_2d(percentis['percentis_tsm']['90'], var)
    
    # Configurar figura e subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    plt.subplots_adjust(hspace=0.01, wspace=0.15)
    
    # Subplots
    temp_a = plot_func(lon, lat, mapa_ef - mapa_medio_norm, levels, cmap, axs[0,0], alpha=0.5, identificador='a')
    temp_b = plot_func(lon, lat, mapa_lf - mapa_medio_norm, levels, cmap, axs[0,1], alpha=0.5, identificador='b')
    temp_c = plot_func(lon, lat, mapa_lq - mapa_medio_norm, levels, cmap, axs[1,0], alpha=0.5, identificador='c')
    temp_d = plot_func(lon, lat, mapa_eq - mapa_medio_norm, levels, cmap, axs[1,1], alpha=0.5, identificador='d')
    
    # Labels centrais
    fig.text(0.45, 0.05, "Longitude", ha='center', fontsize=20)
    fig.text(0.045, 0.5, "Latitude", va='center', rotation='vertical', fontsize=20)
    
    # Barra de cores única
    cbar = fig.colorbar(temp_d, ax=axs, orientation='vertical', shrink=0.5, pad=0.02)
    cbar.set_label(label_cbar, size=15)
    cbar.ax.tick_params(labelsize=12)
    
    # Títulos
    axs[0,0].set_title('a)', loc='left', fontsize=18)
    axs[0,1].set_title('b)', loc='left', fontsize=18)
    axs[1,0].set_title('c)', loc='left', fontsize=18)
    axs[1,1].set_title('d)', loc='left', fontsize=18)
    
    # Ajustar layout e salvar
    fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.2, hspace=0.1, wspace=0.2)
    plt.savefig(nome_arquivo, dpi=360, bbox_inches="tight")
    plt.close()

def plotar_anomalias_percentis_calculados(var_ef,var_lf,var_lq,var_eq,var_norm, 
                                          lon, lat, levels, cmap, label_cbar, nome_arquivo,plot_func):
    # Configurar figura e subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    plt.subplots_adjust(hspace=0.01, wspace=0.15)
    
    # Subplots
    temp_a = plot_func(lon, lat, var_ef - var_norm, levels, cmap, axs[0,0], alpha=0.5, identificador='a')
    temp_b = plot_func(lon, lat, var_lf - var_norm, levels, cmap, axs[0,1], alpha=0.5, identificador='b')
    temp_c = plot_func(lon, lat, var_lq - var_norm, levels, cmap, axs[1,0], alpha=0.5, identificador='c')
    temp_d = plot_func(lon, lat, var_eq - var_norm, levels, cmap, axs[1,1], alpha=0.5, identificador='d')
    
    # Labels centrais
    fig.text(0.45, 0.05, "Longitude", ha='center', fontsize=20)
    fig.text(0.045, 0.5, "Latitude", va='center', rotation='vertical', fontsize=20)
    
    # Barra de cores única
    cbar = fig.colorbar(temp_d, ax=axs, orientation='vertical', shrink=0.5, pad=0.02)
    cbar.set_label(label_cbar, size=15)
    cbar.ax.tick_params(labelsize=12)
    
    # Títulos
    axs[0,0].set_title('a)', loc='left', fontsize=18)
    axs[0,1].set_title('b)', loc='left', fontsize=18)
    axs[1,0].set_title('c)', loc='left', fontsize=18)
    axs[1,1].set_title('d)', loc='left', fontsize=18)
    
    # Ajustar layout e salvar
    fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.2, hspace=0.1, wspace=0.2)
    plt.savefig(nome_arquivo, dpi=360, bbox_inches="tight")
    plt.close()
    
def plotar_anomalias_percentis_calculados_vetorial(var_ef_u,   var_lf_u,   var_lq_u,   var_eq_u,   var_norm_u, 
                                                   var_ef_v,   var_lf_v,   var_lq_v,   var_eq_v,   var_norm_v,
                                                   var_ef_div, var_lf_div, var_lq_div, var_eq_div, var_norm_div, 
                                          lon, lat, label, cmap, spd_def, nome_arquivo,plot_func):
    # Configurar figura e subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    plt.subplots_adjust(hspace=0.01, wspace=0.15)

    # Subplots
    temp_a = plot_func(lon, lat, 
                       var_ef_u - var_norm_u, 
                       var_ef_v - var_norm_v,
                       var_ef_div - var_norm_div, 
                       label, cmap, spd_def, axs[0,0], identificador='a')
    
    temp_b = plot_func(lon, lat,  
                       var_lf_u - var_norm_u, 
                       var_lf_v - var_norm_v,
                       var_lf_div - var_norm_div, 
                       label, cmap, spd_def, axs[0,1], identificador='b')
    temp_c = plot_func(lon, lat,  
                       var_lq_u - var_norm_u, 
                       var_lq_v - var_norm_v,
                       var_lq_div - var_norm_div, 
                       label, cmap, spd_def, axs[1,0], identificador='c')
    temp_d = plot_func(lon, lat, 
                       var_eq_u - var_norm_u, 
                       var_eq_v - var_norm_v,
                       var_eq_div - var_norm_div, 
                      label, cmap, spd_def, axs[1,1], identificador='d')
    
    # Labels centrais
    fig.text(0.45, 0.1, "Longitude", ha='center', fontsize=20)
    fig.text(0.05, 0.5, "Latitude", va='center', rotation='vertical', fontsize=20)
    
    # Barra de cores única
    cbar = fig.colorbar(temp_d, ax=axs, orientation='vertical', shrink=0.5, pad=0.02)
    cbar.set_label(label, size=15)
    cbar.ax.tick_params(labelsize=12)
    
    # Títulos
    axs[0,0].set_title('a)', loc='left', fontsize=18)
    axs[0,1].set_title('b)', loc='left', fontsize=18)
    axs[1,0].set_title('c)', loc='left', fontsize=18)
    axs[1,1].set_title('d)', loc='left', fontsize=18)
    
    # Ajustar layout e salvar
    fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.2, hspace=0.1, wspace=0.2)
    plt.savefig(nome_arquivo, dpi=360, bbox_inches="tight")
    plt.close()
    
def plotar_circulacao_quantis(te, w, u, lvl, lon, resultado, nome_figura):
    te10 = np.mean(np.mean(te[resultado['percentis_tsm']['10'],:,:,:],axis=2),axis=0)
    te25 = np.mean(np.mean(te[resultado['percentis_tsm']['25'],:,:,:],axis=2),axis=0)
    te75 = np.mean(np.mean(te[resultado['percentis_tsm']['75'],:,:,:],axis=2),axis=0)
    te90 = np.mean(np.mean(te[resultado['percentis_tsm']['90'],:,:,:],axis=2),axis=0)
    teno = np.mean(np.mean(te[resultado['percentis_tsm']['normal'],:,:,:],axis=2),axis=0)
    
    w10 = np.mean(np.mean(w[resultado['percentis_tsm']['10'],:,:,:],axis=2),axis=0)
    w25 = np.mean(np.mean(w[resultado['percentis_tsm']['25'],:,:,:],axis=2),axis=0)
    w75 = np.mean(np.mean(w[resultado['percentis_tsm']['75'],:,:,:],axis=2),axis=0)
    w90 = np.mean(np.mean(w[resultado['percentis_tsm']['90'],:,:,:],axis=2),axis=0)
    wno = np.mean(np.mean(w[resultado['percentis_tsm']['normal'],:,:,:],axis=2),axis=0)
    
    u10 = np.mean(np.mean(u[resultado['percentis_tsm']['10'],:,:,:],axis=2),axis=0)
    u25 = np.mean(np.mean(u[resultado['percentis_tsm']['25'],:,:,:],axis=2),axis=0)
    u75 = np.mean(np.mean(u[resultado['percentis_tsm']['75'],:,:,:],axis=2),axis=0)
    u90 = np.mean(np.mean(u[resultado['percentis_tsm']['90'],:,:,:],axis=2),axis=0)
    uno = np.mean(np.mean(u[resultado['percentis_tsm']['normal'],:,:,:],axis=2),axis=0)
    
    [long_,press] = np.meshgrid(lon[0,:],lvl) 
    
    # Crie uma grade igualmente espaçada ao longo do eixo Y
    y_equal_spacing = np.linspace(100, 1000, 60)  # Ajuste os valores mínimo e máximo conforme necessário
    
    # Configurar o tamanho da figura e criar subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    plt.subplots_adjust(hspace=0.01, wspace=0.15)  # Ajuste de espaço horizontal e vertical
    
    #plt.subplots_adjust(hspace=0.05, wspace=0.3)  # Ajuste de espaço horizontal e vertical
    
    # Subplot a) - Topo Esquerda (só Y)
    temp_a = sawp.plotar_circulacao(u10-uno, w10-wno, te10-teno, lvl, y_equal_spacing, long_,
                               'a)', ax=axs[0, 0], identificador='a')
    
    # Subplot b) - Topo Direita (vazio)
    temp_b = sawp.plotar_circulacao(u25-uno, w25-wno, te25-teno, lvl, y_equal_spacing, long_,
                               'b)', ax=axs[0, 1], identificador='b')
    
    # Subplot c) - Baixo Esquerda (ambos X e Y)
    temp_c = sawp.plotar_circulacao(u75-uno, w75-wno, te75-teno, lvl, y_equal_spacing, long_,
                               'c)', ax=axs[1, 0], identificador='c')
    
    # Subplot d) - Baixo Direita (só X)
    temp_d = sawp.plotar_circulacao(u90-uno, w90-wno, te90-teno, lvl, y_equal_spacing, long_,
                               'd)', ax=axs[1, 1], identificador='d')
    
    # Configurar labels centrais se necessário (opcional, como nos exemplos anteriores)
    # Por exemplo:
    fig.text(0.41, -0.03, "Longitude", ha='center', fontsize=20)
    fig.text(-0.03, 0.5, "Pressure Levels (hPa)", va='center', rotation='vertical', fontsize=20)
    
    # Criar uma barra de cores única para todos os subplots
    # Usando temp_d (poderia ser qualquer um dos subplots já que os níveis são iguais)
    cbar = fig.colorbar(temp_d, ax=axs, orientation='vertical', shrink=0.5, pad=-.4)
    cbar.set_label('Equivalent Potential Temperature (K)', size=16)
    cbar.ax.tick_params(labelsize=14)
    
    # Adicionar títulos em cada subplot
    axs[0, 0].set_title('a)', loc='left', fontsize=18)
    axs[0, 1].set_title('b)', loc='left', fontsize=18)
    axs[1, 0].set_title('c)', loc='left', fontsize=18)
    axs[1, 1].set_title('d)', loc='left', fontsize=18)
    
    # Ajustar layout para evitar sobreposição de elementos
    plt.tight_layout()
    plt.savefig(nome_figura, dpi=360, bbox_inches="tight")
    plt.close()

