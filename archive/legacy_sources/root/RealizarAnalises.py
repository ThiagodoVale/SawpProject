#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de processamento de dados SST/TSM
"""

import sawp
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
from scipy import signal
from scipy.stats import linregress, pearsonr
import pickle

# -----------------------------
# Configurações iniciais
# -----------------------------
tendencias = 'não'  # 'sim' ou 'não'
caminho_ = './outputs/' if tendencias == 'não' else './outputs_trended/'

lag_chuva = 'não'
lagc = 0
chuva_ou_anomalia = 'prec anomalia'
tsm_ou_anomalia = 'TSM Anomalia'
anomalias_chuva_tsm = 'sim'

DicionarioAreaEstudo = {'SAWP' : [ -34, -25, -12.5, -4], 
                        'NEB'  : [ -50, -34, -15, 0]}

# Criando dicionário de configuração dos mapas a serem plotados.
dicionario_configuracoes   = { "TSM" : [np.asarray(range(-60,61,10))/100,
                          'bwr',
                          ' SST (ºC)'],
                              
                "ACPC" : [np.asarray(range(-30,31,5))/10,
                          'BrBG',
                          'Mid-level Precipitable Water (mm)'],
                
                "TRSP" : ['89',
                          'k',
                          'Moisture flux convergence (kg m⁻² s⁻¹)'],
                "LH": [
                            np.asarray(range(-30,31,5))/10,
                            'bwr',
                            'Latent Heat Flux (W m$^{-2}$)'
                        ],
                        
                        "SH": [
                            np.asarray(range(-30,31,5))/10,
                            'bwr',
                            'Sensible Heat Flux (W m$^{-2}$)'
                        ],
                }

# -----------------------------
# Carregamento de dados
# -----------------------------
# Configura tempo e área (retorna tempo, lon e lat)
periodo_tempo_convertido, lon, lat = sawp.ConfigurarTempoArea('./data/Era5_Dataset_sst.nc')

# Arquivo TSM/SST dependendo da tendência
tsm_arquivo = 'detrended_sst.npz' if tendencias == 'não' else 'trended_uvq_sst.npz'

# Carregar dados usando função genérica
tsm = sawp.CarregarDados(
    caminho_ + tsm_arquivo,
    tendencias=tendencias,
    nome_var_nc='sst',
    substituir_nan=-999
)

print('Final da exportação dos dados de TSM do ERA5')

# Definindo limites da área SAWP
limites = DicionarioAreaEstudo['SAWP']

lloW = np.argmin(np.abs(lon[0,:] - limites[0]))
lloE = np.argmin(np.abs(lon[0,:] - limites[1]))
llaS = np.argmin(np.abs(lat[:,0] - limites[2]))
llaN = np.argmin(np.abs(lat[:,0] - limites[3]))

# Chamada da função
resultado = sawp.ProcessarTsmv2(tsm, periodo_tempo_convertido, lloW, lloE, llaS, llaN)
#with open('ResultadosSawpPercentis.pkl', 'wb') as f:
#    pickle.dump(resultado, f)
    
    
serie_tsm = resultado['serie_tsm']
print('Final do calculo dos quantis')

# -----------------------------
# Tratamento dos dados de TSM
# -----------------------------
sawp.plotar_anomalias_percentis(tsm, resultado, lon, lat, dicionario_configuracoes['TSM'][0], 
                           dicionario_configuracoes['TSM'][1],
                           'SST (ºC)', 'TSM_AnomaliasQuantis_Subplot.png',
                           sawp.MatrizMedia2dEra5, sawp.plotar_grafico_sombreado)

print('Final das confecções de mapas de TSM')

# -------------------------------------------------
# Calor latente e calor sensível
# -------------------------------------------------

if tendencias == 'não':
    lh_arquivo = 'detrended_uvq_lh.npz'
else:
    lh_arquivo = './data/Era5_Dataset_lhsh.nc'


lh = sawp.CarregarDados(
    caminho_ + lh_arquivo,
    tendencias=tendencias,
    nome_var_nc='lh',       # ajustar para o nome real da variável
    substituir_nan=-999
)


print('Final da exportação dos dados de LH e SH do ERA5')

# -----------------------------
# Tratamento do Calor Latente
# -----------------------------

sawp.plotar_anomalias_percentis(
    lh,
    resultado,
    lon,
    lat,
    np.asarray(range(-10,11,2))/10000000,
    'bwr',
    'Latent Heat Flux (W m$^{-2}$)',
    'LH_AnomaliasQuantis_Subplot.png',
    sawp.MatrizMedia2dEra5,
    sawp.plotar_grafico_sombreado
)

print('Final das confecções de mapas de LH')




print('Final das confecções de mapas de SH')
# -----------------------------
# Tratamento dos dados de Vapor de Água
# -----------------------------

if tendencias == 'não':
    print('Carregando dados sem tendencias  (u, v, q)')

    u_arq = "./outputs/detrended_uvq_u.npz"
    v_arq = "./outputs/detrended_uvq_v.npz"
    q_arq = "./outputs/detrended_uvq_q.npz"

else:
    print('Carregando dados com tendencias  (u, v, q)')

    u_arq = "./outputs_trended/trended_uvq_u.npz"
    v_arq = "./outputs_trended/trended_uvq_v.npz"
    q_arq = "./outputs_trended/trended_uvq_q.npz"

level = np.array(range(6,14)) # Referente aos médios níveis [850 a 400]
resultado_tva_ml = sawp.processarAguaPrecipitavel(u_arq, v_arq, q_arq, lon, lat, resultado,level)

level = np.array(range(0,7)) # Referente aos baixos níveis [1000 a 850]
resultado_tva_bl = sawp.processarAguaPrecipitavel(u_arq, v_arq, q_arq, lon, lat, resultado,level)

var_ef = np.mean(resultado_tva_ml['10']['ap'],axis=0)
var_lf = np.mean(resultado_tva_ml['25']['ap'],axis=0)
var_lq = np.mean(resultado_tva_ml['75']['ap'],axis=0)
var_eq = np.mean(resultado_tva_ml['90']['ap'],axis=0)
var_norm = np.mean(resultado_tva_ml['normal']['ap'],axis=0)

sawp.plotar_anomalias_percentis_calculados(var_ef,var_lf,var_lq,var_eq,var_norm, lon, lat, dicionario_configuracoes['ACPC'][0], 
                           dicionario_configuracoes['ACPC'][1],
                           'Water Vapor Content (mm)', 'ACPC_AnomaliasQuantis_Subplot.png',
                           sawp.plotar_grafico_sombreado)

print('Final das confecções de mapas de Água Precipitável')

# -----------------------------
# Tratamento dos dados de Transporte de Vapor de Água
# -----------------------------

var_ef_u = np.mean(resultado_tva_ml['10']['tvau'],axis=0)
var_lf_u = np.mean(resultado_tva_ml['25']['tvau'],axis=0)
var_lq_u = np.mean(resultado_tva_ml['75']['tvau'],axis=0)
var_eq_u = np.mean(resultado_tva_ml['90']['tvau'],axis=0)
var_norm_u = np.mean(resultado_tva_ml['normal']['tvau'],axis=0)

var_ef_v = np.mean(resultado_tva_ml['10']['tvav'],axis=0)
var_lf_v = np.mean(resultado_tva_ml['25']['tvav'],axis=0)
var_lq_v = np.mean(resultado_tva_ml['75']['tvav'],axis=0)
var_eq_v = np.mean(resultado_tva_ml['90']['tvav'],axis=0)
var_norm_v = np.mean(resultado_tva_ml['normal']['tvav'],axis=0)

var_ef_div = np.mean(resultado_tva_bl['10']['divtva'],axis=0)
var_lf_div = np.mean(resultado_tva_bl['25']['divtva'],axis=0)
var_lq_div = np.mean(resultado_tva_bl['75']['divtva'],axis=0)
var_eq_div = np.mean(resultado_tva_bl['90']['divtva'],axis=0)
var_norm_div = np.mean(resultado_tva_bl['normal']['divtva'],axis=0)

sawp.plotar_anomalias_percentis_calculados_vetorial(var_ef_u,var_lf_u,var_lq_u,var_eq_u,var_norm_u, 
                                           var_ef_v,var_lf_v,var_lq_v,var_eq_v,var_norm_v,
                                           var_ef_div,var_lf_div,var_lq_div,var_eq_div,var_norm_div,
                                           lon, lat, dicionario_configuracoes['TRSP'][2], 
                           dicionario_configuracoes['TRSP'][1],
                           'Water Vapor Transport (mm. m/s)', 'TRSP_AnomaliasQuantis_Subplot.png',
                           sawp.plotar_grafico_sombreado_vetorial)

print('Final das confecções de mapas de Transporte de Umidade e convergência em baixos níveis')

# --------------------------
# BLOCO 1 — Leitura e Pré-processamento
# --------------------------

chuva_periodo_chuvoso = sawp.carregar_chuva_periodo('./data/EstacoesDados.txt', tendencias)
# =============================================================================
# chuva_periodo_chuvoso = chuva_periodo_chuvoso.mask(
#     chuva_periodo_chuvoso < 0,
#     np.nan
# )
# =============================================================================

# Normalizações
chuva_norm = (
    chuva_periodo_chuvoso - chuva_periodo_chuvoso.mean()
) / chuva_periodo_chuvoso.std()

tsm_norm = resultado['media_tsm_anual']['Anomalia'] / resultado['media_tsm_anual']['Anomalia'].std()

#print(chuva_norm.loc[2010:2023])
#print(tsm_norm.loc[2010:2023])

dados = pd.concat(
    [tsm_norm.rename("tsm"), chuva_norm.rename("chuva")],
    axis=1
)

#print(dados.loc[2010:2023])
#print(dados.loc[2010:2023].isna().sum())

# --------------------------
# BLOCO 2 — Classificação ENSO e Regressão
# --------------------------
dados = pd.concat([tsm_norm, chuva_norm], axis=1).dropna()

tsm_norm = dados.iloc[:, 0]
chuva_norm = dados.iloc[:, 1]


resultado_reg = sawp.analise_tsm_regressao(tsm_norm, chuva_norm)

slope = resultado_reg['slope']
intercept = resultado_reg['intercept']
r_value = resultado_reg['r_value']
p_value = resultado_reg['p_value']
std_err = resultado_reg['std_err']

t_q10, t_q90, t_q25, t_q75, t_qno = sawp.ExtrairVariaveisQuantis(resultado_reg,tsm_norm,tsm_norm)
c_q10, c_q90, c_q25, c_q75, c_qno = sawp.ExtrairVariaveisQuantis(resultado_reg,chuva_norm,tsm_norm)

anos_chuva = chuva_norm.index.astype(int)

corremap_ext, pmap_ext, sign_ext, anos_ext = sawp.correlacao_tsm_chuva(
    tsm,
    chuva_norm.values.squeeze(),
    ano_inicial=1959,
    anos_chuva=anos_chuva,
    usar_extremos=True)

sawp.plot_mapa_correlacao_scatter(
    lon, lat,
    corremap_ext, sign_ext,
    t_q10, t_q25, t_qno, t_q75, t_q90,
    c_q10, c_q25, c_qno, c_q75, c_q90,
    slope, intercept, r_value,
    chuva_norm,
    tsm,
    output_file='Combined_Prec_TS_Subplot.png'
)

print('Final das confecções de mapas de Correlações TSM x Chuva ENEB')

if tendencias == 'não':
    print('Carregando dados sem tendencias (te, w)')

    te_arq = "./outputs/detrended_uvq_te.npz"
    w_arq = "./outputs/detrended_uvq_w.npz"
    u_arq = "./outputs/detrended_uvq_u.npz"


else:
    print('Carregando dados com tendencia (te, w)')

    te_arq = "./outputs_trended/trended_uvq_te.npz"
    w_arq = "./outputs_trended/trended_uvq_w.npz"
    u_arq = "./outputs_trended/trended_uvq_u.npz"
    
# Achando as localizações de latitude para realizar média
lat_min, lat_max = -12.5, -4
idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
print(idx[0], idx[-1])

te = np.load(te_arq, mmap_mode='r')['data'][:,:,idx[0]:idx[-1],:]
w = np.load(w_arq, mmap_mode='r')['data'][:,:,idx[0]:idx[-1],:]
u = np.load(u_arq, mmap_mode='r')['data'][:,:,idx[0]:idx[-1],:]

lvl_te = np.load(te_arq, mmap_mode='r')['level']
lvl_w = np.load(w_arq, mmap_mode='r')['level']
lvl_u = np.load(u_arq, mmap_mode='r')['level']

if (lvl_te[0] != lvl_w[0]) | (lvl_te[0] != lvl_u[0]) | (lvl_u[0] != lvl_w[0]):
    
    variaveis = [ w, u]
    niveis = [ lvl_w, lvl_u]
    print(lvl_u)

    variaveis_corrigidas, niveis_corrigidos = sawp.alinhar_niveis(variaveis, niveis, ordem='crescente')#, ref_index=0)
    
    w, u = variaveis_corrigidas
    lvl_w, lvl_u = niveis_corrigidos

import sawp
sawp.plotar_circulacao_quantis(te, w, u, lvl_u, lon, resultado, 'VelVertEPT_Subplot.png')
print('Final das confecções de mapas de Quantis de Circulação')

# -----------------------
# - Já tenho sem tendencias algumas das variáveis que necessito
# - Preciso fazer algumas poucas modificações para as que tem tendencia
# -----------------------
# -----------------------------
# Configurações iniciais
# -----------------------------
tendencias = 'sim'  # 'sim' ou 'não'
caminho_ = './outputs/' if tendencias == 'não' else './outputs_trended/'

# Arquivo TSM/SST dependendo da tendência
tsm_arquivo = 'detrended_sst.npz' if tendencias == 'não' else 'trended_uvq_sst.npz'

# Carregar dados com tendencia
tsm_ct = sawp.CarregarDados(
    caminho_ + tsm_arquivo,
    tendencias=tendencias,
    nome_var_nc='sst',
    substituir_nan=-999
)

# Chamada da função
resultado_ct_ = sawp.ProcessarTsmv2(tsm_ct, periodo_tempo_convertido, lloW, lloE, llaS, llaN)
resultado_ct = resultado_ct_.copy()
resultado_ct['percentis_tsm'] = resultado_ct['percentis_tsm'].copy()

for p in ['10', '25', '75','normal']:
    resultado_ct['percentis_tsm'].pop(p, None)
    
serie_tsm_ct = resultado_ct['serie_tsm']
print('Final do calculo dos quantis')

tsm_st = np.mean(tsm[resultado['percentis_tsm']['90']],axis=0)
tsm_ct = np.mean(tsm_ct[resultado_ct['percentis_tsm']['90']],axis=0)
diff_tsm = tsm_ct - tsm_st
print('Final do calculo diferença percentil 90 para TSM')

u_arq_ct = "./outputs_trended/trended_uvq_u.npz"
v_arq_ct = "./outputs_trended/trended_uvq_v.npz"
q_arq_ct = "./outputs_trended/trended_uvq_q.npz"

level = np.array(range(6,14)) # Referente aos médios níveis [850 a 400]
resultado_tva_ml_ct = sawp.processarAguaPrecipitavel(u_arq_ct, v_arq_ct, q_arq_ct, lon, lat, resultado_ct,level)

level = np.array(range(0,7)) # Referente aos baixos níveis [1000 a 850]
resultado_tva_bl_ct = sawp.processarAguaPrecipitavel(u_arq_ct, v_arq_ct, q_arq_ct, lon, lat, resultado_ct,level)

ap_eq_ct = np.mean(resultado_tva_ml_ct['90']['ap'],axis=0)
ap_eq_st = np.mean(resultado_tva_ml['90']['ap'],axis=0)
diff_ap = ap_eq_ct - ap_eq_st
print('Final do calculo diferença percentil 90 para Agua Precipitável')

tvau_eq_u_ct = np.mean(resultado_tva_ml_ct['90']['tvau'],axis=0)
tvau_eq_u_st = np.mean(resultado_tva_ml['90']['tvau'],axis=0)
diff_tvau = tvau_eq_u_ct - tvau_eq_u_st
print('Final do calculo diferença percentil 90 para Transporte U')

tvav_eq_v_ct = np.mean(resultado_tva_ml_ct['90']['tvav'],axis=0)
tvav_eq_v_st = np.mean(resultado_tva_ml['90']['tvav'],axis=0)
diff_tvav = tvav_eq_v_ct - tvav_eq_v_st
print('Final do calculo diferença percentil 90 para Transporte V')

divtva_eq_div_ct = np.mean(resultado_tva_bl_ct['90']['divtva'],axis=0)
divtva_eq_div_st = np.mean(resultado_tva_bl['90']['divtva'],axis=0)
diff_divtva = divtva_eq_div_ct - divtva_eq_div_st
print('Final do calculo diferença percentil 90 para Divergência')

te_arq_ct = "./outputs_trended/trended_uvq_te.npz"
w_arq_ct = "./outputs_trended/trended_uvq_w.npz"
u_arq_ct = "./outputs_trended/trended_uvq_u.npz"
    
te_ct = np.load(te_arq_ct, mmap_mode='r')['data'][:,:,idx[0]:idx[-1],:]
w_ct = np.load(w_arq_ct, mmap_mode='r')['data'][:,:,idx[0]:idx[-1],:]
u_ct = np.load(u_arq_ct, mmap_mode='r')['data'][:,:,idx[0]:idx[-1],:]

lvl_te_ct = np.load(te_arq, mmap_mode='r')['level']
lvl_w_ct = np.load(w_arq, mmap_mode='r')['level']
lvl_u_ct = np.load(u_arq, mmap_mode='r')['level']

if (lvl_te_ct[0] != lvl_w_ct[0]) | (lvl_te_ct[0] != lvl_u_ct[0]) | (lvl_u_ct[0] != lvl_w_ct[0]):
    
    variaveis = [w_ct, u_ct]
    niveis = [lvl_w_ct, lvl_u_ct]
    print(lvl_u_ct)
    variaveis_corrigidas, niveis_corrigidos = sawp.alinhar_niveis(variaveis, niveis, ordem='crescente')#, ref_index=0)
    
    w_ct, u_ct = variaveis_corrigidas#, w_ct, u_ct = variaveis_corrigidas
    lvl_w_ct, lvl_u_ct = niveis_corrigidos #, lvl_w_ct, lvl_u_ct = niveis_corrigidos
    
lat_min, lat_max = -12.5, -4
idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
print(idx[0], idx[-1])

te_ct = np.mean(np.mean(te_ct[resultado_ct['percentis_tsm']['90'],:,:,:],axis=0),axis=1)
te_st = np.mean(np.mean(te[resultado['percentis_tsm']['90'],:,:,:],axis=0),axis=1)
diff_te = te_ct - te_st
print('Final do calculo diferença percentil 90 para TPE longitudinal')

w_ct   = np.mean(np.mean(w_ct[resultado_ct['percentis_tsm']['90'],:,:,:],axis=0),axis=1)
w_st   = np.mean(np.mean(w[resultado['percentis_tsm']['90'],:,:,:],axis=0),axis=1)
diff_w = w_ct - w_st
print('Final do calculo diferença percentil 90 para w longitudinal')

u_ct   = np.mean(np.mean(u_ct[resultado_ct['percentis_tsm']['90'],:,:,:],axis=0),axis=1)
u_st   = np.mean(np.mean(u[resultado['percentis_tsm']['90'],:,:,:],axis=0),axis=1)
diff_u = u_ct - u_st
print('Final do calculo diferença percentil 90 para u longitudinal')

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 2)  # Cria um grid 2x2 para os subplots

# Subplot a) com projeção
ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
# Subplot b) com projeção
ax_b = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
# Subplot c) com projeção
ax_c = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
# Subplot d) SEM projeção
ax_d = fig.add_subplot(gs[1, 1])

plt.subplots_adjust(hspace=0.1, wspace=0.15)

# a) Diferença de TSM
levels_tsm = np.asarray(range(-5,6,1))/10
cmap_tsm   = dicionario_configuracoes['TSM'][1]
label_tsm  = dicionario_configuracoes['TSM'][2]

temp_a = sawp.plotar_grafico_sombreado(lon, lat, diff_tsm, levels_tsm, cmap_tsm, ax_a, alpha=0.7, identificador='a')
ax_a.set_title('a)', loc='left', fontsize=18)

# Cria eixo interno para o colorbar de a)
cax_a = inset_axes(ax_a,
                   width="5%",      # Largura do colorbar em relação ao eixo
                   height="30%",    # Altura do colorbar em relação ao eixo
                   loc='lower right', 
                   bbox_to_anchor=(-0.15,0.0,1,1),  # Ancorado no canto inferior direito do próprio eixo
                   bbox_transform=ax_a.transAxes,
                   borderpad=0.5)

cbar_a = plt.colorbar(temp_a, cax=cax_a, orientation='vertical')
cbar_a.ax.tick_params(labelsize=8)
cbar_a.ax.set_title('SST (ºC)', fontsize=8)

# b) Diferença de Água Precipitável
levels_acpc = levels_tsm = np.asarray(range(-10,11,2))/10
cmap_acpc   = dicionario_configuracoes["ACPC"][1]
label_acpc  = dicionario_configuracoes["ACPC"][2]

temp_b = sawp.plotar_grafico_sombreado(lon, lat, diff_ap, levels_acpc, 'BrBG', ax_b, alpha=0.7, identificador='b')
ax_b.set_title('b)', loc='left', fontsize=18)

# Eixo interno para o colorbar de b)
cax_b = inset_axes(ax_b,
                   width="5%",  
                   height="30%",
                   loc='lower right',
                   bbox_to_anchor=(-0.15,0.0,1,1),
                   bbox_transform=ax_b.transAxes,
                   borderpad=0.5)

cbar_b = plt.colorbar(temp_b, cax=cax_b, orientation='vertical')
cbar_b.ax.tick_params(labelsize=8)
cbar_b.ax.set_title('Prec. Water (mm)', fontsize=8)

# c) Diferença de Divergência e Transporte
spd_def = np.sqrt(diff_tvau**2 + diff_tvav**2)  # velocidade resultante

temp_c = sawp.plotar_grafico_sombreado_vetorial(lon, lat, diff_tvau, diff_tvav, diff_divtva, 
                                           label='Divergência', cmap='PuOr_r', spd_def=spd_def, 
                                           ax=ax_c, identificador='c')

# Eixo interno para o colorbar de c)
cax_c = inset_axes(ax_c,
                   width="5%",
                   height="30%",
                   loc='lower right',
                   bbox_to_anchor=(-0.15,0.0,1,1),
                   bbox_transform=ax_c.transAxes,
                   borderpad=0.5)

cbar_c = plt.colorbar(temp_c, cax=cax_c, orientation='vertical')
cbar_c.ax.tick_params(labelsize=8)
cbar_c.ax.set_title('Mois. flux conv. (kg m⁻² s⁻¹)',fontsize=8)

# d) Diferença de Circulação SEM projeção
lvl = np.asarray(range(-10,11,2))/10
cmap = 'RdYlBu_r'

[long_,press] = np.meshgrid(lon[0,:],lvl_te) 

# Crie uma grade igualmente espaçada ao longo do eixo Y
y_equal_spacing = np.linspace(100, 1000, 60)  # Ajuste os valores mínimo e máximo conforme necessário

# Subplot d) - Baixo Direita (só X)
contour_d = sawp.plotar_circulacao_diff(diff_u, diff_w, diff_te, lvl_te, y_equal_spacing, long_,
                           'd)', ax=ax_d, identificador='d')
#contour_d = sawp.plotar_circulacao(diff_u, diff_w, diff_te, lvl, cmap, ax=ax_d, identificador='d')

# Eixo interno para o colorbar de d)
cax_d = inset_axes(ax_d,
                   width="5%",
                   height="30%",
                   loc='lower right',
                   bbox_to_anchor=(-0.15,0.0,1,1),
                   bbox_transform=ax_d.transAxes,
                   borderpad=0.5)

# eixo Y à direita
ax_d.yaxis.tick_right()
ax_d.yaxis.set_label_position("right")

# mover eixo Y para a direita
ax_d.yaxis.tick_right()
ax_d.yaxis.set_label_position("right")

ax_d.tick_params(
    axis='y',
    left=False,
    labelleft=False,
    right=True,
    labelright=True
)

cbar_d = plt.colorbar(contour_d, cax=cax_d, orientation='vertical')
cbar_d.ax.tick_params(labelsize=8)
cbar_d.ax.set_title(r' $\theta_e$ (K)',fontsize=8)

# Configurar labels centrais para Latitude e Longitude
fig.text(0.5, 0.05, "Longitude", ha='center', fontsize=20)
fig.text(0.05, 0.5, "Latitude", va='center', rotation='vertical', fontsize=20)

# ------------------------------------------------------------
# Forçar eixo Y do painel d à direita
# ------------------------------------------------------------

ax_d.set_yticks([1000, 850, 700, 500, 300, 200])
ax_d.set_ylim(1000, 200)

ax_d.yaxis.set_ticks_position("right")
ax_d.yaxis.set_label_position("right")

ax_d.tick_params(
    axis="y",
    which="both",
    left=False,
    labelleft=False,
    right=True,
    labelright=True,
    labelsize=12
)

ax_d.spines["left"].set_visible(False)
ax_d.spines["right"].set_visible(True)

ax_d.set_ylabel("Pressure (hPa)", fontsize=14, labelpad=10)

plt.savefig('Subplot_CT-ST.png', dpi=300, bbox_inches="tight")

plt.show()
plt.close()