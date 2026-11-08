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
import gc
import struct
import zipfile
from scipy import stats
from metpy.calc import divergence, lat_lon_grid_deltas

# -----------------------------
# Configurações iniciais
# -----------------------------
tendencias = 'sim'  # 'sim' ou 'não'
caminho_ = './outputs/' if tendencias == 'não' else './outputs_trended/'

lag_chuva = 'não'
lagc = 0
chuva_ou_anomalia = 'prec anomalia'
tsm_ou_anomalia = 'TSM Anomalia'
anomalias_chuva_tsm = 'sim'
metodo_sawp = 'isoterma'

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
                          'Moisture flux convergence (kg m⁻² s⁻¹)']
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

# ============================================================
# DEFINIÇÃO DA SAWP
# ============================================================
# Opções:
# 'quadrado'  -> usa a área fixa original
# 'isoterma'  -> usa apenas os pontos com climatologia de abril > 28.5 °C


limites = DicionarioAreaEstudo['SAWP']

lloW = np.argmin(np.abs(lon[0, :] - limites[0]))
lloE = np.argmin(np.abs(lon[0, :] - limites[1]))
llaS = np.argmin(np.abs(lat[:, 0] - limites[2]))
llaN = np.argmin(np.abs(lat[:, 0] - limites[3]))

if metodo_sawp == 'quadrado':

    resultado = sawp.ProcessarTsmv2(
        tsm,
        periodo_tempo_convertido,
        lloW, lloE,
        llaS, llaN
    )

elif metodo_sawp == 'isoterma':

    tempo = pd.to_datetime(periodo_tempo_convertido)

    # Climatologia de abril
    # Ajuste o período se o revisor pedir outro intervalo climatológico
    clim_ini = 1980
    clim_fim = 2010

    mask_abril_clim = (
        (tempo.month == 4) &
        (tempo.year >= clim_ini) &
        (tempo.year <= clim_fim) 
    )

    clim_abril = np.nanmean(tsm[mask_abril_clim, :, :], axis=0) - 273.16

    # Máscara da SAWP: somente pontos acima de 28.5 °C
    mascara_sawp = clim_abril >= 28.5
    
    # Limite espacial adicional
    # Se você quer limitar ATÉ longitude -25:
    mascara_sawp = mascara_sawp & (lon <= -25)  & (lon >= -40)

    # Restringe a máscara ao domínio original aproximado da SAWP
    #mascara_limites = np.zeros_like(mascara_sawp, dtype=bool)
    #mascara_limites[llaS:llaN+1, lloW:lloE+1] = True

   # mascara_sawp = mascara_sawp & mascara_limites

    print("Pontos na SAWP por isoterma:", np.sum(mascara_sawp))
    
    # Aplica máscara: pontos fora da isoterma viram NaN
    tsm_sawp_masked = np.where(
        mascara_sawp[None, :, :],
        tsm,
        np.nan
    )

    # Usa o menor retângulo que contém a máscara,
    # mas a média será feita apenas nos pontos válidos da isoterma
    idx_lat, idx_lon = np.where(mascara_sawp)

    llaS_iso = idx_lat.min()
    llaN_iso = idx_lat.max()
    lloW_iso = idx_lon.min()
    lloE_iso = idx_lon.max()

    resultado = sawp.ProcessarTsmv2(
        tsm_sawp_masked,
        periodo_tempo_convertido,
        lloW_iso, lloE_iso,
        llaN_iso, llaS_iso
    )

else:
    raise ValueError("metodo_sawp deve ser 'quadrado' ou 'isoterma'")
        
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
    anos_chuva,
    usar_extremos=True
)

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

sawp.plotar_circulacao_quantis(te, w, u, lvl_u, lon, resultado, 'VelVertEPT_Subplot.png')
print('Final das confecções de mapas de Quantis de Circulação')

# ============================================================
# BLOCO FINAL — Tendências AMJJ ponto a ponto
# ============================================================
# Este bloco substitui as abordagens baseadas em:
#     - máximo anual AMJJ;
#     - P90 anual;
#     - frequência acima do P90 histórico.
#
# Agora o diagnóstico é feito apenas com tendência temporal direta.
# Para cada ponto de grade, calcula-se uma série anual AMJJ e depois
# a tendência temporal dessa série.
#
# Produto final:
#     Painel a) tendência da TSM média AMJJ
#     Painel b) tendência da água precipitável média AMJJ
#     Painel c) tendência da convergência de umidade média AMJJ
#     Painel d) tendência de θe, u e w em seção longitude × pressão
#
# As escalas de cores são dinâmicas, isto é, definidas a partir do
# próprio campo de tendência, com limites simétricos em torno de zero.
# ============================================================

# -----------------------------
# Configurações da análise
# -----------------------------
MESES_AMJJ = [4, 5, 6, 7]
MIN_ANOS_VALIDOS = 30
ESCALA_DECADA = 10.0
PERCENTIL_ESCALA = 90
NIVEIS_COR = 21

# ------------------------------------------------------------
# Chaveadores independentes das camadas verticais
# ------------------------------------------------------------
# Agora a água precipitável e a convergência de umidade NÃO usam
# obrigatoriamente a mesma camada. Isso evita misturar diagnósticos
# fisicamente distintos:
#   1) Água precipitável: médios níveis, representando umidade na coluna média.
#   2) Convergência de umidade: baixos níveis, representando alimentação dinâmica da chuva.
#
# Opções para cada chaveador:
#   "toda_coluna"     -> usa todos os níveis disponíveis no arquivo;
#   "baixos"          -> usa 1000–850 hPa;
#   "medios"          -> usa 850–400 hPa;
#   "custom_pressao"  -> usa os limites definidos abaixo;
#   "custom_indices"  -> usa os índices definidos abaixo.
CAMADA_AGUA_PRECIPITAVEL = "medios"
CAMADA_CONVERGENCIA = "baixos"

# Usado apenas se CAMADA_AGUA_PRECIPITAVEL = "custom_pressao".
NIVEL_MIN_HPA_AP = 850
NIVEL_MAX_HPA_AP = 400

# Usado apenas se CAMADA_CONVERGENCIA = "custom_pressao".
NIVEL_MIN_HPA_CONV = 1000
NIVEL_MAX_HPA_CONV = 850

# Usado apenas se algum chaveador for "custom_indices".
INDICES_NIVEIS_AP_CUSTOM = np.array([6, 7, 8, 9, 10, 11, 12, 13])
INDICES_NIVEIS_CONV_CUSTOM = np.array([0, 1, 2, 3, 4, 5, 6])


def selecionar_amjj(data, tempo, meses=MESES_AMJJ):
    """Seleciona apenas os meses AMJJ mantendo a ordem temporal."""
    tempo = pd.to_datetime(tempo)
    mask = tempo.month.isin(meses)
    return data[mask, ...], tempo[mask]


def media_anual_amjj(data, tempo, meses=MESES_AMJJ):
    """
    Calcula a média AMJJ de cada ano.

    Entrada:
        data: array com dimensão temporal no eixo 0.

    Saída:
        anos, campo_anual

    Interpretação:
        tendência da condição média do período chuvoso AMJJ,
        não tendência de máximo e não tendência de P90.
    """
    dados_amjj, tempo_amjj = selecionar_amjj(data, tempo, meses)
    anos = np.asarray(sorted(np.unique(tempo_amjj.year)))

    saida = []
    for ano in anos:
        mask_ano = tempo_amjj.year == ano
        saida.append(np.nanmean(dados_amjj[mask_ano, ...], axis=0))

    return anos, np.asarray(saida)


def tendencia_1d(anos, serie, min_validos=MIN_ANOS_VALIDOS):
    """
    Tendência em uma série anual.

    Retorna:
        slope_linear, p_linear, sen_slope, p_mk

    Observação:
        p_mk é estimado por Kendall tau, equivalente prático ao
        Mann-Kendall para tendência monotônica sem sazonalidade.
    """
    anos = np.asarray(anos, dtype=float)
    serie = np.asarray(serie, dtype=float)
    mask = np.isfinite(anos) & np.isfinite(serie)

    if np.sum(mask) < min_validos:
        return np.nan, np.nan, np.nan, np.nan

    x = anos[mask]
    y = serie[mask]

    lin = stats.linregress(x, y)
    sen = stats.theilslopes(y, x, 0.95)
    mk = stats.kendalltau(x, y, nan_policy='omit')

    return lin.slope, lin.pvalue, sen.slope, mk.pvalue


def tendencia_nd(anos, campo_anual, min_validos=MIN_ANOS_VALIDOS):
    """
    Aplica tendência ao longo do eixo year de um campo anual.

    Entrada:
        campo_anual: (year, ...)

    Saída:
        dicionário com slope_linear, p_linear, sen_slope, p_mk.
    """
    campo = np.asarray(campo_anual)
    shape_out = campo.shape[1:]
    flat = campo.reshape(campo.shape[0], -1)

    slope = np.full(flat.shape[1], np.nan, dtype=np.float32)
    p_lin = np.full(flat.shape[1], np.nan, dtype=np.float32)
    sen = np.full(flat.shape[1], np.nan, dtype=np.float32)
    p_mk = np.full(flat.shape[1], np.nan, dtype=np.float32)

    for i in range(flat.shape[1]):
        slope[i], p_lin[i], sen[i], p_mk[i] = tendencia_1d(
            anos,
            flat[:, i],
            min_validos=min_validos
        )

    return {
        'slope_linear': slope.reshape(shape_out),
        'p_linear': p_lin.reshape(shape_out),
        'sen_slope': sen.reshape(shape_out),
        'p_mk': p_mk.reshape(shape_out),
    }


def abrir_campo_npz_mapeado(caminho, campo='data'):
    """Abre um membro NPY sem compressão de um NPZ como memória mapeada.

    ``np.load(..., mmap_mode='r')`` não mapeia membros de NPZ. Os dados ERA5
    deste projeto são armazenados sem compressão no ZIP, portanto podem ser
    lidos diretamente do disco sem alocar aproximadamente 2 GB por variável.
    """
    caminho = str(caminho)
    nome_membro = f'{campo}.npy'
    with zipfile.ZipFile(caminho) as arquivo_zip:
        info = arquivo_zip.getinfo(nome_membro)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(
                f'O membro {nome_membro!r} em {caminho!r} precisa estar sem compressão '
                'para usar memória mapeada.'
            )

        with open(caminho, 'rb') as arquivo:
            arquivo.seek(info.header_offset)
            cabecalho_zip = arquivo.read(30)
            assinatura, *_resto, tamanho_nome, tamanho_extra = struct.unpack(
                '<IHHHHHIIIHH', cabecalho_zip
            )
            if assinatura != 0x04034B50:
                raise ValueError(f'Cabeçalho ZIP inválido em {caminho!r}.')

            inicio_npy = info.header_offset + 30 + tamanho_nome + tamanho_extra
            arquivo.seek(inicio_npy)
            versao = np.lib.format.read_magic(arquivo)
            if versao == (1, 0):
                shape, ordem_fortran, dtype = np.lib.format.read_array_header_1_0(arquivo)
            else:
                shape, ordem_fortran, dtype = np.lib.format.read_array_header_2_0(arquivo)
            inicio_dados = arquivo.tell()

    return np.memmap(
        caminho,
        mode='r',
        dtype=dtype,
        shape=shape,
        order='F' if ordem_fortran else 'C',
        offset=inicio_dados
    )


def ler_metadado_npz(caminho, campo):
    """Lê um metadado pequeno de um arquivo NPZ sem manter o ZIP aberto."""
    with np.load(caminho) as arquivo:
        return np.array(arquivo[campo])


def resumo_espacial(nome, tendencia, mascara=None, fator=ESCALA_DECADA, unidade=''):
    """Resumo simples da tendência espacial usando Sen's slope."""
    campo = tendencia['sen_slope'] * fator

    if mascara is not None:
        campo = np.where(mascara, campo, np.nan)

    return {
        'variavel': nome,
        'sen_slope_mediana': float(np.nanmedian(campo)),
        'sen_slope_media': float(np.nanmean(campo)),
        'unidade': unidade,
        'pontos_validos': int(np.sum(np.isfinite(campo))),
    }


def niveis_dinamicos(campo, n=NIVEIS_COR, percentil=PERCENTIL_ESCALA, minimo=None):
    """
    Cria níveis de cor a partir do próprio campo.

    A escala é simétrica em torno de zero para facilitar interpretação
    de tendências positivas e negativas.
    """
    campo = np.asarray(campo, dtype=float)
    abs_finite = np.abs(campo[np.isfinite(campo)])

    if abs_finite.size == 0:
        vmax = 1.0
    else:
        vmax = np.nanpercentile(abs_finite, percentil)

    if (not np.isfinite(vmax)) or vmax == 0:
        vmax = np.nanmax(abs_finite) if abs_finite.size else 1.0

    if minimo is not None:
        vmax = max(vmax, minimo)

    if (not np.isfinite(vmax)) or vmax == 0:
        vmax = 1.0

    return np.linspace(-vmax, vmax, n)


def selecionar_indices_niveis(
    niveis_hpa,
    camada="medios",
    nivel_min_hpa=None,
    nivel_max_hpa=None,
    indices_custom=None,
):
    """
    Seleciona os níveis verticais usados na integração.

    A seleção é feita por valor de pressão, não pela posição no array,
    exceto quando camada='custom_indices'. Isso evita erro caso os níveis
    estejam em ordem crescente ou decrescente.
    """
    niveis_hpa = np.asarray(niveis_hpa, dtype=float)

    if camada == "toda_coluna":
        idx = np.arange(niveis_hpa.size)

    elif camada == "baixos":
        pmin, pmax = 850.0, 1000.0
        idx = np.where((niveis_hpa >= pmin) & (niveis_hpa <= pmax))[0]

    elif camada == "medios":
        pmin, pmax = 400.0, 850.0
        idx = np.where((niveis_hpa >= pmin) & (niveis_hpa <= pmax))[0]

    elif camada == "custom_pressao":
        if nivel_min_hpa is None or nivel_max_hpa is None:
            raise ValueError(
                "Para camada='custom_pressao', informe nivel_min_hpa e nivel_max_hpa."
            )
        pmin = min(float(nivel_min_hpa), float(nivel_max_hpa))
        pmax = max(float(nivel_min_hpa), float(nivel_max_hpa))
        idx = np.where((niveis_hpa >= pmin) & (niveis_hpa <= pmax))[0]

    elif camada == "custom_indices":
        if indices_custom is None:
            raise ValueError(
                "Para camada='custom_indices', informe indices_custom."
            )
        idx = np.asarray(indices_custom, dtype=int)

    else:
        raise ValueError(
            "camada deve ser 'toda_coluna', 'baixos', "
            "'medios', 'custom_pressao' ou 'custom_indices'."
        )

    idx = np.asarray(idx, dtype=int)

    if idx.size < 2:
        raise ValueError(
            f"A camada {camada!r} selecionou menos de dois níveis. "
            f"Níveis disponíveis: {niveis_hpa}"
        )

    if np.any(idx < 0) or np.any(idx >= niveis_hpa.size):
        raise IndexError(
            f"Índices de níveis inválidos: {idx}. "
            f"Número de níveis disponíveis: {niveis_hpa.size}"
        )

    return idx


def descricao_camada(niveis_hpa, indices_niveis, camada="medios"):
    """Descrição curta da camada vertical para títulos e arquivos."""
    niveis_sel = np.asarray(niveis_hpa)[np.asarray(indices_niveis, dtype=int)]
    pmin = int(np.nanmin(niveis_sel))
    pmax = int(np.nanmax(niveis_sel))

    if camada == "toda_coluna":
        return f"whole column {pmax}-{pmin} hPa"

    return f"{camada} {pmax}-{pmin} hPa"


def integrar_vertical_q(q, niveis_hpa, indices_niveis, tamanho_lote=24):
    """
    Água precipitável integrada em camada de pressão.

    Entrada:
        q: (time, level, lat, lon), em kg/kg
        niveis_hpa: níveis em hPa

    Saída:
        ap: (time, lat, lon), em kg m⁻², numericamente equivalente a mm.
    """
    g = 9.80665
    idx = np.asarray(indices_niveis)
    p_pa = np.asarray(niveis_hpa[idx], dtype=float) * 100.0

    ordem = np.argsort(p_pa)
    p_pa = p_pa[ordem]
    saida = np.empty((q.shape[0], q.shape[2], q.shape[3]), dtype=np.float32)
    for inicio in range(0, q.shape[0], tamanho_lote):
        fim = min(inicio + tamanho_lote, q.shape[0])
        q_layer = np.asarray(q[inicio:fim, idx, :, :], dtype=np.float32)
        q_layer = q_layer[:, ordem, :, :]
        saida[inicio:fim] = np.trapz(q_layer, x=p_pa, axis=1) / g

    return saida


def integrar_vertical_fluxo(u, v, q, niveis_hpa, indices_niveis, tamanho_lote=24):
    """
    Integra verticalmente o fluxo de umidade em uma camada de pressão.

    Retorna:
        Fu = ∫ u q dp/g
        Fv = ∫ v q dp/g

    Entrada esperada:
        u, v, q: (time, level, lat, lon)
        niveis_hpa: níveis em hPa
        indices_niveis: índices da camada desejada
    """
    g = 9.80665
    idx = np.asarray(indices_niveis)
    p_pa = np.asarray(niveis_hpa[idx], dtype=float) * 100.0

    ordem = np.argsort(p_pa)
    p_pa = p_pa[ordem]

    shape_saida = (u.shape[0], u.shape[2], u.shape[3])
    Fu = np.empty(shape_saida, dtype=np.float32)
    Fv = np.empty(shape_saida, dtype=np.float32)

    for inicio in range(0, u.shape[0], tamanho_lote):
        fim = min(inicio + tamanho_lote, u.shape[0])
        u_lote = np.asarray(u[inicio:fim, idx, :, :], dtype=np.float32)[:, ordem, :, :]
        v_lote = np.asarray(v[inicio:fim, idx, :, :], dtype=np.float32)[:, ordem, :, :]
        q_lote = np.asarray(q[inicio:fim, idx, :, :], dtype=np.float32)[:, ordem, :, :]
        Fu[inicio:fim] = np.trapz(u_lote * q_lote, x=p_pa, axis=1) / g
        Fv[inicio:fim] = np.trapz(v_lote * q_lote, x=p_pa, axis=1) / g

    return Fu, Fv


def divergencia_horizontal_fluxo(Fu, Fv, lon2d, lat2d):
    """
    Calcula a divergência horizontal do fluxo integrado de umidade.

    Entrada:
        Fu, Fv: (time, lat, lon)

    Saída:
        div = dFu/dx + dFv/dy

    Observação:
        Usa MetPy, igual ao cálculo de referência do script antigo.
        A convergência física é definida posteriormente como -div.
    """
    lon_1d = lon2d[0, :] if lon2d.ndim == 2 else lon2d
    lat_1d = lat2d[:, 0] if lat2d.ndim == 2 else lat2d

    dx, dy = lat_lon_grid_deltas(lon_1d, lat_1d)

    div = np.empty_like(Fu, dtype=np.float32)
    for tt in range(Fu.shape[0]):
        div[tt, :, :] = divergence(Fu[tt, :, :], Fv[tt, :, :], dx=dx, dy=dy)

    return div


# -----------------------------
# 1) TSM — tendência média AMJJ
# -----------------------------
# As figuras dos quantis já foram concluídas; libera os campos completos antes
# de iniciar os cálculos de tendências para não duplicar a TSM na memória.
for nome_campo in ('tsm', 'te', 'w', 'u'):
    globals().pop(nome_campo, None)
gc.collect()

tendencias = 'sim'
caminho_ = './outputs_trended/'

tsm_ct = abrir_campo_npz_mapeado(caminho_ + 'trended_uvq_sst.npz')
amostra_tsm = tsm_ct[::120, ::30, ::30]

# Garante saída em °C caso o arquivo esteja em Kelvin.
anos_tsm, tsm_amjj = media_anual_amjj(tsm_ct, periodo_tempo_convertido)
if np.nanmedian(amostra_tsm) > 100:
    tsm_amjj -= 273.15
trend_tsm = tendencia_nd(anos_tsm, tsm_amjj)
trend_tsm_dec = trend_tsm['sen_slope'] * ESCALA_DECADA
del tsm_ct, tsm_amjj, amostra_tsm
gc.collect()

print('Final da tendência média AMJJ da TSM')


# -----------------------------
# 2) Água precipitável e convergência de umidade
# -----------------------------
u_arq_ct = './outputs_trended/trended_uvq_u.npz'
v_arq_ct = './outputs_trended/trended_uvq_v.npz'
q_arq_ct = './outputs_trended/trended_uvq_q.npz'

u_ct = abrir_campo_npz_mapeado(u_arq_ct)
v_ct = abrir_campo_npz_mapeado(v_arq_ct)
q_ct = abrir_campo_npz_mapeado(q_arq_ct)

lvl_u_ct = ler_metadado_npz(u_arq_ct, 'level')
lvl_v_ct = ler_metadado_npz(v_arq_ct, 'level')
lvl_q_ct = ler_metadado_npz(q_arq_ct, 'level')

# Garante ordem/níveis compatíveis quando necessário.
if (lvl_q_ct[0] != lvl_u_ct[0]) | (lvl_q_ct[0] != lvl_v_ct[0]) | (lvl_u_ct[0] != lvl_v_ct[0]):
    variaveis = [u_ct, v_ct, q_ct]
    niveis = [lvl_u_ct, lvl_v_ct, lvl_q_ct]
    variaveis_corrigidas, niveis_corrigidos = sawp.alinhar_niveis(
        variaveis,
        niveis,
        ordem='crescente'
    )
    u_ct, v_ct, q_ct = variaveis_corrigidas
    lvl_u_ct, lvl_v_ct, lvl_q_ct = niveis_corrigidos

# Camadas verticais independentes:
#   - água precipitável em médios níveis;
#   - convergência de umidade em baixos níveis.
indices_ap = selecionar_indices_niveis(
    lvl_q_ct,
    camada=CAMADA_AGUA_PRECIPITAVEL,
    nivel_min_hpa=NIVEL_MIN_HPA_AP,
    nivel_max_hpa=NIVEL_MAX_HPA_AP,
    indices_custom=INDICES_NIVEIS_AP_CUSTOM
)

indices_conv = selecionar_indices_niveis(
    lvl_q_ct,
    camada=CAMADA_CONVERGENCIA,
    nivel_min_hpa=NIVEL_MIN_HPA_CONV,
    nivel_max_hpa=NIVEL_MAX_HPA_CONV,
    indices_custom=INDICES_NIVEIS_CONV_CUSTOM
)

rotulo_camada_ap = descricao_camada(
    lvl_q_ct,
    indices_ap,
    CAMADA_AGUA_PRECIPITAVEL
)

rotulo_camada_conv = descricao_camada(
    lvl_q_ct,
    indices_conv,
    CAMADA_CONVERGENCIA
)

print("Camada da água precipitável:", CAMADA_AGUA_PRECIPITAVEL)
print("Níveis usados na água precipitável:", lvl_q_ct[indices_ap])
print("Camada da convergência:", CAMADA_CONVERGENCIA)
print("Níveis usados na convergência:", lvl_q_ct[indices_conv])

# Água precipitável média AMJJ na camada de médios níveis.
ap_ml = integrar_vertical_q(q_ct, lvl_q_ct, indices_ap)

anos_ap, ap_amjj = media_anual_amjj(
    ap_ml,
    periodo_tempo_convertido
)
trend_ap = tendencia_nd(anos_ap, ap_amjj)
trend_ap_dec = trend_ap['sen_slope'] * ESCALA_DECADA

print('Final da tendência média AMJJ da água precipitável')

# Convergência média AMJJ de umidade em baixos níveis.
Fu_bl, Fv_bl = integrar_vertical_fluxo(
    u_ct,
    v_ct,
    q_ct,
    lvl_q_ct,
    indices_conv
)
div_flux_bl = divergencia_horizontal_fluxo(Fu_bl, Fv_bl, lon, lat)
conv_flux_bl = -div_flux_bl  # positivo = convergência de umidade

anos_conv, conv_amjj = media_anual_amjj(
    conv_flux_bl,
    periodo_tempo_convertido
)
trend_conv = tendencia_nd(anos_conv, conv_amjj)
trend_conv_dec = trend_conv['sen_slope'] * ESCALA_DECADA

print('Final da tendência média AMJJ da convergência de umidade')

# Libera memória dos campos 4D e intermediários grandes.
del u_ct, v_ct, q_ct
try:
    del Fu_bl, Fv_bl, div_flux_bl, conv_flux_bl, ap_ml
except NameError:
    pass


# -----------------------------
# 3) θe, w e u — tendência longitudinal média AMJJ
# -----------------------------
te_arq_ct = './outputs_trended/trended_uvq_te.npz'
w_arq_ct = './outputs_trended/trended_uvq_w.npz'
u_arq_ct = './outputs_trended/trended_uvq_u.npz'

lat_min, lat_max = -12.5, -4
lat_1d = lat[:, 0] if lat.ndim == 2 else lat
idx_lat = np.where((lat_1d >= lat_min) & (lat_1d <= lat_max))[0]

te_ct = abrir_campo_npz_mapeado(te_arq_ct)
w_ct = abrir_campo_npz_mapeado(w_arq_ct)
u_ct = abrir_campo_npz_mapeado(u_arq_ct)

te_ct_full = te_ct[:, :, idx_lat[0]:idx_lat[-1] + 1, :]
w_ct_full = w_ct[:, :, idx_lat[0]:idx_lat[-1] + 1, :]
u_ct_full = u_ct[:, :, idx_lat[0]:idx_lat[-1] + 1, :]

lvl_te_ct = ler_metadado_npz(te_arq_ct, 'level')
lvl_w_ct = ler_metadado_npz(w_arq_ct, 'level')
lvl_u_ct = ler_metadado_npz(u_arq_ct, 'level')

if (lvl_te_ct[0] != lvl_w_ct[0]) | (lvl_te_ct[0] != lvl_u_ct[0]) | (lvl_u_ct[0] != lvl_w_ct[0]):
    variaveis = [w_ct_full, u_ct_full]
    niveis = [lvl_w_ct, lvl_u_ct]
    variaveis_corrigidas, niveis_corrigidos = sawp.alinhar_niveis(
        variaveis,
        niveis,
        ordem='crescente'
    )
    w_ct_full, u_ct_full = variaveis_corrigidas
    lvl_w_ct, lvl_u_ct = niveis_corrigidos

# Média meridional usada no painel vertical: tempo, nível, longitude.
te_lon = np.nanmean(te_ct_full, axis=2)
w_lon = np.nanmean(w_ct_full, axis=2)
u_lon = np.nanmean(u_ct_full, axis=2)

anos_te, te_amjj = media_anual_amjj(
    te_lon,
    periodo_tempo_convertido
)
anos_w, w_amjj = media_anual_amjj(
    w_lon,
    periodo_tempo_convertido
)
anos_u, u_amjj = media_anual_amjj(
    u_lon,
    periodo_tempo_convertido
)

trend_te = tendencia_nd(anos_te, te_amjj)
trend_w = tendencia_nd(anos_w, w_amjj)
trend_u = tendencia_nd(anos_u, u_amjj)

trend_te_dec = trend_te['sen_slope'] * ESCALA_DECADA
trend_w_dec = trend_w['sen_slope'] * ESCALA_DECADA
trend_u_dec = trend_u['sen_slope'] * ESCALA_DECADA

print('Final da tendencia longitudinal media AMJJ de thetae, w e u')

# Libera memória dos campos 4D.
del te_ct_full, w_ct_full, u_ct_full, te_ct, w_ct, u_ct, te_lon, w_lon, u_lon


# -----------------------------
# 4) Resumos numéricos
# -----------------------------
resumos = [
    resumo_espacial(
        'TSM média AMJJ',
        trend_tsm,
        mascara=None,
        unidade='°C/década'
    ),
    resumo_espacial(
        f'Água precipitável média AMJJ ({rotulo_camada_ap})',
        trend_ap,
        unidade='mm/década'
    ),
    resumo_espacial(
        f'Convergência de umidade média AMJJ ({rotulo_camada_conv})',
        trend_conv,
        unidade='kg m⁻² s⁻¹/década'
    ),
]

resumo_tendencias = pd.DataFrame(resumos)
resumo_tendencias.to_csv('Resumo_Tendencias_AMJJ.csv', index=False)
print(resumo_tendencias)
print('Resumo salvo em: Resumo_Tendencias_AMJJ.csv')


# -----------------------------
# 5) Figura síntese das tendências AMJJ
# -----------------------------
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 2)

ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax_b = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax_c = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
ax_d = fig.add_subplot(gs[1, 1])

# Reserva espaço para a chave de vetores sem sobrepor os painéis.
plt.subplots_adjust(hspace=0.1, wspace=0.15, bottom=0.20)

# a) Tendência da TSM média AMJJ
campo_a = trend_tsm_dec

levels_tsm = niveis_dinamicos(campo_a)

temp_a = sawp.plotar_grafico_sombreado(
    lon, lat, campo_a,
    levels_tsm,
    dicionario_configuracoes['TSM'][1],
    ax_a,
    alpha=0.7,
    identificador='a'
)
ax_a.set_title('a)', loc='left', fontsize=12)
cax_a = inset_axes(ax_a, width='5%', height='30%', loc='lower right',
                   bbox_to_anchor=(-0.15, 0.0, 1, 1),
                   bbox_transform=ax_a.transAxes, borderpad=0.5)
cbar_a = plt.colorbar(temp_a, cax=cax_a, orientation='vertical')
cbar_a.ax.tick_params(labelsize=8)
cbar_a.ax.set_title('°C/dec', fontsize=8)

# b) Tendência da água precipitável média AMJJ em médios níveis
levels_ap = niveis_dinamicos(trend_ap_dec)

temp_b = sawp.plotar_grafico_sombreado(
    lon, lat, trend_ap_dec,
    levels_ap,
    'BrBG',
    ax_b,
    alpha=0.7,
    identificador='b'
)
ax_b.set_title(f'b) ', loc='left', fontsize=12)
cax_b = inset_axes(ax_b, width='5%', height='30%', loc='lower right',
                   bbox_to_anchor=(-0.15, 0.0, 1, 1),
                   bbox_transform=ax_b.transAxes, borderpad=0.5)
cbar_b = plt.colorbar(temp_b, cax=cax_b, orientation='vertical')
cbar_b.ax.tick_params(labelsize=8)
cbar_b.ax.set_title('mm/dec', fontsize=8)

# c) Tendência da convergência média AMJJ de umidade em baixos níveis
levels_conv = niveis_dinamicos(trend_conv_dec)

temp_c = sawp.plotar_grafico_sombreado(
    lon, lat, trend_conv_dec,
    levels_conv,
    'PuOr',
    ax_c,
    alpha=0.7,
    identificador='c'
)
ax_c.set_title(f'c)', loc='left', fontsize=12)
cax_c = inset_axes(ax_c, width='5%', height='30%', loc='lower right',
                   bbox_to_anchor=(-0.15, 0.0, 1, 1),
                   bbox_transform=ax_c.transAxes, borderpad=0.5)
cbar_c = plt.colorbar(temp_c, cax=cax_c, orientation='vertical')
cbar_c.ax.tick_params(labelsize=8)
cbar_c.ax.set_title('kg m⁻² s⁻¹/dec', fontsize=8)

# d) Tendência vertical-longitudinal média AMJJ de θe com vetores u/w
long_, press = np.meshgrid(lon[0, :], lvl_te_ct)
levels_te = niveis_dinamicos(trend_te_dec)

contour_d = ax_d.contourf(
    long_,
    press,
    trend_te_dec,
    levels=levels_te,
    cmap='RdYlBu_r',
    extend='both'
)

# Vetores de tendência u/w. A subamostragem evita excesso visual.
step_x = max(1, trend_u_dec.shape[1] // 25)
step_z = max(1, trend_u_dec.shape[0] // 12)
u_setas = trend_u_dec[::step_z, ::step_x]
w_setas = trend_w_dec[::step_z, ::step_x]
trend_arrows = ax_d.quiver(
    long_[::step_z, ::step_x],
    press[::step_z, ::step_x],
    u_setas,
    w_setas * 100,
    scale=None,
    width=0.002
)

# Seleciona uma flecha realmente presente no painel, com comprimento próximo
# ao mediano após a conversão visual de omega (Pa s⁻¹ -> Pa s⁻¹ × 100).
# A chave abaixo reutiliza exatamente a escala que o Matplotlib calculou para
# ``trend_arrows`` e, portanto, tem o mesmo comprimento desta flecha do campo.
modulo_desenhado = np.hypot(u_setas, w_setas * 100)
validos_setas = np.isfinite(modulo_desenhado)
if not np.any(validos_setas):
    raise ValueError('Não há vetores finitos para criar a referência de tendência.')
modulo_referencia = np.nanmedian(modulo_desenhado[validos_setas])
distancia_referencia = np.where(
    validos_setas,
    np.abs(modulo_desenhado - modulo_referencia),
    np.inf
)
idx_referencia = np.unravel_index(
    np.argmin(distancia_referencia), modulo_desenhado.shape
)
u_referencia = float(u_setas[idx_referencia])
w_referencia = float(w_setas[idx_referencia])

ax_d.set_title('d)', loc='left', fontsize=12)

cax_d = inset_axes(ax_d, width='5%', height='30%', loc='lower right',
                   bbox_to_anchor=(-0.15, 0.0, 1, 1),
                   bbox_transform=ax_d.transAxes, borderpad=0.5)
cbar_d = plt.colorbar(contour_d, cax=cax_d, orientation='vertical')
cbar_d.ax.tick_params(labelsize=8)
cbar_d.ax.set_title('K/dec', fontsize=8)

ax_d.set_yticks([1000, 850, 700, 500, 300, 200])
ax_d.set_ylim(1000, 200)
ax_d.yaxis.set_ticks_position('right')
ax_d.yaxis.set_label_position('right')
ax_d.tick_params(axis='y', which='both', left=False, labelleft=False,
                 right=True, labelright=True, labelsize=12)
ax_d.spines['left'].set_visible(False)
ax_d.spines['right'].set_visible(True)
ax_d.set_ylabel('Pressure (hPa)', fontsize=14, labelpad=10)

# Força a determinação da escala automática do quiver antes de criar a chave.
fig.canvas.draw()
escala_setas = trend_arrows.scale
posicao_d = ax_d.get_position()
key_ax = fig.add_axes([.57, .030, posicao_d.width, .05])
key_ax.set(xlim=(0, 1), ylim=(0, 1))
key_ax.axis('off')
key_ax.quiver(
    [.08], [.5], [u_referencia], [w_referencia * 100],
    color='black', pivot=trend_arrows.pivot, angles=trend_arrows.angles,
    scale=escala_setas, scale_units=trend_arrows.scale_units,
    width=trend_arrows.width
)
key_ax.text(
    .18, .5,
    rf'$u$ = {u_referencia:.3g} m s$^{{-1}}$ decade$^{{-1}}$; '
    rf'$\omega$ = {w_referencia:.2e} Pa s$^{{-1}}$ decade$^{{-1}}$',
    va='center', fontsize=8
)

fig.text(0.5, 0.125, 'Longitude', ha='center', fontsize=20)
fig.text(0.05, 0.5, 'Latitude', va='center', rotation='vertical', fontsize=20)

plt.savefig('Subplot_Tendencias_AMJJ.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

print('Figura salva em: Subplot_Tendencias_AMJJ.png')
