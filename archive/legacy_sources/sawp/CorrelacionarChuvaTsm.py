import pandas as pd
from scipy import signal
from scipy.stats import linregress
import numpy as np
from scipy.stats import pearsonr

def carregar_chuva_periodo(caminho, tendencias='não'):
    prec = pd.read_csv(
        caminho,
        header=None,
        sep=r"\s+",
        names=['Date', 'prec'],
        parse_dates=[0]
    )

    prec = prec.set_index('Date')
    prec['prec'] = pd.to_numeric(prec['prec'], errors='coerce')

    # Remove códigos de falha e valores fisicamente inválidos
    prec.loc[prec['prec'] < 0, 'prec'] = np.nan

    # Como os dados já são mensais, não use sum() diretamente sem controle
    chuva_mes = prec['prec'].copy()
    chuva_mes.index = pd.PeriodIndex(chuva_mes.index, freq='M')

    # Abril–Julho
    mask_amjj = (chuva_mes.index.month >= 4) & (chuva_mes.index.month <= 7)

    chuva = chuva_mes[mask_amjj].groupby(
        chuva_mes[mask_amjj].index.year
    ).sum(min_count=1)

    # Se algum mês AMJJ estiver ausente, remove o ano inteiro
    n_valid = chuva_mes[mask_amjj].groupby(
        chuva_mes[mask_amjj].index.year
    ).count()

    chuva = chuva.where(n_valid == 4)

    if tendencias == 'não':
        valid = chuva.notna()
        vals = chuva.copy()
        vals.loc[valid] = signal.detrend(chuva.loc[valid].values)
        chuva = vals

    return pd.Series(chuva.values, index=chuva.index, name='prec')


def analise_tsm_regressao(tsm_norm, chuva_norm):
    
    # Quantis ENSO
    q10, q25, q75, q90 = (
        tsm_norm.quantile(q) for q in (0.10, 0.25, 0.75, 0.90)
    )

    # Máscaras ENSO
    masks = {
        'ef': tsm_norm <= q10,                        # La Niña forte
        'lf': (tsm_norm <= q25) & (tsm_norm > q10),   # La Niña fraca/moderada
        #'no': (tsm_norm > q25) & (tsm_norm < q75),
        'lq': (tsm_norm >= q75) & (tsm_norm < q90),   # El Niño fraco/moderado
        'eq': tsm_norm >= q90                         # El Niño forte
    }

    # Combinação final
    tsm_combinado = pd.concat([tsm_norm[m] for m in masks.values()])
    chuva_combinado = pd.concat([chuva_norm[m] for m in masks.values()])

    # Regressão linear
    x = tsm_combinado.to_numpy()
    y = chuva_combinado.to_numpy()

    slope, intercept, r_value, p_value, std_err = linregress(x, y)

    return {
        'slope': slope,
        'intercept': intercept,
        'r_value': r_value,
        'p_value': p_value,
        'std_err': std_err,
        'q10' : q10,
        'q25' : q25,
        'q75' : q75,
        'q90' : q90
        }

# =============================================================================
# def correlacao_tsm_chuva(
#         tsm,                   # array (tempo x lat x lon)
#         chuva_anom,            # array (anos)
#         ano_inicial=1963,
#         meses_sazonais=[3,4,5,6],   # Abril–Julho
#         periodo_clim=(1980,2010),
#         usar_extremos=False    # True → usa apenas Q25 e Q75
#     ):
#     """
#     Calcula correlação ponto a ponto entre anomalia sazonal AMJJ da TSM (3D)
#     e anomalia da chuva ENEB (1D).
# 
#     Retorna:
#         corremap  – mapa de correlação  (lat x lon)
#         pmap      – p-value             (lat x lon)
#         sign      – significância (1 se p<0.05)
#         indices_anos_usados – lista de anos realmente utilizados
#     """
# 
#     # ==============================
#     # 1. Dimensões gerais
#     # ==============================
#     nt, nx, ny = tsm.shape
#     num_anos = nt // 12
#     anos = np.arange(ano_inicial, ano_inicial + num_anos)
# 
#     # Garante alinhamento
#     chuva_anom = np.array(chuva_anom[:num_anos])
# 
#     # ==============================
#     # 2. Média sazonal AMJJ por ano
#     # ==============================
#     tsm_AMJJ = np.zeros((num_anos, nx, ny))
# 
#     for k in range(num_anos):
#         idx = [k*12 + m for m in meses_sazonais]
#         tsm_AMJJ[k] = np.mean(tsm[idx, :, :], axis=0)
# 
#     # ==============================
#     # 3. Climatologia AMJJ (1980–2010)
#     # ==============================
#     mask_clim = (anos >= periodo_clim[0]) & (anos <= periodo_clim[1])
#     climatologia_AMJJ = np.mean(tsm_AMJJ[mask_clim], axis=0)
# 
#     # ==============================
#     # 4. Anomalias AMJJ
#     # ==============================
#     anomalia_tsm_AMJJ = tsm_AMJJ - climatologia_AMJJ
# 
#     # ==============================
#     # 5. Seleção opcional de anos extremos
#     # ==============================
#     if usar_extremos:
#         q25, q75 = np.quantile(chuva_anom, [0.25, 0.75])
#         indices = np.where((chuva_anom < q25) | (chuva_anom > q75))[0]
#     else:
#         indices = np.arange(num_anos)
# 
#     # ==============================
#     # 6. Correlação ponto a ponto
#     # ==============================
#     corremap  = np.zeros((nx, ny))
#     pmap      = np.zeros((nx, ny))
#     sign      = np.zeros((nx, ny))
# 
#     for i in range(nx):
#         for j in range(ny):
#             serie_tsm = anomalia_tsm_AMJJ[indices, i, j]
#             serie_chu = chuva_anom[indices]
# 
#             corr, p = pearsonr(serie_tsm, serie_chu)
#             corremap[i, j] = corr
#             pmap[i, j] = p
#             sign[i, j] = 1 if p < 0.05 else 0
# 
#     return corremap, pmap, sign, anos[indices]
# =============================================================================
def correlacao_tsm_chuva(
        tsm,
        chuva_anom,
        anos_chuva,
        ano_inicial=1959,
        meses_sazonais=[3, 4, 5, 6],  # Abril–Julho
        periodo_clim=(1990, 2020),
        usar_extremos=False
    ):

# =============================================================================
#     chuva_anom = chuva_norm
#     ano_inicial=1958
#     meses_sazonais=[3, 4, 5, 6]
#     periodo_clim=(1990, 2020)
# =============================================================================
    
    nt, nx, ny = tsm.shape
    num_anos = nt // 12
    anos_tsm = np.arange(ano_inicial, ano_inicial + num_anos)

    chuva_anom = np.asarray(chuva_anom)
    anos_chuva = np.asarray(anos_chuva)

    # Média sazonal da TSM
    tsm_AMJJ = np.zeros((num_anos, nx, ny))

    for k in range(num_anos):
        idx = [k * 12 + m for m in meses_sazonais]
        tsm_AMJJ[k] = np.nanmean(tsm[idx, :, :], axis=0)

    # Climatologia 1980–2010
    mask_clim = (anos_tsm >= periodo_clim[0]) & (anos_tsm <= periodo_clim[1])
    climatologia_AMJJ = np.nanmean(tsm_AMJJ[mask_clim], axis=0)

    climatologia_AMJJ = np.nanmean(tsm_AMJJ[mask_clim], axis=0)

    std_AMJJ = np.nanstd(
        tsm_AMJJ[mask_clim] - climatologia_AMJJ,
        axis=0,
        ddof=1
    )

    anomalia_tsm_AMJJ = (tsm_AMJJ - climatologia_AMJJ) / std_AMJJ

    # Alinhamento real por ano
    anos_comuns = np.intersect1d(anos_tsm, anos_chuva)

    idx_tsm = np.where(np.isin(anos_tsm, anos_comuns))[0]
    idx_chuva = np.where(np.isin(anos_chuva, anos_comuns))[0]

    tsm_sel = anomalia_tsm_AMJJ[idx_tsm, :, :]
    chuva_sel = chuva_anom[idx_chuva]
    anos_sel = anos_comuns

    # Remove qualquer NaN remanescente
    mask_valid = np.isfinite(chuva_sel)
    chuva_sel = chuva_sel[mask_valid]
    tsm_sel = tsm_sel[mask_valid, :, :]
    anos_sel = anos_sel[mask_valid]

    if usar_extremos:
        q25, q75 = np.nanquantile(chuva_sel, [0.25, 0.75])
        indices = np.where((chuva_sel < q25) | (chuva_sel > q75))[0]
    else:
        indices = np.arange(len(chuva_sel))

    corremap = np.full((nx, ny), np.nan)
    pmap = np.full((nx, ny), np.nan)
    sign = np.zeros((nx, ny))

    for i in range(nx):
        for j in range(ny):
            serie_tsm = tsm_sel[indices, i, j]
            serie_chu = chuva_sel[indices]

            mask = np.isfinite(serie_tsm) & np.isfinite(serie_chu)

            if mask.sum() >= 3:
                corr, p = pearsonr(serie_tsm[mask], serie_chu[mask])
                corremap[i, j] = corr
                pmap[i, j] = p
                sign[i, j] = 1 if p < 0.05 else 0

    return corremap, pmap, sign, anos_sel[indices]