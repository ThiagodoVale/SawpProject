# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 18:29:38 2026

@author: silva
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import sawp


# ============================================================
# CONFIGURAÇÕES
# ============================================================

TENDENCIAS = "não"  # "sim" ou "não"

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs" if TENDENCIAS == "não" else BASE_DIR / "outputs_trended"
FIG_DIR = BASE_DIR / "figuras_revisor"
FIG_DIR.mkdir(parents=True, exist_ok=True)

MESES_ESTACAO = [4, 5, 6, 7]

AREA_SAWP = {"lon_min": -34.0, "lon_max": -25.0, "lat_min": -12.5, "lat_max": -4.0}
AREA_NEB  = {"lon_min": -38.0, "lon_max": -34.5, "lat_min": -12.5, "lat_max": -4.5}

LOW_LEVELS = [1000, 925, 900, 875, 850]
MID_LEVELS = [850, 700, 600, 500, 400]
MFC_LEVELS = [1000, 925, 900, 875, 850]
OMEGA_LEVEL = 500

G = 9.80665
R_EARTH = 6_371_000.0

RAIN_FILE = DATA_DIR / "EstacoesDados.txt"
SST_NETCDF_REF = DATA_DIR / "Era5_Dataset_sst.nc"

if TENDENCIAS == "não":
    SST_FILE  = OUT_DIR / "detrended_sst.npz"
    U_FILE    = OUT_DIR / "detrended_uvq_u.npz"
    V_FILE    = OUT_DIR / "detrended_uvq_v.npz"
    Q_FILE    = OUT_DIR / "detrended_uvq_q.npz"
    W_FILE    = OUT_DIR / "detrended_uvq_w.npz"
    SLHF_FILE = OUT_DIR / "detrended_uvq_slhf.npz"
else:
    SST_FILE  = OUT_DIR / "trended_uvq_sst.npz"
    U_FILE    = OUT_DIR / "trended_uvq_u.npz"
    V_FILE    = OUT_DIR / "trended_uvq_v.npz"
    Q_FILE    = OUT_DIR / "trended_uvq_q.npz"
    W_FILE    = OUT_DIR / "trended_uvq_w.npz"
    SLHF_FILE = OUT_DIR / "trended_uvq_slhf.npz"

OUT_FIG = FIG_DIR / "S5C_matriz_correlacao_55_comparacoes.png"
OUT_R = FIG_DIR / "S5C_matriz_correlacao_r.csv"
OUT_P = FIG_DIR / "S5C_matriz_correlacao_p.csv"
OUT_TABLE = FIG_DIR / "S5C_tabela_series_amjj_normalizadas.csv"


# ============================================================
# FUNÇÕES BÁSICAS
# ============================================================

def to_celsius_if_needed(x):
    x = np.asarray(x, dtype=float)
    if np.nanmean(x) > 100:
        return x - 273.15
    return x


def area_mask(lon2d, lat2d, area):
    return (
        (lon2d >= area["lon_min"]) &
        (lon2d <= area["lon_max"]) &
        (lat2d >= area["lat_min"]) &
        (lat2d <= area["lat_max"])
    )


def normalize(s):
    return (s - s.mean()) / s.std()


def annual_amjj_mean(series):
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    s = s[s.index.month.isin(MESES_ESTACAO)]
    return s.groupby(s.index.year).mean()


def time_from_npz(npz):
    if "time" in npz:
        return pd.to_datetime(npz["time"])
    if "valid_time" in npz:
        return pd.to_datetime(npz["valid_time"])

    periodo_tempo_convertido, _, _ = sawp.ConfigurarTempoArea(str(SST_NETCDF_REF))
    return pd.to_datetime(periodo_tempo_convertido)


def get_level_key(npz):
    for key in ["level", "pressure_level", "levels"]:
        if key in npz:
            return key
    raise KeyError("Não encontrei coordenada vertical: level, pressure_level ou levels.")


def load_npz_3d(path):
    npz = np.load(path, mmap_mode="r", allow_pickle=True)
    data = npz["data"]
    time = time_from_npz(npz)

    if data.ndim == 4:
        data = data[:, 0, :, :]

    return data, time


def load_npz_4d(path):
    npz = np.load(path, mmap_mode="r", allow_pickle=True)
    data = npz["data"]
    levels = np.asarray(npz[get_level_key(npz)], dtype=float)
    time = time_from_npz(npz)
    return data, levels, time


def select_level_indices(levels, wanted):
    levels = np.asarray(levels, dtype=float)
    idx = []
    for lev in wanted:
        idx.append(int(np.argmin(np.abs(levels - lev))))
    return np.unique(idx)


def integrate_pressure(field, levels_hpa):
    order = np.argsort(levels_hpa)
    lev_pa = np.asarray(levels_hpa)[order] * 100.0
    fld = field[:, order, :, :]
    return np.trapezoid(fld, x=lev_pa, axis=1) / G


def horizontal_divergence(fx, fy, lon2d, lat2d):
    lat_rad = np.deg2rad(lat2d[:, 0])
    lon_rad = np.deg2rad(lon2d[0, :])

    dy = np.gradient(lat_rad) * R_EARTH
    dx = np.gradient(lon_rad)[None, :] * R_EARTH * np.cos(lat_rad)[:, None]

    dfx_dx = np.gradient(fx, axis=1) / dx
    dfy_dy = np.gradient(fy, axis=0) / dy[:, None]

    return dfx_dx + dfy_dy


# ============================================================
# SÉRIES
# ============================================================

def load_sst_sawp_series(lon2d, lat2d):
    sst, time = load_npz_3d(SST_FILE)
    sst = to_celsius_if_needed(sst)

    mask = area_mask(lon2d, lat2d, AREA_SAWP)

    serie = np.nanmean(
        np.where(mask[None, :, :], sst, np.nan),
        axis=(1, 2)
    )

    return pd.Series(serie, index=time, name="sst_sawp")


def load_slhf_sawp_series(lon2d, lat2d):
    slhf, time = load_npz_3d(SLHF_FILE)

    mask = area_mask(lon2d, lat2d, AREA_SAWP)

    serie = np.nanmean(
        np.where(mask[None, :, :], slhf, np.nan),
        axis=(1, 2)
    )

    return pd.Series(serie, index=time, name="slhf_sawp")


def load_wind_sawp_series(lon2d, lat2d):
    u, lev_u, time_u = load_npz_4d(U_FILE)
    v, lev_v, time_v = load_npz_4d(V_FILE)

    idx_u = int(np.argmin(np.abs(lev_u - 1000)))
    idx_v = int(np.argmin(np.abs(lev_v - 1000)))

    ntime = min(u.shape[0], v.shape[0], len(time_u), len(time_v))

    wind = np.sqrt(
        u[:ntime, idx_u, :, :]**2 +
        v[:ntime, idx_v, :, :]**2
    )

    mask = area_mask(lon2d, lat2d, AREA_SAWP)

    serie = np.nanmean(
        np.where(mask[None, :, :], wind, np.nan),
        axis=(1, 2)
    )

    return pd.Series(serie, index=time_u[:ntime], name="wind_sawp")


def load_rain_eneb_series():
    rain = sawp.carregar_chuva_periodo(str(RAIN_FILE), TENDENCIAS)
    rain = pd.Series(np.asarray(rain).squeeze(), index=rain.index, name="rain_eneb")
    rain = rain.replace(-999, np.nan)
    rain.index = rain.index.astype(int)
    return rain


def load_pw_series(lon2d, lat2d, area, levels, name):
    q, lev_q, time_q = load_npz_4d(Q_FILE)

    idx = select_level_indices(lev_q, levels)
    q_sel = q[:, idx, :, :]
    used_levels = lev_q[idx]

    pw = integrate_pressure(q_sel, used_levels)

    mask = area_mask(lon2d, lat2d, area)

    serie = np.nanmean(
        np.where(mask[None, :, :], pw, np.nan),
        axis=(1, 2)
    )

    return pd.Series(serie, index=time_q, name=name)


def load_omega_series(lon2d, lat2d, area, name):
    w, lev_w, time_w = load_npz_4d(W_FILE)

    idx = int(np.argmin(np.abs(lev_w - OMEGA_LEVEL)))

    mask = area_mask(lon2d, lat2d, area)

    serie = np.nanmean(
        np.where(mask[None, :, :], w[:, idx, :, :], np.nan),
        axis=(1, 2)
    )

    return pd.Series(serie, index=time_w, name=name)


def load_mfc_eneb_series(lon2d, lat2d):
    u, lev_u, time_u = load_npz_4d(U_FILE)
    v, lev_v, time_v = load_npz_4d(V_FILE)
    q, lev_q, time_q = load_npz_4d(Q_FILE)

    ntime = min(u.shape[0], v.shape[0], q.shape[0], len(time_u), len(time_v), len(time_q))

    idx_u = select_level_indices(lev_u, MFC_LEVELS)
    idx_v = select_level_indices(lev_v, MFC_LEVELS)
    idx_q = select_level_indices(lev_q, MFC_LEVELS)

    u_sel = u[:ntime, idx_u, :, :]
    v_sel = v[:ntime, idx_v, :, :]
    q_sel = q[:ntime, idx_q, :, :]

    nlev = min(u_sel.shape[1], v_sel.shape[1], q_sel.shape[1])

    u_sel = u_sel[:, :nlev, :, :]
    v_sel = v_sel[:, :nlev, :, :]
    q_sel = q_sel[:, :nlev, :, :]

    used_levels = lev_q[idx_q][:nlev]

    flux_u = integrate_pressure(q_sel * u_sel, used_levels)
    flux_v = integrate_pressure(q_sel * v_sel, used_levels)

    mask = area_mask(lon2d, lat2d, AREA_NEB)

    valores = []

    for i in range(ntime):
        div = horizontal_divergence(flux_u[i], flux_v[i], lon2d, lat2d)
        conv = -div
        valores.append(np.nanmean(np.where(mask, conv, np.nan)))

    return pd.Series(valores, index=time_q[:ntime], name="mfc_eneb")


# ============================================================
# MATRIZ DE CORRELAÇÃO
# ============================================================

def significance_stars(p):
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def calc_corr_matrix(df):
    # Remove colunas duplicadas, se existirem
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Garante apenas valores numéricos
    df = df.apply(pd.to_numeric, errors="coerce")

    cols = list(df.columns)

    rmat = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    pmat = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)

    for c1 in cols:
        for c2 in cols:
            tmp = df[[c1, c2]].dropna()

            if len(tmp) >= 3:
                x = tmp[c1].to_numpy(dtype=float).ravel()
                y = tmp[c2].to_numpy(dtype=float).ravel()

                r, p = pearsonr(x, y)

                rmat.loc[c1, c2] = float(r)
                pmat.loc[c1, c2] = float(p)

    return rmat, pmat


def plot_corr_matrix(rmat, pmat, labels):
    data = rmat.values.astype(float)
    names = [labels[c] for c in rmat.columns]

    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(
        data,
        vmin=-1,
        vmax=1,
        cmap="RdBu_r"
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Pearson correlation coefficient (r)", fontsize=13)

    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))

    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(names, fontsize=11)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            r = rmat.iloc[i, j]
            p = pmat.iloc[i, j]

            if np.isfinite(r):
                text = f"{r:.2f}{significance_stars(p)}"
            else:
                text = ""

            color = "white" if abs(r) >= 0.55 else "black"

            ax.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color=color
            )

    ax.set_title(
        "Correlation matrix among normalized AMJJ SAWP–ENEB variables",
        fontsize=16,
        pad=20
    )

    ax.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.text(
        0.5,
        0.02,
        "* p < 0.10   ** p < 0.05   *** p < 0.01",
        ha="center",
        fontsize=12
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Figura salva em: {OUT_FIG}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Configurando grade...")
    _, lon2d, lat2d = sawp.ConfigurarTempoArea(str(SST_NETCDF_REF))

    print("Carregando séries mensais...")

    series = {}

    series["sst_norm"] = normalize(annual_amjj_mean(load_sst_sawp_series(lon2d, lat2d)))
    series["rain_norm"] = normalize(load_rain_eneb_series())
    series["slhf_sawp_norm"] = normalize(annual_amjj_mean(load_slhf_sawp_series(lon2d, lat2d)))
    series["wind_norm"] = normalize(annual_amjj_mean(load_wind_sawp_series(lon2d, lat2d)))

    series["pw_low_sawp_norm"] = normalize(
        annual_amjj_mean(load_pw_series(lon2d, lat2d, AREA_SAWP, LOW_LEVELS, "pw_low_sawp"))
    )
    series["pw_low_neb_norm"] = normalize(
        annual_amjj_mean(load_pw_series(lon2d, lat2d, AREA_NEB, LOW_LEVELS, "pw_low_neb"))
    )
    series["pw_mid_sawp_norm"] = normalize(
        annual_amjj_mean(load_pw_series(lon2d, lat2d, AREA_SAWP, MID_LEVELS, "pw_mid_sawp"))
    )
    series["pw_mid_neb_norm"] = normalize(
        annual_amjj_mean(load_pw_series(lon2d, lat2d, AREA_NEB, MID_LEVELS, "pw_mid_neb"))
    )

    series["omega_sawp_norm"] = normalize(
        annual_amjj_mean(load_omega_series(lon2d, lat2d, AREA_SAWP, "omega_sawp"))
    )
    series["omega_neb_norm"] = normalize(
        annual_amjj_mean(load_omega_series(lon2d, lat2d, AREA_NEB, "omega_neb"))
    )

    series["mfc_eneb_norm"] = normalize(
        annual_amjj_mean(load_mfc_eneb_series(lon2d, lat2d))
    )

    df = pd.concat(series.values(), axis=1)
    df.columns = list(series.keys())
    df = df.dropna()

    df.to_csv(OUT_TABLE)

    print(f"Número de anos usados: {len(df)}")
    print(f"Número de variáveis: {len(df.columns)}")
    print(f"Número de comparações únicas: {len(df.columns) * (len(df.columns) - 1) // 2}")

    labels = {
        "sst_norm": "SAWP SST",
        "rain_norm": "ENEB rainfall",
        "slhf_sawp_norm": "SAWP SLHF",
        "wind_norm": "SAWP wind",
        "pw_low_sawp_norm": "SAWP PW\n1000-850",
        "pw_low_neb_norm": "ENEB PW\n1000-850",
        "pw_mid_sawp_norm": "SAWP PW\n850-400",
        "pw_mid_neb_norm": "ENEB PW\n850-400",
        "omega_sawp_norm": "SAWP ω\n500 hPa",
        "omega_neb_norm": "ENEB ω\n500 hPa",
        "mfc_eneb_norm": "ENEB MFC\n1000-850",
    }

    rmat, pmat = calc_corr_matrix(df)

    rmat.to_csv(OUT_R)
    pmat.to_csv(OUT_P)

    plot_corr_matrix(rmat, pmat, labels)

    print("Finalizado.")