#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
import sawp


# ============================================================
# CONFIG
# ============================================================

TENDENCIAS = "não"
MESES = [4, 5, 6, 7]

BASE = Path(".")
DATA = BASE / "data"
OUT = BASE / "outputs" if TENDENCIAS == "não" else BASE / "outputs_trended"
FIG = BASE / "figuras_revisor"
FIG.mkdir(exist_ok=True)

SST_REF = DATA / "Era5_Dataset_sst.nc"
RAIN_FILE = DATA / "EstacoesDados.txt"

if TENDENCIAS == "não":
    SST_FILE  = OUT / "detrended_sst.npz"
    U_FILE    = OUT / "detrended_uvq_u.npz"
    V_FILE    = OUT / "detrended_uvq_v.npz"
    Q_FILE    = OUT / "detrended_uvq_q.npz"
    W_FILE    = OUT / "detrended_uvq_w.npz"
    SLHF_FILE = OUT / "detrended_uvq_slhf.npz"
else:
    SST_FILE  = OUT / "trended_uvq_sst.npz"
    U_FILE    = OUT / "trended_uvq_u.npz"
    V_FILE    = OUT / "trended_uvq_v.npz"
    Q_FILE    = OUT / "trended_uvq_q.npz"
    W_FILE    = OUT / "trended_uvq_w.npz"
    SLHF_FILE = OUT / "trended_uvq_slhf.npz"

AREA_SAWP = [-34.0, -25.0, -12.5, -4.0]
AREA_ENEB = [-38.0, -34.5, -12.5, -4.5]

LOW_LEVELS = [1000, 925, 900, 875, 850]
MID_LEVELS = [850, 700, 600, 500, 400]
MFC_LEVELS = [1000, 925, 900, 875, 850]
OMEGA_LEVEL = 500

G = 9.80665
R = 6_371_000.0
CONVERTER_SLHF = True


# ============================================================
# FUNÇÕES CURTAS
# ============================================================

def norm(s):
    s = pd.Series(s).astype(float)
    return (s - s.mean()) / s.std()


def time_npz(z):
    if "time" in z:
        return pd.to_datetime(z["time"])
    if "valid_time" in z:
        return pd.to_datetime(z["valid_time"])
    tempo, _, _ = sawp.ConfigurarTempoArea(str(SST_REF))
    return pd.to_datetime(tempo)


def level_key(z):
    for k in ["level", "pressure_level", "levels"]:
        if k in z:
            return k
    raise KeyError("Arquivo sem level/pressure_level/levels.")


def load3(path):
    z = np.load(path, mmap_mode="r", allow_pickle=True)
    x = z["data"]
    if x.ndim == 4:
        x = x[:, 0]
    return x, time_npz(z)


def load4(path):
    z = np.load(path, mmap_mode="r", allow_pickle=True)
    return z["data"], np.asarray(z[level_key(z)], float), time_npz(z)


def mask_area(lon, lat, area):
    lon_min, lon_max, lat_min, lat_max = area
    return (
        (lon >= lon_min) & (lon <= lon_max) &
        (lat >= lat_min) & (lat <= lat_max)
    )


def media_area(data, time, lon, lat, area):
    m = mask_area(lon, lat, area)
    s = np.nanmean(np.where(m[None, :, :], data, np.nan), axis=(1, 2))
    return pd.Series(s, index=pd.to_datetime(time))


def amjj_mean(s):
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    s = s[s.index.month.isin(MESES)]
    return s.groupby(s.index.year).mean()

def amjj_sum(s):
    s = s.copy()
    s.index = pd.to_datetime(s.index)
    s = s[s.index.month.isin(MESES)]
    return s.groupby(s.index.year).sum()


def select_levels(levels, wanted):
    return np.unique([int(np.argmin(np.abs(levels - l))) for l in wanted])


def integ_p(field, levels):
    order = np.argsort(levels)
    lev_pa = np.asarray(levels)[order] * 100.0
    return np.trapezoid(field[:, order], x=lev_pa, axis=1) / G


def div_h(fx, fy, lon, lat):
    lat_rad = np.deg2rad(lat[:, 0])
    lon_rad = np.deg2rad(lon[0, :])

    dy = np.gradient(lat_rad) * R
    dx = np.gradient(lon_rad)[None, :] * R * np.cos(lat_rad)[:, None]

    return np.gradient(fx, axis=1) / dx + np.gradient(fy, axis=0) / dy[:, None]


# ============================================================
# SÉRIES ANUAIS
# ============================================================

def serie_sst_original():
    tempo, lon, lat = sawp.ConfigurarTempoArea(str(SST_REF))

    tsm = sawp.CarregarDados(
        str(SST_FILE),
        tendencias=TENDENCIAS,
        nome_var_nc="sst",
        substituir_nan=-999,
    )

    lloW = np.argmin(np.abs(lon[0, :] - AREA_SAWP[0]))
    lloE = np.argmin(np.abs(lon[0, :] - AREA_SAWP[1]))
    llaS = np.argmin(np.abs(lat[:, 0] - AREA_SAWP[2]))
    llaN = np.argmin(np.abs(lat[:, 0] - AREA_SAWP[3]))

    resultado_tsm = sawp.ProcessarTsmv2(tsm, tempo, lloW, lloE, llaS, llaN)

    s = resultado_tsm["media_tsm_anual"]["Anomalia"].copy()
    s.index = s.index.astype(int)
    s = s / s.std()
    s.name = "sst_norm"

    return s


def serie_chuva_original():
    chuva = sawp.carregar_chuva_periodo(str(RAIN_FILE), TENDENCIAS)
    chuva = pd.Series(np.asarray(chuva).squeeze(), index=chuva.index.astype(int))
    chuva = chuva.replace(-999, np.nan)

    s = norm(chuva)
    s.name = "rain_norm"
    return s


def serie_slhf(lon, lat, lag_meses=0, name="slhf_sawp_norm"):
    x, t = load3(SLHF_FILE)

    if CONVERTER_SLHF:
        x = -x

    # Série mensal
    s = media_area(x, t, lon, lat, AREA_SAWP)

    # Para lag -2 físico:
    # SLHF antecede SST em 2 meses.
    # Em pandas isso é shift(+2).
    if lag_meses != 0:
        s = s.shift(lag_meses)

    # Use média sazonal para fluxo W/m²
    s = norm(amjj_mean(s))

    s.name = name
    return s


def serie_wind(lon, lat):
    u, lev_u, t_u = load4(U_FILE)
    v, lev_v, t_v = load4(V_FILE)

    iu = int(np.argmin(np.abs(lev_u - 1000)))
    iv = int(np.argmin(np.abs(lev_v - 1000)))

    n = min(u.shape[0], v.shape[0], len(t_u), len(t_v))
    wind = np.sqrt(u[:n, iu] ** 2 + v[:n, iv] ** 2)

    s = norm(amjj_mean(media_area(wind, t_u[:n], lon, lat, AREA_SAWP)))
    s.name = "wind_sawp_norm"
    return s


def serie_pw(lon, lat, area, levels, name):
    q, lev, t = load4(Q_FILE)
    idx = select_levels(lev, levels)

    pw = integ_p(q[:, idx], lev[idx])

    s = norm(amjj_mean(media_area(pw, t, lon, lat, area)))
    s.name = name
    return s


def serie_omega(lon, lat, area, name):
    w, lev, t = load4(W_FILE)
    idx = int(np.argmin(np.abs(lev - OMEGA_LEVEL)))

    s = norm(amjj_mean(media_area(w[:, idx], t, lon, lat, area)))
    s.name = name
    return s


def serie_mfc(lon, lat):
    u, lev_u, t_u = load4(U_FILE)
    v, lev_v, t_v = load4(V_FILE)
    q, lev_q, t_q = load4(Q_FILE)

    n = min(u.shape[0], v.shape[0], q.shape[0], len(t_u), len(t_v), len(t_q))

    iu = select_levels(lev_u, MFC_LEVELS)
    iv = select_levels(lev_v, MFC_LEVELS)
    iq = select_levels(lev_q, MFC_LEVELS)

    u = u[:n, iu]
    v = v[:n, iv]
    q = q[:n, iq]

    nlev = min(u.shape[1], v.shape[1], q.shape[1])
    u = u[:, :nlev]
    v = v[:, :nlev]
    q = q[:, :nlev]
    lev = lev_q[iq][:nlev]

    fu = integ_p(q * u, lev)
    fv = integ_p(q * v, lev)

    m = mask_area(lon, lat, AREA_ENEB)
    vals = []

    for i in range(n):
        conv = -div_h(fu[i], fv[i], lon, lat)
        vals.append(np.nanmean(np.where(m, conv, np.nan)))

    s = pd.Series(vals, index=pd.to_datetime(t_q[:n]))
    s = norm(amjj_mean(s))
    s.name = "mfc_eneb_norm"
    return s


# ============================================================
# CORRELAÇÃO E FIGURA
# ============================================================

def stats_matriz(df):
    # Remove colunas duplicadas
    df = df.loc[:, ~df.columns.duplicated()].copy()

    # Garante numérico
    df = df.apply(pd.to_numeric, errors="coerce")

    cols = list(df.columns)

    rmat = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    pmat = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)

    smat = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    spmat = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)

    rows = []

    for i, a in enumerate(cols):
        for j, b in enumerate(cols):

            tmp = df[[a, b]].dropna()

            if len(tmp) < 3:
                continue

            x = tmp.iloc[:, 0].to_numpy(dtype=float).reshape(-1)
            y = tmp.iloc[:, 1].to_numpy(dtype=float).reshape(-1)

            r, p = pearsonr(x, y)
            rho, ps = spearmanr(x, y)

            rmat.iat[i, j] = float(r)
            pmat.iat[i, j] = float(p)
            smat.iat[i, j] = float(rho)
            spmat.iat[i, j] = float(ps)

            if j > i:
                tau, pt = kendalltau(x, y)

                rows.append({
                    "var_1": a,
                    "var_2": b,
                    "n": int(len(tmp)),
                    "pearson_r": float(r),
                    "pearson_p": float(p),
                    "spearman_rho": float(rho),
                    "spearman_p": float(ps),
                    "kendall_tau": float(tau),
                    "kendall_p": float(pt),
                })

    return rmat, pmat, smat, spmat, pd.DataFrame(rows)

def stars(p):
    if not np.isfinite(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def plot_matriz(mat, pmat, labels, title, out):
    cols = list(mat.columns)
    names = [labels[c] for c in cols]
    data = mat.values.astype(float)

    fig, ax = plt.subplots(figsize=(14, 12))

    im = ax.imshow(data, vmin=-1, vmax=1, cmap="RdBu_r")
    cb = fig.colorbar(im, ax=ax, shrink=0.82)

    # Título da barra
    cb.set_label("Correlation coefficient", fontsize=18)
    
    # Números da barra
    cb.ax.tick_params(labelsize=16)

    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=16)
    ax.set_yticklabels(names, fontsize=16)

    for i in range(len(cols)):
        for j in range(len(cols)):
            if j > i:
                ax.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        color="white",
                        zorder=3
                    )
                )
                continue

            r = mat.iloc[i, j]
            p = pmat.iloc[i, j]

            txt = "" if not np.isfinite(r) else f"{r:.2f}{stars(p)}"
            color = "white" if np.isfinite(r) and abs(r) >= 0.55 else "black"

            ax.text(j, i, txt, ha="center", va="center", fontsize=14, color=color)

    ax.set_title(title, fontsize=18, pad=20)

    ax.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.text(
        0.5,
        0.02,
        "* p < 0.10   ** p < 0.05   *** p < 0.01",
        ha="center",
        fontsize=14
    )

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(out, dpi=600, bbox_inches="tight")
    plt.close(fig)

    print(f"Figura salva: {out}")


# ============================================================
# MAIN
# ============================================================

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Configurando grade...")
    _, lon, lat = sawp.ConfigurarTempoArea(str(SST_REF))

    print("Gerando séries anuais AMJJ...")
    sst = serie_sst_original()
    rain = serie_chuva_original()

    # SLHF lag 0: usado na matriz contra chuva, vento, PW, omega e MFC
    slhf = serie_slhf(
        lon,
        lat,
        lag_meses=0,
        name="slhf_sawp_norm"
    )

    # SLHF lag -2 físico: SLHF antecede SST em 2 meses.
    # Em pandas, isso é shift(+2).
    slhf_lag2_sst = serie_slhf(
        lon,
        lat,
        lag_meses=3,
        name="slhf_sawp_lag_minus2_norm"
    )

    wind = serie_wind(lon, lat)

    pw_low_sawp = serie_pw(lon, lat, AREA_SAWP, LOW_LEVELS, "pw_low_sawp_norm")
    pw_low_eneb = serie_pw(lon, lat, AREA_ENEB, LOW_LEVELS, "pw_low_eneb_norm")

    pw_mid_sawp = serie_pw(lon, lat, AREA_SAWP, MID_LEVELS, "pw_mid_sawp_norm")
    pw_mid_eneb = serie_pw(lon, lat, AREA_ENEB, MID_LEVELS, "pw_mid_eneb_norm")

    omega_sawp = serie_omega(lon, lat, AREA_SAWP, "omega_sawp_norm")
    omega_eneb = serie_omega(lon, lat, AREA_ENEB, "omega_eneb_norm")

    mfc = serie_mfc(lon, lat)

    # --------------------------------------------------------
    # Matriz geral: SLHF em lag 0
    # --------------------------------------------------------
    df = pd.concat(
        [
            sst,
            rain,
            slhf,
            wind,
            pw_low_sawp,
            pw_low_eneb,
            pw_mid_sawp,
            pw_mid_eneb,
            omega_sawp,
            omega_eneb,
            mfc,
        ],
        axis=1
    )

    df = df.loc[:, ~df.columns.duplicated()].copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna()

    # --------------------------------------------------------
    # Máscara dos quantis da TSM
    # --------------------------------------------------------
    q10 = df["sst_norm"].quantile(0.10)
    q25 = df["sst_norm"].quantile(0.25)
    q75 = df["sst_norm"].quantile(0.75)
    q90 = df["sst_norm"].quantile(0.90)

    mask_q10 = df["sst_norm"] <= q10
    mask_q25 = (df["sst_norm"] <= q25) & (df["sst_norm"] > q10)
    mask_q75 = (df["sst_norm"] >= q75) & (df["sst_norm"] < q90)
    mask_q90 = df["sst_norm"] >= q90

    mask_quantis_tsm = mask_q10 | mask_q25 | mask_q75 | mask_q90
    df_quantis = df.loc[mask_quantis_tsm].copy()

    # --------------------------------------------------------
    # Checagem SST × chuva
    # --------------------------------------------------------
    r0, p0 = pearsonr(
        df_quantis["sst_norm"],
        df_quantis["rain_norm"]
    )

    print("\nQuantis da TSM usados:")
    print(f"Q10 = {q10:.4f}")
    print(f"Q25 = {q25:.4f}")
    print(f"Q75 = {q75:.4f}")
    print(f"Q90 = {q90:.4f}")

    print("\nNúmero de anos por classe:")
    print(f"TSM <= Q10: {mask_q10.sum()}")
    print(f"Q10 < TSM <= Q25: {mask_q25.sum()}")
    print(f"Q75 <= TSM < Q90: {mask_q75.sum()}")
    print(f"TSM >= Q90: {mask_q90.sum()}")

    print("\nAnos usados:")
    print(df_quantis.index.tolist())

    print("\nChecagem SST × chuva usando máscara dos quantis:")
    print(f"n = {len(df_quantis)}")
    print(f"r = {r0:.4f}")
    print(f"p = {p0:.4f}")

    print("\nResumo matriz:")
    print(f"anos usados = {len(df_quantis)}")
    print(f"variáveis = {len(df_quantis.columns)}")
    print(f"comparações únicas = {len(df_quantis.columns) * (len(df_quantis.columns) - 1) // 2}")
    print(f"período = {df_quantis.index.min()}–{df_quantis.index.max()}")

    labels = {
        "sst_norm": "SAWP SST",
        "rain_norm": "ENEB rainfall",
        "slhf_sawp_norm": "SAWP LHF",
        "wind_sawp_norm": "SAWP wind",
        "pw_low_sawp_norm": "SAWP IWV\n1000–850 hPa",
        "pw_low_eneb_norm": "ENEB IWV\n1000–850 hPa",
        "pw_mid_sawp_norm": "SAWP IWV\n850–400 hPa",
        "pw_mid_eneb_norm": "ENEB IWV\n850–400 hPa",
        "omega_sawp_norm": "SAWP ω\n500 hPa",
        "omega_eneb_norm": "ENEB ω\n500 hPa",
        "mfc_eneb_norm": r"ENEB $\nabla \cdot \mathbf{Q}$",
    }

    # --------------------------------------------------------
    # Correlação geral: matriz com SLHF lag 0
    # --------------------------------------------------------
    rmat, pmat, smat, spmat, pares = stats_matriz(df_quantis)

    # --------------------------------------------------------
    # Substitui APENAS a célula SST × SLHF por lag -2 meses
    # --------------------------------------------------------
    df_sst_lhf_lag2 = pd.concat(
        [
            df_quantis["sst_norm"],
            slhf_lag2_sst
        ],
        axis=1
    ).dropna()

    r_lhf_lag2, p_lhf_lag2 = pearsonr(
        df_sst_lhf_lag2["sst_norm"],
        df_sst_lhf_lag2["slhf_sawp_lag_minus2_norm"]
    )

    rho_lhf_lag2, ps_lhf_lag2 = spearmanr(
        df_sst_lhf_lag2["sst_norm"],
        df_sst_lhf_lag2["slhf_sawp_lag_minus2_norm"]
    )

    rmat.loc["slhf_sawp_norm", "sst_norm"] = r_lhf_lag2
    pmat.loc["slhf_sawp_norm", "sst_norm"] = p_lhf_lag2

    smat.loc["slhf_sawp_norm", "sst_norm"] = rho_lhf_lag2
    spmat.loc["slhf_sawp_norm", "sst_norm"] = ps_lhf_lag2

    print("\nChecagem SST × SLHF com lag -2 meses:")
    print(f"n = {len(df_sst_lhf_lag2)}")
    print(f"Pearson r = {r_lhf_lag2:.4f}")
    print(f"Pearson p = {p_lhf_lag2:.4f}")
    print(f"Spearman rho = {rho_lhf_lag2:.4f}")
    print(f"Spearman p = {ps_lhf_lag2:.4f}")

    # --------------------------------------------------------
    # Salvamento
    # --------------------------------------------------------
    suffix = "metodo_original_AMJJ_anual_mascara_quantis_TSM"

    df.to_csv(FIG / f"S5C_series_todos_anos_{suffix}.csv")
    df_quantis.to_csv(FIG / f"S5C_series_anos_quantis_{suffix}.csv")
    df_sst_lhf_lag2.to_csv(FIG / f"S5C_series_SST_LHF_lag_minus2_{suffix}.csv")

    pares.to_csv(FIG / f"S5C_55_comparacoes_{suffix}.csv", index=False)

    rmat.to_csv(FIG / f"S5C_pearson_r_{suffix}.csv")
    pmat.to_csv(FIG / f"S5C_pearson_p_{suffix}.csv")
    smat.to_csv(FIG / f"S5C_spearman_rho_{suffix}.csv")
    spmat.to_csv(FIG / f"S5C_spearman_p_{suffix}.csv")

    plot_matriz(
        rmat,
        pmat,
        labels,
        "Pearson correlation matrix using TSM quantile-mask years\nSST × LHF uses lag -2 months",
        FIG / f"S5C_matriz_pearson_{suffix}.png"
    )

    plot_matriz(
        smat,
        spmat,
        labels,
        "Spearman correlation matrix using TSM quantile-mask years\nSST × LHF uses lag -2 months",
        FIG / f"S5C_matriz_spearman_{suffix}.png"
    )

    print("\nFinalizado.")