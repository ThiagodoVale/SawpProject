#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False

import sawp


# ============================================================
# CONFIGURAÇÕES
# ============================================================

TENDENCIAS = "sim"        # "sim" ou "não"
METODO_SAWP = "quadrado"  # "quadrado" ou "isoterma"
MESES_ESTACAO = [4, 5, 6, 7]  # AMJJ

SST_WARM_THRESHOLD = 1.0
RAIN_WET_THRESHOLD = 1.0
RAIN_DRY_THRESHOLD = 0.0

PW_LEVELS_HPA = [1000, 925, 850, 700, 600, 500, 400]
MFC_LEVELS_HPA = [1000, 925, 900, 875, 850]
OMEGA_LEVEL_HPA = 500

AREA_SAWP = {"lon_min": -34.0, "lon_max": -25.0, "lat_min": -12.5, "lat_max": -4.0}
AREA_MAP  = {"lon_min": -50.0, "lon_max": -25.0, "lat_min": -15.0, "lat_max": 0.0}
AREA_NEB  = {"lon_min": -45.0, "lon_max": -34.0, "lat_min": -15.0, "lat_max": 0.0}

AREA_LOW_LEVEL_PW_SERIES = AREA_SAWP
LOW_LEVEL_PW_LEVELS_HPA = [1000, 925, 900, 875, 850]

ISOTHERM_THRESHOLD_C = 28.5
CLIM_INI = 1980
CLIM_FIM = 2010

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs" if TENDENCIAS == "não" else BASE_DIR / "outputs_trended"
FIG_DIR = BASE_DIR / "figuras_revisor"
FIG_DIR.mkdir(parents=True, exist_ok=True)

if TENDENCIAS == "não":
    SST_FILE  = OUT_DIR / "detrended_sst.npz"
    U_FILE    = OUT_DIR / "detrended_uvq_u.npz"
    V_FILE    = OUT_DIR / "detrended_uvq_v.npz"
    Q_FILE    = OUT_DIR / "detrended_uvq_q.npz"
    W_FILE    = OUT_DIR / "detrended_uvq_w.npz"
    SLHF_FILE = OUT_DIR / "detrended_slhf.npz"
else:
    SST_FILE  = OUT_DIR / "trended_uvq_sst.npz"
    U_FILE    = OUT_DIR / "trended_uvq_u.npz"
    V_FILE    = OUT_DIR / "trended_uvq_v.npz"
    Q_FILE    = OUT_DIR / "trended_uvq_q.npz"
    W_FILE    = OUT_DIR / "trended_uvq_w.npz"
    SLHF_FILE = OUT_DIR / "trended_uvq_slhf.npz"

WIND_FALLBACK_LEVEL_HPA = 1000

if TENDENCIAS == "não":
    U10_CANDIDATES = [
        OUT_DIR / "detrended_u10.npz",
        OUT_DIR / "detrended_uvq_u10.npz",
        OUT_DIR / "detrended_10m_u_component_of_wind.npz",
    ]
    V10_CANDIDATES = [
        OUT_DIR / "detrended_v10.npz",
        OUT_DIR / "detrended_uvq_v10.npz",
        OUT_DIR / "detrended_10m_v_component_of_wind.npz",
    ]
else:
    U10_CANDIDATES = [
        OUT_DIR / "trended_u10.npz",
        OUT_DIR / "trended_uvq_u10.npz",
        OUT_DIR / "trended_10m_u_component_of_wind.npz",
    ]
    V10_CANDIDATES = [
        OUT_DIR / "trended_v10.npz",
        OUT_DIR / "trended_uvq_v10.npz",
        OUT_DIR / "trended_10m_v_component_of_wind.npz",
    ]

RAIN_FILE = DATA_DIR / "EstacoesDados.txt"
SST_NETCDF_REF = DATA_DIR / "Era5_Dataset_sst.nc"

G = 9.80665
R_EARTH = 6_371_000.0

PLOT_INDIVIDUAL_FIGURES = False


# ============================================================
# UTILITÁRIOS
# ============================================================

def to_celsius_if_needed(x):
    x = np.asarray(x, dtype=float)
    if np.nanmean(x) > 100:
        return x - 273.15
    return x


def get_level_key(npz):
    for key in ["level", "pressure_level", "levels"]:
        if key in npz:
            return key
    raise KeyError("Could not find vertical coordinate: expected 'level' or 'pressure_level'.")


def select_level_indices(levels, wanted_hpa):
    levels = np.asarray(levels, dtype=float)
    idx = [int(np.argmin(np.abs(levels - lev))) for lev in wanted_hpa]
    return np.unique(idx)


def area_mask(lon2d, lat2d, area):
    return (
        (lon2d >= area["lon_min"]) & (lon2d <= area["lon_max"]) &
        (lat2d >= area["lat_min"]) & (lat2d <= area["lat_max"])
    )


def annual_amjj_mean(series):
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    s = s[s.index.month.isin(MESES_ESTACAO)]
    return s.groupby(s.index.year).mean()


def normalize(s):
    return (s - s.mean()) / s.std()


def time_from_npz(npz):
    if "time" in npz:
        return pd.to_datetime(npz["time"])
    if "valid_time" in npz:
        return pd.to_datetime(npz["valid_time"])

    periodo_tempo_convertido, _, _ = sawp.ConfigurarTempoArea(str(SST_NETCDF_REF))
    return pd.to_datetime(periodo_tempo_convertido)


def first_existing_path(paths):
    for path in paths:
        if Path(path).exists():
            return Path(path)
    return None


def load_npz_data_and_time(path):
    npz = np.load(path, mmap_mode="r", allow_pickle=True)

    if "data" in npz:
        data = npz["data"]
    else:
        coord_keys = {
            "time", "valid_time", "level", "levels", "pressure_level",
            "lat", "lon", "latitude", "longitude"
        }
        candidates = [k for k in npz.files if k not in coord_keys]
        if len(candidates) == 0:
            raise KeyError(f"Could not find data array in {path}.")
        data = npz[candidates[0]]

    return data, time_from_npz(npz)


def load_4d(path):
    npz = np.load(path, mmap_mode="r", allow_pickle=True)
    data = npz["data"]
    levels = np.asarray(npz[get_level_key(npz)], dtype=float)
    time = time_from_npz(npz)
    return data, levels, time


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
# SÉRIES SAWP / ENEB
# ============================================================

def load_sawp_sst_series():
    periodo_tempo_convertido, lon2d, lat2d = sawp.ConfigurarTempoArea(str(SST_NETCDF_REF))
    time = pd.to_datetime(periodo_tempo_convertido)

    sst = sawp.CarregarDados(
        str(SST_FILE),
        tendencias=TENDENCIAS,
        nome_var_nc="sst",
        substituir_nan=-999,
    )

    sst_c = to_celsius_if_needed(sst)

    if METODO_SAWP == "quadrado":
        mask_sawp = area_mask(lon2d, lat2d, AREA_SAWP)

    elif METODO_SAWP == "isoterma":
        mask_clim = (
            (time.month == 4) &
            (time.year >= CLIM_INI) &
            (time.year <= CLIM_FIM)
        )
        clim_april = np.nanmean(sst_c[mask_clim], axis=0)
        mask_sawp = clim_april >= ISOTHERM_THRESHOLD_C
        mask_sawp = mask_sawp & (lon2d >= -40.0) & (lon2d <= -25.0)
        print(f"SAWP isotherm points: {int(mask_sawp.sum())}")

    else:
        raise ValueError("METODO_SAWP must be 'quadrado' or 'isoterma'.")

    sst_sawp = np.nanmean(
        np.where(mask_sawp[None, :, :], sst_c, np.nan),
        axis=(1, 2)
    )

    return pd.Series(sst_sawp, index=time, name="sawp_sst_c"), time, lon2d, lat2d, mask_sawp


def load_sawp_surface_wind_series(lon2d, lat2d, mask_sawp):
    u10_path = first_existing_path(U10_CANDIDATES)
    v10_path = first_existing_path(V10_CANDIDATES)

    if u10_path is not None and v10_path is not None:
        print(f"Loading surface wind from 10-m files: {u10_path.name}, {v10_path.name}")

        u10, time_u = load_npz_data_and_time(u10_path)
        v10, time_v = load_npz_data_and_time(v10_path)

        ntime = min(len(time_u), len(time_v), u10.shape[0], v10.shape[0])
        time = time_u[:ntime]

        u10 = u10[:ntime]
        v10 = v10[:ntime]

        if u10.ndim == 4:
            u10 = u10[:, 0]
        if v10.ndim == 4:
            v10 = v10[:, 0]

        wind = np.sqrt(u10**2 + v10**2)

    else:
        print(
            "10-m wind files not found. "
            f"Using nearest pressure level to {WIND_FALLBACK_LEVEL_HPA} hPa."
        )

        u, lev_u, time_u = load_4d(U_FILE)
        v, lev_v, time_v = load_4d(V_FILE)

        idx_u = int(np.argmin(np.abs(lev_u - WIND_FALLBACK_LEVEL_HPA)))
        idx_v = int(np.argmin(np.abs(lev_v - WIND_FALLBACK_LEVEL_HPA)))

        print(f"Wind fallback levels: u={lev_u[idx_u]:.0f} hPa, v={lev_v[idx_v]:.0f} hPa")

        ntime = min(len(time_u), len(time_v), u.shape[0], v.shape[0])
        time = time_u[:ntime]

        wind = np.sqrt(
            u[:ntime, idx_u, :, :]**2 +
            v[:ntime, idx_v, :, :]**2
        )

    wind_sawp = np.nanmean(
        np.where(mask_sawp[None, :, :], wind, np.nan),
        axis=(1, 2)
    )

    return pd.Series(wind_sawp, index=pd.to_datetime(time), name="sawp_surface_wind_ms")


def load_sawp_slhf_series(lon2d, lat2d, mask_sawp):
    """
    ERA5 SLHF:
      standard_name = surface_upward_latent_heat_flux
      units = J m-2
    Mantém o sinal original: positivo = fluxo latente para cima.
    """
    slhf, time_slhf = load_npz_data_and_time(SLHF_FILE)

    if slhf.ndim == 4:
        slhf = slhf[:, 0, :, :]

    slhf_sawp = np.nanmean(
        np.where(mask_sawp[None, :, :], slhf, np.nan),
        axis=(1, 2)
    )

    return pd.Series(slhf_sawp, index=pd.to_datetime(time_slhf), name="sawp_slhf_j_m2")


def load_area_omega_series(lon2d, lat2d, area=AREA_NEB):
    w, lev_w, time_w = load_4d(W_FILE)

    idx_w = int(np.argmin(np.abs(lev_w - OMEGA_LEVEL_HPA)))
    print(f"Omega/w time series level: {lev_w[idx_w]:.0f} hPa")

    mask = area_mask(lon2d, lat2d, area)

    omega_area = np.nanmean(
        np.where(mask[None, :, :], w[:, idx_w, :, :], np.nan),
        axis=(1, 2)
    )

    return pd.Series(
        omega_area,
        index=pd.to_datetime(time_w),
        name=f"omega_{int(lev_w[idx_w])}hpa_area_pa_s",
    )


def load_low_level_precipitable_water_series(lon2d, lat2d, area=AREA_LOW_LEVEL_PW_SERIES):
    q, lev_q, time_q = load_4d(Q_FILE)

    idx_q = select_level_indices(lev_q, LOW_LEVEL_PW_LEVELS_HPA)
    used_levels = lev_q[idx_q]

    print(
        "Low-level precipitable-water levels:",
        ", ".join(f"{lev:.0f}" for lev in used_levels),
        "hPa"
    )

    q_sel = q[:, idx_q, :, :]
    pw_low = integrate_pressure(q_sel, used_levels)

    mask = area_mask(lon2d, lat2d, area)

    pw_area = np.nanmean(
        np.where(mask[None, :, :], pw_low, np.nan),
        axis=(1, 2)
    )

    return pd.Series(
        pw_area,
        index=pd.to_datetime(time_q),
        name="low_level_pw_1000_850_kg_m2",
    )


def load_eneb_rainfall_series():
    rain = sawp.carregar_chuva_periodo(str(RAIN_FILE), TENDENCIAS)
    rain = pd.Series(np.asarray(rain).squeeze(), index=rain.index, name="eneb_rain_mm")
    rain = rain.replace(-999, np.nan)
    return rain


# ============================================================
# EVENT TABLE
# ============================================================

def safe_pearson(x, y):
    tmp = pd.concat([x, y], axis=1).dropna()
    if len(tmp) < 3:
        return np.nan, np.nan
    r, p = pearsonr(tmp.iloc[:, 0], tmp.iloc[:, 1])
    return r, p


def build_event_table():
    sst_monthly, _, lon2d, lat2d, mask_sawp = load_sawp_sst_series()

    wind_monthly = load_sawp_surface_wind_series(lon2d, lat2d, mask_sawp)
    slhf_monthly = load_sawp_slhf_series(lon2d, lat2d, mask_sawp)
    omega_neb_monthly = load_area_omega_series(lon2d, lat2d, AREA_NEB)
    pw_low_monthly = load_low_level_precipitable_water_series(
        lon2d, lat2d, AREA_LOW_LEVEL_PW_SERIES
    )
    rain_season = load_eneb_rainfall_series()

    sst_amjj = annual_amjj_mean(sst_monthly)
    wind_amjj = annual_amjj_mean(wind_monthly)
    slhf_amjj = annual_amjj_mean(slhf_monthly)
    omega_neb_amjj = annual_amjj_mean(omega_neb_monthly)
    pw_low_amjj = annual_amjj_mean(pw_low_monthly)

    rain_season.index = rain_season.index.astype(int)

    df = pd.concat(
        [
            normalize(sst_amjj).rename("sst_norm"),
            normalize(wind_amjj).rename("wind_norm"),
            normalize(slhf_amjj).rename("slhf_norm"),
            normalize(omega_neb_amjj).rename("omega_neb_norm"),
            normalize(pw_low_amjj).rename("pw_low_norm"),
            normalize(rain_season).rename("rain_norm"),

            sst_amjj.rename("sst_amjj_c"),
            wind_amjj.rename("wind_amjj_ms"),
            slhf_amjj.rename("slhf_amjj_j_m2"),
            omega_neb_amjj.rename("omega_neb_amjj_pa_s"),
            pw_low_amjj.rename("pw_low_1000_850_amjj_kg_m2"),
            rain_season.rename("rain_amjj_mm"),
        ],
        axis=1
    ).dropna()

    corr_pairs = {
        "sst_rain": ("sst_norm", "rain_norm"),
        "wind_rain": ("wind_norm", "rain_norm"),
        "sst_wind": ("sst_norm", "wind_norm"),
        "sst_slhf": ("sst_norm", "slhf_norm"),
        "slhf_rain": ("slhf_norm", "rain_norm"),
        "wind_slhf": ("wind_norm", "slhf_norm"),
        "slhf_pw_low": ("slhf_norm", "pw_low_norm"),
        "omega_rain": ("omega_neb_norm", "rain_norm"),
        "omega_sst": ("omega_neb_norm", "sst_norm"),
        "sst_pw_low": ("sst_norm", "pw_low_norm"),
        "pw_low_rain": ("pw_low_norm", "rain_norm"),
    }

    for name, (a, b) in corr_pairs.items():
        r, p = safe_pearson(df[a], df[b])
        df.attrs[f"r_{name}"] = r
        df.attrs[f"p_{name}"] = p

    # Compatibilidade com o script antigo
    df.attrs["r"] = df.attrs["r_sst_rain"]
    df.attrs["p"] = df.attrs["p_sst_rain"]

    df["group"] = "other"

    df.loc[
        (df["sst_norm"] >= SST_WARM_THRESHOLD) &
        (df["rain_norm"] >= RAIN_WET_THRESHOLD),
        "group"
    ] = "warm_wet"

    df.loc[
        (df["sst_norm"] >= SST_WARM_THRESHOLD) &
        (df["rain_norm"] <= RAIN_DRY_THRESHOLD),
        "group"
    ] = "warm_dry"

    df.loc[
        (df["sst_norm"] <= -SST_WARM_THRESHOLD) &
        (df["rain_norm"] <= -RAIN_WET_THRESHOLD),
        "group"
    ] = "cold_dry"

    return df


# ============================================================
# COMPOSITES
# ============================================================

def select_amjj_time_indices(time, years):
    years = set(int(y) for y in years)
    mask = np.array([(t.year in years) and (t.month in MESES_ESTACAO) for t in time])
    return np.where(mask)[0]


def composite_fields(years, lon2d, lat2d):
    if len(years) == 0:
        raise ValueError("No years supplied for composite.")

    u, lev_u, time_u = load_4d(U_FILE)
    v, lev_v, time_v = load_4d(V_FILE)
    q, lev_q, time_q = load_4d(Q_FILE)
    w, lev_w, time_w = load_4d(W_FILE)

    if len(time_u) != len(time_q):
        warnings.warn("u and q have different time lengths. Make sure files are aligned.")

    tidx = select_amjj_time_indices(time_q, years)

    if len(tidx) == 0:
        raise ValueError(f"No AMJJ months found for years: {years}")

    idx_pw_q = select_level_indices(lev_q, PW_LEVELS_HPA)
    idx_mfc_q = select_level_indices(lev_q, MFC_LEVELS_HPA)
    idx_mfc_u = select_level_indices(lev_u, MFC_LEVELS_HPA)
    idx_mfc_v = select_level_indices(lev_v, MFC_LEVELS_HPA)
    idx_w = int(np.argmin(np.abs(lev_w - OMEGA_LEVEL_HPA)))

    q_pw = q[tidx][:, idx_pw_q, :, :]
    pw = integrate_pressure(q_pw, lev_q[idx_pw_q])
    pw_comp = np.nanmean(pw, axis=0)

    q_mfc = q[tidx][:, idx_mfc_q, :, :]
    u_mfc = u[tidx][:, idx_mfc_u, :, :]
    v_mfc = v[tidx][:, idx_mfc_v, :, :]

    nlev = min(q_mfc.shape[1], u_mfc.shape[1], v_mfc.shape[1])

    q_mfc = q_mfc[:, :nlev]
    u_mfc = u_mfc[:, :nlev]
    v_mfc = v_mfc[:, :nlev]

    lev_mfc = lev_q[idx_mfc_q][:nlev]

    flux_u = integrate_pressure(q_mfc * u_mfc, lev_mfc)
    flux_v = integrate_pressure(q_mfc * v_mfc, lev_mfc)

    conv_list = []

    for i in range(flux_u.shape[0]):
        div = horizontal_divergence(flux_u[i], flux_v[i], lon2d, lat2d)
        conv_list.append(-div)

    mfc_comp = np.nanmean(np.asarray(conv_list), axis=0)

    tidx_w = select_amjj_time_indices(time_w, years)
    omega_comp = np.nanmean(w[tidx_w][:, idx_w, :, :], axis=0)

    return {
        "pw": pw_comp,
        "mfc": mfc_comp,
        "omega": omega_comp,
        "omega_level": lev_w[idx_w],
        "years": years,
    }


# ============================================================
# PLOTS
# ============================================================

def setup_map_axis(ax, area=AREA_MAP):
    if HAS_CARTOPY:
        ax.set_extent(
            [area["lon_min"], area["lon_max"], area["lat_min"], area["lat_max"]],
            crs=ccrs.PlateCarree()
        )

        ax.add_feature(cfeature.LAND, alpha=0.7)
        ax.add_feature(cfeature.OCEAN, alpha=0.7)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)

        try:
            states = cfeature.NaturalEarthFeature(
                "cultural",
                "admin_1_states_provinces",
                "10m",
                facecolor="none",
            )
            ax.add_feature(states, edgecolor="gray", linewidth=0.4)
        except Exception:
            pass

        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.4,
            color="gray",
            alpha=0.4,
            linestyle="--",
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {"size": 8}
        gl.ylabel_style = {"size": 8}

    else:
        ax.set_xlim(area["lon_min"], area["lon_max"])
        ax.set_ylim(area["lat_min"], area["lat_max"])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.3)


def add_panel_label(ax, label):
    ax.text(
        0.01,
        0.98,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
        zorder=20,
    )


def map_levels(field, diff=False, nlev=13):
    field = np.asarray(field, dtype=float)

    if diff:
        vmax = np.nanpercentile(np.abs(field), 98)
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        return np.linspace(-vmax, vmax, nlev)

    vmin = np.nanpercentile(field, 2)
    vmax = np.nanpercentile(field, 98)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = np.nanmin(field), np.nanmax(field)

    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin, vmax = -1.0, 1.0

    return np.linspace(vmin, vmax, nlev)


def plot_map_subplot(fig, ax, lon2d, lat2d, field, title, label, cmap, diff=False):
    setup_map_axis(ax)
    levels = map_levels(field, diff=diff)

    if HAS_CARTOPY:
        im = ax.contourf(
            lon2d,
            lat2d,
            field,
            levels=levels,
            cmap=cmap,
            extend="both",
            transform=ccrs.PlateCarree(),
        )
    else:
        im = ax.contourf(
            lon2d,
            lat2d,
            field,
            levels=levels,
            cmap=cmap,
            extend="both",
        )

    ax.set_title(title, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.76, pad=0.02)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    return im


def plot_maps_only_panel(event_table, wet, dry, lon2d, lat2d):
    diff_pw = wet["pw"] - dry["pw"]
    diff_mfc = wet["mfc"] - dry["mfc"]
    diff_omega = wet["omega"] - dry["omega"]

    projection = ccrs.PlateCarree() if HAS_CARTOPY else None

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(18, 14),
        subplot_kw={"projection": projection} if HAS_CARTOPY else None,
    )

    axes = axes.ravel()

    plot_map_subplot(fig, axes[0], lon2d, lat2d, wet["pw"], "Warm-wet", "mm", "BrBG")
    plot_map_subplot(fig, axes[1], lon2d, lat2d, dry["pw"], "Warm-dry", "mm", "BrBG")
    plot_map_subplot(fig, axes[2], lon2d, lat2d, diff_pw, "Warm-wet minus warm-dry", "mm", "BrBG", diff=True)
    axes[0].set_ylabel("Precipitable water", fontsize=11)

    plot_map_subplot(fig, axes[3], lon2d, lat2d, wet["mfc"], "Warm-wet", "kg m$^{-2}$ s$^{-1}$", "PuOr")
    plot_map_subplot(fig, axes[4], lon2d, lat2d, dry["mfc"], "Warm-dry", "kg m$^{-2}$ s$^{-1}$", "PuOr")
    plot_map_subplot(fig, axes[5], lon2d, lat2d, diff_mfc, "Warm-wet minus warm-dry", "kg m$^{-2}$ s$^{-1}$", "PuOr", diff=True)
    axes[3].set_ylabel("Moisture convergence", fontsize=11)

    omega_title = f"Vertical velocity at {int(wet['omega_level'])} hPa"

    plot_map_subplot(fig, axes[6], lon2d, lat2d, wet["omega"], "Warm-wet", "Pa s$^{-1}$", "RdBu_r")
    plot_map_subplot(fig, axes[7], lon2d, lat2d, dry["omega"], "Warm-dry", "Pa s$^{-1}$", "RdBu_r")
    plot_map_subplot(fig, axes[8], lon2d, lat2d, diff_omega, "Warm-wet minus warm-dry", "Pa s$^{-1}$", "RdBu_r", diff=True)
    axes[6].set_ylabel(omega_title, fontsize=11)

    for label, ax in zip(list("abcdefghi"), axes):
        add_panel_label(ax, label)

    r = event_table.attrs.get("r", np.nan)
    p = event_table.attrs.get("p", np.nan)

    fig.suptitle(
        "Atmospheric composites for coupled and decoupled warm SAWP years\n"
        f"SST-rainfall relationship: r={r:.2f}, p={p:.3f} | "
        f"Warm-wet: {', '.join(map(str, wet['years']))} | "
        f"Warm-dry: {', '.join(map(str, dry['years']))}",
        fontsize=15,
        y=0.995,
    )

    plt.tight_layout()

    out = FIG_DIR / "S5A_MAPAS_composites_warm_wet_warm_dry_com_slhf.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out}")


def plot_paired_series(ax, event_table, y1, y2, label1, label2, title, r_attr, p_attr):
    ax.plot(event_table.index, event_table[y1], marker="o", linewidth=1.8, label=label1)
    ax.plot(event_table.index, event_table[y2], marker="s", linewidth=1.8, label=label2)
    ax.axhline(0, color="k", linestyle="--", linewidth=1)

    ax.set_title(
        f"{title}\n"
        f"r = {event_table.attrs.get(r_attr, np.nan):.2f}, "
        f"p = {event_table.attrs.get(p_attr, np.nan):.3f}",
        fontsize=12,
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Normalized anomaly")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)


def plot_series_only_panel(event_table):
    fig, axes = plt.subplots(4, 2, figsize=(18, 18), sharex=True)
    axes = axes.ravel()

    plot_paired_series(
        axes[0], event_table,
        "sst_norm", "rain_norm",
        "SAWP SST", "ENEB rainfall",
        "SAWP SST x ENEB rainfall",
        "r_sst_rain", "p_sst_rain",
    )

    plot_paired_series(
        axes[1], event_table,
        "sst_norm", "wind_norm",
        "SAWP SST", "SAWP surface wind speed",
        "SAWP SST x SAWP surface wind speed",
        "r_sst_wind", "p_sst_wind",
    )

    plot_paired_series(
        axes[2], event_table,
        "sst_norm", "slhf_norm",
        "SAWP SST", "SAWP latent heat flux",
        "SAWP SST x SAWP latent heat flux",
        "r_sst_slhf", "p_sst_slhf",
    )

    plot_paired_series(
        axes[3], event_table,
        "slhf_norm", "rain_norm",
        "SAWP latent heat flux", "ENEB rainfall",
        "SAWP latent heat flux x ENEB rainfall",
        "r_slhf_rain", "p_slhf_rain",
    )

    plot_paired_series(
        axes[4], event_table,
        "omega_neb_norm", "rain_norm",
        f"NEB omega/w at {OMEGA_LEVEL_HPA} hPa", "ENEB rainfall",
        "NEB omega/w x ENEB rainfall",
        "r_omega_rain", "p_omega_rain",
    )

    plot_paired_series(
        axes[5], event_table,
        "omega_neb_norm", "sst_norm",
        f"NEB omega/w at {OMEGA_LEVEL_HPA} hPa", "SAWP SST",
        "NEB omega/w x SAWP SST",
        "r_omega_sst", "p_omega_sst",
    )

    plot_paired_series(
        axes[6], event_table,
        "sst_norm", "pw_low_norm",
        "SAWP SST", "Low-level precipitable water 1000-850 hPa",
        "SAWP SST x low-level precipitable water",
        "r_sst_pw_low", "p_sst_pw_low",
    )

    plot_paired_series(
        axes[7], event_table,
        "pw_low_norm", "rain_norm",
        "Low-level precipitable water 1000-850 hPa", "ENEB rainfall",
        "Low-level precipitable water x ENEB rainfall",
        "r_pw_low_rain", "p_pw_low_rain",
    )

    for label, ax in zip(list("abcdefgh"), axes):
        add_panel_label(ax, label)

    fig.suptitle(
        "Normalized AMJJ annual time series used to diagnose SAWP-ENEB coupling",
        fontsize=15,
        y=0.995,
    )

    plt.tight_layout()

    out = FIG_DIR / "S5B_SERIES_diagnosticos_SAWP_ENEB_com_slhf.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out}")


def plot_timeseries(event_table):
    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(event_table.index, event_table["sst_norm"], marker="o", label="SAWP SST")
    ax.plot(event_table.index, event_table["rain_norm"], marker="s", label="ENEB rainfall")
    ax.plot(event_table.index, event_table["wind_norm"], marker="^", label="SAWP wind")
    ax.plot(event_table.index, event_table["slhf_norm"], marker="D", label="SAWP SLHF")

    ax.axhline(0, color="k", linestyle="--", linewidth=1)

    warm_wet = event_table[event_table["group"] == "warm_wet"]
    warm_dry = event_table[event_table["group"] == "warm_dry"]

    ax.scatter(
        warm_wet.index,
        warm_wet["sst_norm"],
        s=90,
        marker="o",
        edgecolor="k",
        zorder=5,
        label="Warm-wet years",
    )

    ax.scatter(
        warm_dry.index,
        warm_dry["sst_norm"],
        s=90,
        marker="X",
        edgecolor="k",
        zorder=5,
        label="Warm-dry years",
    )

    r = event_table.attrs.get("r", np.nan)
    p = event_table.attrs.get("p", np.nan)

    ax.set_title(
        f"SAWP SST, SLHF, wind speed, and ENEB rainfall anomalies during AMJJ\n"
        f"SST-rainfall: r = {r:.2f}, p = {p:.3f}; n = {len(event_table)}"
    )

    ax.set_xlabel("Year")
    ax.set_ylabel("Normalized anomaly")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")

    plt.tight_layout()

    out = FIG_DIR / "S5_SAWP_SST_SLHF_ENEB_rainfall_event_groups.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {out}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("Building SAWP SST / ENEB rainfall event table...")
    event_table = build_event_table()
    event_table.to_csv(FIG_DIR / "S5_event_table_sawp_sst_eneb_rainfall_com_slhf.csv")

    cols_print = [
        "sst_norm",
        "wind_norm",
        "slhf_norm",
        "omega_neb_norm",
        "pw_low_norm",
        "rain_norm",
    ]

    print("\nEvent groups:")
    print(event_table.groupby("group").size())

    print("\nWarm-wet years:")
    print(event_table[event_table["group"] == "warm_wet"][cols_print])

    print("\nWarm-dry years:")
    print(event_table[event_table["group"] == "warm_dry"][cols_print])

    _, _, lon2d, lat2d, _ = load_sawp_sst_series()

    warm_wet_years = event_table.index[event_table["group"] == "warm_wet"].astype(int).tolist()
    warm_dry_years = event_table.index[event_table["group"] == "warm_dry"].astype(int).tolist()

    print("\nSelected years:")
    print("Warm-wet years:", warm_wet_years)
    print("Warm-dry years:", warm_dry_years)

    if len(warm_wet_years) < 2 or len(warm_dry_years) < 2:
        print("\nToo few years using strict thresholds.")
        print("Relaxing thresholds automatically...")

        SST_WARM_THRESHOLD = 0.50
        RAIN_WET_THRESHOLD = 0.50
        RAIN_DRY_THRESHOLD = 0.00

        event_table["group"] = "other"

        event_table.loc[
            (event_table["sst_norm"] >= SST_WARM_THRESHOLD) &
            (event_table["rain_norm"] >= RAIN_WET_THRESHOLD),
            "group"
        ] = "warm_wet"

        event_table.loc[
            (event_table["sst_norm"] >= SST_WARM_THRESHOLD) &
            (event_table["rain_norm"] <= RAIN_DRY_THRESHOLD),
            "group"
        ] = "warm_dry"

        event_table.to_csv(FIG_DIR / "S5_event_table_sawp_sst_eneb_rainfall_com_slhf.csv")

        warm_wet_years = event_table.index[event_table["group"] == "warm_wet"].astype(int).tolist()
        warm_dry_years = event_table.index[event_table["group"] == "warm_dry"].astype(int).tolist()

        print("\nSelected years after relaxing thresholds:")
        print("Warm-wet years:", warm_wet_years)
        print("Warm-dry years:", warm_dry_years)

    if len(warm_wet_years) < 2 or len(warm_dry_years) < 2:
        raise RuntimeError(
            "Still too few years. Inspect event_table.csv and reduce thresholds further."
        )

    print("\nCalculating warm-wet composites...")
    comp_wet = composite_fields(warm_wet_years, lon2d, lat2d)

    print("Calculating warm-dry composites...")
    comp_dry = composite_fields(warm_dry_years, lon2d, lat2d)

    np.savez_compressed(
        FIG_DIR / "S5_composite_fields_warm_wet_warm_dry_com_slhf.npz",
        warm_wet_years=np.asarray(warm_wet_years),
        warm_dry_years=np.asarray(warm_dry_years),
        lon=lon2d,
        lat=lat2d,
        pw_warm_wet=comp_wet["pw"],
        pw_warm_dry=comp_dry["pw"],
        mfc_warm_wet=comp_wet["mfc"],
        mfc_warm_dry=comp_dry["mfc"],
        omega_warm_wet=comp_wet["omega"],
        omega_warm_dry=comp_dry["omega"],
        omega_level=comp_wet["omega_level"],
    )

    print("\nGenerating map-only panel...")
    plot_maps_only_panel(event_table, comp_wet, comp_dry, lon2d, lat2d)

    print("\nGenerating series-only panel...")
    plot_series_only_panel(event_table)

    if PLOT_INDIVIDUAL_FIGURES:
        print("\nGenerating optional legacy figures...")
        plot_timeseries(event_table)

    print("\nDone.")
    print(f"Outputs saved in: {FIG_DIR}")