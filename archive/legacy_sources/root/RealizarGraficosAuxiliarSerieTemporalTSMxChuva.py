#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary analysis requested by the reviewer:
SAWP SST anomaly x ENEB rainfall anomaly and atmospheric composites.

Goal
----
Identify years with:
  1) warm SAWP SST and wet ENEB rainfall  -> coupled warm-wet years
  2) warm SAWP SST and dry/normal ENEB rainfall -> decoupled warm-dry years

Then compare AMJJ composites of:
  - precipitable water, integrated from q
  - low-level moisture flux convergence, integrated from u, v, q
  - vertical velocity at a selected pressure level

This directly tests whether warm SAWP conditions accompanied by weak rainfall
are associated with an unfavorable atmospheric circulation, such as weak
moisture convergence and/or subsidence.

Expected input paths follow the structure used in RealizarAnalises_dev.py.
"""

from __future__ import annotations

import os
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
# USER CONFIGURATION
# ============================================================

TENDENCIAS = "sim"       # "sim" = trended data; "não" = detrended data
METODO_SAWP = "quadrado" # "quadrado" or "isoterma"
MESES_ESTACAO = [4, 5, 6, 7]  # AMJJ

# Thresholds for event classification
SST_WARM_THRESHOLD = 1.0   # SAWP SST anomaly >= +1 std
RAIN_WET_THRESHOLD = 1.0   # ENEB rainfall anomaly >= +1 std
RAIN_DRY_THRESHOLD = 0.0   # ENEB rainfall anomaly <= 0 std

# Composite variables
PW_LEVELS_HPA = [1000, 925, 850, 700, 600, 500, 400]      # column for precipitable water
MFC_LEVELS_HPA = [1000, 925, 900, 875, 850]               # low-level moisture convergence
OMEGA_LEVEL_HPA = 500                                     # vertical velocity level

# Spatial domains
AREA_SAWP = {"lon_min": -34.0, "lon_max": -25.0, "lat_min": -12.5, "lat_max": -4.0}
AREA_MAP  = {"lon_min": -50.0, "lon_max": -25.0, "lat_min": -15.0, "lat_max":  0.0}

# April climatology used only if METODO_SAWP == "isoterma"
ISOTHERM_THRESHOLD_C = 28.5
CLIM_INI = 1980
CLIM_FIM = 2010

# Paths
BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs" if TENDENCIAS == "não" else BASE_DIR / "outputs_trended"
FIG_DIR = BASE_DIR / "figuras_revisor"
FIG_DIR.mkdir(parents=True, exist_ok=True)

if TENDENCIAS == "não":
    SST_FILE = OUT_DIR / "detrended_sst.npz"
    U_FILE   = OUT_DIR / "detrended_uvq_u.npz"
    V_FILE   = OUT_DIR / "detrended_uvq_v.npz"
    Q_FILE   = OUT_DIR / "detrended_uvq_q.npz"
    W_FILE   = OUT_DIR / "detrended_uvq_w.npz"
else:
    SST_FILE = OUT_DIR / "trended_uvq_sst.npz"
    U_FILE   = OUT_DIR / "trended_uvq_u.npz"
    V_FILE   = OUT_DIR / "trended_uvq_v.npz"
    Q_FILE   = OUT_DIR / "trended_uvq_q.npz"
    W_FILE   = OUT_DIR / "trended_uvq_w.npz"

RAIN_FILE = DATA_DIR / "EstacoesDados.txt"
SST_NETCDF_REF = DATA_DIR / "Era5_Dataset_sst.nc"

G = 9.80665
R_EARTH = 6_371_000.0


# ============================================================
# BASIC UTILITIES
# ============================================================

def to_celsius_if_needed(x: np.ndarray) -> np.ndarray:
    """Convert SST from K to °C when values indicate Kelvin."""
    x = np.asarray(x, dtype=float)
    if np.nanmean(x) > 100:
        return x - 273.15
    return x


def get_level_key(npz) -> str:
    for key in ["level", "pressure_level", "levels"]:
        if key in npz:
            return key
    raise KeyError("Could not find vertical coordinate: expected 'level' or 'pressure_level'.")


def select_level_indices(levels: np.ndarray, wanted_hpa: list[float]) -> np.ndarray:
    """Return nearest indices for wanted pressure levels."""
    levels = np.asarray(levels, dtype=float)
    idx = []
    for lev in wanted_hpa:
        idx.append(int(np.argmin(np.abs(levels - lev))))
    return np.unique(idx)


def area_mask(lon2d: np.ndarray, lat2d: np.ndarray, area: dict) -> np.ndarray:
    return (
        (lon2d >= area["lon_min"]) & (lon2d <= area["lon_max"]) &
        (lat2d >= area["lat_min"]) & (lat2d <= area["lat_max"])
    )


def annual_amjj_mean(series: pd.Series) -> pd.Series:
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    s = s[s.index.month.isin(MESES_ESTACAO)]
    return s.groupby(s.index.year).mean()


def normalize(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std()


def time_from_npz(npz) -> pd.DatetimeIndex:
    if "time" in npz:
        return pd.to_datetime(npz["time"])
    if "valid_time" in npz:
        return pd.to_datetime(npz["valid_time"])
    # fallback: same time returned by ERA5 reference file
    periodo_tempo_convertido, _, _ = sawp.ConfigurarTempoArea(str(SST_NETCDF_REF))
    return pd.to_datetime(periodo_tempo_convertido)


# ============================================================
# SAWP SST AND ENEB RAINFALL SERIES
# ============================================================

def load_sawp_sst_series() -> tuple[pd.Series, pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """Return monthly SAWP SST series and grid information."""
    periodo_tempo_convertido, lon2d, lat2d = sawp.ConfigurarTempoArea(str(SST_NETCDF_REF))
    time = pd.to_datetime(periodo_tempo_convertido)

    sst = sawp.CarregarDados(
        str(SST_FILE),
        tendencias=TENDENCIAS,
        nome_var_nc="sst",
        substituir_nan=-999
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
        clim_april = np.nanmean(sst_c[mask_clim, :, :], axis=0)
        mask_sawp = clim_april >= ISOTHERM_THRESHOLD_C
        mask_sawp = mask_sawp & (lon2d >= -40.0) & (lon2d <= -25.0)
        print(f"SAWP isotherm points: {int(mask_sawp.sum())}")

    else:
        raise ValueError("METODO_SAWP must be 'quadrado' or 'isoterma'.")

    sst_sawp = np.nanmean(np.where(mask_sawp[None, :, :], sst_c, np.nan), axis=(1, 2))
    return pd.Series(sst_sawp, index=time, name="sawp_sst_c"), time, lon2d, lat2d, mask_sawp


def load_eneb_rainfall_series() -> pd.Series:
    """Load the seasonal rainfall series using the same routine as the main analysis."""
    rain = sawp.carregar_chuva_periodo(str(RAIN_FILE), TENDENCIAS)
    rain = pd.Series(np.asarray(rain).squeeze(), index=rain.index, name="eneb_rain_mm")
    rain = rain.replace(-999, np.nan)
    return rain


def build_event_table() -> pd.DataFrame:
    sst_monthly, _, _, _, _ = load_sawp_sst_series()
    rain_season = load_eneb_rainfall_series()

    sst_amjj = annual_amjj_mean(sst_monthly)
    rain_season.index = rain_season.index.astype(int)

    df = pd.concat(
        [
            normalize(sst_amjj).rename("sst_norm"),
            normalize(rain_season).rename("rain_norm"),
            sst_amjj.rename("sst_amjj_c"),
            rain_season.rename("rain_amjj_mm"),
        ],
        axis=1
    ).dropna()

    r, p = pearsonr(df["sst_norm"], df["rain_norm"])
    df.attrs["r"] = r
    df.attrs["p"] = p

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
# ATMOSPHERIC COMPOSITES
# ============================================================

def select_amjj_time_indices(time: pd.DatetimeIndex, years: list[int]) -> np.ndarray:
    years = set(int(y) for y in years)
    mask = np.array([(t.year in years) and (t.month in MESES_ESTACAO) for t in time])
    return np.where(mask)[0]


def load_4d(path: Path) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    npz = np.load(path, mmap_mode="r", allow_pickle=True)
    data = npz["data"]
    levels = np.asarray(npz[get_level_key(npz)], dtype=float)
    time = time_from_npz(npz)
    return data, levels, time


def integrate_pressure(field: np.ndarray, levels_hpa: np.ndarray) -> np.ndarray:
    """
    Pressure integration of a field with dimensions time, level, lat, lon.
    Integral field dp/g. Levels can be ascending or descending.
    """
    order = np.argsort(levels_hpa)
    lev_pa = levels_hpa[order] * 100.0
    fld = field[:, order, :, :]
    return np.trapz(fld, x=lev_pa, axis=1) / G


def horizontal_divergence(fx: np.ndarray, fy: np.ndarray, lon2d: np.ndarray, lat2d: np.ndarray) -> np.ndarray:
    """
    Approximate horizontal divergence on a regular lat-lon grid.
    fx and fy are 2D fluxes integrated vertically.
    """
    lat_rad = np.deg2rad(lat2d[:, 0])
    lon_rad = np.deg2rad(lon2d[0, :])

    dy = np.gradient(lat_rad) * R_EARTH
    dx = np.gradient(lon_rad)[None, :] * R_EARTH * np.cos(lat_rad)[:, None]

    dfx_dx = np.gradient(fx, axis=1) / dx
    dfy_dy = np.gradient(fy, axis=0) / dy[:, None]

    return dfx_dx + dfy_dy


def composite_fields(years: list[int], lon2d: np.ndarray, lat2d: np.ndarray) -> dict[str, np.ndarray]:
    """Calculate AMJJ atmospheric composites for selected years."""
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

    # Select nearest levels
    idx_pw_q = select_level_indices(lev_q, PW_LEVELS_HPA)
    idx_mfc_q = select_level_indices(lev_q, MFC_LEVELS_HPA)
    idx_mfc_u = select_level_indices(lev_u, MFC_LEVELS_HPA)
    idx_mfc_v = select_level_indices(lev_v, MFC_LEVELS_HPA)
    idx_w = int(np.argmin(np.abs(lev_w - OMEGA_LEVEL_HPA)))

    # Precipitable water
    q_pw = q[tidx][:, idx_pw_q, :, :]
    pw = integrate_pressure(q_pw, lev_q[idx_pw_q])
    pw_comp = np.nanmean(pw, axis=0)

    # Moisture flux convergence: -div integral(q*u dp/g, q*v dp/g)
    q_mfc = q[tidx][:, idx_mfc_q, :, :]
    u_mfc = u[tidx][:, idx_mfc_u, :, :]
    v_mfc = v[tidx][:, idx_mfc_v, :, :]

    # If nearest-level selections produce same number/order, use them directly.
    nlev = min(q_mfc.shape[1], u_mfc.shape[1], v_mfc.shape[1])
    q_mfc = q_mfc[:, :nlev, :, :]
    u_mfc = u_mfc[:, :nlev, :, :]
    v_mfc = v_mfc[:, :nlev, :, :]
    lev_mfc = lev_q[idx_mfc_q][:nlev]

    flux_u = integrate_pressure(q_mfc * u_mfc, lev_mfc)
    flux_v = integrate_pressure(q_mfc * v_mfc, lev_mfc)

    conv_list = []
    for i in range(flux_u.shape[0]):
        div = horizontal_divergence(flux_u[i], flux_v[i], lon2d, lat2d)
        conv_list.append(-div)  # positive = convergence
    mfc_comp = np.nanmean(np.asarray(conv_list), axis=0)

    # Vertical velocity at selected level
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
# PLOTTING
# ============================================================

def setup_map_axis(ax, area=AREA_MAP):
    if HAS_CARTOPY:
        ax.set_extent([area["lon_min"], area["lon_max"], area["lat_min"], area["lat_max"]], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, alpha=0.7)
        ax.add_feature(cfeature.OCEAN, alpha=0.7)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
        try:
            states = cfeature.NaturalEarthFeature("cultural", "admin_1_states_provinces", "10m", facecolor="none")
            ax.add_feature(states, edgecolor="gray", linewidth=0.4)
        except Exception:
            pass
        gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.4, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False
    else:
        ax.set_xlim(area["lon_min"], area["lon_max"])
        ax.set_ylim(area["lat_min"], area["lat_max"])
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(alpha=0.3)


def plot_composite_difference(wet: dict, dry: dict, lon2d: np.ndarray, lat2d: np.ndarray, event_table: pd.DataFrame):
    """Plot warm-wet minus warm-dry composites."""
    diff_pw = wet["pw"] - dry["pw"]
    diff_mfc = wet["mfc"] - dry["mfc"]
    diff_omega = wet["omega"] - dry["omega"]

    projection = ccrs.PlateCarree() if HAS_CARTOPY else None
    fig, axes = plt.subplots(
        1, 3,
        figsize=(17, 5),
        subplot_kw={"projection": projection} if HAS_CARTOPY else None
    )

    fields = [diff_pw, diff_mfc, diff_omega]
    titles = [
        "a) Precipitable water\nWarm-wet minus warm-dry",
        "b) Low-level moisture convergence\nWarm-wet minus warm-dry",
        f"c) Vertical velocity at {int(wet['omega_level'])} hPa\nWarm-wet minus warm-dry",
    ]
    labels = ["mm", "kg m$^{-2}$ s$^{-1}$", "Pa s$^{-1}$"]
    cmaps = ["BrBG", "PuOr", "RdBu_r"]

    for ax, field, title, label, cmap in zip(axes, fields, titles, labels, cmaps):
        setup_map_axis(ax)
        vmax = np.nanpercentile(np.abs(field), 98)
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        levels = np.linspace(-vmax, vmax, 13)
        if HAS_CARTOPY:
            im = ax.contourf(lon2d, lat2d, field, levels=levels, cmap=cmap, extend="both", transform=ccrs.PlateCarree())
        else:
            im = ax.contourf(lon2d, lat2d, field, levels=levels, cmap=cmap, extend="both")
        ax.set_title(title, fontsize=12)
        cbar = plt.colorbar(im, ax=ax, orientation="vertical", shrink=0.75, pad=0.03)
        cbar.set_label(label)

    r = event_table.attrs.get("r", np.nan)
    p = event_table.attrs.get("p", np.nan)
    fig.suptitle(
        "Atmospheric contrast between coupled and decoupled warm SAWP years\n"
        f"SST-rainfall relationship: r={r:.2f}, p={p:.3f}",
        fontsize=14
    )
    plt.tight_layout()

    out = FIG_DIR / "S5_warm_wet_minus_warm_dry_atmospheric_composites.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_timeseries(event_table: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(event_table.index, event_table["sst_norm"], marker="o", label="SAWP SST anomaly (AMJJ, normalized)")
    ax.plot(event_table.index, event_table["rain_norm"], marker="s", label="ENEB rainfall anomaly (AMJJ, normalized)")
    ax.axhline(0, color="k", linestyle="--", linewidth=1)

    warm_wet = event_table[event_table["group"] == "warm_wet"]
    warm_dry = event_table[event_table["group"] == "warm_dry"]

    ax.scatter(warm_wet.index, warm_wet["sst_norm"], s=90, marker="o", edgecolor="k", zorder=5, label="Warm-wet years")
    ax.scatter(warm_dry.index, warm_dry["sst_norm"], s=90, marker="X", edgecolor="k", zorder=5, label="Warm-dry years")

    r = event_table.attrs.get("r", np.nan)
    p = event_table.attrs.get("p", np.nan)
    ax.set_title(f"SAWP SST and ENEB rainfall anomalies during AMJJ\nr = {r:.2f}, p = {p:.3f}, n = {len(event_table)}")
    ax.set_xlabel("Year")
    ax.set_ylabel("Normalized anomaly")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    plt.tight_layout()

    out = FIG_DIR / "S5_SAWP_SST_ENEB_rainfall_event_groups.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Building SAWP SST / ENEB rainfall event table...")
    event_table = build_event_table()
    event_table.to_csv(FIG_DIR / "S5_event_table_sawp_sst_eneb_rainfall.csv")

    print("\nEvent groups:")
    print(event_table.groupby("group").size())
    print("\nWarm-wet years:")
    print(event_table[event_table["group"] == "warm_wet"][["sst_norm", "rain_norm"]])
    print("\nWarm-dry years:")
    print(event_table[event_table["group"] == "warm_dry"][["sst_norm", "rain_norm"]])

    plot_timeseries(event_table)

    # Reload grid information once for composites
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
        FIG_DIR / "S5_composite_fields_warm_wet_warm_dry.npz",
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

    plot_composite_difference(comp_wet, comp_dry, lon2d, lat2d, event_table)

    print("\nDone.")
    print(f"Outputs saved in: {FIG_DIR}")
