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

TENDENCIAS = "não"       # "sim" = trended data; "não" = detrended data
METODO_SAWP = "quadrado" # "quadrado" or "isoterma"
MESES_ESTACAO = [4, 5, 6, 7]  # AMJJ

# Thresholds for event classification
SST_WARM_THRESHOLD = .5   # SAWP SST anomaly >= +1 std
RAIN_WET_THRESHOLD = .5   # ENEB rainfall anomaly >= +1 std
RAIN_DRY_THRESHOLD = 0.0   # ENEB rainfall anomaly <= 0 std

# Composite variables
PW_LEVELS_HPA = [1000, 925, 850, 700, 600, 500, 400]      # column for precipitable water
MFC_LEVELS_HPA = [1000, 925, 900, 875, 850]               # low-level moisture convergence
OMEGA_LEVEL_HPA = 500                                     # vertical velocity level

# Spatial domains
AREA_SAWP = {"lon_min": -34.0, "lon_max": -25.0, "lat_min": -12.5, "lat_max": -4.0}
AREA_MAP  = {"lon_min": -50.0, "lon_max": -25.0, "lat_min": -15.0, "lat_max":  0.0}
AREA_NEB  = {"lon_min": -38, "lon_max": -34.5, "lat_min": -12.5, "lat_max":  -4.5}

# Levels used for precipitable-water diagnostics
LOW_LEVEL_PW_LEVELS_HPA = [1000, 925, 900, 875, 850]
MID_LEVEL_PW_LEVELS_HPA = [850, 700, 600, 500, 400]

# Area aliases used in the time-series diagnostics
AREA_ENEB = AREA_NEB

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
# Surface wind speed in the SAWP area.
# The script first tries 10-m wind files. If they do not exist,
# it falls back to the nearest pressure level to WIND_FALLBACK_LEVEL_HPA
# using U_FILE and V_FILE.
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


def first_existing_path(paths: list[Path]) -> Path | None:
    """Return the first existing path from a list, or None."""
    for path in paths:
        if Path(path).exists():
            return Path(path)
    return None


def load_npz_data_and_time(path: Path) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Load a 2D/3D variable from npz and its time coordinate."""
    npz = np.load(path, mmap_mode="r", allow_pickle=True)
    if "data" in npz:
        data = npz["data"]
    else:
        # Use the first non-coordinate array as a fallback.
        coord_keys = {"time", "valid_time", "level", "levels", "pressure_level", "lat", "lon", "latitude", "longitude"}
        candidates = [k for k in npz.files if k not in coord_keys]
        if len(candidates) == 0:
            raise KeyError(f"Could not find data array in {path}.")
        data = npz[candidates[0]]
    return data, time_from_npz(npz)


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

def load_sawp_slhf_series(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    mask_sawp: np.ndarray,
) -> pd.Series:
    """
    Return monthly surface latent heat flux averaged over SAWP.

    ERA5 surface latent heat flux is usually accumulated energy flux.
    Here it is kept in the same unit as stored in the npz file.
    """
    slhf, time_slhf = load_npz_data_and_time(SLHF_FILE)

    if slhf.ndim == 4:
        slhf = slhf[:, 0, :, :]

    slhf_sawp = np.nanmean(
        np.where(mask_sawp[None, :, :], slhf, np.nan),
        axis=(1, 2)
    )

    return pd.Series(
        slhf_sawp,
        index=pd.to_datetime(time_slhf),
        name="sawp_slhf"
    )

def load_sawp_surface_wind_series(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    mask_sawp: np.ndarray,
) -> pd.Series:
    """
    Return monthly SAWP surface wind-speed series using the same spatial mask as SST.

    Priority:
      1) 10-m wind components, if u10/v10 npz files exist.
      2) fallback to the nearest pressure level to WIND_FALLBACK_LEVEL_HPA
         from the 4D u/v files.

    Output unit is m s-1.
    """
    u10_path = first_existing_path(U10_CANDIDATES)
    v10_path = first_existing_path(V10_CANDIDATES)

    if (u10_path is not None) and (v10_path is not None):
        print(f"Loading surface wind from 10-m files: {u10_path.name}, {v10_path.name}")
        u10, time_u = load_npz_data_and_time(u10_path)
        v10, time_v = load_npz_data_and_time(v10_path)

        if len(time_u) != len(time_v):
            warnings.warn("u10 and v10 have different time lengths. Using common minimum length.")
            ntime = min(len(time_u), len(time_v), u10.shape[0], v10.shape[0])
            u10 = u10[:ntime]
            v10 = v10[:ntime]
            time = time_u[:ntime]
        else:
            time = time_u

        if u10.ndim == 4:
            u10 = u10[:, 0, :, :]
        if v10.ndim == 4:
            v10 = v10[:, 0, :, :]

        wind = np.sqrt(u10**2 + v10**2)

    else:
        print(
            "10-m wind files not found. "
            f"Using nearest pressure level to {WIND_FALLBACK_LEVEL_HPA} hPa from U_FILE/V_FILE."
        )
        u, lev_u, time_u = load_4d(U_FILE)
        v, lev_v, time_v = load_4d(V_FILE)

        idx_u = int(np.argmin(np.abs(lev_u - WIND_FALLBACK_LEVEL_HPA)))
        idx_v = int(np.argmin(np.abs(lev_v - WIND_FALLBACK_LEVEL_HPA)))
        print(f"Wind fallback levels: u={lev_u[idx_u]:.0f} hPa, v={lev_v[idx_v]:.0f} hPa")

        ntime = min(len(time_u), len(time_v), u.shape[0], v.shape[0])
        time = time_u[:ntime]
        wind = np.sqrt(u[:ntime, idx_u, :, :]**2 + v[:ntime, idx_v, :, :]**2)

    wind_sawp = np.nanmean(np.where(mask_sawp[None, :, :], wind, np.nan), axis=(1, 2))
    return pd.Series(wind_sawp, index=pd.to_datetime(time), name="sawp_surface_wind_ms")



def load_area_omega_series(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    area: dict = AREA_NEB,
) -> pd.Series:
    """
    Return monthly omega/w series averaged over a target area.

    The selected pressure level is the nearest one to OMEGA_LEVEL_HPA.
    ERA5 pressure velocity is usually in Pa s-1. Negative values indicate ascent.
    """
    w, lev_w, time_w = load_4d(W_FILE)
    idx_w = int(np.argmin(np.abs(lev_w - OMEGA_LEVEL_HPA)))
    print(f"Omega/w time series level: {lev_w[idx_w]:.0f} hPa")

    mask = area_mask(lon2d, lat2d, area)
    omega_area = np.nanmean(np.where(mask[None, :, :], w[:, idx_w, :, :], np.nan), axis=(1, 2))
    return pd.Series(
        omega_area,
        index=pd.to_datetime(time_w),
        name=f"omega_{int(lev_w[idx_w])}hpa_area_pa_s",
    )



def load_area_precipitable_water_series(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    area: dict,
    levels_hpa: list[float],
    name: str,
) -> pd.Series:
    """
    Return monthly precipitable-water series integrated from q over a given layer and area.

    Unit is kg m-2, numerically equivalent to mm of precipitable water.
    """
    q, lev_q, time_q = load_4d(Q_FILE)
    idx_q = select_level_indices(lev_q, levels_hpa)
    used_levels = lev_q[idx_q]
    print(
        f"{name} precipitable-water levels:",
        ", ".join(f"{lev:.0f}" for lev in used_levels),
        "hPa",
    )

    q_sel = q[:, idx_q, :, :]
    pw = integrate_pressure(q_sel, used_levels)

    mask = area_mask(lon2d, lat2d, area)
    pw_area = np.nanmean(np.where(mask[None, :, :], pw, np.nan), axis=(1, 2))
    return pd.Series(
        pw_area,
        index=pd.to_datetime(time_q),
        name=name,
    )


def load_low_level_precipitable_water_series(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    area: dict = AREA_SAWP,
) -> pd.Series:
    """
    Compatibility wrapper: low-level precipitable water integrated from 1000 to 850 hPa.
    """
    return load_area_precipitable_water_series(
        lon2d=lon2d,
        lat2d=lat2d,
        area=area,
        levels_hpa=LOW_LEVEL_PW_LEVELS_HPA,
        name="low_level_pw_1000_850_kg_m2",
    )


def load_area_low_level_mfc_series(
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    area: dict,
    levels_hpa: list[float] = MFC_LEVELS_HPA,
    name: str = "mfc_low_1000_850_kg_m2_s",
) -> pd.Series:
    """
    Return monthly low-level moisture flux convergence averaged over a target area.

    The convergence is computed as:

        -div[ integral(q*u dp/g), integral(q*v dp/g) ]

    Positive values indicate moisture convergence.
    """
    u, lev_u, time_u = load_4d(U_FILE)
    v, lev_v, time_v = load_4d(V_FILE)
    q, lev_q, time_q = load_4d(Q_FILE)

    ntime = min(len(time_u), len(time_v), len(time_q), u.shape[0], v.shape[0], q.shape[0])
    time = time_q[:ntime]

    idx_q = select_level_indices(lev_q, levels_hpa)
    idx_u = select_level_indices(lev_u, levels_hpa)
    idx_v = select_level_indices(lev_v, levels_hpa)

    q_sel = q[:ntime][:, idx_q, :, :]
    u_sel = u[:ntime][:, idx_u, :, :]
    v_sel = v[:ntime][:, idx_v, :, :]

    nlev = min(q_sel.shape[1], u_sel.shape[1], v_sel.shape[1])
    q_sel = q_sel[:, :nlev, :, :]
    u_sel = u_sel[:, :nlev, :, :]
    v_sel = v_sel[:, :nlev, :, :]
    used_levels = lev_q[idx_q][:nlev]

    print(
        f"{name} moisture-convergence levels:",
        ", ".join(f"{lev:.0f}" for lev in used_levels),
        "hPa",
    )

    flux_u = integrate_pressure(q_sel * u_sel, used_levels)
    flux_v = integrate_pressure(q_sel * v_sel, used_levels)

    mask = area_mask(lon2d, lat2d, area)
    conv_area = []

    for i in range(flux_u.shape[0]):
        div = horizontal_divergence(flux_u[i], flux_v[i], lon2d, lat2d)
        conv = -div  # positive = convergence
        conv_area.append(np.nanmean(np.where(mask, conv, np.nan)))

    return pd.Series(
        np.asarray(conv_area, dtype=float),
        index=pd.to_datetime(time),
        name=name,
    )


def load_eneb_rainfall_series() -> pd.Series:
    """Load the seasonal rainfall series using the same routine as the main analysis."""
    rain = sawp.carregar_chuva_periodo(str(RAIN_FILE), TENDENCIAS)
    rain = pd.Series(np.asarray(rain).squeeze(), index=rain.index, name="eneb_rain_mm")
    rain = rain.replace(-999, np.nan)
    return rain



def build_event_table() -> pd.DataFrame:
    """
    Build the annual AMJJ table used in the time-series diagnostics.

    Variables included:
      - SAWP SST
      - ENEB rainfall
      - Low-level precipitable water 1000-850 hPa over SAWP and NEB
      - Mid-level precipitable water 850-400 hPa over SAWP and NEB
      - Omega/w at 500 hPa over SAWP and NEB
      - Low-level moisture convergence 1000-850 hPa over ENEB

    All *_norm columns are standardized annual AMJJ anomalies.
    """
    sst_monthly, _, lon2d, lat2d, mask_sawp = load_sawp_sst_series()
    rain_season = load_eneb_rainfall_series()
    
    slhf_sawp_monthly = load_sawp_slhf_series(lon2d, lat2d, mask_sawp)
    slhf_sawp_monthly = slhf_sawp_monthly.shift(2)
    wind_sawp_monthly = load_sawp_surface_wind_series(lon2d, lat2d, mask_sawp)

    # Atmospheric monthly series
    pw_low_sawp_monthly = load_area_precipitable_water_series(
        lon2d, lat2d, AREA_SAWP, LOW_LEVEL_PW_LEVELS_HPA, "pw_low_sawp_1000_850"
    )
    pw_low_neb_monthly = load_area_precipitable_water_series(
        lon2d, lat2d, AREA_NEB, LOW_LEVEL_PW_LEVELS_HPA, "pw_low_neb_1000_850"
    )
    pw_mid_sawp_monthly = load_area_precipitable_water_series(
        lon2d, lat2d, AREA_SAWP, MID_LEVEL_PW_LEVELS_HPA, "pw_mid_sawp_850_400"
    )
    pw_mid_neb_monthly = load_area_precipitable_water_series(
        lon2d, lat2d, AREA_NEB, MID_LEVEL_PW_LEVELS_HPA, "pw_mid_neb_850_400"
    )

    omega_sawp_monthly = load_area_omega_series(lon2d, lat2d, AREA_SAWP)
    omega_sawp_monthly.name = "omega_sawp_500"
    omega_neb_monthly = load_area_omega_series(lon2d, lat2d, AREA_NEB)
    omega_neb_monthly.name = "omega_neb_500"

    mfc_eneb_monthly = load_area_low_level_mfc_series(
        lon2d, lat2d, AREA_ENEB, MFC_LEVELS_HPA, "mfc_eneb_low_1000_850"
    )
    slhf_sawp_amjj = annual_amjj_mean(slhf_sawp_monthly)
    wind_sawp_amjj = annual_amjj_mean(wind_sawp_monthly)

    # Annual AMJJ aggregation
    sst_amjj = annual_amjj_mean(sst_monthly)
    pw_low_sawp_amjj = annual_amjj_mean(pw_low_sawp_monthly)
    pw_low_neb_amjj = annual_amjj_mean(pw_low_neb_monthly)
    pw_mid_sawp_amjj = annual_amjj_mean(pw_mid_sawp_monthly)
    pw_mid_neb_amjj = annual_amjj_mean(pw_mid_neb_monthly)
    omega_sawp_amjj = annual_amjj_mean(omega_sawp_monthly)
    omega_neb_amjj = annual_amjj_mean(omega_neb_monthly)
    mfc_eneb_amjj = annual_amjj_mean(mfc_eneb_monthly)

    rain_season.index = rain_season.index.astype(int)

    raw_series = {
        "sst_amjj_c": sst_amjj,
        "rain_amjj_mm": rain_season,
        "pw_low_sawp_1000_850_kg_m2": pw_low_sawp_amjj,
        "pw_low_neb_1000_850_kg_m2": pw_low_neb_amjj,
        "pw_mid_sawp_850_400_kg_m2": pw_mid_sawp_amjj,
        "pw_mid_neb_850_400_kg_m2": pw_mid_neb_amjj,
        "omega_sawp_500_pa_s": omega_sawp_amjj,
        "omega_neb_500_pa_s": omega_neb_amjj,
        "mfc_eneb_low_1000_850_kg_m2_s": mfc_eneb_amjj,
        "sawp_slhf": slhf_sawp_amjj,
        "sawp_surface_wind_ms": wind_sawp_amjj,
    }

    norm_series = {
        "sst_norm": normalize(sst_amjj),
        "rain_norm": normalize(rain_season),
        "pw_low_sawp_norm": normalize(pw_low_sawp_amjj),
        "pw_low_neb_norm": normalize(pw_low_neb_amjj),
        "pw_mid_sawp_norm": normalize(pw_mid_sawp_amjj),
        "pw_mid_neb_norm": normalize(pw_mid_neb_amjj),
        "omega_sawp_norm": normalize(omega_sawp_amjj),
        "omega_neb_norm": normalize(omega_neb_amjj),
        "mfc_eneb_norm": normalize(mfc_eneb_amjj),
        "slhf_sawp_norm": normalize(slhf_sawp_amjj),
        "wind_norm": normalize(wind_sawp_amjj),
    }

    df = pd.concat(
        [s.rename(name) for name, s in norm_series.items()] +
        [s.rename(name) for name, s in raw_series.items()],
        axis=1
    ).dropna()

    # Main relationship used to classify warm-wet and warm-dry years
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
    return np.trapezoid(fld, x=lev_pa, axis=1) / G


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

PLOT_INDIVIDUAL_FIGURES = False  # True only if you also want the old legacy figures


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
                "cultural", "admin_1_states_provinces", "10m", facecolor="none"
            )
            ax.add_feature(states, edgecolor="gray", linewidth=0.4)
        except Exception:
            pass
        gl = ax.gridlines(
            draw_labels=True, linewidth=0.4, color="gray", alpha=0.4, linestyle="--"
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


def add_panel_label(ax, label: str):
    ax.text(
        0.01, 0.98, label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        va="top",
        ha="left",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=2),
        zorder=20,
    )


def map_levels(field: np.ndarray, diff: bool = False, nlev: int = 13) -> np.ndarray:
    field = np.asarray(field, dtype=float)
    if diff:
        vmax = np.nanpercentile(np.abs(field), 98)
        if not np.isfinite(vmax) or vmax == 0:
            vmax = 1.0
        return np.linspace(-vmax, vmax, nlev)

    vmin = np.nanpercentile(field, 2)
    vmax = np.nanpercentile(field, 98)
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmin == vmax:
        vmin, vmax = np.nanmin(field), np.nanmax(field)
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or vmin == vmax:
        vmin, vmax = -1.0, 1.0
    return np.linspace(vmin, vmax, nlev)


def plot_map_subplot(
    fig,
    ax,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
    field: np.ndarray,
    title: str,
    label: str,
    cmap: str,
    diff: bool = False,
):
    setup_map_axis(ax)
    levels = map_levels(field, diff=diff)

    if HAS_CARTOPY:
        im = ax.contourf(
            lon2d, lat2d, field,
            levels=levels,
            cmap=cmap,
            extend="both",
            transform=ccrs.PlateCarree(),
        )
    else:
        im = ax.contourf(lon2d, lat2d, field, levels=levels, cmap=cmap, extend="both")

    ax.set_title(title, fontsize=10)
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", shrink=0.76, pad=0.02)
    cbar.set_label(label, fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    return im


def plot_full_panel(
    event_table: pd.DataFrame,
    wet: dict,
    dry: dict,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
):
    """
    Generate a complete multi-panel figure with all diagnostics:
      a) normalized AMJJ time series
      b) SAWP SST x ENEB rainfall scatter
      c-e) precipitable water: warm-wet, warm-dry, difference
      f-h) low-level moisture convergence: warm-wet, warm-dry, difference
      i-k) vertical velocity: warm-wet, warm-dry, difference
    """
    diff_pw = wet["pw"] - dry["pw"]
    diff_mfc = wet["mfc"] - dry["mfc"]
    diff_omega = wet["omega"] - dry["omega"]

    r = event_table.attrs.get("r", np.nan)
    p = event_table.attrs.get("p", np.nan)

    warm_wet = event_table[event_table["group"] == "warm_wet"]
    warm_dry = event_table[event_table["group"] == "warm_dry"]
    other = event_table[event_table["group"] == "other"]

    projection = ccrs.PlateCarree() if HAS_CARTOPY else None

    fig = plt.figure(figsize=(18, 20))
    gs = fig.add_gridspec(
        nrows=4,
        ncols=3,
        height_ratios=[0.85, 1.0, 1.0, 1.0],
        hspace=0.35,
        wspace=0.22,
    )

    # --------------------------------------------------------
    # a) Time series
    # --------------------------------------------------------
    ax_ts = fig.add_subplot(gs[0, 0:2])
    ax_ts.plot(
        event_table.index,
        event_table["sst_norm"],
        marker="o",
        linewidth=1.8,
        label="SAWP SST anomaly (AMJJ, normalized)",
    )
    ax_ts.plot(
        event_table.index,
        event_table["rain_norm"],
        marker="s",
        linewidth=1.8,
        label="ENEB rainfall anomaly (AMJJ, normalized)",
    )
    ax_ts.plot(
        event_table.index,
        event_table["wind_norm"],
        marker="^",
        linewidth=1.8,
        label="SAWP surface wind-speed anomaly (AMJJ, normalized)",
    )
    ax_ts.axhline(0, color="k", linestyle="--", linewidth=1)
    ax_ts.scatter(
        warm_wet.index,
        warm_wet["sst_norm"],
        s=90,
        marker="o",
        edgecolor="k",
        zorder=5,
        label="Warm-wet years",
    )
    ax_ts.scatter(
        warm_dry.index,
        warm_dry["sst_norm"],
        s=100,
        marker="X",
        edgecolor="k",
        zorder=5,
        label="Warm-dry years",
    )
    ax_ts.set_title("SAWP SST, surface wind speed, and ENEB rainfall anomalies during AMJJ", fontsize=12)
    ax_ts.set_xlabel("Year")
    ax_ts.set_ylabel("Normalized anomaly")
    ax_ts.grid(alpha=0.3)
    ax_ts.legend(loc="best", fontsize=9)
    add_panel_label(ax_ts, "a")

    # --------------------------------------------------------
    # b) Scatter SST x rainfall
    # --------------------------------------------------------
    ax_sc = fig.add_subplot(gs[0, 2])
    ax_sc.scatter(other["sst_norm"], other["rain_norm"], s=55, alpha=0.75, label="Other")
    ax_sc.scatter(
        warm_wet["sst_norm"],
        warm_wet["rain_norm"],
        s=90,
        marker="o",
        edgecolor="k",
        label="Warm-wet",
    )
    ax_sc.scatter(
        warm_dry["sst_norm"],
        warm_dry["rain_norm"],
        s=100,
        marker="X",
        edgecolor="k",
        label="Warm-dry",
    )
    ax_sc.axhline(0, color="k", linestyle="--", linewidth=1)
    ax_sc.axvline(0, color="k", linestyle="--", linewidth=1)
    ax_sc.axvline(SST_WARM_THRESHOLD, color="k", linestyle=":", linewidth=1)
    ax_sc.axhline(RAIN_WET_THRESHOLD, color="k", linestyle=":", linewidth=1)
    ax_sc.axhline(RAIN_DRY_THRESHOLD, color="k", linestyle=":", linewidth=1)
    ax_sc.set_title(f"SST-rainfall relationship\nr = {r:.2f}, p = {p:.3f}, n = {len(event_table)}", fontsize=12)
    ax_sc.set_xlabel("SAWP SST anomaly (std)")
    ax_sc.set_ylabel("ENEB rainfall anomaly (std)")
    ax_sc.grid(alpha=0.3)
    ax_sc.legend(loc="best", fontsize=8)
    add_panel_label(ax_sc, "b")

    # --------------------------------------------------------
    # Map rows
    # --------------------------------------------------------
    map_axes = []
    for row in range(1, 4):
        for col in range(3):
            if HAS_CARTOPY:
                map_axes.append(fig.add_subplot(gs[row, col], projection=projection))
            else:
                map_axes.append(fig.add_subplot(gs[row, col]))

    # Row 1: precipitable water
    plot_map_subplot(fig, map_axes[0], lon2d, lat2d, wet["pw"], "Warm-wet", "mm", "BrBG", diff=False)
    plot_map_subplot(fig, map_axes[1], lon2d, lat2d, dry["pw"], "Warm-dry", "mm", "BrBG", diff=False)
    plot_map_subplot(fig, map_axes[2], lon2d, lat2d, diff_pw, "Warm-wet minus warm-dry", "mm", "BrBG", diff=True)
    map_axes[0].set_ylabel("Precipitable water", fontsize=11)

    # Row 2: low-level moisture convergence
    plot_map_subplot(fig, map_axes[3], lon2d, lat2d, wet["mfc"], "Warm-wet", "kg m$^{-2}$ s$^{-1}$", "PuOr", diff=False)
    plot_map_subplot(fig, map_axes[4], lon2d, lat2d, dry["mfc"], "Warm-dry", "kg m$^{-2}$ s$^{-1}$", "PuOr", diff=False)
    plot_map_subplot(fig, map_axes[5], lon2d, lat2d, diff_mfc, "Warm-wet minus warm-dry", "kg m$^{-2}$ s$^{-1}$", "PuOr", diff=True)
    map_axes[3].set_ylabel("Moisture convergence", fontsize=11)

    # Row 3: vertical velocity
    omega_title = f"Vertical velocity at {int(wet['omega_level'])} hPa"
    plot_map_subplot(fig, map_axes[6], lon2d, lat2d, wet["omega"], "Warm-wet", "Pa s$^{-1}$", "RdBu_r", diff=False)
    plot_map_subplot(fig, map_axes[7], lon2d, lat2d, dry["omega"], "Warm-dry", "Pa s$^{-1}$", "RdBu_r", diff=False)
    plot_map_subplot(fig, map_axes[8], lon2d, lat2d, diff_omega, "Warm-wet minus warm-dry", "Pa s$^{-1}$", "RdBu_r", diff=True)
    map_axes[6].set_ylabel(omega_title, fontsize=11)

    for label, ax in zip(list("cdefghijk"), map_axes):
        add_panel_label(ax, label)

    fig.suptitle(
        "SAWP-ENEB coupled and decoupled warm years: SST-rainfall-wind relationship and atmospheric composites\n"
        f"Warm-wet years: {', '.join(map(str, wet['years']))} | "
        f"Warm-dry years: {', '.join(map(str, dry['years']))}",
        fontsize=15,
        y=0.995,
    )

    out = FIG_DIR / "S5_full_panel_SAWP_ENEB_composites.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")



def plot_maps_only_panel(
    event_table: pd.DataFrame,
    wet: dict,
    dry: dict,
    lon2d: np.ndarray,
    lat2d: np.ndarray,
):
    """Generate one figure containing only the atmospheric composite maps."""
    diff_pw = wet["pw"] - dry["pw"]
    diff_mfc = wet["mfc"] - dry["mfc"]
    diff_omega = wet["omega"] - dry["omega"]

    projection = ccrs.PlateCarree() if HAS_CARTOPY else None
    fig, axes = plt.subplots(
        3, 3,
        figsize=(18, 14),
        subplot_kw={"projection": projection} if HAS_CARTOPY else None,
    )
    axes = axes.ravel()

    plot_map_subplot(fig, axes[0], lon2d, lat2d, wet["pw"], "Warm-wet", "mm", "BrBG", diff=False)
    plot_map_subplot(fig, axes[1], lon2d, lat2d, dry["pw"], "Warm-dry", "mm", "BrBG", diff=False)
    plot_map_subplot(fig, axes[2], lon2d, lat2d, diff_pw, "Warm-wet minus warm-dry", "mm", "BrBG", diff=True)
    axes[0].set_ylabel("Precipitable water", fontsize=11)

    plot_map_subplot(fig, axes[3], lon2d, lat2d, wet["mfc"], "Warm-wet", "kg m$^{-2}$ s$^{-1}$", "PuOr", diff=False)
    plot_map_subplot(fig, axes[4], lon2d, lat2d, dry["mfc"], "Warm-dry", "kg m$^{-2}$ s$^{-1}$", "PuOr", diff=False)
    plot_map_subplot(fig, axes[5], lon2d, lat2d, diff_mfc, "Warm-wet minus warm-dry", "kg m$^{-2}$ s$^{-1}$", "PuOr", diff=True)
    axes[3].set_ylabel("Moisture convergence", fontsize=11)

    omega_title = f"Vertical velocity at {int(wet['omega_level'])} hPa"
    plot_map_subplot(fig, axes[6], lon2d, lat2d, wet["omega"], "Warm-wet", "Pa s$^{-1}$", "RdBu_r", diff=False)
    plot_map_subplot(fig, axes[7], lon2d, lat2d, dry["omega"], "Warm-dry", "Pa s$^{-1}$", "RdBu_r", diff=False)
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

    out = FIG_DIR / "S5A_MAPAS_apenas_composites_warm_wet_warm_dry_v2.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")



def safe_pearson(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    """Pearson correlation with NaN handling."""
    tmp = pd.concat([x, y], axis=1).dropna()
    if len(tmp) < 3:
        return np.nan, np.nan
    return pearsonr(tmp.iloc[:, 0], tmp.iloc[:, 1])


def plot_paired_series(
    ax,
    event_table: pd.DataFrame,
    y1: str,
    y2: str,
    label1: str,
    label2: str,
    title: str,
):
    """Plot two normalized AMJJ annual series in one subplot and calculate r/p directly."""
    r, p = safe_pearson(event_table[y1], event_table[y2])

    ax.plot(event_table.index, event_table[y1], marker="o", linewidth=1.7, label=label1)
    ax.plot(event_table.index, event_table[y2], marker="s", linewidth=1.7, label=label2)
    ax.axhline(0, color="k", linestyle="--", linewidth=1)

    ax.set_title(f"{title}\nr = {r:.2f}, p = {p:.3f}", fontsize=10)
    ax.set_xlabel("Year")
    ax.set_ylabel("Normalized anomaly")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def plot_series_only_panel(event_table: pd.DataFrame):
    """
    Generate a complete time-series comparison panel.

    Design:
      - One reference comparison: SAWP SST x ENEB rainfall
      - Each atmospheric variable is compared with SAWP SST
      - The same atmospheric variable is compared with ENEB rainfall

    Atmospheric variables:
      - Low-level precipitable water 1000-850 hPa over SAWP
      - Low-level precipitable water 1000-850 hPa over NEB
      - Mid-level precipitable water 850-400 hPa over SAWP
      - Mid-level precipitable water 850-400 hPa over NEB
      - Omega/w at 500 hPa over SAWP
      - Omega/w at 500 hPa over NEB
      - Low-level moisture convergence 1000-850 hPa over ENEB
    """
    comparisons = [
        (
            "sst_norm", "rain_norm",
            "SAWP SST", "ENEB rainfall",
            "SAWP SST x ENEB rainfall",
        ),

        # Low-level precipitable water
        (
            "sst_norm", "pw_low_sawp_norm",
            "SAWP SST", "SAWP low-level PW",
            "SAWP SST x SAWP low-level PW (1000-850 hPa)",
        ),
        (
            "rain_norm", "pw_low_sawp_norm",
            "ENEB rainfall", "SAWP low-level PW",
            "ENEB rainfall x SAWP low-level PW (1000-850 hPa)",
        ),
        (
            "sst_norm", "pw_low_neb_norm",
            "SAWP SST", "ENEB low-level PW",
            "SAWP SST x ENEB low-level PW (1000-850 hPa)",
        ),
        (
            "rain_norm", "pw_low_neb_norm",
            "ENEB rainfall", "ENEB low-level PW",
            "ENEB rainfall x ENEB low-level PW (1000-850 hPa)",
        ),

        # Mid-level precipitable water
        (
            "sst_norm", "pw_mid_sawp_norm",
            "SAWP SST", "SAWP mid-level PW",
            "SAWP SST x SAWP mid-level PW (850-400 hPa)",
        ),
        (
            "rain_norm", "pw_mid_sawp_norm",
            "ENEB rainfall", "SAWP mid-level PW",
            "ENEB rainfall x SAWP mid-level PW (850-400 hPa)",
        ),
        (
            "sst_norm", "pw_mid_neb_norm",
            "SAWP SST", "ENEB mid-level PW",
            "SAWP SST x ENEB mid-level PW (850-400 hPa)",
        ),
        (
            "rain_norm", "pw_mid_neb_norm",
            "ENEB rainfall", "ENEB mid-level PW",
            "ENEB rainfall x ENEB mid-level PW (850-400 hPa)",
        ),

        # Omega
        (
            "sst_norm", "omega_sawp_norm",
            "SAWP SST", "SAWP omega/w 500 hPa",
            "SAWP SST x SAWP omega/w 500 hPa",
        ),
        (
            "rain_norm", "omega_sawp_norm",
            "ENEB rainfall", "SAWP omega/w 500 hPa",
            "ENEB rainfall x SAWP omega/w 500 hPa",
        ),
        (
            "sst_norm", "omega_neb_norm",
            "SAWP SST", "ENEB omega/w 500 hPa",
            "SAWP SST x ENEB omega/w 500 hPa",
        ),
        (
            "rain_norm", "omega_neb_norm",
            "ENEB rainfall", "ENEB omega/w 500 hPa",
            "ENEB rainfall x ENEB omega/w 500 hPa",
        ),

        # Moisture convergence
        (
            "sst_norm", "mfc_eneb_norm",
            "SAWP SST", "ENEB low-level moisture convergence",
            "SAWP SST x ENEB low-level moisture convergence (1000-850 hPa)",
        ),
        (
            "rain_norm", "mfc_eneb_norm",
            "ENEB rainfall", "ENEB low-level moisture convergence",
            "ENEB rainfall x ENEB low-level moisture convergence (1000-850 hPa)",
        ),
        # Surface latent heat flux over SAWP
        (
            "sst_norm", "slhf_sawp_norm",
            "SAWP SST", "SAWP surface latent heat flux",
            "SAWP SST x SAWP surface latent heat flux",
        ),
        (
            "rain_norm", "slhf_sawp_norm",
            "ENEB rainfall", "SAWP surface latent heat flux",
            "ENEB rainfall x SAWP surface latent heat flux",
        ),
        (
            "wind_norm", "slhf_sawp_norm",
            "SAWP surface wind speed", "SAWP surface latent heat flux",
            "SAWP wind speed x SAWP surface latent heat flux",
        ),
    ]

    ncols = 3
    nrows = int(np.ceil(len(comparisons) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(21, 4.0 * nrows), sharex=True)
    axes = np.asarray(axes).ravel()

    for i, (y1, y2, label1, label2, title) in enumerate(comparisons):
        plot_paired_series(
            axes[i],
            event_table,
            y1,
            y2,
            label1,
            label2,
            title,
        )
        add_panel_label(axes[i], chr(ord("a") + i))

    # Hide unused axes if the number of comparisons changes later.
    for j in range(len(comparisons), len(axes)):
        axes[j].axis("off")

    fig.suptitle(
        "Complete normalized AMJJ annual comparison panel: SAWP SST, ENEB rainfall and atmospheric controls",
        fontsize=16,
        y=0.995,
    )
    plt.tight_layout()

    out = FIG_DIR / "S5B_SERIES_comparativos_completos_SAWP_ENEB_v4.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_composite_difference(wet: dict, dry: dict, lon2d: np.ndarray, lat2d: np.ndarray, event_table: pd.DataFrame):
    """Optional old figure: only warm-wet minus warm-dry maps."""
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
        plot_map_subplot(fig, ax, lon2d, lat2d, field, title, label, cmap, diff=True)

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
    """Optional old figure: only normalized time series."""
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(event_table.index, event_table["sst_norm"], marker="o", label="SAWP SST anomaly (AMJJ, normalized)")
    ax.plot(event_table.index, event_table["rain_norm"], marker="s", label="ENEB rainfall anomaly (AMJJ, normalized)")
    ax.plot(event_table.index, event_table["wind_norm"], marker="^", label="SAWP surface wind-speed anomaly (AMJJ, normalized)")
    ax.axhline(0, color="k", linestyle="--", linewidth=1)

    warm_wet = event_table[event_table["group"] == "warm_wet"]
    warm_dry = event_table[event_table["group"] == "warm_dry"]

    ax.scatter(warm_wet.index, warm_wet["sst_norm"], s=90, marker="o", edgecolor="k", zorder=5, label="Warm-wet years")
    ax.scatter(warm_dry.index, warm_dry["sst_norm"], s=90, marker="X", edgecolor="k", zorder=5, label="Warm-dry years")

    r = event_table.attrs.get("r", np.nan)
    p = event_table.attrs.get("p", np.nan)
    r_wind = event_table.attrs.get("r_wind_rain", np.nan)
    p_wind = event_table.attrs.get("p_wind_rain", np.nan)
    ax.set_title(
        f"SAWP SST, surface wind speed, and ENEB rainfall anomalies during AMJJ\n"
        f"SST-rainfall: r = {r:.2f}, p = {p:.3f}; "
        f"wind-rainfall: r = {r_wind:.2f}, p = {p_wind:.3f}; n = {len(event_table)}"
    )
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
    
    cols_print = [
        "sst_norm",
        "rain_norm",
        "slhf_sawp_norm",
        "wind_norm",
        "pw_low_sawp_norm",
        "pw_low_neb_norm",
        "pw_mid_sawp_norm",
        "pw_mid_neb_norm",
        "omega_sawp_norm",
        "omega_neb_norm",
        "mfc_eneb_norm",
    ]
    
    print("\nWarm-wet years:")
    print(event_table[event_table["group"] == "warm_wet"][cols_print])
    
    print("\nWarm-dry years:")
    print(event_table[event_table["group"] == "warm_dry"][cols_print])
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

        # Save the final table again after threshold relaxation
        event_table.to_csv(FIG_DIR / "S5_event_table_sawp_sst_eneb_rainfall.csv")

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

    print("\nGenerating map-only panel...")
    plot_maps_only_panel(event_table, comp_wet, comp_dry, lon2d, lat2d)

    print("\nGenerating series-only panel...")
    plot_series_only_panel(event_table)

    if PLOT_INDIVIDUAL_FIGURES:
        print("\nGenerating optional legacy figures...")
        plot_timeseries(event_table)
        plot_composite_difference(comp_wet, comp_dry, lon2d, lat2d, event_table)

    print("\nDone.")
    print(f"Outputs saved in: {FIG_DIR}")
