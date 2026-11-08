#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Supplementary monthly composite figure for the reviewer.

Goal
----
Evaluate the monthly evolution of SAWP SST anomalies and ENEB rainfall anomalies
around the rainy season, from February to August, for warm SAWP event classes.

Scientific motivation
---------------------
This figure tests whether years with strong rainfall early in the season are
followed by a subsequent reduction of SAWP SST anomalies, and whether extremely
warm SAWP years with weak/normal rainfall keep positive SST anomalies for longer.

Outputs
-------
figuras_revisor/SX_monthly_composite_SAWP_SST_ENEB_rainfall_FebAug.png
figuras_revisor/SX_monthly_composite_values_FebAug.csv

Notes
-----
- Event classification is based on AMJJ normalized SAWP SST and AMJJ normalized
  ENEB rainfall.
- Monthly composite curves are shown from February to August.
- SST and rainfall monthly anomalies are standardized separately before
  compositing, so the two variables can be compared in the same panel.
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

import sawp


# ============================================================
# USER CONFIGURATION
# ============================================================

TENDENCIAS = "sim"        # "sim" = trended data; "não" = detrended data
METODO_SAWP = "quadrado"  # "quadrado" or "isoterma"

# Months used to classify annual warm/rainfall events
MESES_ESTACAO = [4, 5, 6, 7]  # AMJJ

# Months shown in the supplementary monthly evolution figure
MESES_EVOLUCAO = [2, 3, 4, 5, 6, 7, 8]
MONTH_LABELS = {2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun", 7: "Jul", 8: "Aug"}

# Event thresholds using normalized AMJJ anomalies
MODERATE_WARM_MIN = 0.50
EXTREME_WARM_MIN = 1.00
RAIN_WET_MIN = 0.50
RAIN_DRY_MAX = 0.00

# Spatial domains
AREA_SAWP = {"lon_min": -34.0, "lon_max": -25.0, "lat_min": -12.5, "lat_max": -4.0}
AREA_ENEB_BOX = {"lon_min": -40.5, "lon_max": -34.5, "lat_min": -14.5, "lat_max": -2.0}

# April climatology used only if METODO_SAWP == "isoterma"
ISOTHERM_THRESHOLD_C = 28.5
CLIM_INI = 1980
CLIM_FIM = 2010

# Climatology for monthly anomalies
ANOM_CLIM_INI = 1980
ANOM_CLIM_FIM = 2010

BASE_DIR = Path(".")
DATA_DIR = BASE_DIR / "data"
OUT_DIR = BASE_DIR / "outputs" if TENDENCIAS == "não" else BASE_DIR / "outputs_trended"
FIG_DIR = BASE_DIR / "figuras_revisor"
FIG_DIR.mkdir(parents=True, exist_ok=True)

if TENDENCIAS == "não":
    SST_FILE = OUT_DIR / "detrended_sst.npz"
else:
    SST_FILE = OUT_DIR / "trended_uvq_sst.npz"

RAIN_FILE = DATA_DIR / "EstacoesDados.txt"
SST_NETCDF_REF = DATA_DIR / "Era5_Dataset_sst.nc"


# ============================================================
# UTILITIES
# ============================================================

def to_celsius_if_needed(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if np.nanmean(x) > 100:
        return x - 273.15
    return x


def area_mask(lon2d: np.ndarray, lat2d: np.ndarray, area: dict) -> np.ndarray:
    return (
        (lon2d >= area["lon_min"]) & (lon2d <= area["lon_max"]) &
        (lat2d >= area["lat_min"]) & (lat2d <= area["lat_max"])
    )


def normalize(s: pd.Series) -> pd.Series:
    s = pd.Series(s, dtype=float).copy()
    return (s - s.mean()) / s.std(ddof=1)


def monthly_anomaly(series: pd.Series, clim_ini: int = ANOM_CLIM_INI, clim_fim: int = ANOM_CLIM_FIM) -> pd.Series:
    """Remove monthly climatology from a monthly time series."""
    s = series.copy()
    s.index = pd.to_datetime(s.index)

    clim_period = s[(s.index.year >= clim_ini) & (s.index.year <= clim_fim)]
    monthly_clim = clim_period.groupby(clim_period.index.month).mean()

    anom = s.copy() * np.nan
    for month in range(1, 13):
        anom.loc[anom.index.month == month] = s.loc[s.index.month == month] - monthly_clim.loc[month]
    return anom


def annual_amjj_mean(series: pd.Series) -> pd.Series:
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    s = s[s.index.month.isin(MESES_ESTACAO)]
    return s.groupby(s.index.year).mean()


def monthly_composite_by_years(series: pd.Series, years: list[int], months: list[int]) -> pd.Series:
    """Return month-of-year composite for selected years."""
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    years = set(int(y) for y in years)

    out = {}
    for month in months:
        vals = s[(s.index.year.isin(years)) & (s.index.month == month)]
        out[month] = np.nanmean(vals.values.astype(float)) if len(vals) > 0 else np.nan
    return pd.Series(out)


# ============================================================
# LOAD MONTHLY SERIES
# ============================================================

def load_sawp_sst_monthly() -> tuple[pd.Series, pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray]:
    """Return monthly SAWP SST series and grid information."""
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
        clim_april = np.nanmean(sst_c[mask_clim, :, :], axis=0)
        mask_sawp = clim_april >= ISOTHERM_THRESHOLD_C
        mask_sawp = mask_sawp & (lon2d >= -40.0) & (lon2d <= -25.0)
        print(f"SAWP isotherm points: {int(mask_sawp.sum())}")

    else:
        raise ValueError("METODO_SAWP must be 'quadrado' or 'isoterma'.")

    sst_sawp = np.nanmean(np.where(mask_sawp[None, :, :], sst_c, np.nan), axis=(1, 2))
    return pd.Series(sst_sawp, index=time, name="sawp_sst_c"), time, lon2d, lat2d, mask_sawp


def load_eneb_rainfall_monthly() -> pd.Series:
    """
    Load monthly ENEB rainfall.

    This function tries to use sawp.carregar_chuva_periodo_mensal if available.
    If not available, it reads EstacoesDados.txt directly using the common Monitor
    de Secas station format described in the project files:
        #code #x #y
        year month day value

    The direct reader averages/sums station monthly rainfall over ENEB stations.
    """
    if hasattr(sawp, "carregar_chuva_periodo_mensal"):
        rain = sawp.carregar_chuva_periodo_mensal(str(RAIN_FILE), TENDENCIAS)
        rain = pd.Series(np.asarray(rain).squeeze(), index=rain.index, name="eneb_rain_mm")
        return rain.replace(-999, np.nan)

    print("sawp.carregar_chuva_periodo_mensal not found. Reading EstacoesDados.txt directly...")
    return read_monthly_rainfall_from_estacoes(RAIN_FILE)


def read_monthly_rainfall_from_estacoes(path: Path) -> pd.Series:
    """
    Robust fallback reader for EstacoesDados.txt.

    It accepts the most common formats used in this project:

    1) Station blocks:
       # code lon lat
       year month day rainfall

    2) Flat station table:
       code lon lat year month day rainfall
       or
       year month day rainfall code lon lat

    3) Monthly matrix/table:
       date station_1 station_2 ...
       or
       year month station_1 station_2 ...

    Output:
       regional monthly rainfall series in mm, obtained by summing daily values
       within each station/month and then averaging across stations.
    """
    import re

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Rainfall file not found: {path}")

    def inside_eneb(lon, lat) -> bool:
        """Return True when coordinates are missing or inside AREA_ENEB_BOX."""
        if not (np.isfinite(lon) and np.isfinite(lat)):
            return True
        return (
            AREA_ENEB_BOX["lon_min"] <= lon <= AREA_ENEB_BOX["lon_max"] and
            AREA_ENEB_BOX["lat_min"] <= lat <= AREA_ENEB_BOX["lat_max"]
        )

    def clean_numeric_tokens(line: str) -> list[float]:
        """Extract all numeric tokens from a line, including negative coordinates."""
        return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)]

    # --------------------------------------------------------
    # First attempt: flat table/matrix using pandas.
    # This catches files with a header such as:
    # code x y year month day value
    # or date station1 station2 ...
    # --------------------------------------------------------
    try:
        df_try = pd.read_csv(path, sep=r"\s+|;|,", engine="python", comment=None)
        cols_lower = [str(c).lower().replace("#", "") for c in df_try.columns]
        df_try.columns = cols_lower

        # Case A: monthly/daily matrix with a date column.
        date_cols = [c for c in df_try.columns if c in ["date", "data", "time", "valid_time"]]
        if date_cols:
            date_col = date_cols[0]
            idx = pd.to_datetime(df_try[date_col], errors="coerce")
            value_cols = [c for c in df_try.columns if c != date_col]
            mat = df_try[value_cols].apply(pd.to_numeric, errors="coerce")
            mat = mat.where(mat >= 0)
            rain = mat.mean(axis=1, skipna=True)
            rain.index = idx
            rain = rain.dropna()
            if len(rain) > 0:
                rain = rain.resample("ME").sum(min_count=1)
                rain.name = "eneb_rain_mm"
                print(f"Monthly rainfall read as date matrix: {len(value_cols)} columns")
                return rain

        # Case B: table with year/month and station columns.
        if {"year", "month"}.issubset(set(df_try.columns)):
            meta_cols = {"year", "month", "day", "dia", "code", "codigo", "station", "id", "x", "y", "lon", "lat", "longitude", "latitude"}
            value_cols = [c for c in df_try.columns if c not in meta_cols]
            if len(value_cols) >= 1:
                yy = pd.to_numeric(df_try["year"], errors="coerce")
                mm = pd.to_numeric(df_try["month"], errors="coerce")
                valid = yy.between(1900, 2100) & mm.between(1, 12)
                mat = df_try.loc[valid, value_cols].apply(pd.to_numeric, errors="coerce")
                mat = mat.where(mat >= 0)
                idx = pd.to_datetime(dict(year=yy[valid].astype(int), month=mm[valid].astype(int), day=1))
                rain = pd.Series(mat.mean(axis=1, skipna=True).values, index=idx)
                rain = rain.resample("ME").sum(min_count=1)
                rain.name = "eneb_rain_mm"
                if rain.notna().sum() > 0:
                    print(f"Monthly rainfall read as year/month matrix: {len(value_cols)} columns")
                    return rain

        # Case C: flat station table with explicit coordinate/date/value columns.
        possible_year = [c for c in df_try.columns if c in ["year", "ano"]]
        possible_month = [c for c in df_try.columns if c in ["month", "mes", "mês"]]
        possible_day = [c for c in df_try.columns if c in ["day", "dia"]]
        possible_value = [c for c in df_try.columns if c in ["rain", "rainfall", "prec", "precip", "chuva", "valor", "value"]]
        possible_lon = [c for c in df_try.columns if c in ["lon", "longitude", "x"]]
        possible_lat = [c for c in df_try.columns if c in ["lat", "latitude", "y"]]
        possible_code = [c for c in df_try.columns if c in ["code", "codigo", "station", "id"]]

        if possible_year and possible_month and possible_value:
            ycol, mcol, vcol = possible_year[0], possible_month[0], possible_value[0]
            dcol = possible_day[0] if possible_day else None
            loncol = possible_lon[0] if possible_lon else None
            latcol = possible_lat[0] if possible_lat else None
            codecol = possible_code[0] if possible_code else None

            df = df_try.copy()
            df[ycol] = pd.to_numeric(df[ycol], errors="coerce")
            df[mcol] = pd.to_numeric(df[mcol], errors="coerce")
            df[vcol] = pd.to_numeric(df[vcol], errors="coerce")
            df.loc[df[vcol] < 0, vcol] = np.nan
            df = df[df[ycol].between(1900, 2100) & df[mcol].between(1, 12)]

            if loncol and latcol:
                df[loncol] = pd.to_numeric(df[loncol], errors="coerce")
                df[latcol] = pd.to_numeric(df[latcol], errors="coerce")
                df = df[
                    df[loncol].between(AREA_ENEB_BOX["lon_min"], AREA_ENEB_BOX["lon_max"]) &
                    df[latcol].between(AREA_ENEB_BOX["lat_min"], AREA_ENEB_BOX["lat_max"])
                ]

            df["date"] = pd.to_datetime(dict(year=df[ycol].astype(int), month=df[mcol].astype(int), day=1))
            if codecol:
                monthly_station = df.groupby(["date", codecol])[vcol].sum(min_count=1).unstack(codecol)
                rain = monthly_station.mean(axis=1, skipna=True)
            else:
                rain = df.groupby("date")[vcol].mean()
            rain = rain.resample("ME").sum(min_count=1)
            rain.name = "eneb_rain_mm"
            if rain.notna().sum() > 0:
                print("Monthly rainfall read as flat station table with named columns")
                return rain
    except Exception as exc:
        print(f"Pandas rainfall-table reader failed; trying line-by-line parser. Reason: {exc}")

    # --------------------------------------------------------
    # Second attempt: line-by-line parser for station blocks or
    # headerless flat rows.
    # --------------------------------------------------------
    station_series = []
    flat_rows = []

    current_lon = np.nan
    current_lat = np.nan
    current_code = None
    current_values = []

    def flush_station():
        nonlocal current_values, current_lon, current_lat, current_code
        if len(current_values) == 0:
            return
        if not inside_eneb(current_lon, current_lat):
            current_values = []
            return

        df = pd.DataFrame(current_values, columns=["year", "month", "day", "rain"])
        df["rain"] = pd.to_numeric(df["rain"], errors="coerce")
        df.loc[df["rain"] < 0, "rain"] = np.nan
        idx = pd.to_datetime(dict(year=df["year"].astype(int), month=df["month"].astype(int), day=1))
        monthly = pd.Series(df["rain"].values, index=idx).resample("ME").sum(min_count=1)
        station_series.append(monthly.rename(str(current_code) if current_code is not None else f"station_{len(station_series)+1}"))
        current_values = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            nums = clean_numeric_tokens(line)
            if len(nums) == 0:
                continue

            # Metadata/header line beginning with '#'.
            if line.startswith("#"):
                flush_station()
                current_lon = np.nan
                current_lat = np.nan
                current_code = None

                # Common header: #code #x #y or # code lon lat
                if len(nums) >= 3:
                    current_code = int(nums[0]) if abs(nums[0]) > 100 else nums[0]
                    # Prefer the two last numeric values as lon/lat.
                    a, b = nums[-2], nums[-1]
                    # If reversed, fix it.
                    if -20 <= a <= 10 and -60 <= b <= -20:
                        current_lat, current_lon = a, b
                    else:
                        current_lon, current_lat = a, b
                elif len(nums) >= 2:
                    current_lon, current_lat = nums[-2], nums[-1]
                continue

            # Headerless station block data row: year month day value
            if len(nums) >= 4 and 1900 <= nums[0] <= 2100 and 1 <= nums[1] <= 12:
                y, m, d, val = int(nums[0]), int(nums[1]), int(nums[2]), float(nums[3])
                current_values.append([y, m, d, val])
                continue

            # Headerless flat row with coordinates and date/value.
            # Try all combinations looking for year, month, rainfall and lon/lat.
            if len(nums) >= 7:
                # Pattern: code lon lat year month day rain
                cand_patterns = [
                    (3, 4, 5, 6, 1, 2, 0),  # code lon lat year month day rain
                    (0, 1, 2, 3, 5, 6, 4),  # year month day rain lon lat code-ish
                    (0, 1, 2, 6, 4, 5, 3),  # year month day ... lon lat rain
                ]
                for yi, mi, di, vi, loni, lati, codei in cand_patterns:
                    try:
                        y, m, d = int(nums[yi]), int(nums[mi]), int(nums[di])
                        val = float(nums[vi])
                        lon, lat = float(nums[loni]), float(nums[lati])
                        code = nums[codei]
                        if 1900 <= y <= 2100 and 1 <= m <= 12 and inside_eneb(lon, lat):
                            flat_rows.append([code, lon, lat, y, m, d, val])
                            break
                    except Exception:
                        pass

    flush_station()

    if len(station_series) > 0:
        mat = pd.concat(station_series, axis=1)
        rain = mat.mean(axis=1, skipna=True)
        rain.name = "eneb_rain_mm"
        print(f"Monthly rainfall read from station blocks: {mat.shape[1]} stations")
        return rain

    if len(flat_rows) > 0:
        df = pd.DataFrame(flat_rows, columns=["code", "lon", "lat", "year", "month", "day", "rain"])
        df.loc[df["rain"] < 0, "rain"] = np.nan
        df["date"] = pd.to_datetime(dict(year=df["year"].astype(int), month=df["month"].astype(int), day=1))
        monthly_station = df.groupby(["date", "code"])["rain"].sum(min_count=1).unstack("code")
        rain = monthly_station.mean(axis=1, skipna=True)
        rain.name = "eneb_rain_mm"
        print(f"Monthly rainfall read from headerless flat rows: {monthly_station.shape[1]} stations")
        return rain

    # --------------------------------------------------------
    # Last resort: use your annual AMJJ loader only to show a more useful error.
    # --------------------------------------------------------
    try:
        annual = sawp.carregar_chuva_periodo(str(path), TENDENCIAS)
        print("Annual rainfall loader works, but monthly rainfall could not be reconstructed from this file.")
        print("Annual series preview:")
        print(pd.Series(np.asarray(annual).squeeze(), index=annual.index).dropna().head())
    except Exception:
        pass

    raise RuntimeError(
        "Could not read monthly rainfall from EstacoesDados.txt. "
        "The file is probably not in a station-block or flat-table monthly/daily format. "
        "If you have monthly_matrix_qc or ENEB_monthly_precip_qc saved, set RAIN_FILE to that file "
        "or add sawp.carregar_chuva_periodo_mensal()."
    )


# ============================================================
# EVENT CLASSIFICATION
# ============================================================

def build_event_table(sst_monthly: pd.Series, rain_monthly: pd.Series) -> pd.DataFrame:
    """Classify years using AMJJ normalized anomalies."""
    sst_anom = monthly_anomaly(sst_monthly)
    rain_anom = monthly_anomaly(rain_monthly)

    sst_amjj = annual_amjj_mean(sst_anom)
    rain_amjj = annual_amjj_mean(rain_anom)

    df = pd.concat(
        [
            normalize(sst_amjj).rename("sst_amjj_norm"),
            normalize(rain_amjj).rename("rain_amjj_norm"),
            sst_amjj.rename("sst_amjj_anom_c"),
            rain_amjj.rename("rain_amjj_anom_mm"),
        ],
        axis=1,
    ).dropna()

    r, p = pearsonr(df["sst_amjj_norm"], df["rain_amjj_norm"])
    df.attrs["r"] = r
    df.attrs["p"] = p

    df["group"] = "Other"

    # Moderate warm: positive SST anomaly but below extreme threshold.
    df.loc[
        (df["sst_amjj_norm"] >= MODERATE_WARM_MIN) &
        (df["sst_amjj_norm"] < EXTREME_WARM_MIN),
        "group",
    ] = "Moderately warm"

    # Extreme warm: strongest positive SAWP SST anomalies.
    df.loc[
        df["sst_amjj_norm"] >= EXTREME_WARM_MIN,
        "group",
    ] = "Extremely warm"

    # Warm-wet and warm-dry diagnose coupled versus decoupled warm years.
    df.loc[
        (df["sst_amjj_norm"] >= MODERATE_WARM_MIN) &
        (df["rain_amjj_norm"] >= RAIN_WET_MIN),
        "group_coupling",
    ] = "Warm-wet"

    df.loc[
        (df["sst_amjj_norm"] >= MODERATE_WARM_MIN) &
        (df["rain_amjj_norm"] <= RAIN_DRY_MAX),
        "group_coupling",
    ] = "Warm-dry/normal"

    df["group_coupling"] = df["group_coupling"].fillna("Other")

    return df


# ============================================================
# PLOTTING
# ============================================================

def add_line(ax, x, y, label, marker="o"):
    ax.plot(x, y, marker=marker, linewidth=2.0, label=label)


def plot_monthly_composite_figure(event_table: pd.DataFrame, sst_monthly: pd.Series, rain_monthly: pd.Series):
    """Create the supplementary monthly evolution figure."""
    sst_std_anom = normalize(monthly_anomaly(sst_monthly))
    rain_std_anom = normalize(monthly_anomaly(rain_monthly))

    months = MESES_EVOLUCAO
    x = np.arange(len(months))
    xticklabels = [MONTH_LABELS[m] for m in months]

    groups = [
        ("Moderately warm", event_table.index[event_table["group"] == "Moderately warm"].astype(int).tolist()),
        ("Extremely warm", event_table.index[event_table["group"] == "Extremely warm"].astype(int).tolist()),
        ("Warm-wet", event_table.index[event_table["group_coupling"] == "Warm-wet"].astype(int).tolist()),
        ("Warm-dry/normal", event_table.index[event_table["group_coupling"] == "Warm-dry/normal"].astype(int).tolist()),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True)
    axes = axes.ravel()

    composite_rows = []

    for ax, (group_name, years) in zip(axes, groups):
        if len(years) == 0:
            ax.text(0.5, 0.5, f"No years in group\n{group_name}", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(group_name)
            ax.grid(alpha=0.3)
            continue

        sst_comp = monthly_composite_by_years(sst_std_anom, years, months)
        rain_comp = monthly_composite_by_years(rain_std_anom, years, months)

        for month in months:
            composite_rows.append({
                "group": group_name,
                "month": month,
                "month_label": MONTH_LABELS[month],
                "n_years": len(years),
                "years": ",".join(map(str, years)),
                "sawp_sst_standardized_anomaly": sst_comp.loc[month],
                "eneb_rainfall_standardized_anomaly": rain_comp.loc[month],
            })

        add_line(ax, x, sst_comp.values, "SAWP SST anomaly", marker="o")
        add_line(ax, x, rain_comp.values, "ENEB rainfall anomaly", marker="s")

        ax.axhline(0, color="k", linestyle="--", linewidth=1)
        ax.axvline(months.index(4), color="k", linestyle=":", linewidth=1)
        ax.set_title(f"{group_name} years (n={len(years)})\n{', '.join(map(str, years))}", fontsize=11)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels)
        ax.set_ylabel("Standardized monthly anomaly")
        ax.grid(alpha=0.3)
        ax.legend(loc="best", fontsize=9)

    r = event_table.attrs.get("r", np.nan)
    p = event_table.attrs.get("p", np.nan)
    fig.suptitle(
        "Monthly evolution of SAWP SST and ENEB rainfall anomalies before and during AMJJ\n"
        f"Event classes based on AMJJ anomalies; SST-rainfall AMJJ relationship: r={r:.2f}, p={p:.3f}",
        fontsize=14,
        y=0.995,
    )
    plt.tight_layout()

    out_fig = FIG_DIR / "SX_monthly_composite_SAWP_SST_ENEB_rainfall_FebAug.png"
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_fig}")

    out_csv = FIG_DIR / "SX_monthly_composite_values_FebAug.csv"
    pd.DataFrame(composite_rows).to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


def plot_single_panel_combined(event_table: pd.DataFrame, sst_monthly: pd.Series, rain_monthly: pd.Series):
    """Optional compact single-panel figure with all groups in one axis."""
    sst_std_anom = normalize(monthly_anomaly(sst_monthly))
    rain_std_anom = normalize(monthly_anomaly(rain_monthly))

    months = MESES_EVOLUCAO
    x = np.arange(len(months))
    xticklabels = [MONTH_LABELS[m] for m in months]

    groups = [
        ("Moderately warm", event_table.index[event_table["group"] == "Moderately warm"].astype(int).tolist()),
        ("Extremely warm", event_table.index[event_table["group"] == "Extremely warm"].astype(int).tolist()),
        ("Warm-wet", event_table.index[event_table["group_coupling"] == "Warm-wet"].astype(int).tolist()),
        ("Warm-dry/normal", event_table.index[event_table["group_coupling"] == "Warm-dry/normal"].astype(int).tolist()),
    ]

    fig, ax = plt.subplots(figsize=(12, 6))

    for group_name, years in groups:
        if len(years) == 0:
            continue
        sst_comp = monthly_composite_by_years(sst_std_anom, years, months)
        rain_comp = monthly_composite_by_years(rain_std_anom, years, months)
        ax.plot(x, sst_comp.values, marker="o", linewidth=2, label=f"SST - {group_name}")
        ax.plot(x, rain_comp.values, marker="s", linewidth=2, linestyle="--", label=f"Rain - {group_name}")

    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.axvline(months.index(4), color="k", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set_ylabel("Standardized monthly anomaly")
    ax.set_title("Monthly composite evolution of SAWP SST and ENEB rainfall anomalies")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8, ncol=2)
    plt.tight_layout()

    out_fig = FIG_DIR / "SX_monthly_composite_SAWP_SST_ENEB_rainfall_FebAug_combined.png"
    fig.savefig(out_fig, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_fig}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    print("Loading monthly SAWP SST...")
    sst_monthly, _, _, _, _ = load_sawp_sst_monthly()

    print("Loading monthly ENEB rainfall...")
    rain_monthly = load_eneb_rainfall_monthly()

    print("Building AMJJ event table...")
    event_table = build_event_table(sst_monthly, rain_monthly)
    out_table = FIG_DIR / "SX_event_table_monthly_composite_classes.csv"
    event_table.to_csv(out_table)
    print(f"Saved: {out_table}")

    print("\nEvent counts:")
    print(event_table["group"].value_counts())
    print("\nCoupling counts:")
    print(event_table["group_coupling"].value_counts())

    print("\nModerately warm years:")
    print(event_table[event_table["group"] == "Moderately warm"][["sst_amjj_norm", "rain_amjj_norm"]])
    print("\nExtremely warm years:")
    print(event_table[event_table["group"] == "Extremely warm"][["sst_amjj_norm", "rain_amjj_norm"]])
    print("\nWarm-wet years:")
    print(event_table[event_table["group_coupling"] == "Warm-wet"][["sst_amjj_norm", "rain_amjj_norm"]])
    print("\nWarm-dry/normal years:")
    print(event_table[event_table["group_coupling"] == "Warm-dry/normal"][["sst_amjj_norm", "rain_amjj_norm"]])

    print("\nGenerating supplementary monthly composite figure...")
    plot_monthly_composite_figure(event_table, sst_monthly, rain_monthly)
    plot_single_panel_combined(event_table, sst_monthly, rain_monthly)

    print("\nDone.")
    print(f"Outputs saved in: {FIG_DIR}")
