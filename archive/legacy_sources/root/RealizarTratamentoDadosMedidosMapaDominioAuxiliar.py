#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Monthly rain-gauge processing and quality control for calculating
regional mean precipitation over ENEB.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt

# ============================================================
# GLOBAL FIGURE STYLE
# ============================================================

FONTSIZE = 18

plt.rcParams.update({
    "font.size": FONTSIZE,
    "axes.titlesize": FONTSIZE,
    "axes.labelsize": FONTSIZE,
    "xtick.labelsize": FONTSIZE,
    "ytick.labelsize": FONTSIZE,
    "legend.fontsize": 14,
})


# ============================================================
# CONFIGURAÇÕES
# ============================================================

DATA_DIR = Path("./data/monitor")
OUT_DIR = Path("./data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATES = ["AL", "PB", "PE", "RN", "CE", "BA", "SE", "PI", "MA"]

DATE_START = "1959-01-31"
DATE_END = "2023-12-31"
DATE_INDEX = pd.date_range(DATE_START, DATE_END, freq="ME")

LON_MIN, LON_MAX = -38.0, -34.5
LAT_MIN, LAT_MAX = -15.0, 0.0

ABS_MONTHLY_MAX = 2000.0
REGIONAL_FACTOR = 5.0
REGIONAL_MIN_THRESHOLD = 500

MIN_VALID_STATIONS = 30
MIN_VALID_FRACTION_DYNAMIC = 0.10


# ============================================================
# FUNÇÕES
# ============================================================

def read_monitor_file(filename: Path):
    try:
        header = (
            pd.read_csv(filename, nrows=6, header=None, sep=":")
            .set_index(0)
        )
        header.columns = ["value"]

        data = pd.read_csv(
            filename,
            sep=r"\s+",
            skiprows=6,
            header=None
        )

        data["datetime"] = pd.to_datetime(
            data[[0, 1, 2]].astype(str).agg("-".join, axis=1),
            errors="coerce"
        )

        data = data.dropna(subset=["datetime"])
        data = data.set_index("datetime")
        data = data.drop(columns=[0, 1, 2])

        code = str(int(float(header.loc["#code"].value)))
        lon = float(header.loc["#x"].value)
        lat = float(header.loc["#y"].value)

        data.columns = [code]
        data[code] = pd.to_numeric(data[code], errors="coerce")

        return header, data, code, lon, lat

    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None, None, None, None, None


def monthly_basic_qc(df: pd.DataFrame):
    removed = []
    df_qc = df.copy()

    mask_negative = df_qc < 0
    for i, j in zip(*np.where(mask_negative.values)):
        removed.append({
            "date": df_qc.index[i],
            "station": df_qc.columns[j],
            "value": df_qc.iloc[i, j],
            "reason": "negative_value"
        })

    df_qc = df_qc.mask(mask_negative)

    mask_abs = df_qc > ABS_MONTHLY_MAX
    for i, j in zip(*np.where(mask_abs.values)):
        removed.append({
            "date": df_qc.index[i],
            "station": df_qc.columns[j],
            "value": df_qc.iloc[i, j],
            "reason": "above_absolute_monthly_limit"
        })

    df_qc = df_qc.mask(mask_abs)

    return df_qc, removed


def regional_outlier_qc(df: pd.DataFrame):
    removed = []
    df_qc = df.copy()

    regional_median = df_qc.median(axis=1, skipna=True)

    mask = pd.DataFrame(False, index=df_qc.index, columns=df_qc.columns)

    for date in df_qc.index:
        med = regional_median.loc[date]

        if pd.isna(med) or med <= 0:
            continue

        mask.loc[date] = (
            (df_qc.loc[date] > REGIONAL_MIN_THRESHOLD) &
            (df_qc.loc[date] > REGIONAL_FACTOR * med)
        )

    for i, j in zip(*np.where(mask.values)):
        removed.append({
            "date": df_qc.index[i],
            "station": df_qc.columns[j],
            "value": df_qc.iloc[i, j],
            "regional_median": regional_median.iloc[i],
            "reason": "spatially_inconsistent_extreme"
        })

    df_qc = df_qc.mask(mask)

    return df_qc, removed


# ============================================================
# LEITURA DAS ESTAÇÕES
# ============================================================

series = []
metadata = []

for state in STATES:
    files = sorted((DATA_DIR / state).glob("*"))
    print(f"{state}: {len(files)} files found")

    for file in files:
        header, df_station, code, lon, lat = read_monitor_file(file)

        if df_station is None:
            continue

        if not (LON_MIN <= lon <= LON_MAX and LAT_MIN <= lat <= LAT_MAX):
            continue

        df_station = df_station.reindex(DATE_INDEX)

        series.append(df_station)

        metadata.append({
            "code": code,
            "state": state,
            "lon": lon,
            "lat": lat,
            "file": str(file),
            "first_valid_date": df_station.dropna().index.min(),
            "last_valid_date": df_station.dropna().index.max(),
            "n_monthly_valid_raw": int(df_station.notna().sum().iloc[0])
        })


if len(series) == 0:
    raise RuntimeError("No rain gauges found within the defined area.")


# ============================================================
# MATRIZ MENSAL
# ============================================================

df_raw = pd.concat(series, axis=1)
metadata = pd.DataFrame(metadata)

print(f"Stations within the area: {df_raw.shape[1]}")


# ============================================================
# CONTROLE DE QUALIDADE
# ============================================================

removed_all = []

df_qc, removed = monthly_basic_qc(df_raw)
removed_all.extend(removed)

df_qc, removed = regional_outlier_qc(df_qc)
removed_all.extend(removed)

removed_df = pd.DataFrame(removed_all)


# ============================================================
# REMOÇÃO DE ESTAÇÕES SEM DADOS ÚTEIS
# ============================================================

station_valid_fraction = df_qc.notna().mean(axis=0)

metadata["valid_fraction_after_qc"] = metadata["code"].astype(str).map(
    station_valid_fraction
)

metadata["used_after_qc"] = metadata["valid_fraction_after_qc"] > 0

df_qc = df_qc.loc[:, df_qc.notna().any(axis=0)]


# ============================================================
# TESTE DE HOMOGENEIDADE DO REGIME PLUVIOMÉTRICO
# ============================================================

from scipy.spatial import cKDTree
from scipy.stats import pearsonr

meta = metadata.copy()
meta["code"] = meta["code"].astype(str)

meta = meta[meta["code"].isin(df_qc.columns.astype(str))].copy()
meta = meta.set_index("code").loc[df_qc.columns.astype(str)]

coords = meta[["lat", "lon"]].values
tree = cKDTree(coords)

N_NEIGHBORS = 10
MIN_COMMON_MONTHS = 60

results = []

for station in df_qc.columns.astype(str):

    dist, idx = tree.query(
        meta.loc[station, ["lat", "lon"]].values,
        k=N_NEIGHBORS + 1
    )

    neighbors = meta.index[idx[1:]]

    serie_station = df_qc[station]
    serie_neighbors = df_qc[neighbors].mean(axis=1, skipna=True)

    dados = pd.concat(
        [
            serie_station.rename("station"),
            serie_neighbors.rename("neighbors")
        ],
        axis=1
    ).dropna()

    if len(dados) < MIN_COMMON_MONTHS:
        r = np.nan
        p = np.nan
    else:
        r, p = pearsonr(dados["station"], dados["neighbors"])

    annual_mean = serie_station.groupby(serie_station.index.year).sum(min_count=1).mean()
    monthly_cycle = serie_station.groupby(serie_station.index.month).mean()

    peak_month = monthly_cycle.idxmax()
    seasonality_ratio = monthly_cycle.max() / monthly_cycle.mean()

    results.append({
        "station": station,
        "lat": meta.loc[station, "lat"],
        "lon": meta.loc[station, "lon"],
        "annual_mean_mm": annual_mean,
        "peak_month": peak_month,
        "seasonality_ratio": seasonality_ratio,
        "corr_neighbors": r,
        "p_corr_neighbors": p,
        "n_common_months": len(dados)
    })

regime_test = pd.DataFrame(results)

q01 = regime_test["annual_mean_mm"].quantile(0.01)
q99 = regime_test["annual_mean_mm"].quantile(0.99)

regime_test["flag_annual_extreme"] = (
    (regime_test["annual_mean_mm"] < q01) |
    (regime_test["annual_mean_mm"] > q99)
)

regime_test["flag_low_neighbor_corr"] = (
    regime_test["corr_neighbors"] < 0.30
)

regime_test["flag_unusual_peak_month"] = ~regime_test["peak_month"].isin([4, 5, 6, 7])

regime_test["flag_potentially_inconsistent"] = (
    regime_test["flag_annual_extreme"] |
    regime_test["flag_low_neighbor_corr"] |
    regime_test["flag_unusual_peak_month"]
)

regime_test.to_csv(
    OUT_DIR / "rain_gauge_regime_homogeneity_test.csv",
    index=False
)

print(regime_test["flag_potentially_inconsistent"].value_counts())
print(regime_test.sort_values("corr_neighbors").head(20))


# ============================================================
# MAPA DE ESTAÇÕES COM REGIME PLUVIOMÉTRICO DIFERENCIADO
# ============================================================

import cartopy.crs as ccrs
import cartopy.feature as cfeature

unusual = regime_test[
    regime_test["flag_unusual_peak_month"]
].copy()

normal = regime_test[
    ~regime_test["flag_unusual_peak_month"]
].copy()

fig = plt.figure(figsize=(12, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent(
    [
        regime_test["lon"].min() - 1,
        regime_test["lon"].max() + 1,
        regime_test["lat"].min() - 1,
        regime_test["lat"].max() + 1,
    ],
    crs=ccrs.PlateCarree()
)

ax.add_feature(cfeature.LAND, alpha=0.8)
ax.add_feature(cfeature.OCEAN, alpha=0.8)
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.BORDERS, linestyle=":")

states = cfeature.NaturalEarthFeature(
    category="cultural",
    name="admin_1_states_provinces",
    scale="10m",
    facecolor="none"
)

ax.add_feature(
    states,
    edgecolor="gray",
    linewidth=0.5
)

ax.scatter(
    normal["lon"],
    normal["lat"],
    s=8,
    c="black",
    transform=ccrs.PlateCarree(),
    label=f"Compatible rainfall regime (n={len(normal)})",
    zorder=3
)

ax.scatter(
    unusual["lon"],
    unusual["lat"],
    s=20,
    c="red",
    transform=ccrs.PlateCarree(),
    label=f"Distinct rainfall regime (n={len(unusual)})",
    zorder=4
)

gl = ax.gridlines(
    draw_labels=True,
    linewidth=0.5,
    color="gray",
    alpha=0.5,
    linestyle="--"
)

gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {"size": FONTSIZE}
gl.ylabel_style = {"size": FONTSIZE}

ax.set_title(
    "Rainfall Regime Homogeneity\n"
    "Stations with Distinct Seasonality Highlighted in Red",
    fontsize=FONTSIZE
)

ax.legend(
    loc="lower left",
    fontsize=14
)

ax.tick_params(labelsize=FONTSIZE)

plt.tight_layout()

plt.savefig(
    OUT_DIR / "stations_unusual_regime_map.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# REMOÇÃO DAS ESTAÇÕES COM REGIME DIFERENCIADO
# ============================================================

stations_remove = regime_test.loc[
    regime_test["flag_unusual_peak_month"],
    "station"
].astype(str)

print(f"Stations removed due to homogeneity criterion: {len(stations_remove)}")

regime_test.loc[
    regime_test["flag_unusual_peak_month"]
].to_csv(
    OUT_DIR / "stations_removed_homogeneity.csv",
    index=False
)

df_qc = df_qc.drop(
    columns=stations_remove,
    errors="ignore"
)

print(f"Remaining stations: {df_qc.shape[1]}")


# ============================================================
# ESTAÇÕES EFETIVAMENTE UTILIZADAS
# ============================================================

valid_station_codes = df_qc.columns.astype(str)

stations_used = metadata[
    metadata["code"].astype(str).isin(valid_station_codes)
].copy()

stations_used = stations_used[
    [
        "code",
        "state",
        "lat",
        "lon",
        "first_valid_date",
        "last_valid_date",
        "n_monthly_valid_raw",
        "valid_fraction_after_qc"
    ]
]

stations_used = stations_used.sort_values(["state", "code"])

n_stations_used = len(stations_used)

stations_used.to_csv(
    OUT_DIR / "rain_gauge_stations_used_lat_lon.csv",
    index=False
)

print(f"Stations effectively used: {n_stations_used}")
print(f"Used stations file: {OUT_DIR / 'rain_gauge_stations_used_lat_lon.csv'}")


# ============================================================
# MÉDIA REGIONAL COM CONTROLE DE COBERTURA
# ============================================================

n_valid = df_qc.notna().sum(axis=1)
n_total = df_qc.shape[1]

required_stations = max(
    MIN_VALID_STATIONS,
    int(np.ceil(MIN_VALID_FRACTION_DYNAMIC * n_total))
)

coverage_fraction = n_valid / n_total

regional_mean = df_qc.mean(axis=1, skipna=True)
regional_mean = regional_mean.where(n_valid >= required_stations)

regional_output = pd.DataFrame({
    "precipitation_mm": regional_mean,
    "n_valid_stations": n_valid,
    "n_total_stations": n_total,
    "coverage_fraction": coverage_fraction,
    "required_stations": required_stations
})


# ============================================================
# ANOMALIA DO MÁXIMO MENSAL ANUAL
# ============================================================

annual_max = regional_output["precipitation_mm"].groupby(
    regional_output.index.year
).max()

annual_anomaly = (annual_max - annual_max.mean()) / annual_max.mean()

annual_output = pd.DataFrame({
    "annual_max_monthly_precip_mm": annual_max,
    "relative_anomaly": annual_anomaly
})


# ============================================================
# EXPORTAÇÃO
# ============================================================

df_qc.to_csv(
    OUT_DIR / "rain_gauge_monthly_matrix_qc.csv",
    na_rep="-999"
)

regional_output.to_csv(
    OUT_DIR / "ENEB_monthly_precipitation_qc.csv",
    na_rep="-999"
)

annual_output.to_csv(
    OUT_DIR / "ENEB_annual_max_monthly_anomaly_qc.csv",
    na_rep="-999"
)

metadata.to_csv(
    OUT_DIR / "rain_gauge_metadata_qc.csv",
    index=False
)

removed_df.to_csv(
    OUT_DIR / "removed_outliers_qc.csv",
    index=False
)

regional_output["precipitation_mm"].to_csv(
    "./data/EstacoesDados.txt",
    header=False,
    na_rep="-999.",
    sep=" "
)


# ============================================================
# RELATÓRIO FINAL
# ============================================================

print("\nProcessing completed.")
print(f"Stations within the area: {df_raw.shape[1]}")
print(f"Stations with data after QC: {df_qc.shape[1]}")
print(f"Total removed values: {len(removed_df)}")
print(f"Minimum number of stations per month: {required_stations}")
print(f"Valid months in regional series: {regional_output['precipitation_mm'].notna().sum()}")
print(f"Main file: {OUT_DIR / 'ENEB_monthly_precipitation_qc.csv'}")


# ============================================================
# TESTE DE ROBUSTEZ DA SÉRIE REGIONAL
# ============================================================

from scipy.stats import linregress

ROBUST_DIR = OUT_DIR / "robustness"
ROBUST_DIR.mkdir(parents=True, exist_ok=True)


def calc_regional_series(df_subset):
    return df_subset.mean(axis=1, skipna=True)


def calc_trend(series):
    y = series.values.astype(float)
    mask = np.isfinite(y)

    if mask.sum() < 3:
        return np.nan, np.nan

    x = np.arange(mask.sum())
    reg = linregress(x, y[mask])

    return reg.slope, reg.pvalue


def calc_corr(a, b):
    dados = pd.concat(
        [a.rename("a"), b.rename("b")],
        axis=1
    ).dropna()

    if len(dados) < 3:
        return np.nan, np.nan, len(dados)

    r, p = pearsonr(dados["a"], dados["b"])
    return r, p, len(dados)


df_all = df_qc.copy()
serie_all = calc_regional_series(df_all)

station_fraction = df_qc.notna().mean(axis=0)

stations_50 = station_fraction[station_fraction >= 0.50].index
stations_70 = station_fraction[station_fraction >= 0.70].index
stations_90 = station_fraction[station_fraction >= 0.90].index

df_50 = df_qc[stations_50]
df_70 = df_qc[stations_70]
df_90 = df_qc[stations_90]

serie_50 = calc_regional_series(df_50) if df_50.shape[1] > 0 else pd.Series(index=df_qc.index, dtype=float)
serie_70 = calc_regional_series(df_70) if df_70.shape[1] > 0 else pd.Series(index=df_qc.index, dtype=float)
serie_90 = calc_regional_series(df_90) if df_90.shape[1] > 0 else pd.Series(index=df_qc.index, dtype=float)

metadata_tmp = metadata.copy()
metadata_tmp["code"] = metadata_tmp["code"].astype(str)
metadata_tmp["first_valid_date"] = pd.to_datetime(metadata_tmp["first_valid_date"])

available_codes = df_qc.columns.astype(str)

stations_from_1970 = metadata_tmp.loc[
    (metadata_tmp["first_valid_date"] <= "1970-12-31") &
    (metadata_tmp["code"].isin(available_codes)),
    "code"
]

stations_from_1965 = metadata_tmp.loc[
    (metadata_tmp["first_valid_date"] <= "1965-12-31") &
    (metadata_tmp["code"].isin(available_codes)),
    "code"
]

stations_from_1960 = metadata_tmp.loc[
    (metadata_tmp["first_valid_date"] <= "1960-12-31") &
    (metadata_tmp["code"].isin(available_codes)),
    "code"
]

df_1970 = df_qc[stations_from_1970] if len(stations_from_1970) > 0 else pd.DataFrame(index=df_qc.index)
df_1965 = df_qc[stations_from_1965] if len(stations_from_1965) > 0 else pd.DataFrame(index=df_qc.index)
df_1960 = df_qc[stations_from_1960] if len(stations_from_1960) > 0 else pd.DataFrame(index=df_qc.index)

serie_1970 = calc_regional_series(df_1970) if df_1970.shape[1] > 0 else pd.Series(index=df_qc.index, dtype=float)
serie_1965 = calc_regional_series(df_1965) if df_1965.shape[1] > 0 else pd.Series(index=df_qc.index, dtype=float)
serie_1960 = calc_regional_series(df_1960) if df_1960.shape[1] > 0 else pd.Series(index=df_qc.index, dtype=float)

series_dict = {
    "all_stations": serie_all,
    "stations_50pct_complete": serie_50,
    "stations_70pct_complete": serie_70,
    "stations_90pct_complete": serie_90,
    "stations_since_1970": serie_1970,
    "stations_since_1965": serie_1965,
    "stations_since_1960": serie_1960,
}

n_stations_dict = {
    "all_stations": df_all.shape[1],
    "stations_50pct_complete": df_50.shape[1],
    "stations_70pct_complete": df_70.shape[1],
    "stations_90pct_complete": df_90.shape[1],
    "stations_since_1970": df_1970.shape[1],
    "stations_since_1965": df_1965.shape[1],
    "stations_since_1960": df_1960.shape[1],
}

summary = []

for name, serie in series_dict.items():
    slope, p_trend = calc_trend(serie)

    if name == "all_stations":
        r, p_corr, n_common = 1.0, 0.0, serie.notna().sum()
    else:
        r, p_corr, n_common = calc_corr(serie_all, serie)

    summary.append({
        "subset": name,
        "n_stations": n_stations_dict[name],
        "n_valid_months": int(serie.notna().sum()),
        "corr_with_all": r,
        "p_corr_with_all": p_corr,
        "n_common_months": n_common,
        "linear_trend_mm_per_month": slope,
        "p_trend": p_trend
    })

summary_df = pd.DataFrame(summary)

summary_df.to_csv(
    ROBUST_DIR / "robustness_summary.csv",
    index=False
)

robust_series = pd.DataFrame(series_dict)

robust_series.to_csv(
    ROBUST_DIR / "robustness_monthly_series.csv",
    na_rep="-999"
)


def export_station_subset(codes, filename):
    subset = metadata_tmp[
        metadata_tmp["code"].isin(pd.Index(codes).astype(str))
    ].copy()

    subset = subset[
        [
            "code",
            "state",
            "lat",
            "lon",
            "first_valid_date",
            "last_valid_date",
            "n_monthly_valid_raw",
            "valid_fraction_after_qc"
        ]
    ]

    subset.to_csv(ROBUST_DIR / filename, index=False)


export_station_subset(df_all.columns, "stations_all.csv")
export_station_subset(stations_50, "stations_50pct_complete.csv")
export_station_subset(stations_70, "stations_70pct_complete.csv")
export_station_subset(stations_90, "stations_90pct_complete.csv")
export_station_subset(stations_from_1970, "stations_since_1970.csv")
export_station_subset(stations_from_1965, "stations_since_1965.csv")
export_station_subset(stations_from_1960, "stations_since_1960.csv")


# ============================================================
# ROBUSTNESS FIGURES
# ============================================================

plt.figure(figsize=(12, 6))

serie_all.rolling(12, min_periods=6).mean().plot(
    label=f"All stations ({df_all.shape[1]} stations)"
)

if df_50.shape[1] > 0:
    serie_50.rolling(12, min_periods=6).mean().plot(
        label=f">=50% complete ({df_50.shape[1]} stations)"
    )

if df_70.shape[1] > 0:
    serie_70.rolling(12, min_periods=6).mean().plot(
        label=f">=70% complete ({df_70.shape[1]} stations)"
    )

if df_90.shape[1] > 0:
    serie_90.rolling(12, min_periods=6).mean().plot(
        label=f">=90% complete ({df_90.shape[1]} stations)"
    )

plt.ylabel("Regional Mean Monthly Precipitation (mm)", fontsize=FONTSIZE)
plt.xlabel("Year", fontsize=FONTSIZE)
plt.title("Robustness Test of Regional Precipitation", fontsize=FONTSIZE)
plt.legend(fontsize=14)
plt.grid(alpha=0.3)
plt.tick_params(labelsize=FONTSIZE)
plt.tight_layout()

plt.savefig(
    ROBUST_DIR / "robustness_completeness_series.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


plt.figure(figsize=(12, 6))

serie_all.rolling(12, min_periods=6).mean().plot(
    label=f"All stations ({df_all.shape[1]} stations)"
)

if df_1970.shape[1] > 0:
    serie_1970.rolling(12, min_periods=6).mean().plot(
        label=f"Start <=1970 ({df_1970.shape[1]} stations)"
    )

if df_1965.shape[1] > 0:
    serie_1965.rolling(12, min_periods=6).mean().plot(
        label=f"Start <=1965 ({df_1965.shape[1]} stations)"
    )

if df_1960.shape[1] > 0:
    serie_1960.rolling(12, min_periods=6).mean().plot(
        label=f"Start <=1960 ({df_1960.shape[1]} stations)"
    )

plt.ylabel("Regional Mean Monthly Precipitation (mm)", fontsize=FONTSIZE)
plt.xlabel("Year", fontsize=FONTSIZE)
plt.title("Robustness Test Using Long-Term Stations", fontsize=FONTSIZE)
plt.legend(fontsize=14)
plt.grid(alpha=0.3)
plt.tick_params(labelsize=FONTSIZE)
plt.tight_layout()

plt.savefig(
    ROBUST_DIR / "robustness_long_records_series.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


plt.figure(figsize=(12, 4))

df_qc.notna().sum(axis=1).plot()

plt.ylabel("Number of Valid Stations", fontsize=FONTSIZE)
plt.xlabel("Year", fontsize=FONTSIZE)
plt.title("Monthly Availability of the Rain-Gauge Network", fontsize=FONTSIZE)
plt.grid(alpha=0.3)
plt.tick_params(labelsize=FONTSIZE)
plt.tight_layout()

plt.savefig(
    ROBUST_DIR / "valid_stations_per_month.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nRobustness test completed.")
print(summary_df)
print(f"Files saved in: {ROBUST_DIR}")