#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Processamento e controle de qualidade de dados pluviométricos mensais
para cálculo da precipitação regional média sobre o ENEB.

Entrada:
    ./data/monitor/{UF}/*

Saídas:
    ./data/processed/rain_gauge_monthly_matrix_qc.csv
    ./data/processed/ENEB_monthly_precipitation_qc.csv
    ./data/processed/ENEB_annual_max_monthly_anomaly_qc.csv
    ./data/processed/rain_gauge_metadata_qc.csv
    ./data/processed/removed_outliers_qc.csv
    ./data/EstacoesDados.txt
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime


# ============================================================
# CONFIGURAÇÕES
# ============================================================

DATA_DIR = Path("./data/monitor")
OUT_DIR = Path("./data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STATES = ["AL", "PB", "PE", "RN", "CE", "BA", "SE", "PI", "MA"]

DATE_START = "1959-01-31"
DATE_END   = "2023-12-31"
DATE_INDEX = pd.date_range(DATE_START, DATE_END, freq="ME")

# Área de estudo usada no seu script original
LON_MIN, LON_MAX = -38.0, -34.5
LAT_MIN, LAT_MAX = -15.0, 0.0

# Critérios de controle de qualidade mensal
ABS_MONTHLY_MAX = 2000.0      # limite físico conservador
REGIONAL_FACTOR = 5.0         # fator em relação à mediana regional
REGIONAL_MIN_THRESHOLD = 500  # só avalia outlier regional acima desse valor

# Critério mínimo de cobertura espacial
MIN_VALID_STATIONS = 30       # ajuste conforme sensibilidade
MIN_VALID_FRACTION_DYNAMIC = 0.10


# ============================================================
# FUNÇÕES
# ============================================================

def read_monitor_file(filename: Path):
    """
    Lê um arquivo de estação no formato monitor.
    Assume que os dados são acumulados mensais.
    """

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
        print(f"Erro ao ler {filename}: {e}")
        return None, None, None, None, None


def monthly_basic_qc(df: pd.DataFrame):
    """
    Controle básico de qualidade mensal.

    Mantém zero como valor físico válido.
    Remove:
        - valores negativos;
        - acumulados mensais fisicamente implausíveis.
    """

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
    """
    Controle de outliers com consistência regional.

    Remove valores muito altos apenas quando:
        1. excedem REGIONAL_MIN_THRESHOLD;
        2. excedem REGIONAL_FACTOR vezes a mediana regional mensal.

    Isso reduz a chance de remover eventos extremos reais.
    """

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
    print(f"{state}: {len(files)} arquivos encontrados")

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
    raise RuntimeError("Nenhuma estação encontrada na área definida.")


# ============================================================
# MATRIZ MENSAL
# ============================================================

df_raw = pd.concat(series, axis=1)
metadata = pd.DataFrame(metadata)

print(f"Estações dentro da área: {df_raw.shape[1]}")


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
import numpy as np
import pandas as pd

# df_qc: matriz mensal estação x tempo
# metadata: contém code, lat, lon

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

    neighbors = meta.index[idx[1:]]  # remove a própria estação

    serie_station = df_qc[station]

    serie_neighbors = df_qc[neighbors].mean(axis=1, skipna=True)

    dados = pd.concat(
        [serie_station.rename("station"),
         serie_neighbors.rename("neighbors")],
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

# Critérios de alerta, não remoção automática
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

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# estações consideradas "unusual"
unusual = regime_test[
    regime_test["flag_unusual_peak_month"]
].copy()

# estações consideradas normais
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

# ------------------------------------------------------------
# Mapa base
# ------------------------------------------------------------

ax.add_feature(cfeature.LAND, alpha=0.8)
ax.add_feature(cfeature.OCEAN, alpha=0.8)

ax.add_feature(
    cfeature.COASTLINE,
    linewidth=1
)

ax.add_feature(
    cfeature.BORDERS,
    linestyle=":"
)

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

# ------------------------------------------------------------
# Estações normais
# ------------------------------------------------------------

ax.scatter(
    normal["lon"],
    normal["lat"],
    s=8,
    c="black",
    transform=ccrs.PlateCarree(),
    label=f"Regime compatível (n={len(normal)})",
    zorder=3
)

# ------------------------------------------------------------
# Estações com regime diferenciado
# ------------------------------------------------------------

ax.scatter(
    unusual["lon"],
    unusual["lat"],
    s=20,
    c="red",
    transform=ccrs.PlateCarree(),
    label=f"Regime diferenciado (n={len(unusual)})",
    zorder=4
)

# ------------------------------------------------------------
# Grid
# ------------------------------------------------------------

gl = ax.gridlines(
    draw_labels=True,
    linewidth=0.5,
    color="gray",
    alpha=0.5,
    linestyle="--"
)

gl.top_labels = False
gl.right_labels = False

# ------------------------------------------------------------
# Título
# ------------------------------------------------------------

plt.title(
    "Homogeneidade do Regime Pluviométrico\n"
    "Estações com sazonalidade diferenciada destacadas em vermelho",
    fontsize=14
)

plt.legend(
    loc="lower left"
)

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

print(f"Estações removidas por homogeneidade: {len(stations_remove)}")

# Salva lista das removidas
regime_test.loc[
    regime_test["flag_unusual_peak_month"]
].to_csv(
    OUT_DIR / "stations_removed_homogeneity.csv",
    index=False
)

# Remove da matriz principal
df_qc = df_qc.drop(
    columns=stations_remove,
    errors="ignore"
)

print(f"Estações restantes: {df_qc.shape[1]}")

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

print(f"Estações efetivamente utilizadas: {n_stations_used}")
print(f"Arquivo de estações utilizadas: {OUT_DIR / 'rain_gauge_stations_used_lat_lon.csv'}")

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

print("\nProcessamento concluído.")
print(f"Estações dentro da área: {df_raw.shape[1]}")
print(f"Estações com dados após QC: {df_qc.shape[1]}")
print(f"Total de valores removidos: {len(removed_df)}")
print(f"Número mínimo de estações por mês: {required_stations}")
print(f"Meses válidos na série regional: {regional_output['precipitation_mm'].notna().sum()}")
print(f"Arquivo principal: {OUT_DIR / 'ENEB_monthly_precipitation_qc.csv'}")

# ============================================================
# TESTE DE ROBUSTEZ DA SÉRIE REGIONAL
# ============================================================

from scipy.stats import pearsonr, linregress
import matplotlib.pyplot as plt


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


# ------------------------------------------------------------
# 1. Série com todas as estações
# ------------------------------------------------------------

df_all = df_qc.copy()
serie_all = calc_regional_series(df_all)


# ------------------------------------------------------------
# 2. Subconjuntos por completude temporal
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 3. Subconjuntos por início da série
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 4. Comparação estatística
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 5. Exporta séries mensais
# ------------------------------------------------------------

robust_series = pd.DataFrame(series_dict)

robust_series.to_csv(
    ROBUST_DIR / "robustness_monthly_series.csv",
    na_rep="-999"
)


# ------------------------------------------------------------
# 6. Exporta estações de cada subconjunto
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# 7. Gráfico de robustez
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

serie_all.rolling(12, min_periods=6).mean().plot(
    label=f"Todas ({df_all.shape[1]} estações)"
)

if df_50.shape[1] > 0:
    serie_50.rolling(12, min_periods=6).mean().plot(
        label=f">=50% completa ({df_50.shape[1]} estações)"
    )

if df_70.shape[1] > 0:
    serie_70.rolling(12, min_periods=6).mean().plot(
        label=f">=70% completa ({df_70.shape[1]} estações)"
    )

if df_90.shape[1] > 0:
    serie_90.rolling(12, min_periods=6).mean().plot(
        label=f">=90% completa ({df_90.shape[1]} estações)"
    )

plt.ylabel("Precipitação mensal média regional (mm)")
plt.xlabel("Ano")
plt.title("Teste de robustez da precipitação regional")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    ROBUST_DIR / "robustness_completeness_series.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 8. Gráfico por início da série
# ------------------------------------------------------------

plt.figure(figsize=(12, 6))

serie_all.rolling(12, min_periods=6).mean().plot(
    label=f"Todas ({df_all.shape[1]} estações)"
)

if df_1970.shape[1] > 0:
    serie_1970.rolling(12, min_periods=6).mean().plot(
        label=f"Início <=1970 ({df_1970.shape[1]} estações)"
    )

if df_1965.shape[1] > 0:
    serie_1965.rolling(12, min_periods=6).mean().plot(
        label=f"Início <=1965 ({df_1965.shape[1]} estações)"
    )

if df_1960.shape[1] > 0:
    serie_1960.rolling(12, min_periods=6).mean().plot(
        label=f"Início <=1960 ({df_1960.shape[1]} estações)"
    )

plt.ylabel("Precipitação mensal média regional (mm)")
plt.xlabel("Ano")
plt.title("Teste de robustez por estações de longa duração")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    ROBUST_DIR / "robustness_long_records_series.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ------------------------------------------------------------
# 9. Número de estações válidas por mês
# ------------------------------------------------------------

plt.figure(figsize=(12, 4))

df_qc.notna().sum(axis=1).plot()

plt.ylabel("Número de estações válidas")
plt.xlabel("Ano")
plt.title("Disponibilidade mensal da rede pluviométrica")
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    ROBUST_DIR / "valid_stations_per_month.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print("\nTeste de robustez concluído.")
print(summary_df)
print(f"Arquivos salvos em: {ROBUST_DIR}")

# ============================================================
# FIGURA DO DOMÍNIO, SAWP E ESTAÇÕES PLUVIOMÉTRICAS
# ============================================================

import netCDF4 as nc
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.patches import Rectangle


FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# Estações efetivamente usadas após QC/homogeneidade
# ------------------------------------------------------------

stations_used_codes = df_qc.columns.astype(str)

stations_used = metadata[
    metadata["code"].astype(str).isin(stations_used_codes)
].copy()

# Caso exista uma lista de estações removidas por homogeneidade
if "stations_remove" in globals():
    stations_removed = regime_test[
        regime_test["station"].astype(str).isin(pd.Index(stations_remove).astype(str))
    ].copy()
else:
    stations_removed = pd.DataFrame(columns=["station", "lat", "lon"])


# ------------------------------------------------------------
# Leitura da TSM usada para ilustrar o domínio oceânico
# ------------------------------------------------------------

ds1 = nc.Dataset("./data/dominio/sst.nc")

lat_sst = np.array(ds1["lat"][:])
lon_sst = np.array(ds1["lon"][:])
sst = np.array(ds1["sst"][:])

# Ajusta longitude para -180 a 180
lon_sst_adj = np.where(lon_sst > 180, lon_sst - 360, lon_sst)

# Ordena longitudes após ajuste
idx_sort = np.argsort(lon_sst_adj)

lon_sst_adj = lon_sst_adj[idx_sort]
sst_adj = sst[:, :, idx_sort]

lon_sst_grid, lat_sst_grid = np.meshgrid(lon_sst_adj, lat_sst)


# ------------------------------------------------------------
# Leitura da topografia
# ------------------------------------------------------------

ds2 = nc.Dataset("./data/dominio/srtm30plus_v11_land.nc")

lat_topo = np.array(ds2["lat"][:])
lon_topo = np.array(ds2["lon"][:])
hgt = np.array(ds2["elev"][:])

lon_topo_grid, lat_topo_grid = np.meshgrid(lon_topo, lat_topo)


# ------------------------------------------------------------
# Figura
# ------------------------------------------------------------

fig = plt.figure(figsize=(10, 10))
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent([-40, -20, -15, 0], crs=ccrs.PlateCarree())

# Mapa base
ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
ax.add_feature(cfeature.OCEAN, facecolor="white", alpha=1.0)
ax.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=1.0)
ax.add_feature(cfeature.BORDERS, linestyle="--", edgecolor="black", alpha=0.4)

states_feature = cfeature.NaturalEarthFeature(
    category="cultural",
    name="admin_1_states_provinces",
    scale="10m",
    facecolor="none"
)

ax.add_feature(states_feature, edgecolor="black", linewidth=0.5, alpha=0.4)

# ------------------------------------------------------------
# TSM e contorno de 28.5 °C
# ------------------------------------------------------------

# Definindo os níveis (levels) para o intervalo entre 28.5 e 32
levels = np.arange(27, 30, 0.25)


# Plotando o gráfico de contorno preenchido
ct = plt.contourf(lon_sst_grid, lat_sst_grid, sst_adj[3, :, :], levels=levels, cmap='hot_r')
ct2 = plt.contour(lon_sst_grid, lat_sst_grid, sst_adj[3, :, :], levels=[28.5], colors='k', linewidths=2)
plt.clabel(ct2, inline=True, fontsize=18, fmt='%1.1f', colors='k')

ct2 = ax.contour(
    lon_sst_grid,
    lat_sst_grid,
    sst_adj[3, :, :],
    levels=[28.5],
    colors='k',
    linewidths=2.5,
    transform=ccrs.PlateCarree()
)

ax.clabel(
    ct2,
    inline=True,
    fontsize=12,
    fmt="%1.1f"
)

ax.clabel(
    ct2,
    inline=True,
    fontsize=14,
    fmt='%1.1f'
)

# ------------------------------------------------------------
# COLORBAR SST
# ------------------------------------------------------------

cbar_sst = fig.colorbar(
    ct,
    ax=ax,
    orientation='horizontal',
    pad=0.13,
    fraction=0.06,
    shrink=0.85,
    aspect=35
)

cbar_sst.set_label(
    "Sea Surface Temperature (°C)",
    fontsize=16
)

# ------------------------------------------------------------
# Topografia
# ------------------------------------------------------------

sh = ax.pcolormesh(
    lon_topo_grid,
    lat_topo_grid,
    hgt,
    cmap="terrain",
    vmin=0,
    vmax=1200,
    alpha=0.55,
    transform=ccrs.PlateCarree()
)

cbar = plt.colorbar(
    sh,
    ax=ax,
    orientation="vertical",
    pad=0.01,
    fraction=0.025
)

cbar.set_label("Topography (m)", fontsize=13)


# ------------------------------------------------------------
# Estações usadas
# ------------------------------------------------------------

ax.scatter(
    stations_used["lon"],
    stations_used["lat"],
    color="black",
    s=5,
    transform=ccrs.PlateCarree(),
    label=f"Rain gauges used (n={len(stations_used)})",
    zorder=4
)

# ------------------------------------------------------------
# Estações removidas por homogeneidade, se existirem
# ------------------------------------------------------------

# ------------------------------------------------------------
# Retângulo da área SAWP
# ------------------------------------------------------------

rect_sawp = Rectangle(
    xy=(-34, -12.5),
    width=9,
    height=8.5,
    edgecolor="black",
    facecolor="lightgray",
    alpha=0.25,
    linewidth=2.5,
    transform=ccrs.PlateCarree(),
    zorder=6
)

ax.add_patch(rect_sawp)

ax.text(
    -31.8,
    -7.5,
    "SAWP",
    color="black",
    fontsize=12,
    weight="bold",
    transform=ccrs.PlateCarree(),
    zorder=7
)

# ------------------------------------------------------------
# Grade geográfica
# ------------------------------------------------------------

gl = ax.gridlines(
    draw_labels=True,
    crs=ccrs.PlateCarree(),
    color="gray",
    linestyle="--",
    linewidth=0.5,
    alpha=0.6
)

gl.top_labels = False
gl.right_labels = False
gl.xlabel_style = {"size": 12}
gl.ylabel_style = {"size": 12}
gl.xlines = False
gl.ylines = False


# ------------------------------------------------------------
# Finalização
# ------------------------------------------------------------

ax.legend(
    loc="upper right",
    fontsize=10,
    frameon=True
)

fig.text(0.5, 0.2, "Longitude", ha="center", fontsize=20)
fig.text(0.03, 0.5, "Latitude", va="center", rotation="vertical", fontsize=20)

plt.savefig(
    FIG_DIR / "Mapa_dominio_SAWP_ENEB_estacoes.png",
    dpi=360,
    bbox_inches="tight"
)

plt.show()
plt.close()

print(f"Figura de domínio salva em: {FIG_DIR / 'Mapa_dominio_SAWP_ENEB_estacoes.png'}")

# ============================================================
# SUPPLEMENTARY MATERIAL
# TESTE DE MUDANÇA DE TENDÊNCIA E EFEITO DO DETRENDING
# ============================================================

from scipy.stats import linregress, norm, theilslopes
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUPP_DIR = OUT_DIR / "supplementary"
SUPP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Série mensal regional e máximo anual AMJJ
# ============================================================

prec_monthly = regional_output["precipitation_mm"].copy()


def decimal_year_from_datetime(index):
    """
    Converte um DatetimeIndex para ano decimal.

    Isso permite ajustar a tendência mensal em unidade de mm/ano,
    respeitando o espaçamento temporal da série.
    """

    index = pd.DatetimeIndex(index)
    return index.year + (index.dayofyear - 1) / 365.25


def detrend_keep_mean(series, x_values=None):
    """
    Remove tendência linear preservando a média da série.

    A saída é:
        série detrendada = série original - tendência linear + média original

    Se x_values for informado, a regressão é feita usando esses valores
    como eixo temporal. Para série mensal, recomenda-se usar ano decimal,
    de modo que a inclinação fique em mm/ano.
    """

    y = series.astype(float).copy()
    valid = y.notna()

    if valid.sum() < 3:
        return y, None

    if x_values is None:
        x = np.arange(valid.sum(), dtype=float)
    else:
        x_all = np.asarray(x_values, dtype=float)
        x = x_all[valid.values]

    reg = linregress(x, y[valid].values)
    trend = reg.slope * x + reg.intercept

    y_dt = y.copy()
    y_dt.loc[valid] = y[valid].values - trend + y[valid].mean()

    return y_dt, reg


# ------------------------------------------------------------
# 1.1 Detrending primeiro na série mensal regional
# ------------------------------------------------------------

x_monthly = decimal_year_from_datetime(prec_monthly.index)

prec_monthly_dt, reg_monthly = detrend_keep_mean(
    prec_monthly,
    x_values=x_monthly
)

prec_monthly_dt.name = "monthly_precipitation_detrended_mm"


# ------------------------------------------------------------
# 1.2 Máximo anual AMJJ original e máximo anual AMJJ detrendido
# ------------------------------------------------------------

mask_amjj = prec_monthly.index.month.isin([4, 5, 6, 7])

# Máximo anual AMJJ da série original, mantido para comparação e tendência bruta
prec_amjj = (
    prec_monthly[mask_amjj]
    .groupby(prec_monthly[mask_amjj].index.year)
    .max()
)

prec_amjj.name = "AMJJ_precipitation_mm"

# Máximo anual AMJJ calculado APÓS remover a tendência mensal
prec_amjj_dt = (
    prec_monthly_dt[mask_amjj]
    .groupby(prec_monthly_dt[mask_amjj].index.year)
    .max()
)

prec_amjj_dt.name = "AMJJ_precipitation_monthly_detrended_then_annual_max_mm"


# ------------------------------------------------------------
# 2. Tendência da série original de máximos anuais AMJJ
# ------------------------------------------------------------

valid_amjj = prec_amjj.dropna()
years_amjj = valid_amjj.index.values.astype(float)
reg_full = linregress(years_amjj, valid_amjj.values.astype(float))



# ============================================================
# 2.1 Testes não paramétricos de tendência e mudança de regime
# ============================================================

def mann_kendall_test(series):
    """
    Teste de Mann-Kendall bilateral para tendência monotônica.

    Retorna S, Z, p-valor e direção da tendência.
    Implementação sem dependência externa, com correção para empates.
    """

    s = series.dropna().astype(float)
    y = s.values
    n = len(y)

    if n < 3:
        return {
            "n": n,
            "S": np.nan,
            "var_S": np.nan,
            "z": np.nan,
            "p_value": np.nan,
            "trend": "insufficient_data"
        }

    S = 0
    for k in range(n - 1):
        S += np.sign(y[k + 1:] - y[k]).sum()

    _, counts = np.unique(y, return_counts=True)
    tie_counts = counts[counts > 1]

    var_S = (
        n * (n - 1) * (2 * n + 5)
        - np.sum(tie_counts * (tie_counts - 1) * (2 * tie_counts + 5))
    ) / 18.0

    if var_S == 0:
        z = 0.0
    elif S > 0:
        z = (S - 1) / np.sqrt(var_S)
    elif S < 0:
        z = (S + 1) / np.sqrt(var_S)
    else:
        z = 0.0

    p_value = 2 * (1 - norm.cdf(abs(z)))

    if p_value < 0.05 and z > 0:
        trend = "increasing_significant_95pct"
    elif p_value < 0.05 and z < 0:
        trend = "decreasing_significant_95pct"
    elif z > 0:
        trend = "increasing_not_significant_95pct"
    elif z < 0:
        trend = "decreasing_not_significant_95pct"
    else:
        trend = "no_trend"

    return {
        "n": n,
        "S": S,
        "var_S": var_S,
        "z": z,
        "p_value": p_value,
        "trend": trend
    }


def sens_slope_test(series):
    """
    Estimativa robusta Sen/Theil-Sen da inclinação em mm/ano.
    Usa o índice anual da série como eixo temporal.
    """

    s = series.dropna().astype(float)

    if len(s) < 3:
        return {
            "n": len(s),
            "sen_slope_mm_per_year": np.nan,
            "intercept": np.nan,
            "lower_slope_95pct": np.nan,
            "upper_slope_95pct": np.nan
        }

    x = s.index.values.astype(float)
    y = s.values.astype(float)

    slope, intercept, lower, upper = theilslopes(y, x, alpha=0.95)

    return {
        "n": len(s),
        "sen_slope_mm_per_year": slope,
        "intercept": intercept,
        "lower_slope_95pct": lower,
        "upper_slope_95pct": upper
    }


def pettitt_test(series):
    """
    Teste de Pettitt para mudança abrupta na mediana da série.

    Retorna o ano provável de mudança, estatística K e p-valor aproximado.
    """

    s = series.dropna().astype(float)
    y = s.values
    years = s.index.values.astype(int)
    n = len(y)

    if n < 6:
        return {
            "n": n,
            "change_year": np.nan,
            "K": np.nan,
            "p_value": np.nan,
            "median_before": np.nan,
            "median_after": np.nan
        }

    ranks = pd.Series(y).rank(method="average").values
    U = np.array([
        2 * np.sum(ranks[:t]) - t * (n + 1)
        for t in range(1, n + 1)
    ])

    k_idx = int(np.argmax(np.abs(U)))
    K = float(np.abs(U[k_idx]))
    change_year = int(years[k_idx])

    p_value = 2 * np.exp((-6 * K ** 2) / (n ** 3 + n ** 2))
    p_value = min(float(p_value), 1.0)

    before = y[:k_idx + 1]
    after = y[k_idx + 1:]

    return {
        "n": n,
        "change_year": change_year,
        "K": K,
        "p_value": p_value,
        "median_before": float(np.nanmedian(before)) if len(before) else np.nan,
        "median_after": float(np.nanmedian(after)) if len(after) else np.nan
    }


def run_trend_tests(series, series_name):
    """Executa regressão linear, Mann-Kendall, Sen's slope e Pettitt."""

    s = series.dropna().astype(float)

    if len(s) >= 3:
        x = s.index.values.astype(float)
        reg = linregress(x, s.values.astype(float))
        linear = {
            "linear_slope_mm_per_year": reg.slope,
            "linear_intercept": reg.intercept,
            "linear_r": reg.rvalue,
            "linear_p_value": reg.pvalue,
            "linear_stderr": reg.stderr
        }
    else:
        linear = {
            "linear_slope_mm_per_year": np.nan,
            "linear_intercept": np.nan,
            "linear_r": np.nan,
            "linear_p_value": np.nan,
            "linear_stderr": np.nan
        }

    mk = mann_kendall_test(s)
    sen = sens_slope_test(s)
    pettitt = pettitt_test(s)

    return {
        "series": series_name,
        "n": int(len(s)),
        **linear,
        "mann_kendall_S": mk["S"],
        "mann_kendall_z": mk["z"],
        "mann_kendall_p_value": mk["p_value"],
        "mann_kendall_trend": mk["trend"],
        "sen_slope_mm_per_year": sen["sen_slope_mm_per_year"],
        "sen_lower_slope_95pct": sen["lower_slope_95pct"],
        "sen_upper_slope_95pct": sen["upper_slope_95pct"],
        "pettitt_change_year": pettitt["change_year"],
        "pettitt_K": pettitt["K"],
        "pettitt_p_value": pettitt["p_value"],
        "pettitt_median_before": pettitt["median_before"],
        "pettitt_median_after": pettitt["median_after"]
    }


trend_tests_summary = pd.DataFrame([
    run_trend_tests(prec_amjj, "AMJJ_original_annual_max"),
    run_trend_tests(prec_amjj_dt, "AMJJ_monthly_detrended_then_annual_max")
])

trend_tests_summary.to_csv(
    SUPP_DIR / "AMJJ_MannKendall_SenSlope_Pettitt_tests.csv",
    index=False
)

# ============================================================
# 3. Teste simples de mudança de tendência
# ============================================================

def piecewise_trend_test(series, min_years=10):
    """
    Teste exploratório de mudança de tendência.
    Busca o ano de quebra que minimiza o erro quadrático residual
    de duas regressões lineares independentes.
    """

    s = series.dropna()
    years = s.index.values.astype(int)
    y = s.values.astype(float)

    candidates = years[min_years:-min_years]

    results = []

    for break_year in candidates:
        left = years <= break_year
        right = years > break_year

        x_left = years[left]
        y_left = y[left]

        x_right = years[right]
        y_right = y[right]

        reg_left = linregress(x_left, y_left)
        reg_right = linregress(x_right, y_right)

        fit_left = reg_left.slope * x_left + reg_left.intercept
        fit_right = reg_right.slope * x_right + reg_right.intercept

        rss = np.sum((y_left - fit_left) ** 2) + np.sum((y_right - fit_right) ** 2)

        results.append({
            "break_year": break_year,
            "rss": rss,
            "slope_before_mm_per_year": reg_left.slope,
            "p_before": reg_left.pvalue,
            "slope_after_mm_per_year": reg_right.slope,
            "p_after": reg_right.pvalue,
            "r_before": reg_left.rvalue,
            "r_after": reg_right.rvalue
        })

    results = pd.DataFrame(results)

    best = results.loc[results["rss"].idxmin()].copy()

    return results, best


trend_scan, best_break = piecewise_trend_test(prec_amjj)

trend_scan.to_csv(
    SUPP_DIR / "AMJJ_trend_break_scan.csv",
    index=False
)

best_break.to_frame().T.to_csv(
    SUPP_DIR / "AMJJ_best_trend_break.csv",
    index=False
)


# ============================================================
# 4. Figura: série original, detrendada e tendências
# ============================================================

fig, ax = plt.subplots(figsize=(11, 5))

prec_amjj.plot(
    ax=ax,
    color="black",
    linewidth=1.4,
    label="Original AMJJ precipitation"
)

prec_amjj_dt.plot(
    ax=ax,
    color="gray",
    linewidth=1.4,
    linestyle="--",
    label="Monthly detrended → annual AMJJ max"
)

# tendência total
valid = prec_amjj.dropna()
years = valid.index.values.astype(int)
fit_full = reg_full.slope * years + reg_full.intercept

ax.plot(
    years,
    fit_full,
    color="red",
    linewidth=2,
    label=f"Annual-max trend: {reg_full.slope:.2f} mm yr⁻¹"
)

# quebra ótima
break_year = int(best_break["break_year"])
ax.axvline(
    break_year,
    color="blue",
    linestyle=":",
    linewidth=2,
    label=f"Estimated break year: {break_year}"
)

ax.set_ylabel("AMJJ precipitation (mm)")
ax.set_xlabel("Year")
ax.set_title("AMJJ regional precipitation: raw annual max vs monthly-detrended annual max")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    SUPP_DIR / "AMJJ_original_detrended_trend.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 5. Figura: tendência antes e depois da quebra
# ============================================================

fig, ax = plt.subplots(figsize=(11, 5))

prec_amjj.plot(
    ax=ax,
    color="black",
    linewidth=1.2,
    label="AMJJ precipitation"
)

s = prec_amjj.dropna()
years = s.index.values.astype(int)
y = s.values.astype(float)

left = years <= break_year
right = years > break_year

reg_left = linregress(years[left], y[left])
reg_right = linregress(years[right], y[right])

fit_left = reg_left.slope * years[left] + reg_left.intercept
fit_right = reg_right.slope * years[right] + reg_right.intercept

ax.plot(
    years[left],
    fit_left,
    color="red",
    linewidth=2,
    label=f"Before {break_year}: {reg_left.slope:.2f} mm yr⁻¹"
)

ax.plot(
    years[right],
    fit_right,
    color="blue",
    linewidth=2,
    label=f"After {break_year}: {reg_right.slope:.2f} mm yr⁻¹"
)

ax.axvline(
    break_year,
    color="gray",
    linestyle=":",
    linewidth=1.8
)

ax.set_ylabel("AMJJ precipitation (mm)")
ax.set_xlabel("Year")
ax.set_title("Piecewise trend test for AMJJ precipitation")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    SUPP_DIR / "AMJJ_piecewise_trend_test.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 6. Figura: disponibilidade da rede observacional
# ============================================================

fig, ax1 = plt.subplots(figsize=(11, 4))

n_valid_annual = df_qc.notna().sum(axis=1).groupby(df_qc.index.year).mean()

n_valid_annual.plot(
    ax=ax1,
    color="black",
    linewidth=1.6
)

ax1.set_ylabel("Mean number of valid stations")
ax1.set_xlabel("Year")
ax1.set_title("Temporal evolution of rain-gauge availability")
ax1.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(
    SUPP_DIR / "annual_valid_station_count.png",
    dpi=300,
    bbox_inches="tight"
)
plt.close()


# ============================================================
# 7. Exporta séries usadas no suplemento
# ============================================================

supp_series = pd.DataFrame({
    "AMJJ_precipitation_mm": prec_amjj,
    "monthly_precipitation_detrended_mm": prec_monthly_dt,
    "AMJJ_precipitation_monthly_detrended_then_annual_max_mm": prec_amjj_dt
})

supp_series.to_csv(
    SUPP_DIR / "AMJJ_precipitation_original_monthly_detrended.csv",
    na_rep="-999"
)


# ============================================================
# 8. Relatório final
# ============================================================

print("\nSupplementary analyses completed.")
print(f"Monthly trend removed first: {reg_monthly.slope:.3f} mm yr⁻¹ | p = {reg_monthly.pvalue:.3f}")
print(f"Raw annual AMJJ-max trend: {reg_full.slope:.3f} mm yr⁻¹ | p = {reg_full.pvalue:.3f}")
print(f"Estimated break year: {break_year}")
print(
    f"Before {break_year}: {reg_left.slope:.3f} mm yr⁻¹ | p = {reg_left.pvalue:.3f}"
)
print(
    f"After {break_year}: {reg_right.slope:.3f} mm yr⁻¹ | p = {reg_right.pvalue:.3f}"
)
print("\nNon-parametric trend tests:")
print(trend_tests_summary)
print(f"Supplementary files saved in: {SUPP_DIR}")