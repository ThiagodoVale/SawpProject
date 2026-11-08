# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 17:21:17 2026

@author: silva
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree
from scipy.stats import pearsonr


def gerar_serie_chuva_mensal_acumulada(
    data_dir="./data/monitor",
    states=("AL", "PB", "PE", "RN", "CE", "BA", "SE", "PI", "MA"),
    date_start="1959-01-31",
    date_end="2023-12-31",
    lon_min=-38.0,
    lon_max=-34.5,
    lat_min=-15.0,
    lat_max=0.0,
    abs_monthly_max=2000.0,
    regional_factor=5.0,
    regional_min_threshold=500.0,
    min_valid_stations=30,
    min_valid_fraction_dynamic=0.10,
    remove_unusual_peak_month=True,
    n_neighbors=10,
    min_common_months=60,
):
    """
    Gera a série mensal acumulada regional de chuva.

    Retorna:
        regional_output : DataFrame com:
            - precipitation_mm
            - n_valid_stations
            - n_total_stations
            - coverage_fraction
            - required_stations

        df_qc : matriz mensal estação x tempo após QC
        metadata : metadados das estações
        removed_df : valores removidos no QC
        regime_test : teste de homogeneidade pluviométrica
    """

    data_dir = Path(data_dir)
    date_index = pd.date_range(date_start, date_end, freq="ME")

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

            return data, code, lon, lat

        except Exception as e:
            print(f"Erro ao ler {filename}: {e}")
            return None, None, None, None

    def monthly_basic_qc(df):
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

        mask_abs = df_qc > abs_monthly_max
        for i, j in zip(*np.where(mask_abs.values)):
            removed.append({
                "date": df_qc.index[i],
                "station": df_qc.columns[j],
                "value": df_qc.iloc[i, j],
                "reason": "above_absolute_monthly_limit"
            })

        df_qc = df_qc.mask(mask_abs)

        return df_qc, removed

    def regional_outlier_qc(df):
        removed = []
        df_qc = df.copy()

        regional_median = df_qc.median(axis=1, skipna=True)
        mask = pd.DataFrame(False, index=df_qc.index, columns=df_qc.columns)

        for date in df_qc.index:
            med = regional_median.loc[date]

            if pd.isna(med) or med <= 0:
                continue

            mask.loc[date] = (
                (df_qc.loc[date] > regional_min_threshold) &
                (df_qc.loc[date] > regional_factor * med)
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

    # ======================================================
    # 1) LEITURA DAS ESTAÇÕES
    # ======================================================
    series = []
    metadata = []

    for state in states:
        files = sorted((data_dir / state).glob("*"))
        print(f"{state}: {len(files)} arquivos encontrados")

        for file in files:
            df_station, code, lon, lat = read_monitor_file(file)

            if df_station is None:
                continue

            if not (lon_min <= lon <= lon_max and lat_min <= lat <= lat_max):
                continue

            df_station = df_station.reindex(date_index)
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

    df_raw = pd.concat(series, axis=1)
    metadata = pd.DataFrame(metadata)

    print(f"Estações dentro da área: {df_raw.shape[1]}")

    # ======================================================
    # 2) CONTROLE DE QUALIDADE
    # ======================================================
    removed_all = []

    df_qc, removed = monthly_basic_qc(df_raw)
    removed_all.extend(removed)

    df_qc, removed = regional_outlier_qc(df_qc)
    removed_all.extend(removed)

    removed_df = pd.DataFrame(removed_all)

    # Remove estações totalmente vazias após QC
    station_valid_fraction = df_qc.notna().mean(axis=0)

    metadata["valid_fraction_after_qc"] = metadata["code"].astype(str).map(
        station_valid_fraction
    )

    metadata["used_after_qc"] = metadata["valid_fraction_after_qc"] > 0

    df_qc = df_qc.loc[:, df_qc.notna().any(axis=0)]

    # ======================================================
    # 3) TESTE DE HOMOGENEIDADE DO REGIME
    # ======================================================
    meta = metadata.copy()
    meta["code"] = meta["code"].astype(str)

    meta = meta[meta["code"].isin(df_qc.columns.astype(str))].copy()
    meta = meta.set_index("code").loc[df_qc.columns.astype(str)]

    coords = meta[["lat", "lon"]].values
    tree = cKDTree(coords)

    results = []

    for station in df_qc.columns.astype(str):

        dist, idx = tree.query(
            meta.loc[station, ["lat", "lon"]].values,
            k=n_neighbors + 1
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

        if len(dados) < min_common_months:
            r = np.nan
            p = np.nan
        else:
            r, p = pearsonr(dados["station"], dados["neighbors"])

        annual_mean = (
            serie_station
            .groupby(serie_station.index.year)
            .sum(min_count=1)
            .mean()
        )

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

    regime_test["flag_unusual_peak_month"] = ~regime_test["peak_month"].isin(
        [4, 5, 6, 7]
    )

    regime_test["flag_potentially_inconsistent"] = (
        regime_test["flag_annual_extreme"] |
        regime_test["flag_low_neighbor_corr"] |
        regime_test["flag_unusual_peak_month"]
    )

    if remove_unusual_peak_month:
        stations_remove = regime_test.loc[
            regime_test["flag_unusual_peak_month"],
            "station"
        ].astype(str)

        print(f"Estações removidas por homogeneidade: {len(stations_remove)}")

        df_qc = df_qc.drop(
            columns=stations_remove,
            errors="ignore"
        )

    print(f"Estações restantes: {df_qc.shape[1]}")

    # ======================================================
    # 4) SÉRIE REGIONAL MENSAL ACUMULADA
    # ======================================================
    n_valid = df_qc.notna().sum(axis=1)
    n_total = df_qc.shape[1]

    required_stations = max(
        min_valid_stations,
        int(np.ceil(min_valid_fraction_dynamic * n_total))
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

    print("\nSérie mensal acumulada concluída.")
    print(f"Estações finais usadas: {df_qc.shape[1]}")
    print(f"Número mínimo de estações por mês: {required_stations}")
    print(f"Meses válidos: {regional_output['precipitation_mm'].notna().sum()}")

    return regional_output, df_qc, metadata, removed_df, regime_test