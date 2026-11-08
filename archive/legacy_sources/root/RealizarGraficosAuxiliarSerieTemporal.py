# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 07:13:01 2026

@author: silva
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Série temporal espacial média no NEB para variável ERA5
comparando dados com tendência e sem tendência.

Exemplo:
    VARIAVEL = "q"
    AREA_NEB = [-50, -34, -15, 0]
"""

import os
import numpy as np
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURAÇÕES
# ============================================================

BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
FIG_DIR = os.path.join(BASE_DIR, "figuras")

os.makedirs(FIG_DIR, exist_ok=True)

VARIAVEL = "slhf"   # opções: "u", "v", "w", "t", "q", "sst"

AREA_NEB = {
    "lon_min": -34,
    "lon_max": -25,
    "lat_min": -12.5,
    "lat_max": -2.5,
}

# Para variáveis 4D, escolha níveis.
# Use None para média em todos os níveis.
# Exemplo: [1000, 925, 850] se o arquivo tiver níveis físicos.
NIVEIS_ESCOLHIDOS = None


# ============================================================
# CAMINHOS
# ============================================================

ARQUIVOS_ORIGINAIS = {
    "u": os.path.join(DATA_DIR, "Era5_Dataset_u_v_w.nc"),
    "v": os.path.join(DATA_DIR, "Era5_Dataset_u_v_w.nc"),
    "w": os.path.join(DATA_DIR, "Era5_Dataset_u_v_w.nc"),
    "t": os.path.join(DATA_DIR, "Era5_Dataset_t_q.nc"),
    "q": os.path.join(DATA_DIR, "Era5_Dataset_t_q.nc"),
    "sst": os.path.join(DATA_DIR, "Era5_Dataset_sst.nc"),
    "slhf": os.path.join(DATA_DIR, "Era5_Dataset_lh.nc"),
}

ARQUIVOS_DETRENDED = {
    "u": os.path.join(OUT_DIR, "detrended_uvq_u.npz"),
    "v": os.path.join(OUT_DIR, "detrended_uvq_v.npz"),
    "w": os.path.join(OUT_DIR, "detrended_uvq_w.npz"),
    "t": os.path.join(OUT_DIR, "detrended_uvq_t.npz"),
    "q": os.path.join(OUT_DIR, "detrended_uvq_q.npz"),
    "sst": os.path.join(OUT_DIR, "detrended_sst.npz"),
    "slhf": os.path.join(OUT_DIR, "detrended_uvq_slhf.npz"),
}


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def detectar_dim_tempo(da):
    if "time" in da.dims:
        return "time"
    if "valid_time" in da.dims:
        return "valid_time"
    raise ValueError(f"Nenhuma dimensão temporal encontrada em {da.dims}")


def detectar_dim_vertical(da):
    if "level" in da.dims:
        return "level"
    if "pressure_level" in da.dims:
        return "pressure_level"
    return None


def recortar_area_xarray(da, area):
    lat_name = "latitude"
    lon_name = "longitude"

    lat = da[lat_name]
    lon = da[lon_name]

    if lat[0] > lat[-1]:
        da = da.sel(
            latitude=slice(area["lat_max"], area["lat_min"]),
            longitude=slice(area["lon_min"], area["lon_max"])
        )
    else:
        da = da.sel(
            latitude=slice(area["lat_min"], area["lat_max"]),
            longitude=slice(area["lon_min"], area["lon_max"])
        )

    return da


def selecionar_niveis_xarray(da, niveis):
    dim_vertical = detectar_dim_vertical(da)

    if dim_vertical is None or niveis is None:
        return da

    niveis_disponiveis = da[dim_vertical].values

    idx = []
    for n in niveis:
        i = np.argmin(np.abs(niveis_disponiveis - n))
        idx.append(i)

    return da.isel({dim_vertical: idx})


def serie_original_netcdf(varname, area, niveis=None):
    arquivo = ARQUIVOS_ORIGINAIS[varname]

    ds = xr.open_dataset(arquivo)
    da = ds[varname]

    time_dim = detectar_dim_tempo(da)
    dim_vertical = detectar_dim_vertical(da)

    da = recortar_area_xarray(da, area)
    da = selecionar_niveis_xarray(da, niveis)

    dims_media = ["latitude", "longitude"]

    if dim_vertical is not None:
        dims_media.append(dim_vertical)

    serie = da.mean(dim=dims_media, skipna=True).to_pandas()
    serie.name = f"{varname}_com_tendencia"

    return serie


def serie_detrended_npz(varname, area, niveis=None):
    arquivo = ARQUIVOS_DETRENDED[varname]

    dados = np.load(arquivo, mmap_mode="r", allow_pickle=True)

    data = dados["data"]
    lat = dados["lat"]
    lon = dados["lon"]

    if "time" in dados:
        tempo = pd.to_datetime(dados["time"])
    elif "valid_time" in dados:
        tempo = pd.to_datetime(dados["valid_time"])
    else:
        raise ValueError("Arquivo .npz sem coordenada temporal.")

    # -----------------------------
    # Máscara espacial NEB
    # -----------------------------
    idx_lat = np.where(
        (lat >= area["lat_min"]) & (lat <= area["lat_max"])
    )[0]

    idx_lon = np.where(
        (lon >= area["lon_min"]) & (lon <= area["lon_max"])
    )[0]

    if len(idx_lat) == 0 or len(idx_lon) == 0:
        raise ValueError("Recorte espacial vazio. Confira lat/lon.")

    # -----------------------------
    # Caso 3D: time, lat, lon
    # -----------------------------
    if data.ndim == 3:
        recorte = data[:, idx_lat.min():idx_lat.max()+1, idx_lon.min():idx_lon.max()+1]
        serie = np.nanmean(recorte, axis=(1, 2))

    # -----------------------------
    # Caso 4D: time, level, lat, lon
    # -----------------------------
    elif data.ndim == 4:
        level = dados["level"]

        if niveis is not None:
            idx_lev = []
            for n in niveis:
                i = np.argmin(np.abs(level - n))
                idx_lev.append(i)
            idx_lev = np.array(idx_lev)
        else:
            idx_lev = np.arange(len(level))

        recorte = data[
            :,
            idx_lev,
            idx_lat.min():idx_lat.max()+1,
            idx_lon.min():idx_lon.max()+1
        ]

        serie = np.nanmean(recorte, axis=(1, 2, 3))

    else:
        raise ValueError(f"Dimensão inesperada: {data.shape}")

    serie = pd.Series(serie, index=tempo, name=f"{varname}_sem_tendencia")

    return serie


def salvar_e_plotar(serie_ct, serie_st, varname):
    df = pd.concat([serie_ct, serie_st], axis=1)

    out_csv = os.path.join(FIG_DIR, f"serie_NEB_{varname}_com_sem_tendencia.csv")
    out_png = os.path.join(FIG_DIR, f"serie_NEB_{varname}_com_sem_tendencia.png")

    df.to_csv(out_csv)

    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df.iloc[:, 0], label="Com tendência", linewidth=1.8)
    plt.plot(df.index, df.iloc[:, 1], label="Sem tendência", linewidth=1.8)

    plt.title(f"Série temporal média no NEB - {varname}")
    plt.xlabel("Tempo")
    plt.ylabel(varname)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.show()
    plt.close()

    print("Arquivos salvos:")
    print(out_csv)
    print(out_png)

    return df

def normalizar_indice_mensal(serie):
    serie = serie.copy()
    serie.index = pd.to_datetime(serie.index).to_period("M").to_timestamp()
    return serie


# ============================================================
# EXECUÇÃO
# ============================================================

if __name__ == "__main__":

    # -----------------------------
    # Séries SLHF
    # -----------------------------
    print("Processando SLHF...")

    slhf_ct = serie_original_netcdf(
        "slhf",
        AREA_NEB,
        niveis=None
    )

    slhf_st = serie_detrended_npz(
        "slhf",
        AREA_NEB,
        niveis=None
    )

    # -----------------------------
    # Séries SST
    # -----------------------------
    print("Processando SST...")

    sst_ct = serie_original_netcdf(
        "sst",
        AREA_NEB,
        niveis=None
    )

    sst_st = serie_detrended_npz(
        "sst",
        AREA_NEB,
        niveis=None
    )

    slhf_ct = normalizar_indice_mensal(slhf_ct)
    slhf_st = normalizar_indice_mensal(slhf_st)
    sst_ct  = normalizar_indice_mensal(sst_ct)
    sst_st  = normalizar_indice_mensal(sst_st)

    # -----------------------------
    # DataFrames alinhados
    # -----------------------------
    df = pd.concat(
        [
            slhf_ct.rename("slhf_com_tendencia"),
            slhf_st.rename("slhf_sem_tendencia"),
            sst_ct.rename("sst_com_tendencia"),
            sst_st.rename("sst_sem_tendencia"),
        ],
        axis=1
    )
    
    print("\nDiagnóstico antes do dropna:")
    print(df.count())
    print(df.isna().sum())
    print(df.head())
    print(df.tail())
    
    df = df.dropna()
    
    print("\nApós dropna:")
    print(df.shape)
    print(df.count())
    
    if df.empty:
        raise RuntimeError(
            "O DataFrame ficou vazio após o alinhamento. "
            "Provavelmente SLHF e SST estão com datas diferentes ou uma das séries está toda NaN."
        )
    # -----------------------------
    # Correlações
    # -----------------------------
    corr_ct = df["slhf_com_tendencia"].corr(df["sst_com_tendencia"])
    corr_st = df["slhf_sem_tendencia"].corr(df["sst_sem_tendencia"])

    print(f"Correlação SLHF x SST com tendência: {corr_ct:.3f}")
    print(f"Correlação SLHF x SST sem tendência: {corr_st:.3f}")

    # -----------------------------
    # Salvar CSV
    # -----------------------------
    out_csv = os.path.join(FIG_DIR, "serie_NEB_slhf_sst_com_sem_tendencia.csv")
    df.to_csv(out_csv)

    # -----------------------------
    # Figura SLHF x SST
    # -----------------------------
    out_png = os.path.join(FIG_DIR, "scatter_NEB_slhf_x_sst.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    axes[0].scatter(
        df["sst_com_tendencia"],
        df["slhf_com_tendencia"],
        s=45,
        alpha=0.8
    )
    axes[0].set_title(f"Com tendência\nr = {corr_ct:.2f}", fontsize=14)
    axes[0].set_xlabel("SST")
    axes[0].set_ylabel("SLHF")
    axes[0].grid(alpha=0.3)

    axes[1].scatter(
        df["sst_sem_tendencia"],
        df["slhf_sem_tendencia"],
        s=45,
        alpha=0.8
    )
    axes[1].set_title(f"Sem tendência\nr = {corr_st:.2f}", fontsize=14)
    axes[1].set_xlabel("SST")
    axes[1].set_ylabel("SLHF")
    axes[1].grid(alpha=0.3)

    plt.suptitle("Relação entre SLHF e SST média no NEB", fontsize=16)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.show()
    plt.close()

    print("Arquivos salvos:")
    print(out_csv)
    print(out_png)
    
    df["slhf_lag0"] = df["slhf_com_tendencia"].shift(0)
    df["slhf_lag1"] = df["slhf_com_tendencia"].shift(1)
    df["slhf_lag2"] = df["slhf_com_tendencia"].shift(2)
    df["slhf_lag3"] = df["slhf_com_tendencia"].shift(3)

    corr_lag0 = df["slhf_lag0"].corr(df["sst_com_tendencia"])    
    corr_lag1 = df["slhf_lag1"].corr(df["sst_com_tendencia"])
    corr_lag2 = df["slhf_lag2"].corr(df["sst_com_tendencia"])
    corr_lag3 = df["slhf_lag3"].corr(df["sst_com_tendencia"])
    
    print(corr_lag0)
    print(corr_lag1)
    print(corr_lag2)
    print(corr_lag3)