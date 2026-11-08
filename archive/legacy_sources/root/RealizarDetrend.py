#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 21:20:58 2025

@author: simopec
"""

import os
import numpy as np
import xarray as xr
from sawp.DetrendERA5 import detrend_3d, detrend_4d, detrend_4d_gpu


# ---------------------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, ".", "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs")

# Cria outputs/ se não existir
os.makedirs(OUT_DIR, exist_ok=True)

FILE_UVQ = os.path.join(DATA_DIR, "Era5_Dataset_u_v_w.nc")        # 4D
FILE_UVQ_LL = os.path.join(DATA_DIR, "Era5_Dataset_t_q.nc")      # 4D
FILE_SST = os.path.join(DATA_DIR, "Era5_Dataset_sst.nc")        # 3D
FILE_LH = os.path.join(DATA_DIR, "Era5_Dataset_lhsh.nc")        # 3D
FILE_EVP = os.path.join(DATA_DIR, "Era5_Dataset_evp.nc")        # 3D


# =======================================================================================
# REMOÇÃO ARQUIVO 4D
# =======================================================================================
def test_detrend_4d(ARQUIVO,variaveis):

    ds = xr.open_dataset(ARQUIVO)
    

    for varname in variaveis:
        print(varname)
        da = ds[varname]
        
        # --------- aplica detrend ---------
        da_dt = detrend_4d(da)

        # --------- verificações ---------
        assert da_dt.shape == da.shape, f"shape diferente para {varname}"

        # Detectar automaticamente a dimensão de tempo
        if "time" in da.dims:
            time_dim = "time"
            t = np.arange(da.time.size)
            print("Escolhendo Time")
        elif "valid_time" in da.dims:
            time_dim = "valid_time"
            t = np.arange(da.valid_time.size)
            print("Escolhendo valid_time")
        else:
            raise ValueError(
                "Nenhuma dimensão temporal encontrada. "
                "Esperado 'time' ou 'valid_time'. "
                f"Dimensões encontradas: {da.dims}"
            )
        
        # Usa o nome detectado no mean()
        mean_orig = da.mean(dim=time_dim)
        mean_dt   = da_dt.mean(dim=time_dim)
        diff = float(np.abs((mean_dt - mean_orig)).mean())

        assert diff < 1e-3, f"Climatologia muito alterada em {varname} (diff={diff})"
        

       
        # Detecta o nome correto da coordenada vertical
        if "level" in da.coords:
            level_coord = "level"
            level_values = da.level.values
        elif "pressure_level" in da.coords:
            level_coord = "pressure_level"
            level_values = da.pressure_level.values

        else:
            level_values = None  # campos 2D ou superficiais

            raise ValueError("Nenhuma coordenada vertical encontrada: esperado 'level' ou 'pressure_level'.")
            
        
        # Cálculo dos coeficientes usando a coordenada correta
        coef_orig = np.polyfit(t, da.mean((level_coord, "latitude", "longitude")), 1)[0]
        coef_dt   = np.polyfit(t, da_dt.mean((level_coord, "latitude", "longitude")), 1)[0]

        # Extrair vetor de tempo dinamicamente
        time_values = da[time_dim].values
        
        # Converter para datetime64, se necessário
        if np.issubdtype(time_values.dtype, np.datetime64):
            time_values = time_values.astype("datetime64[s]")

        assert abs(coef_dt) < abs(coef_orig), f"Tendência não removida em {varname}"

        # --------- SALVA EM .NPZ ---------
        out_file = os.path.join(OUT_DIR, f"detrended_uvq_{varname}.npz")

        np.savez(
            out_file,
            data=da_dt.values,
            level=level_values,
            lat=da.latitude.values,
            lon=da.longitude.values,
            **{time_dim: time_values}
        )

# =======================================================================================
# TESTE DO ARQUIVO 3D
# =======================================================================================
def test_detrend_3d(FILE_SST,variavel):

    ds = xr.open_dataset(FILE_SST)
    varname = list(ds.data_vars)[0]
    da = ds[varname]

    # --------- aplica detrend ---------
    da_dt = detrend_3d(da)

    # --------- verificações ---------
    assert da_dt.shape == da.shape

    # Detectar automaticamente a dimensão de tempo
    if "time" in da.dims:
        time_dim = "time"
        t = np.arange(da.time.size)
        print("Escolhendo time")

    elif "valid_time" in da.dims:
        time_dim = "valid_time"
        t = np.arange(da.valid_time.size)
        print("Escolhendo valid_time")

    else:
        raise ValueError(
            "Nenhuma dimensão temporal encontrada. "
            "Esperado 'time' ou 'valid_time'. "
            f"Dimensões encontradas: {da.dims}"
        )

    # --------- compara climatologia ---------
    mean_orig = da.mean(dim=time_dim)
    mean_dt = da_dt.mean(dim=time_dim)

    diff = float(np.abs((mean_dt - mean_orig)).mean())

    #assert diff < 1e-3, f"Climatologia muito alterada (diff={diff})"

    # --------- tendência antes e depois ---------
    coef_orig = np.polyfit(
        t,
        da.mean(("latitude", "longitude")),
        1
    )[0]

    coef_dt = np.polyfit(
        t,
        da_dt.mean(("latitude", "longitude")),
        1
    )[0]

    assert abs(coef_dt) < abs(coef_orig), "Tendência não reduzida"

    # --------- extrai tempo dinamicamente ---------
    time_values = da[time_dim].values

    if np.issubdtype(time_values.dtype, np.datetime64):
        time_values = time_values.astype("datetime64[s]")

    # --------- SALVA EM .NPZ ---------
    out_file = os.path.join(OUT_DIR, f"detrended_{variavel}.npz")

    np.savez(
        out_file,
        data=da_dt.values,
        lat=da.latitude.values,
        lon=da.longitude.values,
        **{time_dim: time_values}
    )

    print(f"Arquivo salvo em: {out_file}")

#test_detrend_4d(FILE_UVQ,['u','v','w'])
#test_detrend_4d(FILE_UVQ_LL,['t','q'])
#test_detrend_3d(FILE_SST,'sst')
test_detrend_3d(FILE_LH, 'lh')
test_detrend_3d(FILE_LH, 'sh')
#test_detrend_3d(FILE_LH, 'evp')
