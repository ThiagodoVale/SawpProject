
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 23:59:50 2025
Script para salvar as variáveis originais (COM tendência) em formato .npz
com a mesma estrutura usada para as variáveis detrendadas.

@author: simopec
"""

import os
import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, ".", "data")
OUT_DIR = os.path.join(BASE_DIR, "outputs_trended")

os.makedirs(OUT_DIR, exist_ok=True)

FILE_UVQ = os.path.join(DATA_DIR, "Era5_Dataset_u_v_w.nc")
FILE_UVQ_LL = os.path.join(DATA_DIR, "Era5_Dataset_t_q.nc")
FILE_SST = os.path.join(DATA_DIR, "Era5_Dataset_sst.nc")
FILE_LH = os.path.join(DATA_DIR, "Era5_Dataset_lh.nc")
FILE_EVP = os.path.join(DATA_DIR, "Era5_Dataset_evp.nc")

# =======================================================================================
# FUNÇÃO PARA SALVAR OS DADOS COM TENDÊNCIA
# =======================================================================================
def save_trended_4d(ARQUIVO, variaveis):

    ds = xr.open_dataset(ARQUIVO)

    for varname in variaveis:
        print(f"Salvando variável COM tendência: {varname}")

        da = ds[varname]

        # ------------------------------------------------------------------------------
        # Detecta automaticamente dimensão temporal
        # ------------------------------------------------------------------------------
        if "time" in da.dims:
            time_dim = "time"
            time_values = da.time.values
            print("Escolhendo: time")
        elif "valid_time" in da.dims:
            time_dim = "valid_time"
            time_values = da.valid_time.values
            print("Escolhendo: valid_time")
        else:
            raise ValueError(
                "Nenhuma dimensão temporal encontrada. "
                "Esperado 'time' ou 'valid_time'. "
                f"Dimensões encontradas: {da.dims}"
            )

        # Garante datetime64[s]
        if np.issubdtype(time_values.dtype, np.datetime64):
            time_values = time_values.astype("datetime64[s]")

        # ------------------------------------------------------------------------------
        # Detecta automaticamente o nome da coordenada vertical
        # ------------------------------------------------------------------------------
        if "level" in da.coords:
            level_values = da.level.values
        elif "pressure_level" in da.coords:
            level_values = da.pressure_level.values
        else:
            level_values = None  # campos 2D ou superficiais

        # ------------------------------------------------------------------------------
        # SALVAR EM NPZ
        # ------------------------------------------------------------------------------
        out_file = os.path.join(OUT_DIR, f"trended_uvq_{varname}.npz")

        np.savez(
            out_file,
            data=da.values,             # sem detrend
            level=level_values,
            lat=da.latitude.values,
            lon=da.longitude.values,
            **{time_dim: time_values}
        )

        print(f"→ Arquivo salvo: {out_file}\n")

def save_trended_3d(ARQUIVO, variaveis):
    """
    Salva variáveis de um arquivo NetCDF em formato .npz, preservando a tendência.
    Para variáveis 3D (tempo, lat, lon).

    Parâmetros
    ----------
    ARQUIVO : str
        Caminho do arquivo NetCDF.
    variaveis : list
        Lista com nomes das variáveis a serem exportadas.
    """

    ds = xr.open_dataset(ARQUIVO)

    for varname in variaveis:
        print(f"Salvando variável 3D COM tendência: {varname}")

        da = ds[varname]

        # ------------------------------------------------------------------------------
        # Detecta automaticamente a dimensão temporal
        # ------------------------------------------------------------------------------
        if "time" in da.dims:
            time_dim = "time"
            time_values = da.time.values
            print("→ Dimensão temporal detectada: time")
        elif "valid_time" in da.dims:
            time_dim = "valid_time"
            time_values = da.valid_time.values
            print("→ Dimensão temporal detectada: valid_time")
        else:
            raise ValueError(
                "Nenhuma dimensão temporal encontrada. "
                "Esperado 'time' ou 'valid_time'. "
                f"Dimensões encontradas: {da.dims}"
            )

        # Garante datetime64[s]
        if np.issubdtype(time_values.dtype, np.datetime64):
            time_values = time_values.astype("datetime64[s]")

        # ------------------------------------------------------------------------------
        # Verificação: variável precisa ser 3D
        # ------------------------------------------------------------------------------
        if da.ndim != 3:
            raise ValueError(
                f"A variável {varname} NÃO é 3D. "
                f"Dimensões encontradas: {da.dims} "
                "Use save_trended_4d para variáveis com níveis verticais."
            )

        # ------------------------------------------------------------------------------
        # SALVAR EM NPZ
        # ------------------------------------------------------------------------------
        out_file = os.path.join(OUT_DIR, f"trended_uvq_{varname}.npz")

        np.savez(
            out_file,
            data=da.values,             # dados COM tendência
            lat=da.latitude.values,
            lon=da.longitude.values,
            **{time_dim: time_values}
        )

        print(f"→ Arquivo salvo: {out_file}\n")


# =======================================================================================
# EXECUÇÃO
# =======================================================================================

#save_trended_4d(FILE_UVQ, ['u', 'v', 'w'])
#save_trended_4d(FILE_UVQ_LL, ['t', 'q'])
#save_trended_3d(FILE_SST, ['sst'])
save_trended_3d(FILE_LH, ['slhf'])
save_trended_3d(FILE_EVP, ['e'])

