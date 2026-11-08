#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Função genérica para carregar dados de SST ou TSM,
tanto de arquivos NetCDF (com tendência) quanto NPZ (sem tendência).
"""

import netCDF4 as nc
import numpy as np

def CarregarDados(caminho, tendencias='sim', nome_var_nc='sst', substituir_nan=-999):
    """
    Carrega dados de SST/TSM de arquivo NetCDF ou NPZ.
    
    Parâmetros:
        caminho : str
            Caminho do arquivo (.nc ou .npz)
        tendencias : str
            'sim' para NetCDF com tendência
            'não' para NPZ sem tendência
        nome_var_nc : str
            Nome da variável dentro do NetCDF (default: 'sst')
        substituir_nan : float
            Valor que substituirá NaNs ou valores inválidos (default: -999)
            
    Retorna:
        dados : np.ndarray
            Array com os dados carregados
    """
    # Carregar NPZ sem tendência
    loaded_data = np.load(caminho)
    # Assume que o array relevante está na chave 'detrended'
    dados = loaded_data['data']
    # Substitui NaNs ou valores inválidos se necessário
    dados = np.nan_to_num(dados, nan=substituir_nan)

# =============================================================================
#     if tendencias.lower() == 'não':
#         # Carregar NPZ sem tendência
#         loaded_data = np.load(caminho)
#         # Assume que o array relevante está na chave 'detrended'
#         dados = loaded_data['data']
#         # Substitui NaNs ou valores inválidos se necessário
#         dados = np.nan_to_num(dados, nan=substituir_nan)
#     else:
#         # Carregar NetCDF com tendência
#         ds = nc.Dataset(caminho)
#         dados = ds.variables[nome_var_nc][:]  # ajusta o nome da variável se necessário
#         # Substitui NaNs se houver
#         dados = np.nan_to_num(dados, nan=substituir_nan)
#     
# =============================================================================
    return dados

# Exemplo de uso:
# tsm = carregar_dados('./Dados_sem_tendencia/tsm_medios_niveis.npz', tendencias='não')
# sst = carregar_dados('./data/Era5_Dataset_sst.nc', tendencias='sim')
