#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuração de tempo e área
"""

import netCDF4 as nc
from datetime import timedelta, datetime
import numpy as np

def ConfigurarTempoArea(caminho_arquivo):
    """
    Extrai dados de SST por hora e coordenadas de latitude e longitude.
    
    Retorna:
        sst: array (time, lat, lon)
        tempo: lista de datetime
        lon, lat: meshgrid das coordenadas
    """
    # Abrindo arquivo NetCDF
    ds = nc.Dataset(caminho_arquivo)
    
    # Extraindo variável SST (assumindo primeira variável disponível)
    sst = ds.variables['sst'][:]  # ajustar se o nome da variável for diferente
    
    # Extraindo vetor de tempo e convertendo para datetime
    vetor_tempo = ds.variables['time'][:]
    tempo_inicial = datetime(1900, 1, 1)
    tempo = [tempo_inicial + timedelta(hours=int(t)) for t in vetor_tempo]
    
    # Extraindo coordenadas
    lat_ = ds.variables['latitude'][:]
    lon_ = ds.variables['longitude'][:]
    lon, lat = np.meshgrid(lon_, lat_)
    
    return tempo, lon, lat