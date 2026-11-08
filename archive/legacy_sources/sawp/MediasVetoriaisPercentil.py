#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Módulo: medias_era5
Funções para cálculo de médias de mapas ERA5 (2D e 3D).

@author: simopec
"""

import numpy as np

def MatrizMedia3dEra5(percentil, variavel):
    """
    Calcula a média tridimensional (lev, lat, lon) para os índices fornecidos.

    Parâmetros:
        percentil : list ou np.ndarray
            Lista de índices ou percentis para calcular a média
        variavel : np.ndarray
            Array 4D (time, lev, lat, lon) da variável ERA5

    Retorna:
        np.ndarray 3D (lev, lat, lon) com a média sobre os índices selecionados
    """
    for i, indice in enumerate(percentil[::4]):
        if i == 0:
            var_media = variavel[int(indice):int(indice)+4, :, :, :]
        else:
            var_concatenar = variavel[int(indice):int(indice)+4, :, :, :]
            var_media = np.concatenate((var_media, var_concatenar), axis=0)
    return np.mean(var_media, axis=0)


def MatrizMedia2dEra5(percentil, variavel):
    """
    Calcula a média bidimensional (lat, lon) para os índices fornecidos.

    Parâmetros:
        percentil : list ou np.ndarray
            Lista de índices ou percentis para calcular a média
        variavel : np.ndarray
            Array 3D (time, lat, lon) da variável ERA5

    Retorna:
        np.ndarray 2D (lat, lon) com a média sobre os índices selecionados
    """
    for i, indice in enumerate(percentil[::4]):
        print(f'estou fazendo a média nos índices {i} do {indice}')
        if i == 0:
            var_media = variavel[int(indice):int(indice)+4, :, :]
            print(f'Estou fazendo agora  variavel[{int(indice)}:{int(indice)+4}, :, :]')
        else:
            var_concatenar = variavel[int(indice):int(indice)+4, :, :]
            print(f'Estou fazendo agora  variavel[{int(indice)}:{int(indice)+4}, :, :]')
            var_media = np.concatenate((var_media, var_concatenar), axis=0)
    return np.mean(var_media, axis=0)
