#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 21:28:37 2025

@author: simopec
"""

import numpy as np

def alinhar_niveis_generico(variaveis, niveis, ref_index=0):
    """
    Ajusta automaticamente o eixo de níveis para N variáveis 4D.
    
    Parameters
    ----------
    variaveis : list of np.ndarray
        Lista com arrays 4D sobre tempo,nivel,lat,lon.
    niveis : list of np.ndarray
        Lista com arrays 1D contendo os níveis de pressão.
    ref_index : int
        Índice da variável que será usada como referência.

    Returns
    -------
    variaveis2 : list
        Lista das variáveis ajustadas.
    niveis2 : list
        Lista dos níveis ajustados.
    """

    # Segurança básica
    if len(variaveis) != len(niveis):
        raise ValueError("variaveis e niveis devem ter o mesmo comprimento.")

    # Nível da variável de referência
    lev_ref = niveis[ref_index]

    # Ordem da referência
    ref_crescente = lev_ref[0] < lev_ref[-1]

    variaveis2 = []
    niveis2 = []

    for var, lev in zip(variaveis, niveis):

        if var is None or lev is None:
            variaveis2.append(var)
            niveis2.append(lev)
            continue

        var_crescente = lev[0] < lev[-1]

        # Se a ordem for diferente da referência → inverter
        if var_crescente != ref_crescente:
            var = np.flip(var, axis=1)
            lev = lev[::-1]

        variaveis2.append(var)
        niveis2.append(lev)

    return variaveis2, niveis2

import numpy as np

def alinhar_niveis(variaveis, niveis, ordem='decrescente'):
    """
    Garante que todas as variáveis 4D estejam com níveis na ordem desejada.

    Parameters
    ----------
    variaveis : list of np.ndarray
        Lista de arrays 4D (tempo, nivel, lat, lon).
    niveis : list of np.ndarray
        Lista de arrays 1D contendo os níveis verticais.
    ordem : str, optional
        'decrescente' (ex: 1000 → 100 hPa) ou
        'crescente'   (ex: 100 → 1000 hPa).
        Default é 'decrescente'.

    Returns
    -------
    variaveis2 : list
        Lista das variáveis ajustadas.
    niveis2 : list
        Lista dos níveis ajustados.
    """

    if len(variaveis) != len(niveis):
        raise ValueError("variaveis e niveis devem ter o mesmo comprimento.")

    if ordem not in ('crescente', 'decrescente'):
        raise ValueError("ordem deve ser 'crescente' ou 'decrescente'.")

    variaveis2 = []
    niveis2 = []

    for var, lev in zip(variaveis, niveis):

        if var is None or lev is None:
            variaveis2.append(var)
            niveis2.append(lev)
            continue

        # Verifica se já está na ordem correta
        if ordem == 'decrescente' and lev[0] > lev[-1]:
            variaveis2.append(var)
            niveis2.append(lev)
            continue

        if ordem == 'crescente' and lev[0] < lev[-1]:
            variaveis2.append(var)
            niveis2.append(lev)
            continue

        # Caso contrário, inverte eixo de nível
        var = np.flip(var, axis=1)
        lev = lev[::-1]

        variaveis2.append(var)
        niveis2.append(lev)

    return variaveis2, niveis2

