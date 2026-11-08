#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 20:03:52 2025

@author: simopec
"""

import netCDF4 as nc
import numpy as np
import cupy as cp

# Calculo do ponto de orvalho.
def dewpoint(spum,temp,pres,lati,long):
    pressao_pacial = spum / (1 - spum)
    pressao_saturacao = 6.112 * np.exp((17.67*(temp-273.15))/(temp-29.65))
    razao_de_mistura = np.zeros([len(pres),lati.shape[0],long.shape[0]])
    
    for p in range(len(pres)):
        razao_de_mistura[p,] = 0.622 * pressao_saturacao[p,]/(pres[p]-pressao_saturacao[p,])
        
    umidade_relativa = pressao_pacial / razao_de_mistura  
    pressao_vapor = umidade_relativa * pressao_saturacao 
    #print(pressao_vapor)
    
    val = np.log(pressao_vapor / 6.112)
    ponto_orvalho = 243.5 * val / (17.67 - val)
    return ponto_orvalho

# calculo da temperatura potencial equivalente.
def temperatura_potencial_equivalente(temp, ponto_orvalho, pres, lati,long):
    t = temp
    td = ponto_orvalho + 273.15
    
    # e = pressão de vapor de saturação
    e = 6.112 * np.exp((17.67*(td-273.15))/(td-29.65))
    
    # r = razão de mistura de saturação
    r = np.zeros([len(pres),lati.shape[0],long.shape[0]])
    
    for p in range(len(pres)):
        r[p,] = 0.622 * e[p,]/(pres[p]-e[p,])
    
    t_l = 56 + 1. / (1. / (td - 56) + np.log(t / td) / 800.)
    
    
    th_l = np.zeros([len(pres),lati.shape[0],long.shape[0]])
    
    for p in range(len(pres)):
        th_l[p,] = t[p]*(1000/(pres[p]-e[p]))**0.286*(t[p]/t_l[p])**(0.28*r[p])
        
    EPT = th_l * np.exp(r * (1 + (0.448 * r)) * ((3036. / t_l) - 1.78))
    return EPT

def dewpoint_gpu(spum, temp, pres, lati, long):
    """
    Calcula o ponto de orvalho na GPU.

    Parâmetros
    ----------
    spum : cp.ndarray (nlev, nlat, nlon)
        Umidade específica (kg/kg)
    temp : cp.ndarray (nlev, nlat, nlon)
        Temperatura (K)
    pres : cp.ndarray (nlev,)
        Pressão (hPa)
    lati : cp.ndarray (nlat,)
    long : cp.ndarray (nlon,)
    """

    # Pressão parcial do vapor
    pressao_parcial = spum / (1.0 - spum)

    # Pressão de saturação (hPa)
    pressao_saturacao = 6.112 * cp.exp(
        (17.67 * (temp - 273.15)) / (temp - 29.65)
    )

    # Ajuste de dimensões da pressão
    pres_3d = pres[:, None, None]

    # Razão de mistura
    razao_de_mistura = 0.622 * pressao_saturacao / (pres_3d - pressao_saturacao)

    # Umidade relativa
    umidade_relativa = pressao_parcial / razao_de_mistura

    # Pressão de vapor
    pressao_vapor = umidade_relativa * pressao_saturacao

    # Ponto de orvalho (°C)
    val = cp.log(pressao_vapor / 6.112)
    ponto_orvalho = 243.5 * val / (17.67 - val)

    return ponto_orvalho

def temperatura_potencial_equivalente_gpu(temp, ponto_orvalho, pres, lati, long):
    """
    Calcula a Temperatura Potencial Equivalente (EPT) na GPU.

    Parâmetros
    ----------
    temp : cp.ndarray (nlev, nlat, nlon)
        Temperatura do ar (K)
    ponto_orvalho : cp.ndarray (nlev, nlat, nlon)
        Ponto de orvalho (°C)
    pres : cp.ndarray (nlev,)
        Pressão (hPa)
    lati : cp.ndarray (nlat,)
    long : cp.ndarray (nlon,)
    """

    # Temperatura e ponto de orvalho em Kelvin
    t = temp
    td = ponto_orvalho + 273.15

    # Pressão de vapor de saturação (hPa)
    e = 6.112 * cp.exp(
        (17.67 * (td - 273.15)) / (td - 29.65)
    )

    # Pressão em 3D (broadcast)
    pres_3d = pres[:, None, None]

    # Razão de mistura de saturação
    r = 0.622 * e / (pres_3d - e)

    # Temperatura no nível de condensação por levantamento (LCL)
    t_l = 56.0 + 1.0 / (
        (1.0 / (td - 56.0)) + cp.log(t / td) / 800.0
    )

    # Temperatura potencial no LCL
    th_l = (
        t *
        (1000.0 / (pres_3d - e))**0.286 *
        (t / t_l)**(0.28 * r)
    )

    # Temperatura potencial equivalente (EPT)
    EPT = th_l * cp.exp(
        r * (1.0 + 0.448 * r) * ((3036.0 / t_l) - 1.78)
    )

    return EPT
