#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cálculo de teor de umidade e transporte horizontal de vapor d'água.

Equações:
 - Teor integrado:  ∫ q * dp / g
 - Transporte zonal/meridional: ∫ (u*q) * dp / g , ∫ (v*q) * dp / g
 - Divergência horizontal: div(T)

Inputs:
 - u, v, q : arrays numpy (lev, lat, lon)
 - level   : array de níveis (mesma ordem dos dados)
 - lat, lon: arrays 1D
"""

import numpy as np
from metpy.calc import divergence, lat_lon_grid_deltas


def teor_e_transporte(lat, lon, u, v, q, level):
    """
    Calcula teor de umidade, transporte de vapor e divergência do transporte.

    Parameters
    ----------
    lat : array (nlat)
    lon : array (nlon)
    u   : array (nlev, nlat, nlon)
    v   : array (nlev, nlat, nlon)
    q   : array (nlev, nlat, nlon)
    level : array (nlev) com níveis em hPa

    Returns
    -------
    ap    : teor de umidade integrado (nlat, nlon)
    tvau  : transporte zonal integrado (nlat, nlon)
    tvav  : transporte meridional integrado (nlat, nlon)
    divtva: divergência do transporte (nlat, nlon)
    """

    # Segurança: converter level para array 1D
    level = np.asarray(level)

        # Escolher o sinal de ft dependendo da ordem dos níveis
    if level[0] < level[-1]:
        # níveis crescentes (ex: 200, 300, 500, 700, 850, 1000 hPa)
        ft = -1.0 / 9.81
    else:
        # níveis decrescentes (ex: 1000, 850, 700, 500, 300, 200 hPa)
        ft = 1.0 / 9.81

    # Diferença entre níveis (dp)
    dp = (level[:-1] - level[1:]) * 100  # hPa → Pa, shape (nlev-1)

    # ------------------------------------------------------------
    # 1) Médias entre níveis
    # ------------------------------------------------------------
    q_m = 0.5 * (q[:-1] + q[1:])         # (nlev-1, lat, lon)
    uq_m = 0.5 * (u[:-1] * q[:-1] + u[1:] * q[1:])
    vq_m = 0.5 * (v[:-1] * q[:-1] + v[1:] * q[1:])

    # ------------------------------------------------------------
    # 2) Integração vertical (somatório ∑ f * dp * const)
    # ------------------------------------------------------------
    # reshape dp para broadcast: (nlev-1, 1, 1)
    dp3 = dp[:, None, None]

    ap = ft * np.sum(q_m * dp3, axis=0)
    tvau = ft * np.sum(uq_m * dp3, axis=0)
    tvav = ft * np.sum(vq_m * dp3, axis=0)

    # ------------------------------------------------------------
    # 3) Divergência horizontal do transporte
    # ------------------------------------------------------------
    dx, dy = lat_lon_grid_deltas(lon, lat)
    divtva = divergence(tvau, tvav, dx=dx, dy=dy)

    return ap, tvau, tvav, divtva

def teor_e_transporte_temporal(lat, lon, u, v, q, level):
    """
    Calcula teor de umidade e transporte de vapor ao longo do tempo.
    
    Entrada:
      u, v, q : arrays (ntime, nlev, nlat, nlon)
      level   : array (nlev)
      lat, lon: arrays 1D

    Saída:
      ap_t    : (ntime, nlat, nlon)
      tvau_t  : (ntime, nlat, nlon)
      tvav_t  : (ntime, nlat, nlon)
      divtva_t: (ntime, nlat, nlon)
    """

    # Segurança
    level = np.asarray(level)

        # Escolher o sinal de ft dependendo da ordem dos níveis
    if level[0] < level[-1]:
        # níveis crescentes (ex: 200, 300, 500, 700, 850, 1000 hPa)
        ft = -1.0 / 9.81
    else:
        # níveis decrescentes (ex: 1000, 850, 700, 500, 300, 200 hPa)
        ft = 1.0 / 9.81


    # dp somente depende dos níveis
    dp = (level[:-1] - level[1:]) * 100  # (nlev-1)pressure_level = np.load(u_arq, mmap_mode='r')['level'][level]
    dp4 = dp[None, :, None, None]        # reshape para (1, nlev-1, 1, 1)

    # -------------------------------
    # Médias verticais entre níveis
    # -------------------------------
    q_m  = 0.5 * (q[:, :-1] + q[:, 1:])                # (time, lev-1, lat, lon)
    uq_m = 0.5 * (u[:, :-1] * q[:, :-1] + u[:, 1:] * q[:, 1:])
    vq_m = 0.5 * (v[:, :-1] * q[:, :-1] + v[:, 1:] * q[:, 1:])

    # -------------------------------
    # Integração vertical por tempo
    # -------------------------------
    ap_t   = ft * np.sum(q_m  * dp4, axis=1)    # (time, lat, lon)
    tvau_t = ft * np.sum(uq_m * dp4, axis=1)
    tvav_t = ft * np.sum(vq_m * dp4, axis=1)

    # -------------------------------
    # Divergência para cada tempo
    # -------------------------------
    dx, dy = lat_lon_grid_deltas(lon, lat)

    divtva_t = np.empty_like(ap_t)

    for tt in range(ap_t.shape[0]):
        divtva_t[tt] = divergence(tvau_t[tt], tvav_t[tt], dx=dx, dy=dy)

    return ap_t, tvau_t, tvav_t, divtva_t

def teor_e_transporte_temporal_old(lat, lon, u, v, q, level):
    """
    Versão 4D da função original (time, level, lat, lon)
    Mantém exatamente a mesma lógica da função 3D.
    """

    # Dimensões
    t, uz, ulat, ulon = u.shape
    _, vz, vlat, vlon = v.shape
    _, qz, qlat, qlon = q.shape

    lev = level.size
    
    ft = -1/9.81

    # Matrizes de saída (t, lev-1, lat, lon)
    ap_   = np.zeros((t, qz-1, ulat, ulon))
    tvau_ = np.zeros((t, qz-1, ulat, ulon))
    tvav_ = np.zeros((t, qz-1, ulat, ulon))

    for zz in range(lev-1):

        # --------- Teor de umidade ---------
        ap_[:, zz, :, :] = ft * (
            (q[:, zz+1, :, :] + q[:, zz, :, :]) / 2
        ) * (level[zz] - level[zz+1]) * 100

        # --------- Transporte zonal ---------
        tvau_[:, zz, :, :] = ft * (
            (u[:, zz+1, :, :] * q[:, zz+1, :, :] +
             u[:, zz, :, :]   * q[:, zz, :, :]) / 2
        ) * (level[zz] - level[zz+1]) * 100

        # --------- Transporte meridional ---------
        tvav_[:, zz, :, :] = ft * (
            (v[:, zz+1, :, :] * q[:, zz+1, :, :] +
             v[:, zz, :, :]   * q[:, zz, :, :]) / 2
        ) * (level[zz] - level[zz+1]) * 100


    # Soma vertical → mantém dimensão temporal
    ap   = np.sum(ap_,   axis=1)  # (t, lat, lon)
    tvau = np.sum(tvau_, axis=1)
    tvav = np.sum(tvav_, axis=1)

    # Divergência para cada tempo
    dx, dy = lat_lon_grid_deltas(lon, lat)

    divtva = np.zeros_like(ap)
    for tt in range(t):
        divtva[tt] = divergence(
            tvau[tt], tvav[tt], dx=dx, dy=dy
        )

    return ap, tvau, tvav, divtva
