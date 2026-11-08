#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 20:10:50 2025

@author: simopec
"""
import time
import sawp
import numpy as np
import cupy as cp
import os

# ---------------------------------------------------------------------------------------
# Caminhos
# ---------------------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
#DATA_DIR = os.path.join(BASE_DIR, ".", "data")

tendencias = 'não'

if tendencias == 'não':
    print('Carregando detrended (t, q)')

    t_arq = "./outputs/detrended_uvq_t.npz"
    q_arq = "./outputs/detrended_uvq_q.npz"
    VAR_NOME = 'detrended_uvq_te.npz'
    OUT_DIR = os.path.join(BASE_DIR, "outputs")

else:
    print('Carregando trended (t, q)')

    t_arq = "./outputs_trended/trended_uvq_t.npz"
    q_arq = "./outputs_trended/trended_uvq_q.npz"
    VAR_NOME = 'trended_uvq_te.npz'
    OUT_DIR = os.path.join(BASE_DIR, "outputs_trended")
    
lat = np.load(t_arq, mmap_mode='r')['lat']
lon = np.load(t_arq, mmap_mode='r')['lon']
level = np.load(t_arq, mmap_mode='r')['level']
periodo = np.load(t_arq, mmap_mode='r')['time']

# Achando as localizações de latitude para realizar média
lat_min, lat_max = -12.5, -4
idx = np.where((lat >= lat_min) & (lat <= lat_max))[0]
print(idx[0], idx[-1])

t = np.load(t_arq, mmap_mode='r')['data']
q = np.load(q_arq, mmap_mode='r')['data']

nt, nlev, nlat, nlon = t.shape

EPT_gpu_out = np.empty((nt, nlev, nlat, nlon), dtype=np.float32)

pres_gpu = cp.asarray(level, dtype=cp.float32)
lati_gpu = cp.asarray(lat, dtype=cp.float32)
long_gpu = cp.asarray(lon, dtype=cp.float32)

# 🔹 início do tempo GPU
cp.cuda.Stream.null.synchronize()
t0 = time.perf_counter()

for it in range(nt):

    temp_gpu = cp.asarray(t[it], dtype=cp.float32)
    q_gpu    = cp.asarray(q[it], dtype=cp.float32)

    # 1️⃣ Ponto de orvalho
    td_gpu = sawp.dewpoint_gpu(
        q_gpu, temp_gpu, pres_gpu, lati_gpu, long_gpu
    )

    # 2️⃣ EPT
    EPT_gpu = sawp.temperatura_potencial_equivalente_gpu(
        temp_gpu, td_gpu, pres_gpu, lati_gpu, long_gpu
    )

    # Copia para RAM
    EPT_gpu_out[it] = cp.asnumpy(EPT_gpu)

    # Limpeza de VRAM
    del temp_gpu, q_gpu, td_gpu, EPT_gpu
    cp.get_default_memory_pool().free_all_blocks()

# 🔹 fim do tempo GPU
cp.cuda.Stream.null.synchronize()
t1 = time.perf_counter()

print(f"Tempo total GPU: {t1 - t0:.2f} s")
print(f"Tempo médio por passo de tempo (GPU): {(t1 - t0)/nt:.4f} s")

out_file = os.path.join(OUT_DIR, VAR_NOME)

np.savez(
    out_file,
    data=EPT_gpu_out,
    level=level,
    lat=lat,
    lon=lon,
    time = periodo
)
        

