#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 20:50:24 2025

@author: simopec
"""
import sawp
import numpy as np

def processarAguaPrecipitavel(u_arq, v_arq, q_arq, lon, lat, resultado, level):
    import time
    
    # Tempo total
    t_total_ini = time.perf_counter()
    
    
    resultado_tva = {}   # ← dicionário final
    
    for qq in resultado['percentis_tsm'].keys():
    
        print(f"Processando quantil: {qq}")
        
        # Tempo por quantil
        t_ini = time.perf_counter()
        
        # Índices deste quantil
        idx = resultado['percentis_tsm'][qq].astype(int)
        print(idx)
        
        # ---- Carregamento eficiente (mmap) ----
        u = np.load(u_arq, mmap_mode='r')['data'][idx]
        v = np.load(v_arq, mmap_mode='r')['data'][idx]
        q = np.load(q_arq, mmap_mode='r')['data'][idx]

        lvl_u = np.load(u_arq, mmap_mode='r', allow_pickle=True)['level']
        lvl_v = np.load(v_arq, mmap_mode='r', allow_pickle=True)['level']
        lvl_q = np.load(q_arq, mmap_mode='r', allow_pickle=True)['level']
        
        variaveis = [u, v, q]
        
        print(u.shape)
        variaveis_lvl = [lvl_u, lvl_v, lvl_q]
        
        variaveis_corrigidas, niveis_corrigidos = sawp.alinhar_niveis(variaveis, variaveis_lvl, ordem='decrescente')#, ref_index=0)
        
        u, v, q = variaveis_corrigidas
        

        lvl_u, lvl_v, lvl_q = niveis_corrigidos
        print(lvl_u)
        
        # Level vem só do último load
        pressure_level = lvl_q[level]
        
        # ---- Cálculo do Teor e Transporte ----
        ap, tvau, tvav, divtva = sawp.teor_e_transporte_temporal(lat, lon, 
                                                        u[:,level], 
                                                        v[:,level], 
                                                        q[:,level], pressure_level)
    
        # ---- Armazena no dicionário ----
        resultado_tva[qq] = {
            "ap": ap,
            "tvau": tvau,
            "tvav": tvav,
            "divtva": divtva
        }
        
        # Tempo final por quantil
        t_fim = time.perf_counter()
        print(f"⏱️ Tempo do quantil {qq}: {t_fim - t_ini:.3f} s")
    
    # Tempo total
    t_total_fim = time.perf_counter()
    print("\n===================================")
    print(f"⏱️ Tempo total de processamento: {t_total_fim - t_total_ini:.3f} s")
    print("===================================")
    return resultado_tva