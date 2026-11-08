#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 13 19:32:23 2025

@author: simopec
"""

def ExtrairVariaveisQuantis(resultado_reg,var,tsm_norm):
    # Usando quantis para retirar variáveis contidas nos intervalos interquantis
    q10     = var[tsm_norm <= resultado_reg['q10']]
    q90     = var[tsm_norm >= resultado_reg['q90']]
    
    q25     = var[(tsm_norm > resultado_reg['q10']) & 
                         (tsm_norm <= resultado_reg['q25'])]
    
    q75     = var[(tsm_norm >= resultado_reg['q75']) &
                         (tsm_norm < resultado_reg['q90'])]
    
    qno     = var[(tsm_norm > resultado_reg['q25']) & 
                         (tsm_norm < resultado_reg['q75'])]
    return q10, q90, q25, q75, qno
