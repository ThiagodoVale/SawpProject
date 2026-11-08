# -*- coding: utf-8 -*-
"""
Created on Sat Jun 27 16:08:43 2026

@author: silva
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download do fluxo de calor latente superficial (Surface Latent Heat Flux)
ERA5 Monthly Means
Período: 1959–2023
"""

from sawp import download_era5_single_monthly

# ---------------------------------------------------------------------
# Configurações
# ---------------------------------------------------------------------

OUTFILE = "./data/Era5_Dataset_meanHF.nc"

YEARS = [str(y) for y in range(1959, 2024)]

AREA = (
    15,     # Norte
    -60,    # Oeste
    -30,    # Sul
    -10     # Leste
)

VARIABLES = [
        "mean_surface_latent_heat_flux",
        "mean_surface_sensible_heat_flux"
]

# ---------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------

download_era5_single_monthly(
    out_filepath=OUTFILE,
    variables=VARIABLES,
    years=YEARS,
    area=AREA,
    retry=5,
    wait=30,
)

print("\nDownload finalizado com sucesso!")
print(f"Arquivo salvo em:\n{OUTFILE}")