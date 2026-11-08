#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Módulo para download de dados ERA5 via CDS API.
Inclui:
- Níveis de pressão (monthly means)
- Níveis simples (single levels, ex: SST)

Funções principais:
- download_era5_pressure_monthly
- download_era5_single_monthly
"""

from __future__ import annotations
import os
import time
import logging
from typing import Optional, Sequence
import cdsapi

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_PRESSURE_LEVELS = [
    "400", "450", "500", "550", "600",
    "650", "700", "750", "775", "800",
    "825", "850",
]

DEFAULT_MONTHS = [f"{m:02d}" for m in range(1, 13)]


def _make_client(client: Optional[cdsapi.Client]) -> cdsapi.Client:
    """Retorna um cliente CDS, criando um se necessário."""
    if client is None:
        return cdsapi.Client()
    return client


# -------------------------------------------------------------------
# Núcleo genérico
# -------------------------------------------------------------------
def _retrieve_with_retry(
    dataset: str,
    request: dict,
    out_filepath: str,
    client: Optional[cdsapi.Client],
    retry: int,
    wait: int,
) -> str:
    """Função interna (não exportada) que executa o retrieve com retries."""
    c = _make_client(client)
    os.makedirs(os.path.dirname(out_filepath) or ".", exist_ok=True)

    last_err = None
    for attempt in range(1, retry + 1):
        try:
            logger.info(f"Solicitando CDS API ({dataset}), tentativa {attempt}/{retry}")
            c.retrieve(dataset, request, out_filepath)
            logger.info(f"Download concluído: {out_filepath}")
            return out_filepath
        except Exception as e:
            logger.warning(f"Falha no CDS ({attempt}/{retry}): {e}")
            last_err = e
            if attempt < retry:
                time.sleep(wait)

    logger.error(f"Todas as tentativas falharam. Último erro: {last_err}")
    raise last_err


# -------------------------------------------------------------------
# Função para pressure-levels
# -------------------------------------------------------------------
def download_era5_pressure_monthly(
    out_filepath: str,
    variables: Sequence[str],
    years: Sequence[str],
    months: Sequence[str] = DEFAULT_MONTHS,
    pressure_levels: Sequence[str] = DEFAULT_PRESSURE_LEVELS,
    area: Sequence[float] = (15, -60, -30, -10),
    client: Optional[cdsapi.Client] = None,
    retry: int = 3,
    wait: int = 10,
) -> str:
    """
    Baixa ERA5 (pressure levels) mensal.
    """
    request = {
        "format": "netcdf",
        "product_type": "monthly_averaged_reanalysis",
        "variable": list(variables),
        "pressure_level": list(pressure_levels),
        "year": list(years),
        "month": list(months),
        "time": "00:00",
        "area": list(area),
    }
    return _retrieve_with_retry(
        "reanalysis-era5-pressure-levels-monthly-means",
        request,
        out_filepath,
        client,
        retry,
        wait,
    )


# -------------------------------------------------------------------
# Função para single-levels (ex: SST)
# -------------------------------------------------------------------
def download_era5_single_monthly(
    out_filepath: str,
    variables: Sequence[str],
    years: Sequence[str],
    months: Sequence[str] = DEFAULT_MONTHS,
    area: Sequence[float] = (15, -60, -30, -10),
    client: Optional[cdsapi.Client] = None,
    retry: int = 3,
    wait: int = 10,
) -> str:
    """
    Baixa ERA5 (single levels) mensal.
    Exemplo de variável: 'sea_surface_temperature'
    """
    request = {
        "format": "netcdf",
        "product_type": "monthly_averaged_reanalysis",
        "variable": list(variables),
        "year": list(years),
        "month": list(months),
        "time": "00:00",
        "area": list(area),
    }

    return _retrieve_with_retry(
        "reanalysis-era5-single-levels-monthly-means",
        request,
        out_filepath,
        client,
        retry,
        wait,
    )


# -------------------------------------------------------------------
# CLI opcional (python -m sawp.era5)
# -------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Download ERA5 via CDS API")
    parser.add_argument("--out", required=True, help="Arquivo de saída")
    parser.add_argument("--type", required=True, choices=["pressure", "single"])
    parser.add_argument("--variables", nargs="+", required=True)
    parser.add_argument("--start-year", required=True)
    parser.add_argument("--end-year", required=True)
    parser.add_argument("--area", nargs=4, type=float, default=[15, -60, -30, -10])
    parser.add_argument("--retry", type=int, default=3)
    parser.add_argument("--wait", type=int, default=10)

    args = parser.parse_args()

    years = [str(y) for y in range(int(args.start_year), int(args.end_year) + 1)]

    if args.type == "pressure":
        download_era5_pressure_monthly(
            out_filepath=args.out,
            variables=args.variables,
            years=years,
            area=args.area,
            retry=args.retry,
            wait=args.wait,
        )

    else:  # single
        download_era5_single_monthly(
            out_filepath=args.out,
            variables=args.variables,
            years=years,
            area=args.area,
            retry=args.retry,
            wait=args.wait,
        )
