#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recria o painel CT-ST com vetores e chaves de referencia nos paineis c e d.

A figura mantem os mesmos dados, quantil 90 e diferenca CT - ST de
RealizarAnalises.py. Somente a representacao vetorial dos paineis c e d e
especifica deste script.
"""

from pathlib import Path
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import sawp

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
OUTPUT_FILE = PROJECT_ROOT / "figuras" / "Subplot_CT_ST_quiver_referencia" / "Subplot_CT-ST_quiver_referencia_w_corrigido_unit.png"
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# Parametros comuns aos vetores e respectivas chaves de referencia.
TRANSPORT_SCALE = 300
TRANSPORT_WIDTH = 0.0025
TRANSPORT_REFERENCE = 10.0  # kg m^-1 s^-1
CIRCULATION_SCALE = 100
CIRCULATION_WIDTH = 0.0025
CIRCULATION_REFERENCE = 1.0  # 1 desvio padrao das componentes normalizadas
# As componentes do painel d sao padronizadas antes do quiver.


def configurar_mapa(ax, identificador):
    """Aplica ao painel c a mesma base cartografica do script principal."""
    ax.set_extent([-60, -10, -30, 15], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    ax.add_feature(cfeature.LAND, edgecolor="black", facecolor="lightgrey", alpha=0.3)
    estados = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lakes",
        scale="10m",
        facecolor="none",
    )
    ax.add_feature(estados, edgecolor="black", linewidth=0.3, alpha=0.3)

    if identificador == "a":
        ax.set_yticks(np.arange(-30, 20, 10), crs=ccrs.PlateCarree())
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=16)
    elif identificador == "b":
        ax.set_xticks([])
        ax.set_yticks([])
    elif identificador == "c":
        ax.set_xticks(np.arange(-60, -5, 10), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(-30, 20, 10), crs=ccrs.PlateCarree())
        ax.tick_params(axis="both", labelsize=16)


def plotar_transporte_com_quiver(lon, lat, u, v, convergencia, ax):
    """Painel c: transporte de vapor e convergencia, com quiver rastreavel."""
    configurar_mapa(ax, "c")
    passo = 10
    quiver = ax.quiver(
        lon[::passo, ::passo],
        lat[::passo, ::passo],
        u[::passo, ::passo],
        v[::passo, ::passo],
        transform=ccrs.PlateCarree(),
        scale=TRANSPORT_SCALE,
        width=TRANSPORT_WIDTH,
    )
    niveis = np.asarray(range(-30, 31, 10)) / 1e6
    sombreado = ax.contourf(
        lon,
        lat,
        convergencia,
        transform=ccrs.PlateCarree(),
        extend="both",
        cmap="PuOr_r",
        alpha=0.5,
        levels=niveis,
    )
    ax.set_title("c)", loc="left", fontsize=18)
    return sombreado, quiver


def plotar_circulacao_com_quiver(u, w, theta_e, level, y_equal_spacing, lon, ax):
    """Painel d: secao longitudinal com o mesmo campo, usando setas discretas."""
    n_pressao = len(y_equal_spacing)
    n_lon = u.shape[1]
    u_interp = np.empty((n_pressao, n_lon))
    w_interp = np.empty((n_pressao, n_lon))
    theta_e_interp = np.empty((n_pressao, n_lon))

    for indice_lon in range(n_lon):
        u_interp[:, indice_lon] = np.interp(y_equal_spacing, level, u[:, indice_lon])
        w_interp[:, indice_lon] = np.interp(y_equal_spacing, level, w[:, indice_lon])
        theta_e_interp[:, indice_lon] = np.interp(y_equal_spacing, level, theta_e[:, indice_lon])

    longitude, pressao = np.meshgrid(lon[0, :], y_equal_spacing)
    niveis = np.asarray(range(-10, 11, 2)) / 10
    sombreado = ax.contourf(
        longitude,
        pressao,
        theta_e_interp,
        cmap="RdYlBu_r",
        extend="both",
        levels=niveis,
        alpha=0.4,
    )

    # u e w chegam a esta funcao ja divididos por seus proprios desvios padrao.
    passo_pressao, passo_lon = 4, 5
    quiver = ax.quiver(
        longitude[::passo_pressao, ::passo_lon],
        pressao[::passo_pressao, ::passo_lon],
        u_interp[::passo_pressao, ::passo_lon],
        w_interp[::passo_pressao, ::passo_lon],
        scale=CIRCULATION_SCALE,
        width=CIRCULATION_WIDTH,
        color="0.35",
    )

    ax.set_xlim([-60, -10])
    ax.set_ylim([1000, 200])
    ax.set_xticks(np.linspace(lon.min(), lon.max(), 5))
    ax.tick_params(axis="x", labelsize=16)
    ax.set_title("d)", loc="left", fontsize=18)
    ax.fill_betweenx([1000, 200], -40, -30, color="gray", alpha=0.2)
    return sombreado, quiver


# Dados sem tendencia (ST): mesmas escolhas de RealizarAnalises.py.
tendencias = "não"
caminho = "./outputs/"
area_sawp = [-34, -25, -12.5, -4]
periodo_tempo, lon, lat = sawp.ConfigurarTempoArea("./data/Era5_Dataset_sst.nc")
tsm_st_dados = sawp.CarregarDados(
    caminho + "detrended_sst.npz",
    tendencias=tendencias,
    nome_var_nc="sst",
    substituir_nan=-999,
)

llo_w = np.argmin(np.abs(lon[0, :] - area_sawp[0]))
llo_e = np.argmin(np.abs(lon[0, :] - area_sawp[1]))
lla_s = np.argmin(np.abs(lat[:, 0] - area_sawp[2]))
lla_n = np.argmin(np.abs(lat[:, 0] - area_sawp[3]))
resultado_st = sawp.ProcessarTsmv2(tsm_st_dados, periodo_tempo, llo_w, llo_e, lla_s, lla_n)
resultado_st_90 = resultado_st.copy()
resultado_st_90["percentis_tsm"] = {"90": resultado_st["percentis_tsm"]["90"]}
print("Quantis ST calculados.")

u_st_file = "./outputs/detrended_uvq_u.npz"
v_st_file = "./outputs/detrended_uvq_v.npz"
q_st_file = "./outputs/detrended_uvq_q.npz"
resultado_tva_ml_st = sawp.processarAguaPrecipitavel(
    u_st_file, v_st_file, q_st_file, lon, lat, resultado_st_90, np.array(range(6, 14))
)
resultado_tva_bl_st = sawp.processarAguaPrecipitavel(
    u_st_file, v_st_file, q_st_file, lon, lat, resultado_st_90, np.array(range(0, 7))
)

# Dados com tendencia (CT): mesmo quantil 90 e mesma diferenca CT - ST.
tsm_ct_dados = sawp.CarregarDados(
    "./outputs_trended/trended_uvq_sst.npz",
    tendencias="sim",
    nome_var_nc="sst",
    substituir_nan=-999,
)
resultado_ct = sawp.ProcessarTsmv2(tsm_ct_dados, periodo_tempo, llo_w, llo_e, lla_s, lla_n)
for percentil in ["10", "25", "75", "normal"]:
    resultado_ct["percentis_tsm"].pop(percentil, None)
print("Quantil CT calculado.")

resultado_tva_ml_ct = sawp.processarAguaPrecipitavel(
    "./outputs_trended/trended_uvq_u.npz",
    "./outputs_trended/trended_uvq_v.npz",
    "./outputs_trended/trended_uvq_q.npz",
    lon,
    lat,
    resultado_ct,
    np.array(range(6, 14)),
)
resultado_tva_bl_ct = sawp.processarAguaPrecipitavel(
    "./outputs_trended/trended_uvq_u.npz",
    "./outputs_trended/trended_uvq_v.npz",
    "./outputs_trended/trended_uvq_q.npz",
    lon,
    lat,
    resultado_ct,
    np.array(range(0, 7)),
)

# Diferencas horizontais.
diff_tsm = (
    np.mean(tsm_ct_dados[resultado_ct["percentis_tsm"]["90"]], axis=0)
    - np.mean(tsm_st_dados[resultado_st["percentis_tsm"]["90"]], axis=0)
)
diff_ap = (
    np.mean(resultado_tva_ml_ct["90"]["ap"], axis=0)
    - np.mean(resultado_tva_ml_st["90"]["ap"], axis=0)
)
diff_tvau = (
    np.mean(resultado_tva_ml_ct["90"]["tvau"], axis=0)
    - np.mean(resultado_tva_ml_st["90"]["tvau"], axis=0)
)
diff_tvav = (
    np.mean(resultado_tva_ml_ct["90"]["tvav"], axis=0)
    - np.mean(resultado_tva_ml_st["90"]["tvav"], axis=0)
)
diff_divtva = (
    np.mean(resultado_tva_bl_ct["90"]["divtva"], axis=0)
    - np.mean(resultado_tva_bl_st["90"]["divtva"], axis=0)
)

# Diferencas na secao longitudinal sobre a SAWP.
idx_lat = np.where((lat >= -12.5) & (lat <= -4))[0]
inicio_lat, fim_lat = idx_lat[0], idx_lat[-1]

def carregar_secao(caminho_te, caminho_w, caminho_u):
    te = np.load(caminho_te, mmap_mode="r")["data"][:, :, inicio_lat:fim_lat, :]
    w = np.load(caminho_w, mmap_mode="r")["data"][:, :, inicio_lat:fim_lat, :]
    u = np.load(caminho_u, mmap_mode="r")["data"][:, :, inicio_lat:fim_lat, :]
    nivel_te = np.load(caminho_te, mmap_mode="r")["level"]
    nivel_w = np.load(caminho_w, mmap_mode="r")["level"]
    nivel_u = np.load(caminho_u, mmap_mode="r")["level"]
    if (nivel_te[0] != nivel_w[0]) or (nivel_te[0] != nivel_u[0]) or (nivel_u[0] != nivel_w[0]):
        (w, u), (nivel_w, nivel_u) = sawp.alinhar_niveis([w, u], [nivel_w, nivel_u], ordem="crescente")
    return te, w, u, nivel_te

te_st, w_st, u_st, nivel_te = carregar_secao(
    "./outputs/detrended_uvq_te.npz",
    "./outputs/detrended_uvq_w.npz",
    "./outputs/detrended_uvq_u.npz",
)
te_ct, w_ct, u_ct, _ = carregar_secao(
    "./outputs_trended/trended_uvq_te.npz",
    "./outputs_trended/trended_uvq_w.npz",
    "./outputs_trended/trended_uvq_u.npz",
)

indices_st = resultado_st["percentis_tsm"]["90"]
indices_ct = resultado_ct["percentis_tsm"]["90"]
diff_te = np.mean(np.mean(te_ct[indices_ct], axis=0), axis=1) - np.mean(np.mean(te_st[indices_st], axis=0), axis=1)
diff_w = np.mean(np.mean(w_ct[indices_ct], axis=0), axis=1) - np.mean(np.mean(w_st[indices_st], axis=0), axis=1)
diff_u = np.mean(np.mean(u_ct[indices_ct], axis=0), axis=1) - np.mean(np.mean(u_st[indices_st], axis=0), axis=1)

# Padronizacao exclusiva do painel d: cada componente e dividida pelo
# proprio desvio padrao espacial da diferenca CT - ST.
std_diff_u = np.nanstd(diff_u)
std_diff_w = np.nanstd(diff_w)
if not (np.isfinite(std_diff_u) and std_diff_u > 0 and np.isfinite(std_diff_w) and std_diff_w > 0):
    raise ValueError("Os desvios padrao de u e w devem ser finitos e positivos.")
diff_u_padronizado = diff_u / std_diff_u
diff_w_padronizado = diff_w / std_diff_w
print(f"Painel d padronizado: std(diff_u)={std_diff_u:.6g}; std(diff_w)={std_diff_w:.6g}")

# Figura CT-ST.
fig = plt.figure(figsize=(12, 10))
gs = fig.add_gridspec(2, 2)
ax_a = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
ax_b = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
ax_c = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
ax_d = fig.add_subplot(gs[1, 1])
fig.subplots_adjust(left=0.08, right=0.95, bottom=0.16, top=0.96, hspace=0.24, wspace=0.15)

# Paineis a e b permanecem iguais ao painel CT-ST atual.
temp_a = sawp.plotar_grafico_sombreado(
    lon, lat, diff_tsm, np.asarray(range(-5, 6, 1)) / 10, "bwr", ax_a, alpha=0.7, identificador="a"
)
ax_a.set_title("a)", loc="left", fontsize=18)
cax_a = inset_axes(ax_a, width="5%", height="30%", loc="lower right", bbox_to_anchor=(-0.15, 0.0, 1, 1), bbox_transform=ax_a.transAxes, borderpad=0.5)
cbar_a = plt.colorbar(temp_a, cax=cax_a, orientation="vertical")
cbar_a.ax.tick_params(labelsize=8)
cbar_a.ax.set_title("SST (degC)", fontsize=8)

temp_b = sawp.plotar_grafico_sombreado(
    lon, lat, diff_ap, np.asarray(range(-10, 11, 2)) / 10, "BrBG", ax_b, alpha=0.7, identificador="b"
)
ax_b.set_title("b)", loc="left", fontsize=18)
cax_b = inset_axes(ax_b, width="5%", height="30%", loc="lower right", bbox_to_anchor=(-0.15, 0.0, 1, 1), bbox_transform=ax_b.transAxes, borderpad=0.5)
cbar_b = plt.colorbar(temp_b, cax=cax_b, orientation="vertical")
cbar_b.ax.tick_params(labelsize=8)
cbar_b.ax.set_title("Prec. Water (mm)", fontsize=8)

# Painel c: quiver e chave usam exatamente o mesmo objeto, escala e largura.
temp_c, quiver_c = plotar_transporte_com_quiver(lon, lat, diff_tvau, diff_tvav, diff_divtva, ax_c)
cax_c = inset_axes(ax_c, width="5%", height="30%", loc="lower right", bbox_to_anchor=(-0.15, 0.0, 1, 1), bbox_transform=ax_c.transAxes, borderpad=0.5)
cbar_c = plt.colorbar(temp_c, cax=cax_c, orientation="vertical")
cbar_c.ax.tick_params(labelsize=8)
cbar_c.ax.set_title("Mois. flux conv. (kg m-2 s-1)", fontsize=8)
ax_c.quiverkey(
    quiver_c, X=0.50, Y=-0.15, U=TRANSPORT_REFERENCE,
    label=r"10 kg m$^{-1}$ s$^{-1}$", labelpos="E", coordinates="axes", fontproperties={"size": 9}
)

# Painel d: as duas componentes sao adimensionais e padronizadas.
# u permanece no sentido direto; somente w e invertido para corresponder
# ao sentido vertical da figura de referencia.
pressao_regular = np.linspace(100, 1000, 60)
temp_d, quiver_d = plotar_circulacao_com_quiver(diff_u_padronizado, -diff_w_padronizado, diff_te, nivel_te, pressao_regular, lon, ax_d)
cax_d = inset_axes(ax_d, width="5%", height="30%", loc="lower right", bbox_to_anchor=(-0.15, 0.0, 1, 1), bbox_transform=ax_d.transAxes, borderpad=0.5)
cbar_d = plt.colorbar(temp_d, cax=cax_d, orientation="vertical")
cbar_d.ax.tick_params(labelsize=8)
cbar_d.ax.set_title(r"theta_e (K)", fontsize=8)
ax_d.yaxis.tick_right()
ax_d.yaxis.set_label_position("right")
ax_d.set_yticks([1000, 850, 700, 500, 300, 200])
ax_d.tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True, labelsize=12)
ax_d.spines["left"].set_visible(False)
ax_d.spines["right"].set_visible(True)
ax_d.set_ylabel("Pressure (hPa)", fontsize=14, labelpad=10)
ax_d.quiverkey(
    quiver_d, X=0.50, Y=-0.15, U=CIRCULATION_REFERENCE,
    label="1 unit", labelpos="E", coordinates="axes", fontproperties={"size": 9}
)

fig.text(0.5, 0.035, "Longitude", ha="center", fontsize=20)
fig.text(0.025, 0.5, "Latitude", va="center", rotation="vertical", fontsize=20)
fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Figura salva em: {OUTPUT_FILE}")








