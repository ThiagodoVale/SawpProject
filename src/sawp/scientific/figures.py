# ruff: noqa: RUF001
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.interpolate import interp1d
from scipy.stats import kendalltau, linregress, pearsonr, spearmanr

from sawp.config import ProjectConfig

from .atmosphere import AtmosphereProducts
from .common import (
    open_canonical,
    regional_mean,
    subset_domain,
    theil_sen_slope,
)
from .ocean import OceanProducts
from .rainfall import RainfallProducts

EVENT_CLASSES = (
    ("extreme_cold", "Q10 cold", "#2166ac"),
    ("moderate_cold", "Q10–Q25 cool", "#67a9cf"),
    ("moderate_warm", "Q75–Q90 warm", "#ef8a62"),
    ("extreme_warm", "Q90 warm", "#b2182b"),
)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.grid": False,
        }
    )


def _save(
    fig: plt.Figure,
    path: Path,
    *,
    dpi: int = 300,
    pad_inches: float | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    options = {
        "dpi": dpi,
        "bbox_inches": "tight",
        "metadata": {"Software": "SAWP 1.0"},
    }
    if pad_inches is not None:
        options["pad_inches"] = pad_inches
    fig.savefig(path, **options)
    plt.close(fig)
    return path


def _map(
    axis: plt.Axes,
    value: xr.DataArray,
    *,
    title: str,
    cmap: str = "RdBu_r",
    symmetric: bool = True,
    units: str = "",
):
    array = value.to_numpy()
    finite = array[np.isfinite(array)]
    if finite.size:
        if symmetric:
            limit = float(np.nanquantile(np.abs(finite), 0.98)) or 1.0
            vmin, vmax = -limit, limit
        else:
            vmin, vmax = np.nanquantile(finite, [0.02, 0.98])
    else:
        vmin, vmax = -1.0, 1.0
    image = axis.pcolormesh(
        value.lon,
        value.lat,
        value,
        shading="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axis.set(title=title, xlabel="Longitude", ylabel="Latitude")
    axis.set_xlim(float(value.lon.min()), float(value.lon.max()))
    axis.set_ylim(float(value.lat.min()), float(value.lat.max()))
    axis.grid(alpha=0.2)
    colorbar = axis.figure.colorbar(image, ax=axis, shrink=0.82)
    if units:
        colorbar.set_label(units)
    return image


def _class_years(series: pd.DataFrame, label: str) -> np.ndarray:
    return series.index[series["sst_class"] == label].to_numpy(int)


def _class_composite(value: xr.DataArray, series: pd.DataFrame, label: str) -> xr.DataArray:
    years = np.intersect1d(value.year.to_numpy().astype(int), _class_years(series, label))
    if not years.size:
        return value.mean("year") * np.nan
    return value.sel(year=years).mean("year", skipna=True)


def _column_composite(
    value: xr.DataArray,
    series: pd.DataFrame,
    label: str,
    column: str,
) -> xr.DataArray:
    selected = series.index[series[column] == label].to_numpy(int)
    years = np.intersect1d(value.year.to_numpy().astype(int), selected)
    if not years.size:
        return value.mean("year") * np.nan
    return value.sel(year=years).mean("year", skipna=True)


def _class_contrast(value: xr.DataArray, series: pd.DataFrame, label: str) -> xr.DataArray:
    return _class_composite(value, series, label) - _class_composite(value, series, "neutral")


def _legacy_map_axis(
    axis: plt.Axes,
    value: xr.DataArray,
    *,
    levels: np.ndarray,
    cmap: str,
    panel: str,
    alpha: float = 0.5,
):
    axis.set_extent([-60, -10, -30, 15], crs=ccrs.PlateCarree())
    axis.add_feature(cfeature.COASTLINE, linewidth=0.5)
    axis.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.5)
    axis.add_feature(
        cfeature.LAND,
        edgecolor="black",
        facecolor="lightgrey",
        alpha=0.3,
    )
    states = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lakes",
        scale="10m",
        facecolor="none",
    )
    axis.add_feature(states, edgecolor="black", linewidth=0.3, alpha=0.3)
    if panel == "a":
        axis.set_yticks(np.arange(-30, 20, 10), crs=ccrs.PlateCarree())
        axis.set_xticks([])
        axis.tick_params(axis="y", labelsize=16)
    elif panel == "b":
        axis.set_xticks([])
        axis.set_yticks([])
    elif panel == "c":
        axis.set_xticks(np.arange(-60, -5, 10), crs=ccrs.PlateCarree())
        axis.set_yticks(np.arange(-30, 20, 10), crs=ccrs.PlateCarree())
        axis.tick_params(axis="both", labelsize=16)
    else:
        axis.set_xticks(np.arange(-60, -5, 10), crs=ccrs.PlateCarree())
        axis.set_yticks([])
        axis.tick_params(axis="x", labelsize=16)
    image = axis.contourf(
        value.lon,
        value.lat,
        value,
        levels=levels,
        cmap=cmap,
        extend="both",
        transform=ccrs.PlateCarree(),
        alpha=alpha,
    )
    box = Rectangle(
        (-34, -12.5),
        9,
        8.5,
        linewidth=1.2,
        edgecolor="black",
        facecolor="lightgray",
        hatch="///",
        alpha=0.18,
        transform=ccrs.PlateCarree(),
        zorder=5,
    )
    axis.add_patch(box)
    axis.legend(
        handles=[
            Rectangle(
                (0, 0),
                1,
                1,
                facecolor="lightgray",
                edgecolor="black",
                hatch="///",
                alpha=0.35,
                label="SAWP",
            )
        ],
        loc="upper right",
        fontsize=8,
        frameon=True,
        facecolor="white",
        edgecolor="black",
        framealpha=1.0,
        handlelength=1.5,
        borderpad=0.4,
    )
    axis.set_title(f"{panel})", loc="left", fontsize=18)
    return image


def _legacy_four_maps(
    values: list[xr.DataArray],
    destination: Path,
    *,
    levels: np.ndarray,
    cmap: str,
    colorbar_label: str,
) -> Path:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    fig.subplots_adjust(hspace=0.01, wspace=0.15)
    image = None
    for axis, value, panel in zip(axes.flat, values, "abcd", strict=True):
        image = _legacy_map_axis(axis, value, levels=levels, cmap=cmap, panel=panel)
    fig.text(0.45, 0.05, "Longitude", ha="center", fontsize=20)
    fig.text(0.045, 0.5, "Latitude", va="center", rotation="vertical", fontsize=20)
    colorbar = fig.colorbar(image, ax=axes, orientation="vertical", shrink=0.5, pad=0.02)
    colorbar.set_label(colorbar_label, size=15)
    colorbar.ax.tick_params(labelsize=12)
    fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.2, hspace=0.1, wspace=0.2)
    return _save(fig, destination, dpi=360)


def render_rainfall_figures(
    config: ProjectConfig,
    rainfall: RainfallProducts,
    destination: Path,
) -> list[Path]:
    _style()
    metadata = pd.read_csv(rainfall.metadata, index_col=0)
    regional = pd.read_csv(rainfall.regional, index_col=0, parse_dates=True)
    robustness = pd.read_csv(rainfall.robustness, index_col=0, parse_dates=True)
    homogeneity = pd.read_csv(rainfall.homogeneity, index_col=0)
    outputs: list[Path] = []

    plt.rcParams.update(
        {
            "font.size": 18,
            "axes.titlesize": 18,
            "axes.labelsize": 18,
            "xtick.labelsize": 18,
            "ytick.labelsize": 18,
            "legend.fontsize": 14,
        }
    )

    domain_sst = open_canonical(
        config.paths.raw_domain / "sst.nc", chunks={"lat": 360}, decode_times=False
    )
    topography = open_canonical(
        config.paths.raw_domain / "srtm30plus_v11_land.nc", chunks={"lat": 360}
    )
    april = subset_domain(domain_sst.sst.isel(time=3), west=-40, east=-20, south=-15, north=0)
    elevation = subset_domain(topography.elev, west=-40, east=-20, south=-15, north=0)
    fig = plt.figure(figsize=(10, 10))
    axis = plt.axes(projection=ccrs.PlateCarree())
    axis.set_extent([-40, -20, -15, 0], crs=ccrs.PlateCarree())
    axis.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.5)
    axis.add_feature(cfeature.OCEAN, facecolor="white", alpha=1.0)
    axis.add_feature(cfeature.COASTLINE, edgecolor="black", linewidth=1.0)
    axis.add_feature(cfeature.BORDERS, linestyle="--", edgecolor="black", alpha=0.4)
    states = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces",
        scale="10m",
        facecolor="none",
    )
    axis.add_feature(states, edgecolor="black", linewidth=0.5, alpha=0.4)
    levels = np.arange(27, 30, 0.25)
    image = axis.contourf(
        april.lon,
        april.lat,
        april,
        levels=levels,
        cmap="hot_r",
        transform=ccrs.PlateCarree(),
    )
    contour = axis.contour(
        april.lon,
        april.lat,
        april,
        levels=[28.5],
        colors="black",
        linewidths=2.5,
        transform=ccrs.PlateCarree(),
    )
    axis.clabel(contour, inline=True, fontsize=14, fmt="%1.1f")
    sst_colorbar = fig.colorbar(
        image,
        ax=axis,
        orientation="horizontal",
        pad=0.13,
        fraction=0.06,
        shrink=0.85,
        aspect=35,
    )
    sst_colorbar.set_label("Sea Surface Temperature (°C)", fontsize=16)
    relief = axis.pcolormesh(
        elevation.lon,
        elevation.lat,
        elevation,
        cmap="terrain",
        vmin=0,
        vmax=1200,
        alpha=0.55,
        transform=ccrs.PlateCarree(),
        shading="auto",
    )
    relief_colorbar = fig.colorbar(
        relief, ax=axis, orientation="vertical", pad=0.01, fraction=0.025
    )
    relief_colorbar.set_label("Topography (m)", fontsize=13)
    axis.scatter(
        metadata.longitude,
        metadata.latitude,
        s=5,
        c="black",
        transform=ccrs.PlateCarree(),
        label=f"Rain gauges used (n={len(metadata)})",
        zorder=4,
    )
    sawp = config.domains["sawp_figure1"]
    axis.add_patch(
        Rectangle(
            (sawp.west, sawp.south),
            sawp.east - sawp.west,
            sawp.north - sawp.south,
            edgecolor="black",
            facecolor="lightgray",
            alpha=0.25,
            linewidth=2.5,
            transform=ccrs.PlateCarree(),
            zorder=6,
        )
    )
    axis.text(
        -31.8,
        -7.5,
        "SAWP",
        color="black",
        fontsize=12,
        weight="bold",
        transform=ccrs.PlateCarree(),
        zorder=7,
    )
    gridlines = axis.gridlines(
        draw_labels=True,
        crs=ccrs.PlateCarree(),
        color="gray",
        linestyle="--",
        linewidth=0.5,
        alpha=0.6,
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 12}
    gridlines.ylabel_style = {"size": 12}
    gridlines.xlines = False
    gridlines.ylines = False
    axis.legend(loc="upper right", fontsize=10, frameon=True)
    fig.text(0.5, 0.2, "Longitude", ha="center", fontsize=20)
    fig.text(0.03, 0.5, "Latitude", va="center", rotation="vertical", fontsize=20)
    outputs.append(_save(fig, destination / "Figure1.png", dpi=360))
    domain_sst.close()
    topography.close()

    fig, axis = plt.subplots(figsize=(12, 4))
    regional.n_valid_stations.plot(ax=axis)
    axis.set_ylabel("Number of Valid Stations", fontsize=18)
    axis.set_xlabel("Year", fontsize=18)
    axis.set_title("Monthly Availability of the Rain-Gauge Network", fontsize=18)
    axis.grid(alpha=0.3)
    axis.tick_params(labelsize=18)
    fig.tight_layout()
    outputs.append(_save(fig, destination / "FigureS1.png"))

    fig, axis = plt.subplots(figsize=(12, 6))
    labels = {
        "all": f"All stations ({len(metadata)} stations)",
        "completeness_50": (
            f">=50% complete ({int((metadata.completeness >= 0.50).sum())} stations)"
        ),
        "completeness_70": (
            f">=70% complete ({int((metadata.completeness >= 0.70).sum())} stations)"
        ),
        "completeness_90": (
            f">=90% complete ({int((metadata.completeness >= 0.90).sum())} stations)"
        ),
    }
    for column, label in labels.items():
        robustness[column].rolling(12, min_periods=6).mean().plot(ax=axis, label=label)
    axis.set_ylabel("Regional Mean Monthly Precipitation (mm)", fontsize=18)
    axis.set_xlabel("Year", fontsize=18)
    axis.set_title("Robustness Test of Regional Precipitation", fontsize=18)
    axis.legend(fontsize=14)
    axis.grid(alpha=0.3)
    axis.tick_params(labelsize=18)
    fig.tight_layout()
    outputs.append(_save(fig, destination / "FigureS2.png"))

    fig, axis = plt.subplots(figsize=(12, 6))
    labels = {
        "all": f"All stations ({len(metadata)} stations)",
        "start_1970": (f"Start <=1970 ({int((metadata.first_year <= 1970).sum())} stations)"),
        "start_1965": (f"Start <=1965 ({int((metadata.first_year <= 1965).sum())} stations)"),
        "start_1960": (f"Start <=1960 ({int((metadata.first_year <= 1960).sum())} stations)"),
    }
    for column, label in labels.items():
        robustness[column].rolling(12, min_periods=6).mean().plot(ax=axis, label=label)
    axis.set_ylabel("Regional Mean Monthly Precipitation (mm)", fontsize=18)
    axis.set_xlabel("Year", fontsize=18)
    axis.set_title("Robustness Test Using Long-Term Stations", fontsize=18)
    axis.legend(fontsize=14)
    axis.grid(alpha=0.3)
    axis.tick_params(labelsize=18)
    fig.tight_layout()
    outputs.append(_save(fig, destination / "FigureS3.png"))

    flagged = (
        homogeneity["flag_unusual_peak_month"].astype(str).str.lower().isin({"true", "1", "yes"})
    )
    normal = homogeneity.loc[~flagged]
    unusual = homogeneity.loc[flagged]
    fig = plt.figure(figsize=(12, 10))
    axis = plt.axes(projection=ccrs.PlateCarree())
    axis.set_extent(
        [
            homogeneity.longitude.min() - 1,
            homogeneity.longitude.max() + 1,
            homogeneity.latitude.min() - 1,
            homogeneity.latitude.max() + 1,
        ],
        crs=ccrs.PlateCarree(),
    )
    axis.add_feature(cfeature.LAND, alpha=0.8)
    axis.add_feature(cfeature.OCEAN, alpha=0.8)
    axis.add_feature(cfeature.COASTLINE, linewidth=1)
    axis.add_feature(cfeature.BORDERS, linestyle=":")
    states = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces",
        scale="10m",
        facecolor="none",
    )
    axis.add_feature(states, edgecolor="gray", linewidth=0.5)
    axis.scatter(
        normal.longitude,
        normal.latitude,
        s=8,
        c="black",
        transform=ccrs.PlateCarree(),
        label=f"Compatible rainfall regime (n={len(normal)})",
        zorder=3,
    )
    axis.scatter(
        unusual.longitude,
        unusual.latitude,
        s=20,
        c="red",
        transform=ccrs.PlateCarree(),
        label=f"Distinct rainfall regime (n={len(unusual)})",
        zorder=4,
    )
    gridlines = axis.gridlines(
        draw_labels=True,
        linewidth=0.5,
        color="gray",
        alpha=0.5,
        linestyle="--",
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {"size": 18}
    gridlines.ylabel_style = {"size": 18}
    axis.set_title(
        "Rainfall Regime Homogeneity\nStations with Distinct Seasonality Highlighted in Red",
        fontsize=18,
    )
    axis.legend(loc="lower left", fontsize=14)
    axis.tick_params(labelsize=18)
    fig.tight_layout()
    outputs.append(_save(fig, destination / "FigureS4.png"))
    return outputs


def render_ocean_figures(
    config: ProjectConfig,
    rainfall: RainfallProducts,
    ocean: OceanProducts,
    atmosphere: AtmosphereProducts,
    destination: Path,
) -> list[Path]:
    _style()
    series = pd.read_csv(ocean.series, index_col="year")
    fields = xr.open_dataset(ocean.annual_fields)
    composites = xr.open_dataset(ocean.composites)
    atmospheric_fields = xr.open_dataset(atmosphere.annual_fields)
    outputs: list[Path] = []

    monthly = pd.read_csv(ocean.monthly_series, index_col="time", parse_dates=True)

    def legacy_monthly_anomaly(column: str) -> pd.Series:
        values = monthly[column].rolling(window=3, center=False).mean()
        climate = values.loc["1980":"2010"].groupby(values.loc["1980":"2010"].index.month).mean()
        return values - values.index.month.map(climate)

    raw_monthly = legacy_monthly_anomaly("sst_c")
    detrended_monthly = legacy_monthly_anomaly("sst_detrended_c")
    trend_values = raw_monthly.loc["1980":"2023"].dropna()
    decimal_year = np.asarray([date.year + date.dayofyear / 365.25 for date in trend_values.index])
    fit = linregress(decimal_year, trend_values.to_numpy())
    fitted = fit.intercept + fit.slope * decimal_year
    p90 = float(np.percentile(detrended_monthly.dropna(), 90))
    plt.rcParams.update({"font.size": 20})
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), sharex=True)
    axis = axes[0]
    axis.plot(raw_monthly.index, raw_monthly, color="k", linewidth=1.8, label="SST")
    axis.plot(
        trend_values.index,
        fitted,
        color="blue",
        linestyle="--",
        linewidth=3,
        label="Trend 1980–2023",
    )
    axis.axhline(p90, color="lightgrey", linestyle="--", linewidth=2, label="Percentile 90")
    axis.axhline(0, color="grey", alpha=0.5, linewidth=1)
    axis.fill_between(
        raw_monthly.index,
        p90,
        raw_monthly,
        where=raw_monthly > p90,
        color="red",
        alpha=0.8,
    )
    axis.set_title("a)", loc="left", fontsize=24)
    axis.set_ylabel("SST anomaly (°C)", fontsize=22)
    axis.set_ylim([-1.25, 1.25])
    axis.legend(loc="upper left", fontsize=16)
    axis.spines[["top", "right"]].set_visible(False)
    axis = axes[1]
    axis.plot(
        detrended_monthly.index,
        detrended_monthly,
        color="k",
        linewidth=1.8,
        label="Detrended SST",
    )
    axis.axhline(p90, color="lightgrey", linestyle="--", linewidth=2, label="Percentile 90")
    axis.axhline(0, color="grey", alpha=0.5, linewidth=1)
    axis.fill_between(
        detrended_monthly.index,
        p90,
        detrended_monthly,
        where=detrended_monthly > p90,
        color="red",
        alpha=0.8,
    )
    axis.set_title("b)", loc="left", fontsize=24)
    axis.set_xlabel("Period", fontsize=22)
    axis.set_ylabel("SST anomaly (°C)", fontsize=22)
    axis.set_ylim([-1.25, 1.25])
    axis.legend(loc="upper left", fontsize=16)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    outputs.append(_save(fig, destination / "Figure2.png", dpi=360, pad_inches=0.1))

    outputs.append(
        _legacy_four_maps(
            [
                composites[f"detrended_{label}"] - composites["detrended_neutral"]
                for label, _title, _color in EVENT_CLASSES
            ],
            destination / "Figure3.png",
            levels=np.arange(-0.6, 0.61, 0.1),
            cmap="bwr",
            colorbar_label="SST (ºC)",
        )
    )

    raw_sst = open_canonical(config.paths.raw_era5 / "Era5_Dataset_sst.nc", chunks={"time": 195})
    sst_c = raw_sst.sst - 273.15
    sawp_monthly = (
        sst_c.where(
            (sst_c.lon >= -34) & (sst_c.lon < -25) & (sst_c.lat > -12.5) & (sst_c.lat <= -4),
            drop=True,
        )
        .mean(("lat", "lon"), skipna=True)
        .compute()
        .to_series()
    )
    sawp_monthly.index = pd.to_datetime(sawp_monthly.index)
    s7_sawp_monthly = (
        sst_c.where(
            (sst_c.lon >= -34) & (sst_c.lon < -25) & (sst_c.lat > -12.5) & (sst_c.lat <= -4),
            drop=True,
        )
        .mean(("lat", "lon"), skipna=True)
        .compute()
        .to_series()
    )
    s7_sawp_monthly.index = pd.to_datetime(s7_sawp_monthly.index)
    s7_reference = s7_sawp_monthly.loc["1980":"2010"]
    s7_climatology = s7_reference.groupby(s7_reference.index.month).mean()
    s7_sst_anomaly = s7_sawp_monthly - s7_sawp_monthly.index.month.map(s7_climatology)
    rainy_values = sawp_monthly[sawp_monthly.index.month.isin([3, 4, 5, 6])]
    first_period = rainy_values[rainy_values.index.year < 1990]
    second_period = rainy_values[rainy_values.index.year > 1990]
    april = sst_c.where(sst_c.time.dt.month == 4, drop=True)
    first_april = april.where(april.time.dt.year < 1990, drop=True).mean("time").compute()
    second_april = april.where(april.time.dt.year > 1990, drop=True).mean("time").compute()
    first_mean, first_std = float(first_period.mean()), float(first_period.std(ddof=0))
    second_mean, second_std = float(second_period.mean()), float(second_period.std(ddof=0))
    x_pdf = np.linspace(26.5, 30, 200)
    first_pdf = np.exp(-0.5 * ((x_pdf - first_mean) / first_std) ** 2) / (
        first_std * np.sqrt(2 * np.pi)
    )
    second_pdf = np.exp(-0.5 * ((x_pdf - second_mean) / second_std) ** 2) / (
        second_std * np.sqrt(2 * np.pi)
    )
    spacing = x_pdf[1] - x_pdf[0]
    first_pdf /= first_pdf.sum() * spacing
    second_pdf /= second_pdf.sum() * spacing
    first_p90 = float(np.percentile(first_period, 90))
    second_p90 = float(np.percentile(second_period, 90))
    fig = plt.figure(figsize=(13, 14))
    grid_spec = fig.add_gridspec(3, 2, height_ratios=[1, 1.5, 1.2], width_ratios=[1, 1])
    pdf_axis = fig.add_subplot(grid_spec[0, :])
    pdf_axis.plot(x_pdf, first_pdf, color="#1f77b4", linewidth=2.5, label="1961–1990")
    pdf_axis.plot(x_pdf, second_pdf, color="#ff7f0e", linewidth=2.5, label="1991–2020")
    pdf_axis.fill_between(x_pdf, first_pdf, color="#1f77b4", alpha=0.15)
    pdf_axis.fill_between(x_pdf, second_pdf, color="#ff7f0e", alpha=0.15)
    pdf_axis.axvline(first_p90, color="#1f77b4", linestyle="--", linewidth=2, alpha=0.8)
    pdf_axis.axvline(second_p90, color="#ff7f0e", linestyle="--", linewidth=2, alpha=0.8)
    pdf_axis.text(first_p90 + 0.02, max(first_pdf) * 0.85, "P90", color="#1f77b4", fontsize=10)
    pdf_axis.text(second_p90 + 0.02, max(second_pdf) * 0.85, "P90", color="#ff7f0e", fontsize=10)
    pdf_axis.set_xlim(26.5, 30)
    pdf_axis.set_ylim(0, 1.25)
    pdf_axis.set_title("(a) SAWP Temperature PDF", fontsize=13, loc="left", color="dimgrey")
    pdf_axis.set_xlabel("SAWP Temperature (°C)", fontsize=13)
    pdf_axis.set_ylabel("Probability Density", fontsize=13)
    pdf_axis.legend(frameon=False, fontsize=10)
    pdf_axis.tick_params(labelsize=11)
    pdf_axis.spines[["top", "right"]].set_visible(False)
    projection = ccrs.PlateCarree()
    extent = [
        float(sst_c.lon.min()),
        float(sst_c.lon.max()),
        float(sst_c.lat.min()),
        float(sst_c.lat.max()),
    ]
    difference_axis = fig.add_subplot(grid_spec[1, 0], projection=projection)
    difference = second_april - first_april
    difference_image = difference_axis.contourf(
        difference.lon,
        difference.lat,
        difference,
        levels=np.linspace(-1.5, 1.5, 21),
        cmap="coolwarm",
        extend="both",
    )
    difference_axis.coastlines(resolution="50m", color="black", linewidth=0.8)
    difference_axis.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="gray")
    difference_axis.add_feature(cfeature.LAND, color="lightgrey", zorder=0)
    difference_axis.set_extent(extent)
    difference_axis.set_title(
        "(b) ΔSST (1991–2020 minus 1961–1990)",
        fontsize=13,
        loc="left",
        color="dimgrey",
    )
    color_axis = fig.add_axes([0.45, 0.5, 0.02, 0.10])
    difference_colorbar = fig.colorbar(difference_image, cax=color_axis)
    difference_colorbar.set_label("ΔSST (°C)", fontsize=11, labelpad=8)
    difference_colorbar.ax.tick_params(labelsize=9)
    difference_colorbar.outline.set_visible(False)
    contour_axis = fig.add_subplot(grid_spec[1, 1], projection=projection)
    contour_axis.set_extent(extent)
    contour_axis.add_feature(cfeature.LAND, facecolor="white")
    contour_axis.add_feature(cfeature.COASTLINE, linewidth=0.8, edgecolor="grey")
    contour_axis.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor="lightgrey")
    first_contour = contour_axis.contour(
        first_april.lon,
        first_april.lat,
        first_april,
        levels=[28.5],
        colors="royalblue",
        linewidths=2,
        transform=projection,
    )
    second_contour = contour_axis.contour(
        second_april.lon,
        second_april.lat,
        second_april,
        levels=[28.5],
        colors="darkorange",
        linewidths=2,
        transform=projection,
    )
    contour_axis.clabel(first_contour, fmt="28.5°C", inline=True, fontsize=9, colors="royalblue")
    contour_axis.clabel(second_contour, fmt="28.5°C", inline=True, fontsize=9, colors="darkorange")
    contour_axis.legend(
        handles=[
            Line2D([0], [0], color="royalblue", lw=2),
            Line2D([0], [0], color="darkorange", lw=2),
        ],
        labels=["1961–1990", "1991–2020"],
        loc="lower left",
        frameon=False,
        fontsize=9,
    )
    contour_axis.set_title(
        "(c) 28.5°C Isotherm Extension",
        fontsize=13,
        loc="left",
        color="dimgrey",
    )
    monthly_axis = fig.add_subplot(grid_spec[2, :])
    years = sorted(set(sawp_monthly.index.year))
    year_cmap = plt.colormaps["PRGn"].resampled(len(years))
    for index, year in enumerate(years):
        values = sawp_monthly[sawp_monthly.index.year == year]
        if len(values) == 12:
            interpolator = interp1d(np.arange(1, 13), values.to_numpy(), kind="cubic")
            x_smooth = np.linspace(1, 12, 1000)
            y_smooth = interpolator(x_smooth)
            opacity = np.where(y_smooth <= 28.5, 0.1, 0.8)
            for segment in range(len(x_smooth) - 1):
                monthly_axis.plot(
                    x_smooth[segment : segment + 2],
                    y_smooth[segment : segment + 2],
                    color=year_cmap(index),
                    alpha=opacity[segment],
                    linewidth=1.5,
                )
    monthly_axis.axhline(28.5, color="lightgrey", linestyle="--", linewidth=1.8, alpha=0.7)
    monthly_axis.set_title(
        "(d) Monthly SST Evolution (SAWP)",
        fontsize=13,
        loc="left",
        color="dimgrey",
    )
    monthly_axis.set_xlabel("Month", fontsize=13)
    monthly_axis.set_ylabel("SST (°C)", fontsize=13)
    monthly_axis.set_xticks(np.arange(1, 13))
    monthly_axis.set_xticklabels(
        ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    )
    monthly_axis.spines[["top", "right"]].set_visible(False)
    monthly_axis.tick_params(labelsize=11)
    scalar = cm.ScalarMappable(
        cmap=year_cmap, norm=mcolors.Normalize(vmin=min(years), vmax=max(years))
    )
    inset = inset_axes(monthly_axis, width="20%", height="3%", loc="upper right", borderpad=1.0)
    year_colorbar = fig.colorbar(scalar, cax=inset, orientation="horizontal")
    year_colorbar.set_label("Years", fontsize=10, labelpad=5)
    year_colorbar.set_ticks([years[0], years[-1]], labels=[str(years[0]), str(years[-1])])
    year_colorbar.ax.tick_params(labelsize=9)
    fig.subplots_adjust(hspace=0.4, wspace=0.05)
    fig.suptitle(
        "SST Variability and Trends over the SAWP Region",
        fontsize=16,
        color="grey",
        y=0.95,
        x=0.3,
    )
    outputs.append(_save(fig, destination / "Figure8.png", dpi=400))
    raw_sst.close()

    raw_class = "sst_raw_class"
    detrended_class = "sst_class"
    diff_sst = _column_composite(
        fields.sst_anomaly, series, "extreme_warm", raw_class
    ) - _column_composite(fields.sst_detrended, series, "extreme_warm", detrended_class)
    diff_pw = _column_composite(
        atmospheric_fields.raw_precipitable_water_mid,
        series,
        "extreme_warm",
        raw_class,
    ) - _column_composite(
        atmospheric_fields.precipitable_water_mid,
        series,
        "extreme_warm",
        detrended_class,
    )
    diff_tu = _column_composite(
        atmospheric_fields.raw_transport_u_mid,
        series,
        "extreme_warm",
        raw_class,
    ) - _column_composite(
        atmospheric_fields.transport_u_mid,
        series,
        "extreme_warm",
        detrended_class,
    )
    diff_tv = _column_composite(
        atmospheric_fields.raw_transport_v_mid,
        series,
        "extreme_warm",
        raw_class,
    ) - _column_composite(
        atmospheric_fields.transport_v_mid,
        series,
        "extreme_warm",
        detrended_class,
    )
    diff_mfc = _column_composite(
        atmospheric_fields.raw_moisture_flux_convergence_low,
        series,
        "extreme_warm",
        raw_class,
    ) - _column_composite(
        atmospheric_fields.moisture_flux_convergence_low,
        series,
        "extreme_warm",
        detrended_class,
    )
    diff_theta = _column_composite(
        atmospheric_fields.raw_theta_e_cross,
        series,
        "extreme_warm",
        raw_class,
    ) - _column_composite(
        atmospheric_fields.theta_e_cross,
        series,
        "extreme_warm",
        detrended_class,
    )
    diff_u = _column_composite(
        atmospheric_fields.raw_u_cross, series, "extreme_warm", raw_class
    ) - _column_composite(atmospheric_fields.u_cross, series, "extreme_warm", detrended_class)
    diff_w = _column_composite(
        atmospheric_fields.raw_omega_cross, series, "extreme_warm", raw_class
    ) - _column_composite(atmospheric_fields.omega_cross, series, "extreme_warm", detrended_class)

    fig = plt.figure(figsize=(12, 10))
    grid_spec = fig.add_gridspec(2, 2)
    axes = [
        fig.add_subplot(grid_spec[0, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(grid_spec[0, 1], projection=ccrs.PlateCarree()),
        fig.add_subplot(grid_spec[1, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(grid_spec[1, 1]),
    ]
    fig.subplots_adjust(
        left=0.08,
        right=0.95,
        bottom=0.16,
        top=0.96,
        hspace=0.24,
        wspace=0.15,
    )
    for axis, value, levels, cmap, panel, title in (
        (axes[0], diff_sst, np.arange(-0.5, 0.51, 0.1), "bwr", "a", "SST (degC)"),
        (
            axes[1],
            diff_pw,
            np.arange(-1.0, 1.01, 0.2),
            "BrBG",
            "b",
            "Prec. Water (mm)",
        ),
    ):
        image = _legacy_map_axis(
            axis,
            value,
            levels=levels,
            cmap=cmap,
            panel=panel,
            alpha=0.7,
        )
        axis.set_title(f"{panel})", loc="left", fontsize=18)
        color_axis = inset_axes(
            axis,
            width="5%",
            height="30%",
            loc="lower right",
            bbox_to_anchor=(-0.15, 0.0, 1, 1),
            bbox_transform=axis.transAxes,
            borderpad=0.5,
        )
        colorbar = fig.colorbar(image, cax=color_axis, orientation="vertical")
        colorbar.ax.tick_params(labelsize=8)
        colorbar.ax.set_title(title, fontsize=8)

    image_c = _legacy_map_axis(
        axes[2],
        diff_mfc,
        levels=np.arange(-30, 31, 10) / 1e6,
        cmap="PuOr",
        panel="c",
    )
    step = 10
    quiver_c = axes[2].quiver(
        diff_tu.lon[::step],
        diff_tu.lat[::step],
        diff_tu[::step, ::step],
        diff_tv[::step, ::step],
        transform=ccrs.PlateCarree(),
        scale=300,
        width=0.0025,
    )
    color_axis_c = inset_axes(
        axes[2],
        width="5%",
        height="30%",
        loc="lower right",
        bbox_to_anchor=(-0.15, 0.0, 1, 1),
        bbox_transform=axes[2].transAxes,
        borderpad=0.5,
    )
    colorbar_c = fig.colorbar(image_c, cax=color_axis_c, orientation="vertical")
    colorbar_c.ax.tick_params(labelsize=8)
    colorbar_c.ax.set_title("Mois. flux conv.\n(kg m⁻² s⁻¹)", fontsize=8)
    axes[2].quiverkey(
        quiver_c,
        X=0.50,
        Y=-0.15,
        U=10.0,
        label="10 kg m⁻¹ s⁻¹",
        labelpos="E",
        coordinates="axes",
        fontproperties={"size": 9},
    )

    # The legacy Figure 9 standardizes both circulation components on their
    # native pressure-longitude grid before vertical interpolation.
    u_scale = float(np.nanstd(diff_u.to_numpy())) or 1.0
    w_scale = float(np.nanstd(diff_w.to_numpy())) or 1.0
    pressure_grid = np.linspace(100, 1000, 60)
    section_values = {}
    for name, value in {"theta": diff_theta, "u": diff_u, "w": diff_w}.items():
        section_values[name] = np.column_stack(
            [
                np.interp(pressure_grid, value.level, value[:, column])
                for column in range(value.sizes["lon"])
            ]
        )
    longitude, pressure = np.meshgrid(diff_theta.lon, pressure_grid)
    image_d = axes[3].contourf(
        longitude,
        pressure,
        section_values["theta"],
        cmap="RdYlBu_r",
        extend="both",
        levels=np.arange(-1.0, 1.01, 0.2),
        alpha=0.4,
    )
    quiver_d = axes[3].quiver(
        longitude[::4, ::5],
        pressure[::4, ::5],
        section_values["u"][::4, ::5] / u_scale,
        -section_values["w"][::4, ::5] / w_scale,
        scale=100,
        width=0.0025,
        color="0.35",
    )
    axes[3].set(xlim=(-60, -10), ylim=(1000, 200))
    axes[3].set_xticks(np.linspace(float(diff_theta.lon.min()), float(diff_theta.lon.max()), 5))
    axes[3].fill_betweenx([1000, 200], -40, -30, color="gray", alpha=0.2)
    axes[3].set_title("d)", loc="left", fontsize=18)
    axes[3].yaxis.tick_right()
    axes[3].yaxis.set_label_position("right")
    axes[3].set_yticks([1000, 850, 700, 500, 300, 200])
    axes[3].tick_params(axis="y", left=False, labelleft=False, right=True, labelright=True)
    axes[3].spines["left"].set_visible(False)
    axes[3].set_ylabel("Pressure (hPa)", fontsize=14, labelpad=10)
    color_axis_d = inset_axes(
        axes[3],
        width="5%",
        height="30%",
        loc="lower right",
        bbox_to_anchor=(-0.15, 0.0, 1, 1),
        bbox_transform=axes[3].transAxes,
        borderpad=0.5,
    )
    colorbar_d = fig.colorbar(image_d, cax=color_axis_d, orientation="vertical")
    colorbar_d.ax.tick_params(labelsize=8)
    colorbar_d.ax.set_title(r"$\theta_e$ (K)", fontsize=8)
    axes[3].quiverkey(
        quiver_d,
        X=0.50,
        Y=-0.15,
        U=1.0,
        label="1 sigma (standardized u/w)",
        labelpos="E",
        coordinates="axes",
        fontproperties={"size": 9},
    )
    fig.text(0.5, 0.035, "Longitude", ha="center", fontsize=20)
    fig.text(0.025, 0.5, "Latitude", va="center", rotation="vertical", fontsize=20)
    outputs.append(_save(fig, destination / "Figure9.png"))

    monthly_rain_source = pd.read_csv(rainfall.regional, index_col=0, parse_dates=True)[
        "precipitation_mm"
    ]
    monthly_rain_source.index = monthly_rain_source.index.to_period("M").to_timestamp()
    # Figure S5 uses the spatial maximum inside the inclusive SAWP box for
    # every month. It must not reuse the regional-mean SST index.
    s5_mask = (
        (sst_c.lon >= -34.0) & (sst_c.lon <= -25.0) & (sst_c.lat >= -12.5) & (sst_c.lat <= -4.0)
    )
    monthly_sst_source = (
        sst_c.where(s5_mask, drop=True).max(("lat", "lon"), skipna=True).compute().to_series()
    )
    with xr.open_dataset(ocean.trend_coefficients) as sst_coefficients:
        centered_time = xr.DataArray(
            np.arange(sst_c.sizes["time"], dtype=float),
            dims="time",
            coords={"time": sst_c.time},
        )
        centered_time = centered_time - centered_time.mean("time")
        detrended_sst_grid = sst_c - sst_coefficients.sst_slope_per_month.load() * centered_time
        monthly_sst_detrended = (
            detrended_sst_grid.where(s5_mask, drop=True)
            .max(("lat", "lon"), skipna=True)
            .compute()
            .to_series()
        )
    monthly_sst_source.index = monthly_sst_source.index.to_period("M").to_timestamp()
    monthly_sst_detrended.index = monthly_sst_detrended.index.to_period("M").to_timestamp()

    def detrend_by_calendar_month(value: pd.Series) -> pd.Series:
        result = pd.Series(index=value.index, dtype=float)
        for month_number in range(1, 13):
            selected = value[value.index.month == month_number].dropna()
            x_values = np.arange(len(selected), dtype=float)
            fit = linregress(x_values, selected.to_numpy())
            result.loc[selected.index] = (
                selected - (fit.intercept + fit.slope * x_values) + selected.mean()
            )
        return result

    def monthly_anomaly(value: pd.Series) -> pd.Series:
        reference = value.loc["1980":"2010"]
        climate = reference.groupby(reference.index.month).mean()
        return value - value.index.month.map(climate)

    def maxima_table(sst_value: pd.Series, rain_value: pd.Series) -> pd.DataFrame:
        joined_monthly = pd.concat(
            [monthly_anomaly(sst_value).rename("sst"), monthly_anomaly(rain_value).rename("rain")],
            axis=1,
        ).dropna()
        joined_monthly = joined_monthly[
            joined_monthly.index.month.isin(config.period.season_months)
        ]
        rows = []
        for year, group in joined_monthly.groupby(joined_monthly.index.year):
            rows.append(
                {
                    "year": int(year),
                    "rain_max_anom_amjj": group.rain.max(),
                    "sst_max_anom_amjj": group.sst.max(),
                }
            )
        return pd.DataFrame(rows).set_index("year")

    original_table = maxima_table(monthly_sst_source, monthly_rain_source)
    detrended_table = maxima_table(
        monthly_sst_detrended, detrend_by_calendar_month(monthly_rain_source)
    )

    def statistics(table: pd.DataFrame, label: str) -> dict[str, float | str]:
        x_values = table.rain_max_anom_amjj.to_numpy(float)
        y_values = table.sst_max_anom_amjj.to_numpy(float)
        pearson_r, pearson_p = pearsonr(x_values, y_values)
        spearman_r, spearman_p = spearmanr(x_values, y_values)
        kendall_r, kendall_p = kendalltau(x_values, y_values)
        regression = linregress(x_values, y_values)
        return {
            "case": label,
            "n": len(table),
            "Pearson r": pearson_r,
            "Pearson p": pearson_p,
            "R²": pearson_r**2,
            "Slope (°C/mm)": regression.slope,
            "Slope p": regression.pvalue,
            "Spearman rho": spearman_r,
            "Spearman p": spearman_p,
            "Kendall tau": kendall_r,
            "Kendall p": kendall_p,
        }

    original_statistics = statistics(original_table, "Original")
    detrended_statistics = statistics(detrended_table, "Detrended")
    fig = plt.figure(figsize=(18, 20))
    grid_spec = fig.add_gridspec(
        3,
        2,
        height_ratios=[3.2, 1.8, 1.15],
        hspace=0.36,
        wspace=0.22,
    )
    scatter_axes = [fig.add_subplot(grid_spec[0, 0]), fig.add_subplot(grid_spec[0, 1])]

    def p_text(value: float) -> str:
        return "<0.001" if value < 0.001 else f"{value:.3f}"

    for axis, table, stats, panel in zip(
        scatter_axes,
        [original_table, detrended_table],
        [original_statistics, detrended_statistics],
        "ab",
        strict=True,
    ):
        x_values = table.sst_max_anom_amjj / table.sst_max_anom_amjj.std(ddof=1)
        y_values = table.rain_max_anom_amjj / table.rain_max_anom_amjj.std(ddof=1)
        regression = linregress(x_values, y_values)
        x_line = np.linspace(x_values.min(), x_values.max(), 100)
        axis.scatter(x_values, y_values, s=70)
        axis.plot(x_line, regression.intercept + regression.slope * x_line, linewidth=2.5)
        axis.axhline(0, linestyle="--", linewidth=1.2)
        axis.axvline(0, linestyle="--", linewidth=1.2)
        axis.grid(alpha=0.3)
        axis.set_title(
            f"{panel}) {stats['case']}\n"
            f"r = {stats['Pearson r']:.2f}, p = {p_text(float(stats['Pearson p']))}, "
            f"n = {stats['n']}"
        )
        axis.set_xlabel("Normalized SST anomaly (σ)", fontsize=18)
        axis.set_ylabel("Normalized rainfall anomaly (σ)", fontsize=18)
        axis.set_xlim(-3, 3)
        axis.set_ylim(-2, np.ceil(max(np.max(np.abs(x_values)), np.max(np.abs(y_values)))))
        axis.set_aspect("equal")
    bar_axis = fig.add_subplot(grid_spec[1, :])
    top_ten = detrended_table.sort_values("rain_max_anom_amjj", ascending=False).head(10)
    top_ten = top_ten.sort_values("rain_max_anom_amjj")
    years = top_ten.index.to_numpy(int)
    positions = np.arange(len(years))
    width = 0.38
    detrended_values = top_ten.rain_max_anom_amjj.to_numpy()
    original_values = original_table.loc[years, "rain_max_anom_amjj"].to_numpy()
    bar_axis.bar(
        positions - width / 2,
        detrended_values,
        width=width,
        color="tab:blue",
        label="Detrended",
    )
    bar_axis.bar(
        positions + width / 2,
        original_values,
        width=width,
        color="tab:orange",
        label="Original",
    )
    bar_axis.set_xticks(
        positions,
        [f"{index + 1}\n{year}" for index, year in enumerate(years)],
        fontsize=16,
    )
    bar_axis.set_ylabel("Maximum rainfall anomaly (mm)", fontsize=18)
    bar_axis.set_xlabel(
        "Ranking based on detrended rainfall anomaly\n(year shown below rank)",
        fontsize=18,
    )
    bar_axis.set_title(
        "c) Ten largest annual rainfall maxima: original versus detrended",
        fontsize=18,
        pad=8,
    )
    bar_axis.grid(axis="y", alpha=0.3)
    bar_axis.legend(fontsize=16)
    for index, (detrended_value, original_value) in enumerate(
        zip(detrended_values, original_values, strict=True)
    ):
        bar_axis.text(
            index - width / 2,
            detrended_value,
            f"{detrended_value:.0f}",
            ha="center",
            va="bottom",
            fontsize=14,
        )
        bar_axis.text(
            index + width / 2,
            original_value,
            f"{original_value:.0f}",
            ha="center",
            va="bottom",
            fontsize=14,
        )
    table_axis = fig.add_subplot(grid_spec[2, :])
    table_axis.axis("off")
    statistics_table = pd.DataFrame([original_statistics, detrended_statistics])
    columns = [
        "case",
        "n",
        "Pearson r",
        "Pearson p",
        "R²",
        "Slope (°C/mm)",
        "Slope p",
        "Spearman rho",
        "Spearman p",
        "Kendall tau",
        "Kendall p",
    ]
    display = statistics_table[columns].copy()
    for column in columns[1:]:
        if column == "n":
            display[column] = display[column].astype(int).astype(str)
        elif "p" in column.lower():
            display[column] = display[column].map(lambda value: p_text(float(value)))
        else:
            display[column] = display[column].map(lambda value: f"{float(value):.3f}")
    table = table_axis.table(
        cellText=display.values,
        colLabels=display.columns,
        loc="center",
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(16)
    table.scale(1, 1.65)
    table_axis.set_title("d) Statistical summary and correlation tests", fontsize=18, pad=8)
    fig.suptitle(
        "Annual maximum ENEB rainfall anomaly versus annual maximum SAWP SST anomaly\n"
        "Independent annual maxima within AMJJ for rainfall and SST anomalies",
        fontsize=24,
        y=0.98,
    )
    outputs.append(_save(fig, destination / "FigureS5.png"))

    monthly_rain = pd.read_csv(rainfall.regional, index_col=0, parse_dates=True)["precipitation_mm"]
    monthly_sst = s7_sst_anomaly.copy()
    monthly_sst.index = monthly_sst.index.to_period("M").to_timestamp("M")
    monthly_rain.index = monthly_rain.index.to_period("M").to_timestamp("M")
    decimal_year = monthly_rain.index.year + (monthly_rain.index.dayofyear - 1) / 365.25
    rain_valid = monthly_rain.notna()
    rain_fit = linregress(decimal_year[rain_valid], monthly_rain.loc[rain_valid].to_numpy())
    rain_processed = (
        monthly_rain
        - (rain_fit.slope * decimal_year + rain_fit.intercept)
        + monthly_rain.loc[rain_valid].mean()
    )
    rain_reference = rain_processed.loc["1980":"2010"]
    rain_climate = rain_reference.groupby(rain_reference.index.month).mean()
    rain_anomaly = rain_processed - rain_processed.index.month.map(rain_climate)
    rain_anomaly = rain_anomaly / rain_anomaly.loc["1980":"2010"].std(ddof=1)
    aligned_monthly = pd.concat([monthly_sst.rename("sst"), rain_anomaly.rename("rain")], axis=1)
    rows = []
    ranking_values = pd.Series(index=monthly_sst.index, dtype=float)
    for month_number in range(1, 13):
        selected = monthly_sst[monthly_sst.index.month == month_number].dropna()
        selected_decimal_year = selected.index.year + (selected.index.dayofyear - 1) / 365.25
        selected_fit = linregress(selected_decimal_year, selected.to_numpy())
        ranking_values.loc[selected.index] = (
            selected.to_numpy()
            - (selected_fit.slope * selected_decimal_year + selected_fit.intercept)
            + selected.mean()
        )
    amjj_ranking = ranking_values[ranking_values.index.month.isin(config.period.season_months)]
    ranked_sst = amjj_ranking.groupby(amjj_ranking.index.year).mean().dropna().sort_values()
    extreme_count = max(1, round(0.10 * len(ranked_sst)))
    moderate_count = max(1, round(0.15 * len(ranked_sst)))
    warm_rank_years = {
        "Moderately warm": ranked_sst.index[
            -(extreme_count + moderate_count) : -extreme_count
        ].to_numpy(int),
        "Extremely warm": ranked_sst.index[-extreme_count:].to_numpy(int),
    }
    class_specs = (
        ("moderate_warm", "Moderately warm"),
        ("extreme_warm", "Extremely warm"),
    )
    for _event_class, class_title in class_specs:
        for year in warm_rank_years[class_title]:
            annual = aligned_monthly[aligned_monthly.index.year == year]
            amjj = annual.loc[annual.index.month.isin(config.period.season_months), "sst"].dropna()
            if amjj.empty:
                continue
            peak = amjj.idxmax()
            for lag in (-2, -1, 0, 1, 2):
                date = (peak + pd.DateOffset(months=lag)).to_period("M").to_timestamp("M")
                previous = (date - pd.DateOffset(months=1)).to_period("M").to_timestamp("M")
                if date not in aligned_monthly.index or previous not in aligned_monthly.index:
                    continue
                rows.append(
                    {
                        "year": year,
                        "class": class_title,
                        "lag": lag,
                        "cooling": aligned_monthly.loc[previous, "sst"]
                        - aligned_monthly.loc[date, "sst"],
                        "rain": aligned_monthly.loc[date, "rain"],
                    }
                )
    lag_frame = pd.DataFrame(rows)
    lag_frame["cooling_normalized"] = lag_frame.cooling / lag_frame.cooling.std(ddof=1)
    composite = (
        lag_frame.groupby(["class", "lag"])
        .agg(
            rain_mean=("rain", "mean"),
            rain_std=("rain", "std"),
            cooling_mean=("cooling_normalized", "mean"),
            cooling_std=("cooling_normalized", "std"),
            n=("year", "nunique"),
        )
        .reset_index()
    )
    composite["rain_se"] = composite.rain_std / np.sqrt(composite.n)
    composite["cooling_se"] = composite.cooling_std / np.sqrt(composite.n)
    fig, axes = plt.subplots(1, 2, figsize=(12.80, 5.15), sharey=True)
    for axis, (_event_class, class_title), panel_title in zip(
        axes,
        class_specs,
        ["(a) Moderately warm", "(b) Extremely warm"],
        strict=True,
    ):
        values = composite[composite["class"] == class_title].sort_values("lag")
        x_values = values.lag.to_numpy(float)
        axis.bar(
            x_values,
            values.rain_mean,
            width=0.55,
            alpha=0.60,
            label="ENEB rainfall anomaly",
        )
        axis.errorbar(
            x_values,
            values.rain_mean,
            yerr=values.rain_se,
            fmt="none",
            linewidth=1.0,
            capsize=3,
            alpha=0.8,
        )
        axis.plot(
            x_values,
            values.cooling_mean,
            marker="o",
            linewidth=2.5,
            label="Monthly SAWP SST cooling",
        )
        axis.fill_between(
            x_values,
            values.cooling_mean - values.cooling_se,
            values.cooling_mean + values.cooling_se,
            alpha=0.20,
        )
        axis.axhline(0, color="black", linewidth=1.1)
        axis.axvline(0, color="gray", linestyle="--", linewidth=1.1)
        axis.set_title(panel_title, fontsize=14)
        axis.set_xlabel("Months relative to SAWP SST peak", fontsize=16)
        axis.set_xticks([-2, -1, 0, 1, 2])
        axis.grid(axis="y", alpha=0.3)
        axis.text(
            0.03,
            0.95,
            f"n = {int(values.n.max())}",
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=14,
            bbox={"facecolor": "white", "edgecolor": "gray", "alpha": 0.85},
        )
    for axis in axes:
        axis.tick_params(axis="both", labelsize=14)
    axes[0].set_ylabel(
        "Normalized anomaly (standard deviations)\nMonthly SST cooling (positive = ocean cooling)",
        fontsize=16,
    )
    axes[0].legend(frameon=False, loc="best")
    fig.suptitle(
        "Composite centered on maximum SAWP SST anomaly\n"
        "ENEB rainfall anomaly and monthly SAWP SST cooling",
        fontsize=16,
    )
    fig.tight_layout()
    outputs.append(_save(fig, destination / "FigureS7.png"))
    fields.close()
    composites.close()
    atmospheric_fields.close()
    return outputs


def render_atmosphere_figures(
    config: ProjectConfig,
    rainfall: RainfallProducts,
    ocean: OceanProducts,
    atmosphere: AtmosphereProducts,
    destination: Path,
) -> list[Path]:
    _style()
    series = pd.read_csv(ocean.series, index_col="year")
    rain = pd.read_csv(rainfall.annual, index_col="year")
    ocean_fields = xr.open_dataset(ocean.annual_fields)
    fields = xr.open_dataset(atmosphere.annual_fields)
    outputs: list[Path] = []

    rain_series = xr.DataArray(
        rain.rainfall_detrended_mm.to_numpy(),
        dims="year",
        coords={"year": rain.index.to_numpy(int)},
    )
    aligned_sst, aligned_rain = xr.align(ocean_fields.sst_detrended, rain_series, join="inner")
    aligned_years = aligned_sst.year.to_numpy().astype(int)
    regional_sst = series.loc[aligned_years, "sst_detrended_c"].to_numpy(float)
    sst_standardized = (regional_sst - np.nanmean(regional_sst)) / np.nanstd(regional_sst, ddof=1)
    rain_values = aligned_rain.to_numpy()
    rain_standardized = (rain_values - np.nanmean(rain_values)) / np.nanstd(rain_values, ddof=1)
    # The legacy map selects rainfall tails, whereas the scatter regression
    # selects SST tails. Both selections deliberately exclude neutral years.
    rain_q25, rain_q75 = np.nanquantile(rain_standardized, [0.25, 0.75])
    rainfall_extremes = (rain_standardized < rain_q25) | (rain_standardized > rain_q75)
    extreme_years = aligned_years[rainfall_extremes]
    correlation = xr.corr(
        aligned_sst.sel(year=extreme_years),
        aligned_rain.sel(year=extreme_years),
        dim="year",
    )
    sst_q10, sst_q25, sst_q75, sst_q90 = np.nanquantile(sst_standardized, [0.10, 0.25, 0.75, 0.90])
    classes = np.full(aligned_years.shape, "neutral", dtype="U16")
    classes[sst_standardized <= sst_q10] = "extreme_cold"
    classes[(sst_standardized > sst_q10) & (sst_standardized <= sst_q25)] = "moderate_cold"
    classes[(sst_standardized >= sst_q75) & (sst_standardized < sst_q90)] = "moderate_warm"
    classes[sst_standardized >= sst_q90] = "extreme_warm"
    sst_extremes = classes != "neutral"
    fit = linregress(sst_standardized[sst_extremes], rain_standardized[sst_extremes])
    fig = plt.figure(figsize=(14, 7))
    map_axis = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
    map_axis.set_title("a)", loc="left", fontsize=20)
    map_axis.set_extent([-60, -10, -30, 15], crs=ccrs.PlateCarree())
    map_axis.coastlines(resolution="50m")
    map_axis.add_feature(cfeature.BORDERS)
    map_axis.set_xticks(np.arange(-60, -5, 10), crs=ccrs.PlateCarree())
    map_axis.set_yticks(np.arange(-30, 20, 10), crs=ccrs.PlateCarree())
    map_axis.tick_params(axis="both", labelsize=16)
    states = cfeature.NaturalEarthFeature(
        category="cultural",
        name="admin_1_states_provinces_lakes",
        scale="10m",
        facecolor="none",
    )
    map_axis.add_feature(states, edgecolor="black", linewidth=0.3, alpha=0.3)
    map_axis.set_ylabel("Latitude", fontsize=18)
    map_axis.set_xlabel("Longitude", fontsize=18)
    map_image = map_axis.contourf(
        correlation.lon,
        correlation.lat,
        correlation,
        levels=np.arange(0, 1.1, 0.1),
        transform=ccrs.PlateCarree(),
        extend="min",
        cmap="CMRmap_r",
        alpha=0.5,
    )
    contour_lines = map_axis.contour(
        correlation.lon,
        correlation.lat,
        correlation,
        levels=[0.4, 0.6],
        colors="black",
        linewidths=0.8,
        transform=ccrs.PlateCarree(),
    )
    map_axis.clabel(contour_lines, fmt="%.1f", inline=True, fontsize=10)
    map_colorbar = fig.colorbar(
        map_image,
        ax=map_axis,
        shrink=0.5,
        fraction=0.046,
        pad=0.04,
        orientation="vertical",
    )
    map_colorbar.set_label("Correlation", fontsize=14, color="k")
    map_colorbar.ax.tick_params(labelsize=14)
    scatter_axis = fig.add_subplot(1, 2, 2)
    scatter_axis.set_title("b)", loc="left", fontsize=20)
    scatter_styles = {
        "extreme_cold": ("b", 1.0, "La Niña (forte)"),
        "moderate_cold": ("b", 0.2, "La Niña (moderado)"),
        "neutral": ("lightgrey", 0.2, "Neutro"),
        "moderate_warm": ("r", 0.2, "El Niño (moderado)"),
        "extreme_warm": ("r", 1.0, "El Niño (forte)"),
    }
    for event_class, (color, alpha, label) in scatter_styles.items():
        mask = classes == event_class
        scatter_axis.scatter(
            sst_standardized[mask],
            rain_standardized[mask],
            c=color,
            alpha=alpha,
            label=label,
        )
    mean_rain = float(np.nanmean(rain_standardized))
    scatter_axis.axhline(mean_rain, linestyle="dashed", color="lightgrey", alpha=0.6)
    scatter_axis.axvline(mean_rain, linestyle="dashed", color="lightgrey", alpha=0.6)
    scatter_axis.axvline(mean_rain - 0.7, linestyle="dashed", color="b", alpha=0.6)
    scatter_axis.axvline(mean_rain + 0.7, linestyle="dashed", color="r", alpha=0.3)
    scatter_axis.set_xlim([-3, 3])
    scatter_axis.set_ylim([-3, 3])
    x_values = np.asarray(scatter_axis.get_xlim())
    y_values = fit.intercept + fit.slope * x_values
    scatter_axis.plot(
        x_values,
        y_values,
        "--",
        color="k",
        alpha=0.5,
        label=f"Regressão Linear: y={fit.slope:.2f}x+{fit.intercept:.2f}",
    )
    scatter_axis.text(
        -2.7,
        y_values[0] + (y_values[1] - y_values[0]) * 1.5,
        f"r = {fit.rvalue:.2f}",
        fontsize=20,
        color="white",
        weight="bold",
        bbox={"facecolor": "black", "edgecolor": "none", "alpha": 0.7, "boxstyle": "round,pad=0.3"},
    )
    scatter_axis.set_xlabel("SAWP SST anomaly", fontsize=22)
    scatter_axis.set_ylabel("ENEB Anomaly acumulated Precipitation", fontsize=18)
    scatter_axis.yaxis.set_label_position("right")
    scatter_axis.yaxis.tick_right()
    scatter_axis.tick_params(axis="both", labelsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    outputs.append(_save(fig, destination / "Figure4.png"))

    outputs.append(
        _legacy_four_maps(
            [
                _class_contrast(fields.precipitable_water_mid, series, label)
                for label, _title, _color in EVENT_CLASSES
            ],
            destination / "Figure5.png",
            levels=np.arange(-3.0, 3.01, 0.5),
            cmap="BrBG",
            colorbar_label="Water Vapor Content (mm)",
        )
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12, 10),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    fig.subplots_adjust(hspace=0.01, wspace=0.15)
    image = None
    arrows = []
    for axis, (label, _title, _color), panel in zip(axes.flat, EVENT_CLASSES, "abcd", strict=True):
        convergence = _class_contrast(fields.moisture_flux_convergence_low, series, label)
        u = _class_contrast(fields.transport_u_mid, series, label)
        v = _class_contrast(fields.transport_v_mid, series, label)
        image = _legacy_map_axis(
            axis,
            convergence,
            levels=np.arange(-30, 31, 10) / 1e6,
            cmap="PuOr",
            panel=panel,
        )
        step = 10
        arrows.append(
            axis.quiver(
                u.lon[::step],
                u.lat[::step],
                u[::step, ::step],
                v[::step, ::step],
                transform=ccrs.PlateCarree(),
                scale=300,
                width=0.0025,
            )
        )
    fig.text(0.45, 0.05, "Longitude", ha="center", fontsize=20)
    fig.text(0.045, 0.5, "Latitude", va="center", rotation="vertical", fontsize=20)
    colorbar = fig.colorbar(image, ax=axes, orientation="vertical", shrink=0.5, pad=0.02)
    colorbar.set_label("Moisture flux convergence (kg m⁻² s⁻¹)", size=15)
    colorbar.ax.tick_params(labelsize=12)
    axes.flat[3].quiverkey(
        arrows[3],
        0.70,
        0.055,
        10.0,
        "10 kg m⁻¹ s⁻¹",
        coordinates="figure",
        labelpos="E",
        fontproperties={"size": 12},
    )
    outputs.append(_save(fig, destination / "Figure6.png", dpi=360))

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    fig = plt.figure(figsize=(12, 12))
    grid_spec = fig.add_gridspec(
        2,
        2,
        left=0.10,
        right=0.83,
        bottom=0.34,
        top=0.96,
        hspace=0.12,
        wspace=0.15,
    )
    axes = [fig.add_subplot(grid_spec[index // 2, index % 2]) for index in range(4)]
    pressure_grid = np.linspace(100, 1000, 60)
    theta_values: list[xr.DataArray] = []
    reference_quiver = None
    image = None
    for index, ((label, _title, _color), axis) in enumerate(zip(EVENT_CLASSES, axes, strict=True)):
        theta = _class_contrast(fields.theta_e_cross, series, label)
        u = _class_contrast(fields.u_cross, series, label)
        omega = _class_contrast(fields.omega_cross, series, label)
        theta_values.append(theta)
        interpolated = {}
        for name, value in {"theta": theta, "u": u, "w": omega}.items():
            interpolated[name] = np.column_stack(
                [
                    np.interp(pressure_grid, value.level, value[:, column])
                    for column in range(value.sizes["lon"])
                ]
            )
        image = axis.contourf(
            theta.lon,
            pressure_grid,
            interpolated["theta"],
            levels=np.arange(-3, 3.01, 0.5),
            cmap="RdYlBu_r",
            extend="both",
            alpha=0.4,
        )
        scales = {name: float(np.nanstd(interpolated[name])) or 1.0 for name in ("u", "w")}
        quiver = axis.quiver(
            theta.lon[::8],
            pressure_grid[::2],
            2 * interpolated["u"][::2, ::8] / scales["u"],
            -interpolated["w"][::2, ::8] / scales["w"],
            color="black",
            pivot="mid",
            scale=50,
            width=0.002,
        )
        if index == 0:
            reference_quiver = quiver
        axis.fill_betweenx([1000, 200], -40, -30, color="gray", alpha=0.2)
        axis.set(xlim=(-60, -10), ylim=(1000, 200))
        axis.set_title(f"{'abcd'[index]})", loc="left", fontsize=18)
        axis.set_yticks(np.arange(1000, 200, -200) if index % 2 == 0 else [])
        axis.set_xticks(
            np.linspace(float(theta.lon.min()), float(theta.lon.max()), 5) if index >= 2 else []
        )
        axis.tick_params(labelsize=14)
    fig.text(0.465, 0.295, "Longitude", ha="center", fontsize=18)
    fig.text(
        0.025,
        0.63,
        "Pressure Levels (hPa)",
        va="center",
        rotation="vertical",
        fontsize=18,
    )
    axes[0].quiverkey(
        reference_quiver,
        X=0.15,
        Y=0.255,
        U=2,
        label="1 (Standardized u–w anomalies)",
        coordinates="figure",
        labelpos="E",
        labelsep=0.05,
        fontproperties={"size": 10},
    )
    color_axis = fig.add_axes([0.86, 0.465, 0.018, 0.33])
    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.set_label("Equivalent Potential Temperature (K)", size=14)
    colorbar.ax.tick_params(labelsize=12)

    def layer_mean_difference(value: xr.DataArray) -> float:
        profile = value.sel(lon=slice(-34, -25)).mean("lon", skipna=True)
        pressure = profile.level.to_numpy().astype(float)
        values = profile.to_numpy().astype(float)

        def integrate(top: float, bottom: float) -> float:
            integration_grid = np.unique(
                np.r_[top, pressure[(pressure > top) & (pressure < bottom)], bottom]
            )
            return float(
                np.trapezoid(np.interp(integration_grid, pressure, values), integration_grid)
            )

        return integrate(100, 600) / 500 - integrate(600, 1000) / 400

    bar_axis = fig.add_axes([0.10, 0.065, 0.73, 0.13])
    bar_labels = ["Very cold", "Moderately cold", "Moderately warm", "Extremely warm"]
    bar_values = [layer_mean_difference(value) for value in theta_values]
    bar_axis.bar(
        np.arange(4),
        bar_values,
        color=[color for _label, _title, color in EVENT_CLASSES],
        width=0.55,
        zorder=3,
    )
    bar_axis.axhline(0, color="#555555", linewidth=0.8)
    bar_axis.set_xticks(np.arange(4), bar_labels, fontsize=11)
    bar_axis.set_ylim(-2, 2)
    bar_axis.set_yticks([-2, -1, 0, 1, 2])
    bar_axis.set_ylabel("K", fontsize=12)
    bar_axis.set_title(r"$\theta_e$ gradient", fontsize=15, pad=8)
    bar_axis.set_title("e)", loc="right", fontsize=18, pad=8)
    bar_axis.spines[["top", "right"]].set_visible(False)
    bar_axis.grid(axis="y", color="#eeeeee", linewidth=0.6, zorder=0)
    for index, value in enumerate(bar_values):
        bar_axis.annotate(
            f"{value:.2f}",
            (index, value),
            xytext=(0, 5 if value >= 0 else -5),
            textcoords="offset points",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=10,
        )
    bar_axis.text(
        0.5,
        -0.36,
        "Layer means: 600–100 hPa − 1000–600 hPa | Anomalies relative to the normal category",
        ha="center",
        va="top",
        transform=bar_axis.transAxes,
        fontsize=10,
    )
    outputs.append(_save(fig, destination / "Figure7.png"))

    sawp_domain = config.domains["sawp_index"]

    def normalized_region(value: xr.DataArray, domain_name: str) -> np.ndarray:
        if domain_name == "sawp":
            bounds = {
                "west": sawp_domain.west,
                "east": sawp_domain.east,
                "south": sawp_domain.south,
                "north": sawp_domain.north,
            }
        else:
            # The correlation-matrix script used a narrower ENEB coastal box
            # than the broader domain employed by the rainfall products.
            bounds = {"west": -38.0, "east": -34.5, "south": -12.5, "north": -4.5}
        values = regional_mean(
            value,
            **bounds,
        ).to_numpy()
        return (values - np.nanmean(values)) / np.nanstd(values, ddof=1)

    matrix_data = pd.DataFrame(index=fields.year.to_numpy().astype(int))
    matrix_data["sst_norm"] = (
        series.loc[matrix_data.index, "sst_detrended_c"] - series.sst_detrended_c.mean()
    ) / series.sst_detrended_c.std(ddof=1)
    matrix_data["rain_norm"] = (
        rain.loc[matrix_data.index, "rainfall_detrended_mm"] - rain.rainfall_detrended_mm.mean()
    ) / rain.rainfall_detrended_mm.std(ddof=1)
    matrix_data["slhf_sawp_norm"] = normalized_region(fields.latent_heat_flux, "sawp")
    matrix_data["wind_sawp_norm"] = normalized_region(fields.wind1000_speed, "sawp")
    matrix_data["pw_low_sawp_norm"] = normalized_region(
        fields.correlation_precipitable_water_low, "sawp"
    )
    matrix_data["pw_low_eneb_norm"] = normalized_region(
        fields.correlation_precipitable_water_low, "eneb"
    )
    matrix_data["pw_mid_sawp_norm"] = normalized_region(
        fields.correlation_precipitable_water_mid, "sawp"
    )
    matrix_data["pw_mid_eneb_norm"] = normalized_region(
        fields.correlation_precipitable_water_mid, "eneb"
    )
    matrix_data["omega_sawp_norm"] = normalized_region(fields.omega500, "sawp")
    matrix_data["omega_eneb_norm"] = normalized_region(fields.omega500, "eneb")
    matrix_data["mfc_eneb_norm"] = normalized_region(
        fields.correlation_moisture_flux_convergence_low, "eneb"
    )
    selected_years = series.index[series.sst_class != "neutral"]
    matrix_data = matrix_data.loc[matrix_data.index.intersection(selected_years)].dropna()
    correlation_matrix = matrix_data.corr()
    p_values = pd.DataFrame(
        np.nan,
        index=correlation_matrix.index,
        columns=correlation_matrix.columns,
    )
    for row, first in enumerate(correlation_matrix.columns):
        for column, second in enumerate(correlation_matrix.columns):
            _coefficient, p_value = pearsonr(matrix_data[first], matrix_data[second])
            p_values.iat[row, column] = p_value

    # Reproduce the special legacy SST-LHF cell. The historical script calls
    # ``shift(3)`` before the AMJJ mean (while labelling the physical lag as
    # -2 months), then replaces only this cell in the lag-zero matrix.
    heat_monthly = open_canonical(
        config.paths.raw_era5 / "Era5_Dataset_meanHF.nc",
        chunks={"time": 195},
        config=config,
    )
    latent_region = regional_mean(
        -heat_monthly["avg_slhtf"],
        west=sawp_domain.west,
        east=sawp_domain.east,
        south=sawp_domain.south,
        north=sawp_domain.north,
    )
    with xr.open_dataset(atmosphere.trend_coefficients) as heat_coefficients:
        latent_slope_region = regional_mean(
            heat_coefficients.latent_heat_flux_slope,
            west=sawp_domain.west,
            east=sawp_domain.east,
            south=sawp_domain.south,
            north=sawp_domain.north,
        ).load()
    heat_index = xr.DataArray(
        np.arange(latent_region.sizes["time"], dtype=float),
        dims="time",
        coords={"time": latent_region.time},
    )
    heat_index = heat_index - heat_index.mean("time")
    latent_detrended_region = latent_region - latent_slope_region * heat_index
    latent_lagged = latent_detrended_region.shift(time=3)
    latent_lagged = (
        latent_lagged.where(
            latent_lagged.time.dt.month.isin(config.period.season_months), drop=True
        )
        .groupby("time.year")
        .mean("time", skipna=True)
    )
    latent_lagged_series = latent_lagged.compute().to_series()
    heat_monthly.close()
    latent_lagged_series = (
        latent_lagged_series - latent_lagged_series.mean()
    ) / latent_lagged_series.std(ddof=1)
    lagged_pair = pd.concat(
        [matrix_data["sst_norm"], latent_lagged_series.rename("slhf_lagged")],
        axis=1,
        join="inner",
    ).dropna()
    lagged_r, lagged_p = pearsonr(lagged_pair.sst_norm, lagged_pair.slhf_lagged)
    correlation_matrix.loc["slhf_sawp_norm", "sst_norm"] = lagged_r
    correlation_matrix.loc["sst_norm", "slhf_sawp_norm"] = lagged_r
    p_values.loc["slhf_sawp_norm", "sst_norm"] = lagged_p
    p_values.loc["sst_norm", "slhf_sawp_norm"] = lagged_p
    labels = {
        "sst_norm": "SAWP SST",
        "rain_norm": "ENEB rainfall",
        "slhf_sawp_norm": "SAWP LHF",
        "wind_sawp_norm": "SAWP wind",
        "pw_low_sawp_norm": "SAWP IWV\n1000–850 hPa",
        "pw_low_eneb_norm": "ENEB IWV\n1000–850 hPa",
        "pw_mid_sawp_norm": "SAWP IWV\n850–400 hPa",
        "pw_mid_eneb_norm": "ENEB IWV\n850–400 hPa",
        "omega_sawp_norm": "SAWP ω\n500 hPa",
        "omega_eneb_norm": "ENEB ω\n500 hPa",
        "mfc_eneb_norm": r"ENEB $\nabla \cdot \mathbf{Q}$",
    }

    def significance_stars(p_value: float) -> str:
        if p_value < 0.01:
            return "***"
        if p_value < 0.05:
            return "**"
        if p_value < 0.10:
            return "*"
        return ""

    names = [labels[column] for column in correlation_matrix.columns]
    fig, axis = plt.subplots(figsize=(14, 12))
    image = axis.imshow(correlation_matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    colorbar = fig.colorbar(image, ax=axis, shrink=0.82)
    colorbar.set_label("Correlation coefficient", fontsize=18)
    colorbar.ax.tick_params(labelsize=16)
    axis.set_xticks(np.arange(len(names)), names, rotation=45, ha="right", fontsize=16)
    axis.set_yticks(np.arange(len(names)), names, fontsize=16)
    for row in range(len(names)):
        for column in range(len(names)):
            if column > row:
                axis.add_patch(
                    Rectangle(
                        (column - 0.5, row - 0.5),
                        1,
                        1,
                        color="white",
                        zorder=3,
                    )
                )
                continue
            coefficient = correlation_matrix.iloc[row, column]
            text_color = "white" if abs(coefficient) >= 0.55 else "black"
            axis.text(
                column,
                row,
                f"{coefficient:.2f}{significance_stars(p_values.iloc[row, column])}",
                ha="center",
                va="center",
                fontsize=14,
                color=text_color,
            )
    axis.set_title(
        "Pearson correlation matrix using TSM quantile-mask years\nSST × LHF uses lag -2 months",
        fontsize=18,
        pad=20,
    )
    axis.set_xticks(np.arange(-0.5, len(names), 1), minor=True)
    axis.set_yticks(np.arange(-0.5, len(names), 1), minor=True)
    axis.grid(which="minor", color="white", linewidth=1.2)
    axis.tick_params(which="minor", bottom=False, left=False)
    fig.text(
        0.5,
        0.02,
        "* p < 0.10   ** p < 0.05   *** p < 0.01",
        ha="center",
        fontsize=14,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    outputs.append(_save(fig, destination / "FigureS6.png", dpi=600))

    # Figure S8 uses the robust Theil-Sen slope of the annual AMJJ fields,
    # exactly as the legacy trend script, rather than an OLS monthly slope.
    sst_trend = theil_sen_slope(ocean_fields.sst_anomaly) * 10.0
    pw_trend = theil_sen_slope(fields.raw_precipitable_water_mid) * 10.0
    convergence_trend = theil_sen_slope(fields.raw_moisture_flux_convergence_low) * 10.0
    theta_trend = theil_sen_slope(fields.raw_theta_e_cross) * 10.0
    u_section_trend = theil_sen_slope(fields.raw_u_cross) * 10.0
    omega_section_trend = theil_sen_slope(fields.raw_omega_cross) * 10.0

    def dynamic_levels(value: xr.DataArray) -> np.ndarray:
        finite = np.abs(value.to_numpy())
        finite = finite[np.isfinite(finite)]
        limit = float(np.nanpercentile(finite, 90)) if finite.size else 1.0
        if not np.isfinite(limit) or limit == 0:
            limit = float(np.nanmax(finite)) if finite.size else 1.0
        if not np.isfinite(limit) or limit == 0:
            limit = 1.0
        return np.linspace(-limit, limit, 21)

    fig = plt.figure(figsize=(12, 10))
    grid_spec = fig.add_gridspec(2, 2)
    trend_axes = [
        fig.add_subplot(grid_spec[0, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(grid_spec[0, 1], projection=ccrs.PlateCarree()),
        fig.add_subplot(grid_spec[1, 0], projection=ccrs.PlateCarree()),
        fig.add_subplot(grid_spec[1, 1]),
    ]
    fig.subplots_adjust(hspace=0.1, wspace=0.15, bottom=0.20)
    for axis, value, cmap, panel, unit in (
        (trend_axes[0], sst_trend, "bwr", "a", "°C/dec"),
        (trend_axes[1], pw_trend, "BrBG", "b", "mm/dec"),
        (
            trend_axes[2],
            convergence_trend,
            "PuOr",
            "c",
            "kg m⁻² s⁻¹/dec",
        ),
    ):
        image = _legacy_map_axis(
            axis,
            value,
            levels=dynamic_levels(value),
            cmap=cmap,
            panel=panel,
            alpha=0.7,
        )
        axis.set_title(f"{panel})", loc="left", fontsize=12)
        color_axis = inset_axes(
            axis,
            width="5%",
            height="30%",
            loc="lower right",
            bbox_to_anchor=(-0.15, 0.0, 1, 1),
            bbox_transform=axis.transAxes,
            borderpad=0.5,
        )
        colorbar = fig.colorbar(image, cax=color_axis, orientation="vertical")
        colorbar.ax.tick_params(labelsize=8)
        colorbar.ax.set_title(unit, fontsize=8)

    longitude, pressure = np.meshgrid(theta_trend.lon, theta_trend.level)
    image_d = trend_axes[3].contourf(
        longitude,
        pressure,
        theta_trend,
        levels=dynamic_levels(theta_trend),
        cmap="RdYlBu_r",
        extend="both",
    )
    step_x = max(1, u_section_trend.sizes["lon"] // 25)
    step_z = max(1, u_section_trend.sizes["level"] // 12)
    u_arrows = u_section_trend.to_numpy()[::step_z, ::step_x]
    w_arrows = omega_section_trend.to_numpy()[::step_z, ::step_x]
    trend_arrows = trend_axes[3].quiver(
        longitude[::step_z, ::step_x],
        pressure[::step_z, ::step_x],
        u_arrows,
        w_arrows * 100,
        scale=None,
        width=0.002,
    )
    trend_axes[3].set_title("d)", loc="left", fontsize=12)
    color_axis_d = inset_axes(
        trend_axes[3],
        width="5%",
        height="30%",
        loc="lower right",
        bbox_to_anchor=(-0.15, 0.0, 1, 1),
        bbox_transform=trend_axes[3].transAxes,
        borderpad=0.5,
    )
    colorbar_d = fig.colorbar(image_d, cax=color_axis_d, orientation="vertical")
    colorbar_d.ax.tick_params(labelsize=8)
    colorbar_d.ax.set_title("K/dec", fontsize=8)
    trend_axes[3].set_yticks([1000, 850, 700, 500, 300, 200])
    trend_axes[3].set_ylim(1000, 200)
    trend_axes[3].yaxis.tick_right()
    trend_axes[3].yaxis.set_label_position("right")
    trend_axes[3].tick_params(
        axis="y",
        left=False,
        labelleft=False,
        right=True,
        labelright=True,
        labelsize=12,
    )
    trend_axes[3].spines["left"].set_visible(False)
    trend_axes[3].set_ylabel("Pressure (hPa)", fontsize=14, labelpad=10)
    fig.canvas.draw()
    vector_magnitude = np.hypot(u_arrows, w_arrows * 100)
    valid = np.isfinite(vector_magnitude)
    median_magnitude = float(np.nanmedian(vector_magnitude[valid])) if np.any(valid) else 1.0
    distance = np.where(valid, np.abs(vector_magnitude - median_magnitude), np.inf)
    reference_index = np.unravel_index(np.argmin(distance), vector_magnitude.shape)
    reference_u = float(u_arrows[reference_index])
    reference_w = float(w_arrows[reference_index])
    axis_position = trend_axes[3].get_position()
    key_axis = fig.add_axes([0.57, 0.030, axis_position.width, 0.05])
    key_axis.set(xlim=(0, 1), ylim=(0, 1))
    key_axis.axis("off")
    key_axis.quiver(
        [0.08],
        [0.5],
        [reference_u],
        [reference_w * 100],
        color="black",
        pivot=trend_arrows.pivot,
        angles=trend_arrows.angles,
        scale=trend_arrows.scale,
        scale_units=trend_arrows.scale_units,
        width=trend_arrows.width,
    )
    key_axis.text(
        0.18,
        0.5,
        rf"$u$ = {reference_u:.3g} m s$^{{-1}}$ decade$^{{-1}}$; "
        rf"$\omega$ = {reference_w:.2e} Pa s$^{{-1}}$ decade$^{{-1}}$",
        va="center",
        fontsize=8,
    )
    fig.text(0.5, 0.125, "Longitude", ha="center", fontsize=20)
    fig.text(0.05, 0.5, "Latitude", va="center", rotation="vertical", fontsize=20)
    outputs.append(_save(fig, destination / "FigureS8.png"))

    lhf_values = [
        _class_contrast(fields.latent_heat_flux, series, label)
        for label, _title, _color in EVENT_CLASSES
    ]
    limit = max(
        5.0,
        float(np.ceil(max(np.nanmax(np.abs(value)) for value in lhf_values) / 5) * 5),
    )
    levels = np.linspace(-limit, limit, 13)
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(11.6, 9.0),
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    fig.subplots_adjust(
        left=0.085,
        right=0.865,
        bottom=0.085,
        top=0.970,
        wspace=0.05,
        hspace=0.11,
    )
    states = cfeature.NaturalEarthFeature(
        "cultural",
        "admin_1_states_provinces_lakes",
        "10m",
        facecolor="none",
    )
    image = None
    for index, (axis, value) in enumerate(zip(axes.flat, lhf_values, strict=True)):
        axis.set_extent([-60, -10, -30, 15], crs=ccrs.PlateCarree())
        image = axis.contourf(
            value.lon,
            value.lat,
            value,
            levels=levels,
            cmap="bwr",
            extend="both",
            transform=ccrs.PlateCarree(),
            alpha=0.5,
        )
        axis.add_feature(cfeature.LAND, facecolor="#f3f3fa", edgecolor="none", zorder=3)
        axis.coastlines(resolution="50m", linewidth=0.55, zorder=4)
        axis.add_feature(
            cfeature.BORDERS.with_scale("50m"),
            linewidth=0.4,
            linestyle=":",
            zorder=4,
        )
        axis.add_feature(states, edgecolor="black", linewidth=0.3, alpha=0.3, zorder=4)
        axis.add_patch(
            Rectangle(
                (-34, -12.5),
                9,
                8.5,
                facecolor="lightgray",
                edgecolor="black",
                linewidth=1.2,
                hatch="///",
                alpha=0.18,
                transform=ccrs.PlateCarree(),
                zorder=5,
            )
        )
        axis.legend(
            handles=[
                Rectangle(
                    (0, 0),
                    1,
                    1,
                    facecolor="lightgray",
                    edgecolor="black",
                    hatch="///",
                    alpha=0.35,
                    label="SAWP",
                )
            ],
            loc="upper right",
            fontsize=8,
            framealpha=1,
            edgecolor="black",
            handlelength=1.5,
        )
        axis.set_xticks(np.arange(-60, -5, 10) if index >= 2 else [], crs=ccrs.PlateCarree())
        axis.set_yticks(
            np.arange(-30, 20, 10) if index % 2 == 0 else [],
            crs=ccrs.PlateCarree(),
        )
        axis.tick_params(labelsize=13)
        axis.set_title(f"{'abcd'[index]})", loc="left", fontsize=16, pad=4)
    fig.supxlabel("Longitude", fontsize=19, y=0.015)
    fig.supylabel("Latitude", fontsize=19, x=0.015)
    color_axis = fig.add_axes([0.895, 0.20, 0.024, 0.60])
    colorbar = fig.colorbar(image, cax=color_axis, orientation="vertical")
    colorbar.set_label("Latent heat flux anomaly (W m$^{-2}$)", fontsize=15)
    colorbar.ax.tick_params(labelsize=12)
    outputs.append(_save(fig, destination / "FigureS9.png", dpi=240, pad_inches=0.02))
    fields.close()
    ocean_fields.close()
    return outputs
