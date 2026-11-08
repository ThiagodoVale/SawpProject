from __future__ import annotations

import inspect
import unittest

import numpy as np
import xarray as xr

from sawp.scientific.atmosphere import (
    _legacy_cross_latitude_mean,
    _theta_e_xarray,
    prepare_atmosphere,
)
from sawp.scientific.common import (
    legacy_indexed_box_mean,
    linear_detrend_annual,
    linear_detrend_time,
    regional_mean,
    seasonal_annual,
    theil_sen_slope,
)
from sawp.scientific.figures import render_atmosphere_figures, render_ocean_figures
from sawp.scientific.pipeline import ALL_PRODUCTS, product_filename
from sawp.workflows import product_runner


class ScientificPipelineTests(unittest.TestCase):
    def test_public_product_names_cover_all_eighteen_figures(self) -> None:
        self.assertEqual(len(ALL_PRODUCTS), 18)
        self.assertEqual(product_filename("figure-1"), "Figure1.png")
        self.assertEqual(product_filename("figure-s9"), "FigureS9.png")

    def test_public_product_runner_has_no_reference_fallback(self) -> None:
        source = inspect.getsource(product_runner)
        self.assertNotIn("reference_figures", source)
        self.assertNotIn("materialize_single_product", source)
        self.assertIn("ScientificPipeline", source)

    def test_seasonal_mean_and_linear_detrending(self) -> None:
        time = np.arange("2000-01", "2004-01", dtype="datetime64[M]")
        values = xr.DataArray(np.arange(time.size, dtype=float), dims="time", coords={"time": time})
        annual = seasonal_annual(values, (4, 5, 6, 7))
        self.assertEqual(annual.sizes["year"], 4)
        detrended, slope, _trend = linear_detrend_annual(annual)
        self.assertAlmostEqual(float(slope), 12.0)
        np.testing.assert_allclose(detrended, detrended.mean("year"))

    def test_complete_series_detrend_matches_nan_safe_formula(self) -> None:
        values = xr.DataArray(
            np.arange(60, dtype=float).reshape(10, 2, 3),
            dims=("time", "lat", "lon"),
        )
        safe = linear_detrend_time(values)
        complete = linear_detrend_time(values, assume_complete=True)
        for expected, actual in zip(safe, complete, strict=True):
            xr.testing.assert_allclose(expected, actual)

    def test_legacy_indexed_box_uses_half_open_unweighted_grid(self) -> None:
        values = xr.DataArray(
            np.arange(16, dtype=float).reshape(4, 4),
            dims=("lat", "lon"),
            coords={
                "lat": [-12.5, -12.25, -4.0, -3.75],
                "lon": [-34.0, -33.75, -25.0, -24.75],
            },
        )
        indexed = legacy_indexed_box_mean(values, west=-34.0, east=-25.0, south=-12.5, north=-4.0)
        np.testing.assert_allclose(float(indexed), np.mean([4.0, 5.0, 8.0, 9.0]))
        inclusive = regional_mean(values, west=-34.0, east=-25.0, south=-12.5, north=-4.0)
        np.testing.assert_allclose(float(inclusive), np.mean(values[:3, :3]))

    def test_theil_sen_field_slope_uses_median_pairwise_slope(self) -> None:
        years = np.arange(2000, 2005)
        values = xr.DataArray(
            np.column_stack([2.0 * years + 1.0, [-2.0, 0.0, 2.0, 100.0, 6.0]]),
            dims=("year", "point"),
            coords={"year": years, "point": [0, 1]},
        )
        slopes = theil_sen_slope(values, minimum_count=3, chunk_size=1)
        np.testing.assert_allclose(slopes, [2.0, 2.0])

    def test_atmospheric_figures_use_legacy_layers_lag_and_trend_method(self) -> None:
        source = inspect.getsource(render_atmosphere_figures)
        self.assertIn("fields.precipitable_water_mid", source)
        self.assertIn("fields.transport_u_mid", source)
        self.assertIn("fields.moisture_flux_convergence_low", source)
        self.assertIn("fields.wind1000_speed", source)
        self.assertIn("fields.correlation_precipitable_water_low", source)
        self.assertIn("fields.correlation_precipitable_water_mid", source)
        self.assertIn("fields.correlation_moisture_flux_convergence_low", source)
        self.assertIn("shift(time=3)", source)
        self.assertIn("theil_sen_slope", source)
        self.assertIn(
            "convergence = _class_contrast(fields.moisture_flux_convergence_low, series, label)",
            source,
        )
        self.assertNotIn(
            "divergence = -_class_contrast(fields.moisture_flux_convergence_low",
            source,
        )
        self.assertIn('cmap="PuOr",', source)

    def test_vertical_vectors_retain_the_legacy_longitude_domain(self) -> None:
        atmosphere_source = inspect.getsource(prepare_atmosphere)
        self.assertNotIn("west=-45.0", atmosphere_source)
        self.assertNotIn("east=-20.0", atmosphere_source)
        self.assertIn('west=analysis_bounds["west"]', atmosphere_source)
        self.assertIn('east=analysis_bounds["east"]', atmosphere_source)

        figure_source = inspect.getsource(render_ocean_figures)
        native_u_scale = figure_source.index("np.nanstd(diff_u.to_numpy())")
        interpolation = figure_source.index("section_values = {}")
        self.assertLess(native_u_scale, interpolation)
        self.assertIn(
            "np.linspace(float(diff_theta.lon.min()), float(diff_theta.lon.max()), 5)",
            figure_source,
        )
        self.assertIn('axes[2],\n        diff_mfc,\n        levels=', figure_source)
        self.assertNotIn("axes[2],\n        -diff_mfc,", figure_source)
        self.assertIn('cmap="PuOr",', figure_source)

    def test_cross_section_matches_the_legacy_half_open_latitude_slice(self) -> None:
        values = xr.DataArray(
            [100.0, 2.0, 4.0],
            dims="lat",
            coords={"lat": [-12.5, -12.25, -4.0]},
        )
        result = _legacy_cross_latitude_mean(values, southern_boundary=-12.5)
        self.assertEqual(float(result), 3.0)

    def test_theta_e_uses_the_exact_legacy_dew_point_lcl_formula(self) -> None:
        temperature_value = 300.0
        humidity_value = 0.01
        pressure_value = 850.0
        temperature = xr.DataArray(
            [[[temperature_value]]],
            dims=("level", "lat", "lon"),
            coords={"level": [pressure_value], "lat": [0.0], "lon": [0.0]},
        )
        humidity = xr.full_like(temperature, humidity_value)

        partial_pressure = humidity_value / (1.0 - humidity_value)
        saturation_pressure = 6.112 * np.exp(
            17.67 * (temperature_value - 273.15) / (temperature_value - 29.65)
        )
        saturation_mixing_ratio = (
            0.622 * saturation_pressure / (pressure_value - saturation_pressure)
        )
        vapor_pressure = partial_pressure / saturation_mixing_ratio * saturation_pressure
        logarithm = np.log(vapor_pressure / 6.112)
        dew_point = 243.5 * logarithm / (17.67 - logarithm) + 273.15
        dew_point_pressure = 6.112 * np.exp(17.67 * (dew_point - 273.15) / (dew_point - 29.65))
        mixing_ratio = 0.622 * dew_point_pressure / (pressure_value - dew_point_pressure)
        lcl_temperature = 56.0 + 1.0 / (
            1.0 / (dew_point - 56.0) + np.log(temperature_value / dew_point) / 800.0
        )
        expected = (
            temperature_value
            * (1000.0 / (pressure_value - dew_point_pressure)) ** 0.286
            * (temperature_value / lcl_temperature) ** (0.28 * mixing_ratio)
            * np.exp(
                mixing_ratio * (1.0 + 0.448 * mixing_ratio) * (3036.0 / lcl_temperature - 1.78)
            )
        )
        actual = float(_theta_e_xarray(temperature, humidity).item())
        self.assertAlmostEqual(actual, expected, places=12)


if __name__ == "__main__":
    unittest.main()
