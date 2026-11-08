from __future__ import annotations

import unittest

import numpy as np

from sawp.physics import (
    equivalent_potential_temperature,
    horizontal_divergence,
    integrate_pressure,
    water_vapor_transport,
)


class PhysicsTests(unittest.TestCase):
    def test_pressure_integral_is_order_independent(self) -> None:
        pressure = np.array([1000.0, 850.0])
        humidity = np.full((2, 3, 4), 0.01)
        expected = 0.01 * 15_000.0 / 9.80665
        descending = integrate_pressure(humidity, pressure)
        ascending = integrate_pressure(humidity[::-1], pressure[::-1])
        np.testing.assert_allclose(descending, expected)
        np.testing.assert_allclose(ascending, expected)

    def test_transport(self) -> None:
        pressure = np.array([1000.0, 850.0, 400.0])
        q = np.full((3, 2, 2), 0.01)
        u = np.full_like(q, 5.0)
        v = np.full_like(q, -2.0)
        content, transport_u, transport_v = water_vapor_transport(q, u, v, pressure)
        np.testing.assert_allclose(transport_u, 5.0 * content)
        np.testing.assert_allclose(transport_v, -2.0 * content)

    def test_constant_flux_has_zero_divergence(self) -> None:
        longitude = np.linspace(-40.0, -20.0, 6)
        latitude = np.linspace(-15.0, -4.0, 5)
        u = np.zeros((5, 6))
        v = np.zeros((5, 6))
        np.testing.assert_allclose(horizontal_divergence(u, v, longitude, latitude), 0.0)

    def test_equivalent_potential_temperature_is_physical(self) -> None:
        theta_e = equivalent_potential_temperature(300.0, 0.015, 1000.0)
        self.assertGreater(float(theta_e), 330.0)
        self.assertLess(float(theta_e), 380.0)


if __name__ == "__main__":
    unittest.main()
