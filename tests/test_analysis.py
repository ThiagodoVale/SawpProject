from __future__ import annotations

import unittest

import numpy as np

from sawp.analysis import (
    EventClass,
    classify_quantiles,
    composite_by_class,
    linear_detrend,
    monthly_anomalies,
    seasonal_reduce,
    standardize,
)


class AnalysisTests(unittest.TestCase):
    def test_linear_detrend_preserves_mean_and_removes_slope(self) -> None:
        time = np.arange(20, dtype=float)
        data = np.stack((3.0 * time + 4.0, -2.0 * time + 9.0), axis=1)
        data[5, 1] = np.nan
        detrended = linear_detrend(data, chunk_size=1)
        self.assertAlmostEqual(np.polyfit(time, detrended[:, 0], 1)[0], 0.0, places=12)
        valid = np.isfinite(data[:, 1])
        self.assertAlmostEqual(np.polyfit(time[valid], detrended[valid, 1], 1)[0], 0.0, places=12)
        np.testing.assert_allclose(np.nanmean(detrended, axis=0), np.nanmean(data, axis=0))

    def test_linear_detrend_supports_nonleading_axis_and_out(self) -> None:
        time = np.arange(8, dtype=float)
        data = np.stack((2.0 * time + 1.0, -time + 4.0), axis=0)
        destination = np.empty_like(data)
        result = linear_detrend(data, axis=1, out=destination, chunk_size=1)
        self.assertIs(result, destination)
        for row in result:
            self.assertAlmostEqual(np.polyfit(time, row, 1)[0], 0.0, places=12)

    def test_monthly_anomalies(self) -> None:
        times = np.arange("1980-01", "1982-01", dtype="datetime64[M]")
        values = np.tile(np.arange(1, 13, dtype=float), 2)
        anomalies, climatology = monthly_anomalies(
            values, times, climatology_start=1980, climatology_end=1981
        )
        np.testing.assert_allclose(anomalies, 0.0)
        np.testing.assert_allclose(climatology, np.arange(1, 13, dtype=float))

    def test_quantile_classification(self) -> None:
        values = np.arange(100, dtype=float)
        labels, thresholds = classify_quantiles(values)
        self.assertEqual(labels[0], EventClass.EXTREME_COLD)
        self.assertEqual(labels[-1], EventClass.EXTREME_WARM)
        self.assertLess(thresholds["q10"], thresholds["q25"])
        self.assertLess(thresholds["q75"], thresholds["q90"])

    def test_seasonal_reduce_and_composites(self) -> None:
        times = np.arange("2000-01", "2002-01", dtype="datetime64[M]")
        values = np.arange(24, dtype=float)
        annual, years = seasonal_reduce(values, times, months=(4, 5, 6, 7), reducer="mean")
        np.testing.assert_array_equal(years, [2000, 2001])
        np.testing.assert_allclose(annual, [4.5, 16.5])
        composites = composite_by_class(annual, np.array(["cold", "warm"]))
        self.assertEqual(composites["cold"], 4.5)
        self.assertEqual(composites["warm"], 16.5)

    def test_standardize(self) -> None:
        result = standardize(np.array([1.0, 2.0, 3.0]), ddof=0)
        self.assertAlmostEqual(float(np.mean(result)), 0.0)
        self.assertAlmostEqual(float(np.std(result)), 1.0)


if __name__ == "__main__":
    unittest.main()
