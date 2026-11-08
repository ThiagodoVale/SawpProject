from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from sawp.rainfall import (
    basic_quality_control,
    read_station,
    regional_monthly_series,
    spatial_outlier_control,
    station_completeness,
)


class RainfallTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = pd.date_range("2000-01-01", periods=3, freq="MS")
        self.frame = pd.DataFrame(
            {"A": [100.0, -1.0, 100.0], "B": [110.0, 2200.0, 1000.0], "C": [90.0, 100.0, 100.0]},
            index=self.index,
        )

    def test_basic_quality_control(self) -> None:
        cleaned = basic_quality_control(self.frame)
        self.assertTrue(np.isnan(cleaned.loc[self.index[1], "A"]))
        self.assertTrue(np.isnan(cleaned.loc[self.index[1], "B"]))

    def test_spatial_outlier_control(self) -> None:
        cleaned = basic_quality_control(self.frame)
        cleaned, removed = spatial_outlier_control(cleaned)
        self.assertTrue(np.isnan(cleaned.loc[self.index[2], "B"]))
        self.assertEqual(len(removed), 1)

    def test_regional_series_and_completeness(self) -> None:
        cleaned = basic_quality_control(self.frame)
        result = regional_monthly_series(
            cleaned, minimum_valid_stations=1, minimum_valid_fraction=0.0
        )
        self.assertAlmostEqual(result.loc[self.index[0], "precipitation_mm"], 100.0)
        completeness = station_completeness(cleaned)
        self.assertAlmostEqual(completeness["A"], 2 / 3)

    def test_station_reader_separates_metadata_from_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "station.txt"
            path.write_text(
                "# code: 123\n# name: Test station\n# x: -36.0\n# y: -8.0\n"
                "2000 1 1 12.5\n2000 2 1 20.0\n",
                encoding="utf-8",
            )
            station = read_station(path)
            self.assertEqual(station.station_id, "123")
            self.assertEqual(station.name, "Test station")
            self.assertEqual(len(station.precipitation), 2)
            self.assertAlmostEqual(station.precipitation.iloc[0], 12.5)


if __name__ == "__main__":
    unittest.main()
