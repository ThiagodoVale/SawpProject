from __future__ import annotations

import numpy as np
import pandas as pd

import sawp
from sawp.analysis import EventClass, classify_quantiles
from sawp.config import load_config
from sawp.physics import horizontal_divergence, integrate_pressure, water_vapor_transport
from sawp.rainfall import basic_quality_control, regional_monthly_series
from sawp.scientific.pipeline import ALL_PRODUCTS, ScientificPipeline


def test_release_and_product_inventory() -> None:
    config = load_config()
    assert sawp.__version__ == "1.1.0"
    assert config.version == "1.1.0"
    assert len(ALL_PRODUCTS) == 18


def test_minimal_required_input_inventory() -> None:
    names = {path.name for path in ScientificPipeline().required_inputs()}
    assert names >= {
        "Era5_Dataset_meanHF.nc",
        "Era5_Dataset_sst.nc",
        "Era5_Dataset_t_q.nc",
        "Era5_Dataset_u_v_w.nc",
        "sst.nc",
        "srtm30plus_v11_land.nc",
        "monitor",
    }
    assert "Era5_Dataset_decomposit.nc" not in names


def test_moisture_transport_and_divergence() -> None:
    pressure = np.array([1000.0, 850.0, 400.0])
    humidity = np.full((3, 2, 2), 0.01)
    u = np.full_like(humidity, 5.0)
    v = np.full_like(humidity, -2.0)
    content, transport_u, transport_v = water_vapor_transport(humidity, u, v, pressure)
    np.testing.assert_allclose(transport_u, 5.0 * content)
    np.testing.assert_allclose(transport_v, -2.0 * content)
    np.testing.assert_allclose(
        horizontal_divergence(
            np.zeros((5, 6)),
            np.zeros((5, 6)),
            np.linspace(-40.0, -20.0, 6),
            np.linspace(-15.0, -4.0, 5),
        ),
        0.0,
    )
    assert np.isfinite(integrate_pressure(humidity, pressure)).all()


def test_event_classification_and_rainfall_quality_control() -> None:
    labels, thresholds = classify_quantiles(np.arange(100, dtype=float))
    assert labels[0] == EventClass.EXTREME_COLD
    assert labels[-1] == EventClass.EXTREME_WARM
    assert thresholds["q10"] < thresholds["q90"]

    index = pd.date_range("2000-01-01", periods=2, freq="MS")
    frame = pd.DataFrame({"A": [100.0, -1.0], "B": [110.0, 120.0]}, index=index)
    cleaned = basic_quality_control(frame)
    assert np.isnan(cleaned.loc[index[1], "A"])
    regional = regional_monthly_series(
        cleaned,
        minimum_valid_stations=1,
        minimum_valid_fraction=0.0,
    )
    assert regional.loc[index[0], "precipitation_mm"] == 105.0
