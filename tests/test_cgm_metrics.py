"""Synthetic checks of the CGM outcome and feature definitions."""
import numpy as np
import pandas as pd

from cgmprofiles import cgm_features as cf
from cgmprofiles.cgm_metrics import CV, MEAN, TIR, window_metrics
from cgmprofiles.io import GLUCOSE, PID


def frame(values, step_h=5 / 60):
    return pd.DataFrame({PID: "x", GLUCOSE: values, "elapsed_h": np.arange(len(values)) * step_h})


def test_tir_bounds_are_inclusive():
    out = window_metrics(frame([69.0, 70.0, 180.0, 181.0]), None).iloc[0]
    assert out[TIR] == 50.0


def test_cv_uses_sample_sd():
    values = np.array([100.0, 120.0, 140.0])
    out = window_metrics(frame(values), None).iloc[0]
    assert np.isclose(out[CV], values.std(ddof=1) / values.mean() * 100)
    assert np.isclose(out[MEAN], 120.0)


def test_48h_window_is_inclusive_and_skips_missing():
    values = np.full(600, 100.0)
    values[3] = np.nan
    out = window_metrics(frame(values), 48.0).iloc[0]
    assert out.n_readings == 577 - 1  # 0 to 48 h inclusive at 5 minutes, one missing


def test_single_reading_has_no_cv():
    assert np.isnan(window_metrics(frame([120.0]), None).iloc[0][CV])


def test_mage_of_sine_is_peak_to_nadir():
    assert abs(cf.mage_sine_check() - 100.0) < 1.0


def test_mage_of_flat_series_is_missing():
    assert np.isnan(cf.mage_ma(np.full(200, 120.0)))


def test_runs_split_at_gaps_over_30_minutes():
    x = np.concatenate([np.full(100, 100.0), np.full(6, np.nan), np.full(100, 110.0)])
    assert len(cf.split_runs(x)) == 2
    y = np.concatenate([np.full(100, 100.0), np.full(5, np.nan), np.full(100, 110.0)])
    assert len(cf.split_runs(y)) == 1


def test_acf_of_constant_period_signal():
    t = np.arange(24 * 12 * 3)
    x = 140 + 30 * np.sin(2 * np.pi * t / (24 * 12))
    runs = cf.split_runs(x)
    assert cf.pooled_acf(runs, 24 * 12) > 0.9
    assert cf.pooled_acf(runs, 12 * 12) < -0.9
