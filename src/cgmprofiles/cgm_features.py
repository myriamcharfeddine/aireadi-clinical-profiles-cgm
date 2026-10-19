"""Extended CGM features for the reverse direction (Methods 2.3).

Summary features pool all valid readings in the window. Sequence features (MAGE,
autocorrelation) use contiguous runs: a run is split at any gap longer than 30 minutes,
shorter gaps are linearly interpolated (for MAGE only), and runs shorter than 6 hours are
dropped. The Lomb-Scargle circadian share uses all valid readings of the full record. The
24-hour daily profile uses each participant's own time zone.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lombscargle

from .cgm_metrics import CV, MEAN, TIR
from .config import get_config
from .io import GLUCOSE, PID, TIMESTAMP

CFG = get_config().features
CGM_CFG = get_config().cgm
WINDOW_48H, WINDOW_FULL = CGM_CFG.short_window, CGM_CFG.full_window
READINGS_PER_HOUR = 60 // CFG.sampling_min
EXPECTED_READINGS_48H = int(CGM_CFG.windows[WINDOW_48H]) * READINGS_PER_HOUR
MAX_MISSING_ROWS_INTERPOLATED = CFG.max_interpolation_gap_min // CFG.sampling_min - 1
MIN_RUN_READINGS = CFG.min_run_h * READINGS_PER_HOUR
MINIMAL_FEATURES = [MEAN, TIR, CV]
SUMMARY_FEATURES = MINIMAL_FEATURES + ["tbr_70_pct", "tar_180_pct", "tar_250_pct", "sd_glucose"]


def profile_columns() -> list[str]:
    """Names of the 24 hourly daily-profile columns."""
    return [f"profile_h{h:02d}" for h in range(CFG.hours_per_day)]


def extended_scalar_features(window: str) -> list[str]:
    """Scalar features of the extended set (24 h autocorrelation and circadian share: full record only)."""
    names = SUMMARY_FEATURES + ["mage", "acf_1h", "acf_3h"]
    return names + list(CFG.full_only) if window == WINDOW_FULL else names


def split_runs(values: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """(interpolated run, observed mask) pairs; a run ends after more than 5 consecutive missing rows."""
    valid = ~np.isnan(values)
    if not valid.any():
        return []
    runs, start, last_valid, missing = [], None, None, 0
    for i, ok in enumerate(valid):
        if ok:
            if start is None:
                start = i
            elif missing > MAX_MISSING_ROWS_INTERPOLATED:
                runs.append((start, last_valid))
                start = i
            last_valid, missing = i, 0
        else:
            missing += 1
    runs.append((start, last_valid))
    out = []
    for s, e in runs:
        segment = values[s:e + 1]
        if np.sum(~np.isnan(segment)) < MIN_RUN_READINGS:
            continue
        observed = ~np.isnan(segment)
        index = np.arange(len(segment))
        out.append((np.interp(index, index[observed], segment[observed]), observed))
    return out


def trailing_mean(x: np.ndarray, width: int) -> np.ndarray:
    """Trailing moving average (NaN for the first width - 1 positions)."""
    cumulative = np.concatenate([[0.0], np.cumsum(x)])
    out = np.full(len(x), np.nan)
    out[width - 1:] = (cumulative[width:] - cumulative[:-width]) / width
    return out


def mage_ma(x: np.ndarray) -> float:
    """Moving-average MAGE (Fernandes et al. 2022, iglu mage_ma): crossings of the 5- and 32-reading
    trailing means delimit excursions; amplitudes between successive extremes larger than 1 SD of
    the run are averaged, both directions pooled."""
    short, long = CFG.mage_short_ma, CFG.mage_long_ma
    if len(x) <= long:
        return np.nan
    diff = (trailing_mean(x, short) - trailing_mean(x, long))[long - 1:]
    raw = x[long - 1:]
    sign = np.sign(diff)
    for i in range(1, len(sign)):  # carry the sign through exact zeros
        if sign[i] == 0:
            sign[i] = sign[i - 1]
    crossings = np.flatnonzero(sign[1:] != sign[:-1]) + 1
    bounds = np.concatenate([[0], crossings, [len(raw)]])
    extremes = []
    for s, e in zip(bounds[:-1], bounds[1:]):
        if e <= s or sign[s] == 0:
            continue
        extremes.append(raw[s:e].max() if sign[s] > 0 else raw[s:e].min())
    if len(extremes) < 2:
        return np.nan
    amplitudes = np.abs(np.diff(extremes))
    kept = amplitudes[amplitudes > CFG.mage_sd_multiplier * np.std(x, ddof=1)]
    return float(kept.mean()) if len(kept) else np.nan


def pooled_acf(runs: list[tuple[np.ndarray, np.ndarray]], lag: int) -> float:
    """Autocorrelation at a lag, pooled over runs, observed readings only."""
    numerator, n_pairs, denominator, n_values = 0.0, 0, 0.0, 0
    for filled, observed in runs:
        x = np.where(observed, filled, np.nan)
        x = x - np.nanmean(x)
        denominator += np.nansum(x ** 2)
        n_values += int(observed.sum())
        if len(x) > lag:
            products = x[:-lag] * x[lag:]
            numerator += np.nansum(products)
            n_pairs += int(np.sum(~np.isnan(products)))
    if n_pairs == 0 or denominator == 0:
        return np.nan
    return float((numerator / n_pairs) / (denominator / n_values))


def circadian_share(hours: np.ndarray, values: np.ndarray) -> float:
    """Share of Lomb-Scargle power in the 20 to 28 h band, among periods of 2 to 28 h."""
    centred = values - values.mean()
    periods = np.linspace(CFG.spectrum_band_h[0], CFG.spectrum_band_h[1], CFG.n_spectrum_frequencies)
    frequencies = np.sort(1.0 / periods)
    power = lombscargle(hours, centred, 2 * np.pi * frequencies)
    in_band = (1.0 / frequencies >= CFG.circadian_band_h[0]) & (1.0 / frequencies <= CFG.circadian_band_h[1])
    return float(power[in_band].sum() / power.sum())


def participant_features(block: pd.DataFrame, timezone: str, window: str, min_readings_full: int | None = None) -> dict:
    """All features for one participant's rows in one window; coverage floor 70% of 48 h or 1,000 readings."""
    floor_full = CFG.min_readings_full if min_readings_full is None else min_readings_full
    values_all = block[GLUCOSE].to_numpy(float)
    valid = ~np.isnan(values_all)
    readings = values_all[valid]
    n = len(readings)
    enough = n >= (CFG.min_coverage_48h * EXPECTED_READINGS_48H if window == WINDOW_48H else floor_full)
    row = {"n_readings": n, "passes_coverage": bool(enough)}
    if not enough:
        return row
    sd = readings.std(ddof=1)
    low, high = CGM_CFG.tir_mgdl
    row.update({MEAN: readings.mean(), TIR: np.mean((readings >= low) & (readings <= high)) * 100, CV: sd / readings.mean() * 100,
                "tbr_70_pct": np.mean(readings < CFG.tbr_mgdl) * 100, "tar_180_pct": np.mean(readings > CFG.tar_mgdl) * 100,
                "tar_250_pct": np.mean(readings > CFG.tar_high_mgdl) * 100, "sd_glucose": sd})
    runs = split_runs(values_all)
    run_mage = [(mage_ma(filled), len(filled)) for filled, _ in runs]
    run_mage = [(m, w) for m, w in run_mage if not np.isnan(m)]
    row["mage"] = float(np.average([m for m, _ in run_mage], weights=[w for _, w in run_mage])) if run_mage else np.nan
    row["n_runs"] = len(runs)
    for name, lag_h in CFG.acf_lags_h.raw.items():
        if name in CFG.full_only and window != WINDOW_FULL:
            continue
        row[name] = pooled_acf(runs, lag_h * READINGS_PER_HOUR)
    local = block[TIMESTAMP].dt.tz_convert(timezone)
    hourly = pd.Series(values_all[valid]).groupby(local.dt.hour.to_numpy()[valid]).mean().reindex(range(CFG.hours_per_day))
    hourly = hourly.fillna(readings.mean())
    for hour, value in enumerate(hourly.to_numpy() - hourly.mean()):
        row[f"profile_h{hour:02d}"] = value
    if window == WINDOW_FULL:
        row["circadian_power_share"] = circadian_share(block["elapsed_h"].to_numpy(float)[valid], readings)
    return row


def build_features(cgm: pd.DataFrame, zones: pd.Series, participant_ids: list[str]) -> dict[str, pd.DataFrame]:
    """Features in the first 48 hours and over the full record, rows in the given participant order."""
    out = {}
    for window, hours in CGM_CFG.windows.raw.items():
        selected = cgm if hours is None else cgm[cgm.elapsed_h.le(hours)]
        records = [{PID: pid, **participant_features(block, zones[pid], window)} for pid, block in selected.groupby(PID, sort=False)]
        out[window] = pd.DataFrame(records).set_index(PID).loc[participant_ids].reset_index()
    return out


def mage_sine_check() -> float:
    """MAGE of a 4 h-period sine of amplitude 50 mg/dL (expected 100 mg/dL)."""
    t = np.arange(0, 48 * READINGS_PER_HOUR)
    return mage_ma(140 + 50 * np.sin(2 * np.pi * t / (4 * READINGS_PER_HOUR)))
