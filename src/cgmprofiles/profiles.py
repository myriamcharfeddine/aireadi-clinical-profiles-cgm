"""Clinical profiles (Methods 2.2): fit on training participants, assign the rest to frozen centroids.

Per stratum: iterative imputation (or complete cases for healthy), log(1 + x) for factors with
|skew| > 1 on the training matrix, standardization, k-means (100 initializations, seed 42) on
the training participants in split-file order. Display labels C1, C2, ... order the clusters by
the training members' mean raw HbA1c (then BMI, then TG/HDL). Every participant, training ones
included, is then assigned to the nearest frozen centroid. The imputer is replayed from its
fitted parameters, which reproduces the record's frozen-pipeline arithmetic exactly.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler

from .config import get_config
from .io import FACTORS, PID, SPLIT_RANK, STRATA

CFG = get_config().profiles
COMPLETE_CASE = "complete_case"


def fit_profile_model(train: pd.DataFrame, stratum: str) -> dict:
    """Fit the preprocessing and k-means for one stratum on its training rows (in the given order)."""
    complete_case = CFG.missing_data[stratum] == COMPLETE_CASE
    if complete_case:
        train = train[train[FACTORS].notna().all(axis=1)].reset_index(drop=True)
    raw = train[FACTORS].astype(float).to_numpy()
    imputer = None
    if not complete_case:
        imputer = IterativeImputer(max_iter=CFG.imputer_max_iter, random_state=CFG.seed, sample_posterior=False)
        raw = imputer.fit_transform(raw)
    data = pd.DataFrame(raw, columns=FACTORS)
    log_transformed = []
    for factor in FACTORS:
        if abs(float(data[factor].skew())) > CFG.skew_log_threshold and data[factor].min() > CFG.log_min_value:
            data[factor] = np.log1p(data[factor])
            log_transformed.append(factor)
    scaler = StandardScaler().fit(data.to_numpy())
    kmeans = KMeans(n_clusters=CFG.k[stratum], n_init=CFG.kmeans_n_init, random_state=CFG.seed).fit(scaler.transform(data.to_numpy()))
    order = train.assign(cluster=kmeans.labels_).groupby("cluster")[list(CFG.display_order_by)].mean().sort_values(list(CFG.display_order_by))
    display = {int(original): rank + 1 for rank, original in enumerate(order.index)}
    by_display = sorted(display.items(), key=lambda item: item[1])
    return {"stratum": stratum, "missing_data": CFG.missing_data[stratum], "imputer": imputer, "log_transformed": log_transformed,
            "scaler_mean": scaler.mean_.astype(float), "scaler_scale": scaler.scale_.astype(float),
            "display_clusters": np.array([d for _, d in by_display]),
            "centroids": np.array([kmeans.cluster_centers_[original] for original, _ in by_display], dtype=float),
            "original_to_display": display, "n_train_fit": len(train)}


def apply_imputer(raw: np.ndarray, imputer: IterativeImputer) -> np.ndarray:
    """Replay a fitted IterativeImputer (sample_posterior=False) from its stored parameters:
    mean fill, then each stored round of linear predictions, clipped to the fitted bounds."""
    missing = np.isnan(raw)
    filled = np.where(missing, imputer.initial_imputer_.statistics_[None, :], raw)
    for step in imputer.imputation_sequence_:
        rows = missing[:, step.feat_idx]
        if not rows.any():
            continue
        predicted = filled[rows][:, step.neighbor_feat_idx] @ step.estimator.coef_ + step.estimator.intercept_
        filled[rows, step.feat_idx] = np.clip(predicted, imputer._min_value[step.feat_idx], imputer._max_value[step.feat_idx])
    filled[~missing] = raw[~missing]
    return filled


def impute_and_log(frame: pd.DataFrame, model: dict) -> np.ndarray:
    """Factor matrix after the frozen imputation and log(1 + x), before scaling."""
    values = frame[FACTORS].astype(float).to_numpy()
    if model["imputer"] is not None:
        values = apply_imputer(values, model["imputer"])
    for index, factor in enumerate(FACTORS):
        if factor in model["log_transformed"]:
            values[:, index] = np.log1p(values[:, index])
    return values


def transform(frame: pd.DataFrame, model: dict) -> np.ndarray:
    """Standardized clustering matrix under the frozen pipeline."""
    return (impute_and_log(frame, model) - model["scaler_mean"]) / model["scaler_scale"]


def assign_to_frozen_centroids(matrix: np.ndarray, model: dict) -> pd.DataFrame:
    """Nearest centroid, its distance, the second distance and the relative assignment margin."""
    distances = np.sqrt(((matrix[:, None, :] - model["centroids"][None, :, :]) ** 2).sum(axis=2))
    order = np.argsort(distances, axis=1)
    d1 = distances[np.arange(len(matrix)), order[:, 0]]
    d2 = distances[np.arange(len(matrix)), order[:, 1]]
    return pd.DataFrame({"label": model["display_clusters"][order[:, 0]], "nearest_distance": d1,
                         "second_distance": d2, "relative_margin": (d2 - d1) / d1})


def build_profiles(clinical: pd.DataFrame, split: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict]]:
    """Fit every stratum's profiles and assign every eligible participant (complete cases for healthy)."""
    frame = split.merge(clinical, on=PID, how="left", validate="one_to_one")
    rows, models = [], {}
    for stratum in STRATA:
        subset = frame[frame.canonical_stratum == stratum].copy()
        model = fit_profile_model(subset[subset.split == "train"].reset_index(drop=True), stratum)
        models[stratum] = model
        subset["n_missing_factors"] = subset[FACTORS].isna().sum(axis=1)
        if model["missing_data"] == COMPLETE_CASE:
            subset = subset[subset.n_missing_factors == 0]
        assigned = assign_to_frozen_centroids(transform(subset, model), model)
        assigned.index = subset.index
        rows.append(pd.concat([subset[[PID, "split", "split_file_row", "canonical_stratum", "n_missing_factors", *FACTORS]], assigned], axis=1))
    labels = pd.concat(rows, ignore_index=True)
    labels["split_rank"] = labels["split"].map(SPLIT_RANK)
    return labels, models


def compare_with_frozen_centroids(models: dict[str, dict], centroid_dir: Path) -> dict[str, float]:
    """Largest absolute difference between refit centroids and the tracked frozen centroid files."""
    out = {}
    for stratum, model in models.items():
        payload = json.loads((centroid_dir / f"{stratum}_centroids.json").read_text())
        if payload["factors"] != FACTORS or payload["log_transformed"] != model["log_transformed"]:
            out[stratum] = float("inf")
            continue
        frozen = np.array([payload["centroids_by_display_cluster"][str(d)] for d in model["display_clusters"]])
        out[stratum] = float(np.abs(frozen - model["centroids"]).max())
    return out


def frozen_model_from_files(stratum: str, refit: dict, centroid_dir: Path) -> dict:
    """The refit pipeline with the tracked frozen centroids substituted (used by the refit test)."""
    payload = json.loads((centroid_dir / f"{stratum}_centroids.json").read_text())
    centroids = np.array([payload["centroids_by_display_cluster"][str(d)] for d in refit["display_clusters"]])
    return {**refit, "centroids": centroids}
