"""CGM groups (Methods 2.3, reverse direction): training-only preprocessing and k-means,
nearest-centroid assignment for everyone else.

Preprocessing, fit on training participants: daily-profile PCA (components up to 80% of the
variance, at most 4), training-median imputation of missing features, log(1 + x) for |skew| > 1,
standardization. Display labels G1, G2, ... order groups by the training members' mean glucose.
The same fitted pipeline can be applied unchanged to another window (temporal analysis).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from .cgm_features import MINIMAL_FEATURES, extended_scalar_features, profile_columns
from .cgm_metrics import MEAN
from .config import get_config

CFG = get_config().cgm_groups


def fit_preprocessing(frame: pd.DataFrame, is_train: np.ndarray, feature_set: str, window: str) -> dict:
    """Fit the feature pipeline on the training rows and return it with the standardized matrix."""
    scalars = list(MINIMAL_FEATURES) if feature_set == "minimal" else extended_scalar_features(window)
    data = frame[scalars].astype(float).copy()
    info = {"scalars": scalars, "imputed_with_train_median": {}}
    if feature_set == "extended":
        profiles = frame[profile_columns()].to_numpy(float)
        cumulative = np.cumsum(PCA(random_state=CFG.seed).fit(profiles[is_train]).explained_variance_ratio_)
        n_pcs = int(min(CFG.max_profile_pcs, np.searchsorted(cumulative, CFG.profile_variance_target) + 1))
        pca = PCA(n_components=n_pcs, random_state=CFG.seed).fit(profiles[is_train])
        scores = pca.transform(profiles)
        for i in range(n_pcs):
            data[f"profile_pc{i + 1}"] = scores[:, i]
        info.update({"profile_pcs": n_pcs, "profile_variance_retained": float(pca.explained_variance_ratio_.sum()), "pca": pca})
    for column in data.columns:
        missing = data[column].isna()
        if missing.any():
            info["imputed_with_train_median"][column] = int(missing.sum())
            data.loc[missing, column] = float(data.loc[is_train, column].median())
    logged = [c for c in data.columns if abs(float(data.loc[is_train, c].skew())) > CFG.skew_log_threshold and data.loc[is_train, c].min() > -1]
    for column in logged:
        data[column] = np.log1p(data[column])
    scaler = StandardScaler().fit(data.loc[is_train].to_numpy())
    info.update({"columns": list(data.columns), "log1p": logged, "scaler": scaler})
    return info | {"matrix": scaler.transform(data.to_numpy())}


def display_map(raw_mean_glucose: np.ndarray, train_labels: np.ndarray) -> dict[int, int]:
    """k-means label -> display number, by increasing mean glucose of the training members."""
    means = pd.Series(raw_mean_glucose).groupby(train_labels).mean().sort_values()
    return {int(original): rank + 1 for rank, original in enumerate(means.index)}


def kmeans_partition(matrix: np.ndarray, is_train: np.ndarray, raw_mean_glucose: np.ndarray, k: int) -> dict:
    """k-means on the training rows, nearest centroid for every row, display labels G1..Gk."""
    model = KMeans(n_clusters=k, n_init=CFG.n_init, random_state=CFG.seed).fit(matrix[is_train])
    distances = np.sqrt(((matrix[:, None, :] - model.cluster_centers_[None, :, :]) ** 2).sum(axis=2))
    nearest = distances.argmin(axis=1)
    mapping = display_map(raw_mean_glucose[is_train], model.labels_)
    centroids = np.array([model.cluster_centers_[o] for o, _ in sorted(mapping.items(), key=lambda item: item[1])])
    return {"model": model, "labels": np.array([mapping[c] for c in nearest]), "centroids": centroids, "mapping": mapping,
            "train_silhouette": float(silhouette_score(matrix[is_train], model.labels_))}


def silhouette_scan(matrix_train: np.ndarray) -> tuple[dict[int, float], int]:
    """Training silhouette for k = 2..6; the smallest k within 0.02 of the best is selected."""
    scores = {k: float(silhouette_score(matrix_train, KMeans(n_clusters=k, n_init=CFG.n_init, random_state=CFG.seed).fit_predict(matrix_train)))
              for k in CFG.k_range}
    best = max(scores.values())
    return scores, min(k for k, s in scores.items() if s >= best - CFG.silhouette_tolerance)


def bootstrap_stability(matrix: np.ndarray, is_train: np.ndarray, frozen_labels: np.ndarray, seed: int) -> np.ndarray:
    """ARI between the frozen training labels and k-means refit on 200 bootstrap samples of the training rows."""
    rng = np.random.default_rng(seed)
    train, reference = matrix[is_train], frozen_labels[is_train]
    aris = []
    for _ in range(CFG.n_stability_bootstrap):
        idx = rng.integers(0, len(train), size=len(train))
        model = KMeans(n_clusters=CFG.k_primary, n_init=CFG.n_init, random_state=CFG.seed).fit(train[idx])
        aris.append(adjusted_rand_score(reference, model.predict(train)))
    return np.array(aris)


# ---------------------------------------------------------------- frozen pipeline reused on another window
def assemble(info: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Scalar features plus profile PC scores from the fitted PCA, before imputation."""
    data = frame[info["scalars"]].astype(float).copy()
    if "pca" in info:
        scores = info["pca"].transform(frame[profile_columns()].to_numpy(float))
        for i in range(info["profile_pcs"]):
            data[f"profile_pc{i + 1}"] = scores[:, i]
    return data[info["columns"]]


def fit_frozen(frame: pd.DataFrame, is_train: np.ndarray, window_key: str) -> dict:
    """Fit pipeline and k-means on one window, keeping the training medians for reuse."""
    info = fit_preprocessing(frame, is_train, "extended", window_key)
    info["medians"] = assemble(info, frame).loc[is_train].median()
    partition = kmeans_partition(info["matrix"], is_train, frame[MEAN].to_numpy(float), CFG.k_primary)
    return {"info": info, "partition": partition}


def apply_frozen(info: dict, frame: pd.DataFrame) -> np.ndarray:
    """Standardized matrix of another window under the frozen pipeline."""
    data = assemble(info, frame).fillna(info["medians"])
    for column in info["log1p"]:
        data[column] = np.log1p(data[column])
    return info["scaler"].transform(data.to_numpy())


def clustering_scale(info: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Features on the clustering scale (log1p where chosen), without imputation (for ICCs)."""
    data = assemble(info, frame)
    for column in info["log1p"]:
        data[column] = np.log1p(data[column])
    return data


def nearest_labels(matrix: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Display label (1-based) of the nearest centroid."""
    return ((matrix[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2).argmin(axis=1) + 1
