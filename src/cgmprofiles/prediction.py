"""Prediction in both directions (Methods 2.3): repeated stratified cross-validation (5 folds x 20
repeats, stratified by profile, seed 42); every transform is refit inside each outer training fold;
ridge alpha and logistic C come from an inner 5-fold CV; out-of-fold predictions are averaged over
repeats. Permutation nulls rerun the whole CV with the hyperparameter fixed at its most frequent
value. Clinical-to-CGM prediction is the Section 3.2 linear model with fold-wise iterative imputation.
"""
from __future__ import annotations

import warnings
from collections import Counter

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.exceptions import ConvergenceWarning
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import balanced_accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import KFold, RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

from .cgm_groups import fit_preprocessing
from .config import get_config
from .stats import TAILS

CFG = get_config().prediction
N_BOOT = get_config().bootstrap.n_default
GRID = np.logspace(*CFG.grid_log10)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def folds(profile: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """The Section 3.2 splitter: 5 folds x 20 repeats, stratified by profile."""
    splitter = RepeatedStratifiedKFold(n_splits=CFG.n_folds, n_repeats=CFG.n_repeats, random_state=CFG.fold_seed)
    return list(splitter.split(np.zeros(len(profile)), profile))


def clinical_to_cgm_oof(clinical: pd.DataFrame, factors: list[str], y: np.ndarray, fold_list) -> np.ndarray:
    """Out-of-fold CGM predictions from clinical factors (imputation seed 42 + fold index, scaling, OLS)."""
    raw = clinical[factors].to_numpy(float)
    total, count = np.zeros(len(y)), np.zeros(len(y))
    for fold_index, (train, test) in enumerate(fold_list):
        imputer = IterativeImputer(max_iter=CFG.imputer_max_iter, random_state=CFG.fold_seed + fold_index, sample_posterior=False)
        x_train, x_test = imputer.fit_transform(raw[train]), imputer.transform(raw[test])
        scaler = StandardScaler().fit(x_train)
        total[test] += LinearRegression().fit(scaler.transform(x_train), y[train]).predict(scaler.transform(x_test))
        count[test] += 1
    return total / count


def fold_matrices(features: pd.DataFrame, feature_set: str, window: str, fold_list) -> list[np.ndarray]:
    """CGM matrix per outer fold, with the feature pipeline refit on that fold's training rows."""
    out = []
    for train, _ in fold_list:
        mask = np.zeros(len(features), bool)
        mask[train] = True
        out.append(fit_preprocessing(features, mask, feature_set, window)["matrix"])
    return out


def hba1c_matrices(hba1c: np.ndarray, fold_list) -> list[np.ndarray]:
    """HbA1c alone: training-fold mean for missing values, standardized on the training fold."""
    out = []
    for train, _ in fold_list:
        filled = np.where(np.isnan(hba1c), np.nanmean(hba1c[train]), hba1c)
        out.append(((filled - filled[train].mean()) / filled[train].std())[:, None])
    return out


def ridge_fit_predict(x_train, y_train, x_test, alpha):
    """Closed-form ridge with an unpenalized intercept; y may hold several columns."""
    mean_x, mean_y = x_train.mean(axis=0), y_train.mean(axis=0)
    xc = x_train - mean_x
    coef = np.linalg.solve(xc.T @ xc + alpha * np.eye(xc.shape[1]), xc.T @ (y_train - mean_y))
    return (x_test - mean_x) @ coef + mean_y


def inner_alpha(x, y) -> float:
    """Ridge alpha with the smallest inner-CV squared error."""
    errors = np.zeros(len(GRID))
    for tr, te in KFold(n_splits=CFG.inner_folds, shuffle=True, random_state=CFG.inner_seed).split(x):
        for i, alpha in enumerate(GRID):
            errors[i] += np.sum((ridge_fit_predict(x[tr], y[tr], x[te], alpha) - y[te]) ** 2)
    return float(GRID[int(np.argmin(errors))])


def ridge_cv(matrices, y: np.ndarray, fold_list, alpha_fixed: float | None = None, y_columns: np.ndarray | None = None):
    """Out-of-fold ridge predictions (all columns of y_columns at once when alpha is fixed)."""
    observed = ~np.isnan(y)
    target = y_columns if y_columns is not None else y[:, None]
    total, count, chosen = np.zeros(target.shape), np.zeros(len(y)), []
    for (train, test), matrix in zip(fold_list, matrices):
        train = train[observed[train]]
        alpha = alpha_fixed if alpha_fixed is not None else inner_alpha(matrix[train], y[train])
        chosen.append(alpha)
        total[test] += ridge_fit_predict(matrix[train], target[train], matrix[test], alpha)
        count[test] += 1
    return total / count[:, None], chosen


def logistic(c: float) -> LogisticRegression:
    """Class-balanced multinomial logistic regression."""
    return LogisticRegression(C=c, class_weight="balanced", max_iter=CFG.logistic_max_iter)


def inner_c(x, y) -> float:
    """Logistic C with the best inner-CV balanced accuracy."""
    scores = np.zeros(len(GRID))
    for tr, te in StratifiedKFold(n_splits=CFG.inner_folds, shuffle=True, random_state=CFG.inner_seed).split(x, y):
        for i, c in enumerate(GRID):
            scores[i] += balanced_accuracy_score(y[te], logistic(c).fit(x[tr], y[tr]).predict(x[te]))
    return float(GRID[int(np.argmax(scores))])


def logistic_cv(matrices, y: np.ndarray, fold_list, c_fixed: float | None = None) -> tuple[np.ndarray, list[float]]:
    """Out-of-fold class probabilities averaged over repeats."""
    classes = np.unique(y)
    total, count, chosen = np.zeros((len(y), len(classes))), np.zeros(len(y)), []
    for (train, test), matrix in zip(fold_list, matrices):
        c = c_fixed if c_fixed is not None else inner_c(matrix[train], y[train])
        chosen.append(c)
        total[test] += logistic(c).fit(matrix[train], y[train]).predict_proba(matrix[test])
        count[test] += 1
    return total / count[:, None], chosen


def classification_metrics(y: np.ndarray, proba: np.ndarray, classes: np.ndarray | None = None) -> dict[str, float]:
    """Balanced accuracy and macro one-vs-rest AUROC (plain AUROC for two classes)."""
    classes = np.unique(y) if classes is None else classes
    if len(classes) == 2:
        auroc = roc_auc_score(y == classes[1], proba[:, 1])
    else:
        auroc = roc_auc_score(y, proba, multi_class="ovr", average="macro", labels=classes)
    return {"balanced_accuracy": balanced_accuracy_score(y, classes[proba.argmax(axis=1)]), "macro_auroc": auroc}


def bootstrap_ci(statistic, n: int, seed: int, skip_errors: bool = False) -> tuple[float, float]:
    """Participant bootstrap of a statistic of resampled indices (optionally skipping failed resamples)."""
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        if not skip_errors:
            draws.append(statistic(idx))
            continue
        try:
            draws.append(statistic(idx))
        except ValueError:  # a resample missing a class
            continue
    return tuple(np.percentile(draws, TAILS))


def most_common(values: list[float]) -> float:
    """Most frequently chosen hyperparameter."""
    return Counter(values).most_common(1)[0][0]


def ridge_permutation_null(matrices, y, fold_list, alpha, n_perm, seed) -> np.ndarray:
    """Out-of-fold R2 for permuted targets (one generator, permutations drawn in order)."""
    observed = ~np.isnan(y)
    rng = np.random.default_rng(seed)
    columns = np.tile(y[:, None], (1, n_perm))
    obs_idx = np.flatnonzero(observed)
    for j in range(n_perm):
        columns[obs_idx, j] = y[rng.permutation(obs_idx)]
    oof, _ = ridge_cv(matrices, y, fold_list, alpha_fixed=alpha, y_columns=columns)
    return np.array([r2_score(columns[observed, j], oof[observed, j]) for j in range(n_perm)])


def logistic_permutation_null(matrices, y, fold_list, c, n_perm, seed) -> dict[str, np.ndarray]:
    """Classification metrics for permuted profiles (one spawned seed per permutation, so n_jobs does not matter)."""
    classes = np.unique(y)

    def one(s):
        permuted = np.random.default_rng(s).permutation(y)
        proba, _ = logistic_cv(matrices, permuted, fold_list, c_fixed=c)
        return classification_metrics(permuted, proba, classes)

    results = Parallel(n_jobs=CFG.n_jobs)(delayed(one)(s) for s in np.random.SeedSequence(seed).spawn(n_perm))
    return {k: np.array([r[k] for r in results]) for k in ("balanced_accuracy", "macro_auroc")}


def needs_escalation(p_value: float) -> bool:
    """Pre-registered rule: rerun with 1,000 permutations when 0.01 <= p <= 0.10."""
    low, high = get_config().permutation.escalation_range
    return low <= p_value <= high
