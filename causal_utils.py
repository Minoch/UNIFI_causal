"""
causal_utils.py -- methods library for:

    "A Causal Machine Learning Framework for Treatment Personalization in
     Clinical Trials: Application to Ulcerative Colitis"

This module implements every method used in the paper. It is imported by
`run_analysis.py`, which runs the full pipeline and reproduces the paper's
tables and figures.

Contents
--------
- Data loading and preprocessing                    (load_unifi_data, build_feature_matrices, preprocess_all)
- X-learner CATE estimation with cross-fitting       (xlearner_binary_crossfit)          -> Methods, Eq. 1
- Permutation importance                             (permimp_with_ci)                   -> Table 3, Fig. 1
- Best linear predictor (BLP) testing                (blp_test_multiplier)               -> Table 4
- Doubly robust policy evaluation, binary & 3-arm    (dr_value_binary, dr_value_three_arm,
                                                      nested_three_arm_oof_values_foldsafe)       -> Fig. 2, Eq. 4
- Prognostic performance & incremental Brier score   (oof_proba, incremental_brier, ...) -> Table 5, Eqs. 5-6
- Subgroup average treatment effects (model-free)    (compute_subgroup_ates, ...)        -> Table 6, Eq. 7
- Bootstrap inference utilities                       (bootstrap_delta, summarize_from_dr)
- Figure generation                                  (plot_*)

Reproducibility
---------------
All randomness is seeded via DEFAULT_RANDOM_STATE. Gradient-boosted models can
still vary slightly across XGBoost versions / thread counts.
"""

from typing import Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from scipy.stats import norm, ttest_rel, spearmanr, ttest_ind, chi2
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor

# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_RANDOM_STATE = 42

# XGBoost hyperparameters for outcome models
# tree_method="exact" + n_jobs=1 make fitting deterministic and stable across
# XGBoost versions (the default "hist" method, introduced in XGBoost 2.0, builds
# different trees on small data and is a source of run-to-run drift).
OUTCOME_MODEL_PARAMS = {
    "n_estimators": 800,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "learning_rate": 0.05,
    "reg_lambda": 1.0,
    "tree_method": "exact",
    "n_jobs": 1,
}

# XGBoost hyperparameters for CATE models
CATE_MODEL_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "exact",
    "n_jobs": 1,
}

# XGBoost hyperparameters for prognostic models
PROGNOSTIC_MODEL_PARAMS = {
    "n_estimators": 600,
    "max_depth": 4,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "learning_rate": 0.05,
    "reg_lambda": 1.0,
    "tree_method": "exact",
    "n_jobs": 1,
}

# OPTIONAL logistic variant of the outcome models. 
# Usage:
#     mu = XGBRegressor(**OUTCOME_MODEL_PARAMS_LOGISTIC, random_state=seed)
OUTCOME_MODEL_PARAMS_LOGISTIC = {
    "objective": "binary:logistic",
    "n_estimators": 800,
    "max_depth": 4,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "learning_rate": 0.05,
    "reg_lambda": 1.0,
    "tree_method": "exact",
    "n_jobs": 1,
}


# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================


def load_unifi_data(
    excel_path: str = "Maintenance_ITT_all_ehr_and_video_allmed.xlsx",
    sheet: str = "Sheet1",
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load UNIFI maintenance trial data from Excel file.

    Parameters
    ----------
    excel_path : str
        Path to the Excel file containing trial data.
    sheet : str
        Sheet name to read from.

    Returns
    -------
    df : pd.DataFrame
        Full dataframe with all variables.
    T3 : np.ndarray
        Three-arm treatment indicator (0=Placebo, 1=Q12, 2=Q8).
    Tbin : np.ndarray
        Binary treatment indicator (0=Placebo, 1=UST pooled).
    Y : np.ndarray
        Primary outcome (week-44 remission).
    """
    df = pd.read_excel(excel_path, sheet_name=sheet)

    def infer_treatment(row: pd.Series) -> int:
        """Infer treatment arm from indicator variables."""
        q12 = row["MaintMed_UST_12"]
        q8 = row["MaintMed_UST_Q8"]
        if q12 == 1 and q8 == 0:
            return 1  # Q12
        if q12 == 0 and q8 == 1:
            return 2  # Q8
        if q12 == 0 and q8 == 0:
            return 0  # Placebo
        return np.nan

    df["T3"] = df.apply(infer_treatment, axis=1)
    df = df.dropna(subset=["T3"]).reset_index(drop=True)

    T3 = df["T3"].astype(int).values
    Y = df["Remission_Full_CALC_wk44_ITT"].astype(int).values
    Tbin = ((T3 == 1) | (T3 == 2)).astype(int)

    return df, T3, Tbin, Y


def build_feature_matrices(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List[str], List[str], List[str]]:
    """
    Build feature matrices for different analysis scenarios.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataframe from load_unifi_data.

    Returns
    -------
    X_all : pd.DataFrame
        All features (clinical + endoscopic).
    X_histwk : pd.DataFrame
        Clinical features only (history + week-8 non-endoscopic).
    X_endo : pd.DataFrame
        Endoscopic features only.
    all_features : list
        Names of all features.
    endo_cols : list
        Names of endoscopic features.
    binary_cols : list
        Names of binary features (for preprocessing).
    """
    # Define feature groups
    binary_hist = [
        "BIONAIVE",
        "BIMM",
        "RDIALL",
        "RDICORT",
        "SEX_M",
        "InductionMed_UST_SC",
    ]
    cont_hist = ["AGE"]

    wk8_cols = [
        "CALPL_wk8",
        "CRPL_wk8",
        "ABSSTOOL_wk8",
        "PGSCORE_wk8",
        "PMAYO_wk8",
        "RBSCORE_wk8",
        "SFSCORE_wk8",
    ]

    endo_cols = [
        "CDS_wk8",
        "MES_0_perc_wk8",
        "MES_1_perc_wk8",
        "MES_2_perc_wk8",
        "MES_3_perc_wk8",
        "CDS_abs_0_8",
        "MES_0_perc_Abs_0_8",
        "MES_1_perc_Abs_0_8",
        "MES_2_perc_Abs_0_8",
        "MES_3_perc_Abs_0_8",
    ]

    delta_cols = [
        "pMayo_abs_0_8",
        "RBscore_abs_0_8",
        "SFscore_abs_0_8",
        "PGscore_abs_0_8",
        "ABSstool_abs_0_8",
    ]

    hist_cols = cont_hist + binary_hist
    all_features = list(dict.fromkeys(hist_cols + wk8_cols + endo_cols + delta_cols))

    # Sanity check
    assert "CDS_wk8" in endo_cols and "CDS_wk8" not in wk8_cols

    X_all = df[all_features].copy()
    X_histwk = df[hist_cols + wk8_cols + delta_cols].copy()
    X_endo = df[endo_cols].copy()

    binary_cols = [c for c in binary_hist if c in df.columns]

    return X_all, X_histwk, X_endo, all_features, endo_cols, binary_cols


def build_preprocessor(
    X: pd.DataFrame, binary_features: List[str]
) -> ColumnTransformer:
    """
    Build sklearn ColumnTransformer for preprocessing.

    Numeric features: median imputation + z-score standardization.
    Binary features: mode imputation only.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    binary_features : list
        Names of binary features.

    Returns
    -------
    preprocessor : ColumnTransformer
        Fitted preprocessor.
    """
    num_cols = [
        c for c in X.columns if is_numeric_dtype(X[c]) and c not in binary_features
    ]
    bin_cols = [c for c in X.columns if c in binary_features]

    num_trans = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    bin_trans = Pipeline(steps=[("imputer", SimpleImputer(strategy="most_frequent"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_trans, num_cols),
            ("bin", bin_trans, bin_cols),
        ]
    )
    return preprocessor


def preprocess_all(
    X_all: pd.DataFrame,
    X_histwk: pd.DataFrame,
    X_endo: pd.DataFrame,
    binary_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[int], List[int]]:
    """
    Preprocess all feature matrices and compute feature indices.

    Parameters
    ----------
    X_all : pd.DataFrame
        All features.
    X_histwk : pd.DataFrame
        Clinical features only.
    X_endo : pd.DataFrame
        Endoscopic features only.
    binary_cols : list
        Names of binary features.

    Returns
    -------
    X_all_np : np.ndarray
        Preprocessed all features.
    X_histwk_np : np.ndarray
        Preprocessed clinical features.
    X_endo_np : np.ndarray
        Preprocessed endoscopic features.
    transformed_cols : list
        Column names after transformation.
    endo_idx : list
        Indices of endoscopic features in X_all_np.
    nonendo_idx : list
        Indices of non-endoscopic features in X_all_np.
    """
    pre_all = build_preprocessor(X_all, binary_cols)
    X_all_np = pre_all.fit_transform(X_all)

    pre_histwk = build_preprocessor(X_histwk, binary_cols)
    X_histwk_np = pre_histwk.fit_transform(X_histwk)

    pre_endo = build_preprocessor(X_endo, binary_cols)
    X_endo_np = pre_endo.fit_transform(X_endo)

    # Get actual column order after ColumnTransformer transformation
    num_cols_all = [
        c for c in X_all.columns if is_numeric_dtype(X_all[c]) and c not in binary_cols
    ]
    bin_cols_all = [c for c in X_all.columns if c in binary_cols]
    transformed_cols = num_cols_all + bin_cols_all

    # Compute indices for endoscopic vs non-endoscopic features
    endo_col_set = set(X_endo.columns)
    endo_idx = [i for i, c in enumerate(transformed_cols) if c in endo_col_set]
    nonendo_idx = [i for i, c in enumerate(transformed_cols) if c not in endo_col_set]

    return X_all_np, X_histwk_np, X_endo_np, transformed_cols, endo_idx, nonendo_idx


def group_summaries(X: np.ndarray) -> np.ndarray:
    """
    Compute group-level summary statistics (mean and std) for BLP testing.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (n_samples, n_features).

    Returns
    -------
    summaries : np.ndarray
        Array of shape (n_samples, 2) with mean and std per sample.
    """
    return np.c_[X.mean(axis=1), X.std(axis=1)]


# =============================================================================
# PERMUTATION IMPORTANCE AND BLP TESTING
# =============================================================================


def permimp_with_ci(
    estimator,
    X: np.ndarray,
    y: np.ndarray,
    repeats: int = 10,
    n_repeats: int = 10,
    seed: int = DEFAULT_RANDOM_STATE,
    return_matrix: bool = False,
):
    """
    Compute permutation importance with confidence intervals.

    Parameters
    ----------
    estimator : sklearn estimator
        Fitted model.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target variable.
    repeats : int
        Number of repetitions for CI estimation.
    n_repeats : int
        Number of permutations per repetition.
    seed : int
        Random seed.
    return_matrix : bool
        If True, also return the (repeats x n_features) matrix of per-repeat mean
        importances, from which group-level importance totals and covariance-aware
        standard errors can be computed.

    Returns
    -------
    importance_mean : np.ndarray
        Mean importance per feature.
    importance_se : np.ndarray
        Standard error of importance per feature.
    M : np.ndarray, optional
        Only when ``return_matrix`` is True. Shape (repeats, n_features).

    """
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(repeats):
        perm = permutation_importance(
            estimator, X, y, n_repeats=n_repeats,
            random_state=int(rng.integers(1_000_000_000)),
        )
        means.append(perm.importances_mean)
    M = np.vstack(means)
    mean = M.mean(axis=0)
    se = M.std(axis=0, ddof=1)
    if return_matrix:
        return mean, se, M
    return mean, se


def blp_test_multiplier(
    tau: np.ndarray,
    G_endo: np.ndarray,
    G_clinical: np.ndarray,
    B: int = 1000,
    seed: int = DEFAULT_RANDOM_STATE,
) -> Tuple[float, float]:
    """
    Joint test for whether the endoscopic summaries add linear predictive signal
    for the estimated CATE beyond the clinical summaries.

    The estimated CATE ``tau`` is projected (OLS) onto the clinical and
    endoscopic summaries, and we test whether the endoscopic coefficient block is
    jointly zero. A multiplier bootstrap gives the covariance of the
    coefficients; the test is a Wald quadratic form referenced to a chi-square
    distribution with ``k`` = number of endoscopic coefficients.

    The multiplier bootstrap treats ``tau`` as fixed data and
    reweights the second-stage regression residuals, so the resulting uncertainty
    is CONDITIONAL on the estimated CATE and does not propagate first-stage
    (X-learner) estimation error.

    Parameters
    ----------
    tau : np.ndarray
        Estimated CATEs.
    G_endo : np.ndarray
        Endoscopic feature summaries (n_samples, k_endo).
    G_clinical : np.ndarray
        Clinical feature summaries (n_samples, k_clinical).
    B : int
        Number of multiplier-bootstrap iterations.
    seed : int
        Random seed.

    Returns
    -------
    stat : float
        sqrt of the joint Wald statistic (a non-negative magnitude). This is not
        a z-score; the p-value below is the chi-square tail probability.
    p_value : float
        P(chi^2_k > Wald).
    """
    G = np.c_[G_clinical, G_endo]
    k_endo = G_endo.shape[1]

    lin = LinearRegression().fit(G, tau)
    coef = lin.coef_
    fitted = lin.predict(G)
    resid = tau - fitted

    rng = np.random.default_rng(seed)
    coefs = np.empty((B, G.shape[1]))
    for b in range(B):
        w = rng.normal(size=len(resid))
        yb = fitted + resid * w
        coefs[b] = LinearRegression().fit(G, yb).coef_

    coef_endo = coef[-k_endo:]
    cov_endo = np.atleast_2d(np.cov(coefs[:, -k_endo:], rowvar=False))

    # Joint Wald with the full bootstrap covariance (pinv is robust to
    # near-collinear summaries)
    wald = float(coef_endo @ np.linalg.pinv(cov_endo) @ coef_endo)
    p_endo = float(chi2.sf(wald, df=k_endo))

    return float(np.sqrt(wald)), p_endo


# =============================================================================
# BOOTSTRAP AND INFERENCE UTILITIES
# =============================================================================


def bootstrap_delta(
    drA: np.ndarray,
    drB: np.ndarray,
    n_boot: int = 5000,
    seed: int = DEFAULT_RANDOM_STATE,
    ci: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for the difference in doubly robust estimates.

    Parameters
    ----------
    drA : np.ndarray
        DR contributions for policy A.
    drB : np.ndarray
        DR contributions for policy B.
    n_boot : int
        Number of bootstrap samples.
    seed : int
        Random seed.
    ci : float
        Confidence level.

    Returns
    -------
    delta_mean : float
        Point estimate of difference.
    ci_lo : float
        Lower CI bound.
    ci_hi : float
        Upper CI bound.

    """
    rng = np.random.default_rng(seed)
    diff = drA - drB
    n = len(diff)

    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots.append(diff[idx].mean())
    boots = np.asarray(boots)

    alpha = 1.0 - ci
    lo = np.percentile(boots, 100 * (alpha / 2.0))
    hi = np.percentile(boots, 100 * (1 - alpha / 2.0))

    return diff.mean(), lo, hi


def summarize_from_dr(dr: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute point estimate and Wald CI from DR contributions.

    Parameters
    ----------
    dr : np.ndarray
        Doubly robust contributions.

    Returns
    -------
    value : float
        Point estimate (mean).
    ci_lo : float
        Lower 95% CI.
    ci_hi : float
        Upper 95% CI.
    """
    val = dr.mean()
    se = dr.std(ddof=1) / np.sqrt(len(dr))
    return val, val - 1.96 * se, val + 1.96 * se


# =============================================================================
# X-LEARNER BINARY CATE ESTIMATION
# =============================================================================


def xlearner_binary_crossfit(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_folds: int = 5,
    seed: int = DEFAULT_RANDOM_STATE,
) -> Dict:
    """
    X-learner with cross-fitting for binary treatment comparison.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    T : np.ndarray
        Binary treatment indicator.
    Y : np.ndarray
        Outcome variable.
    n_folds : int
        Number of cross-validation folds.
    seed : int
        Random seed.

    Returns
    -------
    results : dict
        Dictionary with keys:
        - mu0: Cross-fitted outcome predictions under control.
        - mu1: Cross-fitted outcome predictions under treatment.
        - tau: Cross-fitted CATE estimates.
        - g: Full-sample CATE model fit on treated.
        - p_assign: Propensity score matrix.
    """
    p1 = T.mean() if 0 < T.mean() < 1 else 0.5
    p_assign = np.tile([1 - p1, p1], (len(T), 1))

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    n = len(Y)
    mu0_hat = np.zeros(n)
    mu1_hat = np.zeros(n)
    tau_hat = np.zeros(n)

    for train_idx, test_idx in kf.split(X):
        # Stage 1: Fit outcome models
        mu0 = XGBRegressor(**OUTCOME_MODEL_PARAMS, random_state=seed)
        mu1 = XGBRegressor(**OUTCOME_MODEL_PARAMS, random_state=seed)

        train_ctrl = train_idx[T[train_idx] == 0]
        train_treat = train_idx[T[train_idx] == 1]

        mu0.fit(X[train_ctrl], Y[train_ctrl])
        mu1.fit(X[train_treat], Y[train_treat])

        mu0_hat[test_idx] = mu0.predict(X[test_idx])
        mu1_hat[test_idx] = mu1.predict(X[test_idx])

        # Stage 2: Compute imputed treatment effects
        Xt_train = X[train_treat]
        D1_train = Y[train_treat] - mu0.predict(Xt_train)

        Xc_train = X[train_ctrl]
        D0_train = mu1.predict(Xc_train) - Y[train_ctrl]

        # Fit CATE models on pseudo-outcomes
        g = XGBRegressor(**CATE_MODEL_PARAMS, random_state=seed)
        g.fit(Xt_train, D1_train)

        h = XGBRegressor(**CATE_MODEL_PARAMS, random_state=seed)
        h.fit(Xc_train, D0_train)

        # Stage 3: Combine the two X-learner estimates with propensity weighting.
        #   g = treated-group CATE estimate  (tau^(1), regressed on D1 = Y - mu0)
        #   h = control-group CATE estimate  (tau^(0), regressed on D0 = mu1 - Y)
        # Treatment was randomized, so the propensity e = P(T = 1) is constant.
        # Following Kuenzel et al. (2019), the propensity weights the control-group
        # estimate:
        #   tau(x) = e * tau^(0)(x) + (1 - e) * tau^(1)(x)   (Methods, Eq. 1)
        w = p_assign[test_idx, 1]  # e = P(T = 1)
        tau_hat[test_idx] = w * h.predict(X[test_idx]) + (1 - w) * g.predict(
            X[test_idx]
        )

    # Fit full-sample model for importance analysis
    mu0_full = XGBRegressor(**OUTCOME_MODEL_PARAMS, random_state=seed)
    mu0_full.fit(X[T == 0], Y[T == 0])

    treated_mask = T == 1
    y_target = Y[treated_mask] - mu0_full.predict(X[treated_mask])

    g_full = XGBRegressor(**CATE_MODEL_PARAMS, random_state=seed)
    g_full.fit(X[treated_mask], y_target)

    return {
        "mu0": mu0_hat,
        "mu1": mu1_hat,
        "tau": tau_hat,
        "g": g_full,
        "p_assign": p_assign,
    }


# =============================================================================
# DOUBLY ROBUST POLICY EVALUATION
# =============================================================================


def dr_value_binary(
    T: np.ndarray,
    Y: np.ndarray,
    pi: np.ndarray,
    mu0: np.ndarray,
    mu1: np.ndarray,
    p_assign: np.ndarray,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Doubly robust policy value estimation for binary treatment.

    Parameters
    ----------
    T : np.ndarray
        Observed treatment.
    Y : np.ndarray
        Observed outcome.
    pi : np.ndarray
        Policy assignments (0 or 1).
    mu0 : np.ndarray
        Outcome predictions under control.
    mu1 : np.ndarray
        Outcome predictions under treatment.
    p_assign : np.ndarray
        Propensity score matrix (n, 2).

    Returns
    -------
    value : float
        Policy value estimate.
    ci_lo : float
        Lower 95% CI.
    ci_hi : float
        Upper 95% CI.
    dr : np.ndarray
        Individual DR contributions.
    """
    mu_pi = np.where(pi == 1, mu1, mu0)
    p_pi = np.where(pi == 1, p_assign[:, 1], p_assign[:, 0])
    I = (T == pi).astype(float)

    dr = mu_pi + I * (Y - mu_pi) / np.clip(p_pi, 1e-6, 1.0)
    val, lo, hi = summarize_from_dr(dr)

    return val, lo, hi, dr


# =============================================================================
# THREE-ARM POLICY EVALUATION
# =============================================================================


def fit_three_arm_models(
    X: np.ndarray, T3: np.ndarray, Y: np.ndarray, seed: int = DEFAULT_RANDOM_STATE
) -> List[XGBRegressor]:
    """
    Fit separate outcome models for each of three treatment arms.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    T3 : np.ndarray
        Three-arm treatment indicator (0, 1, 2).
    Y : np.ndarray
        Outcome variable.
    seed : int
        Random seed.

    Returns
    -------
    models : list
        List of three fitted XGBRegressor models [placebo, Q12, Q8].
    """
    models = []
    for arm in [0, 1, 2]:
        m = XGBRegressor(**OUTCOME_MODEL_PARAMS, random_state=seed)
        mask = T3 == arm
        m.fit(X[mask], Y[mask])
        models.append(m)
    return models


def optimal_arm_assignment(
    muP: np.ndarray, mu12: np.ndarray, mu8: np.ndarray
) -> np.ndarray:
    """
    Compute optimal policy assignment for three-arm comparison.

    Parameters
    ----------
    muP : np.ndarray
        Predicted outcomes under placebo.
    mu12 : np.ndarray
        Predicted outcomes under Q12.
    mu8 : np.ndarray
        Predicted outcomes under Q8.

    Returns
    -------
    pi : np.ndarray
        Optimal arm assignment (0, 1, or 2).
    """
    mu_all = np.column_stack([muP, mu12, mu8])
    return np.argmax(mu_all, axis=1)


def dr_value_three_arm(
    T: np.ndarray,
    Y: np.ndarray,
    pi: np.ndarray,
    muP: np.ndarray,
    mu12: np.ndarray,
    mu8: np.ndarray,
    p_assign3: np.ndarray,
) -> Tuple[float, float, float, np.ndarray]:
    """
    Doubly robust policy value estimation for three-arm comparison.

    Parameters
    ----------
    T : np.ndarray
        Observed treatment (0, 1, 2).
    Y : np.ndarray
        Observed outcome.
    pi : np.ndarray
        Policy assignments (0, 1, or 2).
    muP, mu12, mu8 : np.ndarray
        Outcome predictions for each arm.
    p_assign3 : np.ndarray
        Propensity score matrix (n, 3).

    Returns
    -------
    value : float
        Policy value estimate.
    ci_lo : float
        Lower 95% CI.
    ci_hi : float
        Upper 95% CI.
    dr : np.ndarray
        Individual DR contributions.
    """
    mu_all = np.column_stack([muP, mu12, mu8])
    n = len(Y)

    mu_pi = mu_all[np.arange(n), pi]
    p_pi = p_assign3[np.arange(n), pi]
    I = (T == pi).astype(float)

    dr = mu_pi + I * (Y - mu_pi) / np.clip(p_pi, 1e-6, 1.0)
    val, lo, hi = summarize_from_dr(dr)

    return val, lo, hi, dr


def compute_three_arm_pseudo_outcome(
    T: np.ndarray, muP: np.ndarray, mu12: np.ndarray, mu8: np.ndarray
) -> np.ndarray:
    """
    Compute optimality gap pseudo-outcome for three-arm analysis.

    Parameters
    ----------
    T : np.ndarray
        Observed treatment.
    muP, mu12, mu8 : np.ndarray
        Outcome predictions for each arm.

    Returns
    -------
    tau_gap : np.ndarray
        Optimality gap (max predicted - assigned predicted).
    """
    mu_all = np.column_stack([muP, mu12, mu8])
    mu_T = mu_all[np.arange(len(T)), T]
    mu_best = mu_all.max(axis=1)
    return mu_best - mu_T


def nested_three_arm_oof_values_foldsafe(
    X_all_df,
    X_histwk_df,
    T3: np.ndarray,
    Y: np.ndarray,
    binary_cols: List[str],
    p_assign3: np.ndarray,
    n_splits: int = 5,
    seed: int = DEFAULT_RANDOM_STATE,
) -> Dict:

    X_all_df = X_all_df.reset_index(drop=True)
    X_histwk_df = X_histwk_df.reset_index(drop=True)
    T3 = np.asarray(T3)
    Y = np.asarray(Y)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    dr_all_list = []
    dr_clinical_list = []

    for tr, te in skf.split(X_all_df, T3):
        pre_all = build_preprocessor(X_all_df.iloc[tr], binary_cols)
        Xa_tr = pre_all.fit_transform(X_all_df.iloc[tr])
        Xa_te = pre_all.transform(X_all_df.iloc[te])

        pre_clin = build_preprocessor(X_histwk_df.iloc[tr], binary_cols)
        Xc_tr = pre_clin.fit_transform(X_histwk_df.iloc[tr])
        Xc_te = pre_clin.transform(X_histwk_df.iloc[te])

        models_all = fit_three_arm_models(Xa_tr, T3[tr], Y[tr], seed=seed)
        models_clinical = fit_three_arm_models(Xc_tr, T3[tr], Y[tr], seed=seed)

        muP_all_te = models_all[0].predict(Xa_te)
        mu12_all_te = models_all[1].predict(Xa_te)
        mu8_all_te = models_all[2].predict(Xa_te)

        muP_clinical_te = models_clinical[0].predict(Xc_te)
        mu12_clinical_te = models_clinical[1].predict(Xc_te)
        mu8_clinical_te = models_clinical[2].predict(Xc_te)

        pi_all_te = optimal_arm_assignment(muP_all_te, mu12_all_te, mu8_all_te)
        pi_clinical_te = optimal_arm_assignment(
            muP_clinical_te, mu12_clinical_te, mu8_clinical_te
        )

        _, _, _, dr_all_te = dr_value_three_arm(
            T3[te], Y[te], pi_all_te,
            muP_all_te, mu12_all_te, mu8_all_te, p_assign3[te],
        )
        _, _, _, dr_clinical_te = dr_value_three_arm(
            T3[te], Y[te], pi_clinical_te,
            muP_clinical_te, mu12_clinical_te, mu8_clinical_te, p_assign3[te],
        )

        dr_all_list.append(dr_all_te)
        dr_clinical_list.append(dr_clinical_te)

    dr_all = np.concatenate(dr_all_list)
    dr_clinical = np.concatenate(dr_clinical_list)

    val_all, lo_all, hi_all = summarize_from_dr(dr_all)
    val_clinical, lo_clinical, hi_clinical = summarize_from_dr(dr_clinical)
    delta_mean, d_lo, d_hi = bootstrap_delta(dr_all, dr_clinical, n_boot=5000, seed=seed)

    return {
        "dr_all": dr_all,
        "dr_clinical": dr_clinical,
        "value_all": val_all,
        "ci_all": (lo_all, hi_all),
        "value_clinical": val_clinical,
        "ci_clinical": (lo_clinical, hi_clinical),
        "delta": delta_mean,
        "delta_ci": (d_lo, d_hi),
    }


# =============================================================================
# PROGNOSTIC MODEL EVALUATION
# =============================================================================


def oof_proba(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 5,
    seed: int = DEFAULT_RANDOM_STATE,
) -> np.ndarray:
    """
    Compute out-of-fold predicted probabilities using calibrated XGBoost.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Binary outcome.
    n_splits : int
        Number of CV folds.
    seed : int
        Random seed.

    Returns
    -------
    proba : np.ndarray
        Out-of-fold predicted probabilities.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    proba = np.zeros_like(y, dtype=float)

    for tr, te in skf.split(X, y):
        clf = CalibratedClassifierCV(
            XGBClassifier(**PROGNOSTIC_MODEL_PARAMS, random_state=seed),
            cv=3,
            method="isotonic",
        )
        clf.fit(X[tr], y[tr])
        proba[te] = clf.predict_proba(X[te])[:, 1]

    return proba


def bootstrap_ci_delta_perf(
    y: np.ndarray,
    p_all: np.ndarray,
    p_clinical: np.ndarray,
    metric: str = "brier",
    n_boot: int = 5000,
    seed: int = DEFAULT_RANDOM_STATE,
) -> Tuple[float, float, float]:
    """
    Bootstrap CI for difference in performance metrics between two models.

    Parameters
    ----------
    y : np.ndarray
        True outcomes.
    p_all : np.ndarray
        Predictions from all-features model.
    p_clinical : np.ndarray
        Predictions from clinical-only model.
    metric : str
        One of 'brier', 'log', or 'auc'.
    n_boot : int
        Number of bootstrap samples.
    seed : int
        Random seed.

    Returns
    -------
    point : float
        Point estimate of difference.
    ci_lo : float
        Lower 95% CI.
    ci_hi : float
        Upper 95% CI.
    """
    rng = np.random.default_rng(seed)

    if metric == "brier":
        contrib = (y - p_clinical) ** 2 - (y - p_all) ** 2
        point = contrib.mean()
    elif metric == "log":
        eps = 1e-12
        pall = np.clip(p_all, eps, 1 - eps)
        pclinical = np.clip(p_clinical, eps, 1 - eps)
        contrib = (
            -y * np.log(pclinical)
            - (1 - y) * np.log(1 - pclinical)
            + y * np.log(pall)
            + (1 - y) * np.log(1 - pall)
        )
        point = contrib.mean()
    elif metric == "auc":

        def auc_p(p):
            try:
                return roc_auc_score(y, p)
            except Exception:
                return np.nan

        base_all = auc_p(p_all)
        base_clinical = auc_p(p_clinical)
        point = base_all - base_clinical
        contrib = None
    else:
        raise ValueError("metric must be 'brier', 'log', or 'auc'.")

    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), size=len(y))
        if metric == "auc":
            try:
                a = roc_auc_score(y[idx], p_all[idx])
                b = roc_auc_score(y[idx], p_clinical[idx])
                boots.append(a - b)
            except Exception:
                continue
        else:
            boots.append(contrib[idx].mean())

    boots = np.array(boots)
    if len(boots) == 0:
        return point, np.nan, np.nan

    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def summarize_perf(
    y: np.ndarray, p: np.ndarray, label: str
) -> Tuple[float, float, float, float]:
    """
    Compute and print standard performance metrics.

    Parameters
    ----------
    y : np.ndarray
        True outcomes.
    p : np.ndarray
        Predicted probabilities.
    label : str
        Label for printing.

    Returns
    -------
    brier : float
        Brier score.
    ll : float
        Log loss.
    auc : float
        AUROC.
    aupr : float
        AUPRC.
    """
    brier = brier_score_loss(y, p)
    ll = log_loss(y, p, labels=[0, 1])
    auc = roc_auc_score(y, p)
    aupr = average_precision_score(y, p)
    print(f"{label} Brier {brier:.3f} LogLoss {ll:.3f} AUROC {auc:.3f} AUPRC {aupr:.3f}")
    return brier, ll, auc, aupr


def incremental_brier(
    y: np.ndarray, p_all: np.ndarray, p_clinical: np.ndarray
) -> float:
    """
    Compute the incremental Brier score (manuscript Eqs. 5 and 7), defined as
    the incremental IPA:
    (Brier_clinical - Brier_all) / Brier_null = IPA_all - IPA_clinical.

    Positive values indicate endoscopic features improve calibration; values at
    or below zero indicate no incremental benefit. Brier_null = ybar(1 - ybar)
    is positive for any non-degenerate outcome, so the metric is always defined
    and stable.

    Parameters
    ----------
    y : np.ndarray
        True outcomes.
    p_all : np.ndarray
        Predictions from all-features model.
    p_clinical : np.ndarray
        Predictions from clinical-only model.

    Returns
    -------
    incremental : float
        Incremental Brier score (incremental IPA). ``np.nan`` only in the
        degenerate case of a constant outcome (Brier_null = 0).
    """
    b_all = brier_score_loss(y, p_all)
    b_clinical = brier_score_loss(y, p_clinical)
    b_null = brier_score_loss(y, np.repeat(y.mean(), len(y)))

    if b_null <= 1e-12:
        return np.nan
    return (b_clinical - b_all) / b_null


def bootstrap_ci_incremental_brier(
    y: np.ndarray,
    p_all: np.ndarray,
    p_clinical: np.ndarray,
    n_boot: int = 5000,
    seed: int = DEFAULT_RANDOM_STATE,
) -> Tuple[float, float, float, float]:
    """
    Paired patient-level bootstrap CI for the incremental Brier score
    (``incremental_brier``). Resamples patients (the same indices applied to
    ``y``, ``p_all`` and ``p_clinical``), recomputes the metric on each resample,
    and returns the 2.5th/97.5th percentile interval.

    Returns
    -------
    (point, ci_lo, ci_hi, frac_defined)
    """
    y = np.asarray(y)
    p_all = np.asarray(p_all)
    p_clinical = np.asarray(p_clinical)
    point = incremental_brier(y, p_all, p_clinical)

    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        v = incremental_brier(y[idx], p_all[idx], p_clinical[idx])
        if np.isfinite(v):
            vals.append(v)
    frac_defined = len(vals) / n_boot if n_boot else 0.0

    if not np.isfinite(point) or frac_defined < 0.5 or len(vals) < 2:
        return point, np.nan, np.nan, frac_defined
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return point, float(lo), float(hi), frac_defined


def policy_agreement_analysis(
    pi_all: np.ndarray, pi_clinical: np.ndarray, comparison_name: str
) -> Dict:
    """
    Analyze agreement between all-features and clinical-only policies.

    Parameters
    ----------
    pi_all : np.ndarray
        Policy decisions using all features.
    pi_clinical : np.ndarray
        Policy decisions using clinical features only.
    comparison_name : str
        Label for the comparison.

    Returns
    -------
    results : dict
        Dictionary with n_agree, n_disagree, and agreement_rate.
    """
    n_total = len(pi_all)
    agree_mask = pi_all == pi_clinical
    n_agree = agree_mask.sum()
    n_disagree = n_total - n_agree
    agreement_rate = n_agree / n_total

    print(f"\n{comparison_name} Policy Agreement:")
    print(f"  N agree: {n_agree}, N disagree: {n_disagree}")
    print(f"  Agreement rate: {agreement_rate:.1%}")

    return {
        "n_agree": n_agree,
        "n_disagree": n_disagree,
        "agreement_rate": agreement_rate,
    }


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================


def plot_feature_importance(
    imp_df: pd.DataFrame,
    endo_cols: List[str],
    title: str,
    filename: str,
    top_n: int = 15,
) -> None:
    """
    Create horizontal bar chart of feature importance, color-coded by category.

    Parameters
    ----------
    imp_df : pd.DataFrame
        DataFrame with columns 'feature', 'importance_mean', 'importance_se'.
    endo_cols : list
        Names of endoscopic features.
    title : str
        Plot title.
    filename : str
        Output filename.
    top_n : int
        Number of top features to display.
    """
    df = imp_df.head(top_n).copy()
    df = df.iloc[::-1]  # Reverse for horizontal bar plot

    endo_set = set(endo_cols)
    colors = ["#E74C3C" if f in endo_set else "#3498DB" for f in df["feature"]]

    fig, ax = plt.subplots(figsize=(10, 8))

    ax.barh(
        df["feature"],
        df["importance_mean"],
        xerr=1.96 * df["importance_se"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
        capsize=3,
        error_kw={"elinewidth": 1, "capthick": 1},
    )

    ax.set_xlabel("Permutation Importance", fontsize=12)
    ax.tick_params(axis="y", labelsize=10)
    ax.tick_params(axis="x", labelsize=10)

    clinical_patch = mpatches.Patch(color="#3498DB", label="Clinical/Lab")
    endo_patch = mpatches.Patch(color="#E74C3C", label="Endoscopic")
    ax.legend(handles=[clinical_patch, endo_patch], loc="lower right", fontsize=10)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def plot_policy_forest(policy_results: List[Dict], filename: str) -> None:
    """
    Create forest plot of policy values with 95% CIs.

    Parameters
    ----------
    policy_results : list
        List of dicts with keys: label, value, ci_lo, ci_hi, color.
    filename : str
        Output filename.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = []
    values = []
    ci_los = []
    ci_his = []
    colors = []

    for result in policy_results:
        labels.append(result["label"])
        values.append(result["value"])
        ci_los.append(result["ci_lo"])
        ci_his.append(result["ci_hi"])
        colors.append(result["color"])

    y_pos = np.arange(len(labels))

    for i, (val, lo, hi, color) in enumerate(zip(values, ci_los, ci_his, colors)):
        ax.errorbar(
            val,
            i,
            xerr=[[val - lo], [hi - val]],
            fmt="o",
            color=color,
            markersize=10,
            capsize=5,
            capthick=2,
            elinewidth=2,
        )

    ax.axvline(x=0.3, color="gray", linestyle="--", alpha=0.5, linewidth=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel("Expected Remission Rate (Doubly Robust)", fontsize=12)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*100:.0f}%"))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    all_patch = mpatches.Patch(color="#2E86AB", label="All Features")
    clin_patch = mpatches.Patch(color="#A23B72", label="Clinical Only")
    ax.legend(handles=[all_patch, clin_patch], loc="lower right", fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def plot_cate_distributions(
    tau_ust: np.ndarray, tau_q8: np.ndarray, tau_3arm: np.ndarray, filename: str
) -> None:
    """
    Create CATE distribution plots for all three comparisons.

    Parameters
    ----------
    tau_ust : np.ndarray
        CATEs for UST vs Placebo.
    tau_q8 : np.ndarray
        CATEs for Q8 vs Q12.
    tau_3arm : np.ndarray
        Optimality gaps for three-arm.
    filename : str
        Output filename.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    hist_kwargs = dict(bins=30, alpha=0.7, edgecolor="black", linewidth=0.5)

    # UST vs Placebo
    ax = axes[0]
    ax.hist(tau_ust, color="#3498DB", **hist_kwargs)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="No effect")
    ax.axvline(
        x=tau_ust.mean(),
        color="darkblue",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {tau_ust.mean():.3f}",
    )
    ax.set_xlabel("CATE (τ)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("UST vs Placebo", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Q8 vs Q12
    ax = axes[1]
    ax.hist(tau_q8, color="#E74C3C", **hist_kwargs)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="No effect")
    ax.axvline(
        x=tau_q8.mean(),
        color="darkred",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {tau_q8.mean():.3f}",
    )
    ax.set_xlabel("CATE (τ)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Q8 vs Q12", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Three-arm optimality gap
    ax = axes[2]
    ax.hist(tau_3arm, color="#27AE60", **hist_kwargs)
    ax.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Optimal assignment")
    ax.axvline(
        x=tau_3arm.mean(),
        color="darkgreen",
        linestyle="-",
        linewidth=2,
        label=f"Mean: {tau_3arm.mean():.3f}",
    )
    ax.set_xlabel("Optimality Gap", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title("Multi-Arm Policy", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


# =============================================================================
# EXTENDED VALIDATION ANALYSES
# GATES, CLAN, and alternative CATE estimators (paper Tables 5-7, Eqs. 8-9)
# =============================================================================


def gates_analysis(
    tau: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    n_groups: int = 5,
    seed: int = DEFAULT_RANDOM_STATE,
) -> Dict:
    """
    GATES: Sorted Group Average Treatment Effects (paper Table 5, Eq. 8).

    Bins patients by predicted CATE into ``n_groups`` quantile groups, estimates
    the observed ATE within each group (difference in means), and tests for
    heterogeneity across groups with a treatment-permutation test. Also reports
    the Spearman correlation between predicted group CATE means and observed
    group ATEs.
    """
    tau = np.asarray(tau)
    T = np.asarray(T)
    Y = np.asarray(Y)

    quantiles = np.percentile(tau, np.linspace(0, 100, n_groups + 1))
    quantiles[0] = tau.min() - 1e-10
    quantiles[-1] = tau.max() + 1e-10
    groups = np.clip(np.digitize(tau, quantiles) - 1, 0, n_groups - 1)

    group_ates, group_cis, group_ns, group_tau_means = [], [], [], []
    for g in range(n_groups):
        mask = groups == g
        group_ns.append(int(mask.sum()))
        group_tau_means.append(tau[mask].mean())
        y1 = Y[mask][T[mask] == 1]
        y0 = Y[mask][T[mask] == 0]
        if len(y1) > 0 and len(y0) > 0:
            ate_g = y1.mean() - y0.mean()
            se_g = np.sqrt(y1.var(ddof=1) / len(y1) + y0.var(ddof=1) / len(y0))
            group_ates.append(ate_g)
            group_cis.append((ate_g - 1.96 * se_g, ate_g + 1.96 * se_g))
        else:
            group_ates.append(np.nan)
            group_cis.append((np.nan, np.nan))

    valid = [a for a in group_ates if not np.isnan(a)]
    observed_range = (max(valid) - min(valid)) if len(valid) >= 2 else np.nan

    # Omnibus heterogeneity test: permute treatment, compare range of group ATEs
    rng = np.random.default_rng(seed)
    null_ranges = []
    for _ in range(1000):
        T_perm = rng.permutation(T)
        perm_ates = []
        for g in range(n_groups):
            mask = groups == g
            y1 = Y[mask][T_perm[mask] == 1]
            y0 = Y[mask][T_perm[mask] == 0]
            if len(y1) > 0 and len(y0) > 0:
                perm_ates.append(y1.mean() - y0.mean())
        if len(perm_ates) >= 2:
            null_ranges.append(max(perm_ates) - min(perm_ates))
    null_ranges = np.array(null_ranges)
    p_heterogeneity = float((null_ranges >= observed_range).mean())

    valid_idx = ~np.isnan(np.array(group_ates))
    if valid_idx.sum() >= 3:
        corr, corr_p = spearmanr(
            np.array(group_tau_means)[valid_idx], np.array(group_ates)[valid_idx]
        )
    else:
        corr, corr_p = np.nan, np.nan

    return {
        "n_groups": n_groups,
        "group_ns": group_ns,
        "group_tau_means": group_tau_means,
        "group_ates": group_ates,
        "group_cis": group_cis,
        "observed_range": observed_range,
        "p_heterogeneity": p_heterogeneity,
        "correlation": corr,
        "correlation_p": corr_p,
    }


def clan_analysis(
    tau: np.ndarray,
    X: np.ndarray,
    feature_names: List[str],
    n_groups: int = 3,
    seed: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """
    CLAN: Characteristics of Affected Subgroups (paper Table 6).

    Compares feature distributions between the top and bottom CATE tertiles
    using two-sample t-tests, identifying which features differentiate patients
    predicted to benefit most vs. least.
    """
    tau = np.asarray(tau)
    q_low = np.percentile(tau, 100 / n_groups)
    q_high = np.percentile(tau, 100 - 100 / n_groups)
    low_mask = tau <= q_low
    high_mask = tau >= q_high

    rows = []
    for i, feat in enumerate(feature_names):
        x_low = X[low_mask, i]
        x_high = X[high_mask, i]
        if len(x_low) > 1 and len(x_high) > 1:
            t_stat, p_val = ttest_ind(x_high, x_low)
        else:
            t_stat, p_val = np.nan, np.nan
        rows.append({
            "feature": feat,
            "mean_low_cate": x_low.mean(),
            "mean_high_cate": x_high.mean(),
            "difference": x_high.mean() - x_low.mean(),
            "t_statistic": t_stat,
            "p_value": p_val,
        })

    df = pd.DataFrame(rows)
    df["abs_t"] = df["t_statistic"].abs()
    return df.sort_values("abs_t", ascending=False)


def tlearner_crossfit(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    n_folds: int = 5, seed: int = DEFAULT_RANDOM_STATE,
) -> Dict:
    """T-learner: separate outcome models, tau(x) = mu1(x) - mu0(x)."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    n = len(Y)
    mu0_hat, mu1_hat = np.zeros(n), np.zeros(n)
    for tr, te in kf.split(X):
        mu0 = XGBRegressor(**OUTCOME_MODEL_PARAMS, random_state=seed)
        mu1 = XGBRegressor(**OUTCOME_MODEL_PARAMS, random_state=seed)
        mu0.fit(X[tr[T[tr] == 0]], Y[tr[T[tr] == 0]])
        mu1.fit(X[tr[T[tr] == 1]], Y[tr[T[tr] == 1]])
        mu0_hat[te] = mu0.predict(X[te])
        mu1_hat[te] = mu1.predict(X[te])
    return {"tau": mu1_hat - mu0_hat, "mu0": mu0_hat, "mu1": mu1_hat}


def slearner_crossfit(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    n_folds: int = 5, seed: int = DEFAULT_RANDOM_STATE,
) -> Dict:
    """S-learner: single model with treatment as a feature."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    n = len(Y)
    mu0_hat, mu1_hat, tau_hat = np.zeros(n), np.zeros(n), np.zeros(n)
    for tr, te in kf.split(X):
        mu = XGBRegressor(**OUTCOME_MODEL_PARAMS, random_state=seed)
        mu.fit(np.column_stack([X[tr], T[tr]]), Y[tr])
        mu0_hat[te] = mu.predict(np.column_stack([X[te], np.zeros(len(te))]))
        mu1_hat[te] = mu.predict(np.column_stack([X[te], np.ones(len(te))]))
        tau_hat[te] = mu1_hat[te] - mu0_hat[te]
    return {"tau": tau_hat, "mu0": mu0_hat, "mu1": mu1_hat}


def drlearner_crossfit(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray,
    n_folds: int = 5, seed: int = DEFAULT_RANDOM_STATE,
) -> Dict:
    """
    DR-learner (paper Eq. 9): doubly robust pseudo-outcome regression with a
    constant (randomized) propensity.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    n = len(Y)
    mu0_hat, mu1_hat, tau_hat = np.zeros(n), np.zeros(n), np.zeros(n)
    for tr, te in kf.split(X):
        mu0 = XGBRegressor(**OUTCOME_MODEL_PARAMS, random_state=seed)
        mu1 = XGBRegressor(**OUTCOME_MODEL_PARAMS, random_state=seed)
        mu0.fit(X[tr[T[tr] == 0]], Y[tr[T[tr] == 0]])
        mu1.fit(X[tr[T[tr] == 1]], Y[tr[T[tr] == 1]])
        mu0_hat[te] = mu0.predict(X[te])
        mu1_hat[te] = mu1.predict(X[te])

    e = T.mean()
    y_dr = (mu1_hat - mu0_hat
            + T * (Y - mu1_hat) / e
            - (1 - T) * (Y - mu0_hat) / (1 - e))

    for tr, te in kf.split(X):
        tau_model = XGBRegressor(**CATE_MODEL_PARAMS, random_state=seed)
        tau_model.fit(X[tr], y_dr[tr])
        tau_hat[te] = tau_model.predict(X[te])
    return {"tau": tau_hat, "mu0": mu0_hat, "mu1": mu1_hat}


def compare_cate_estimators(
    X: np.ndarray, T: np.ndarray, Y: np.ndarray, seed: int = DEFAULT_RANDOM_STATE
) -> Dict:
    """
    Compare X-, T-, S-, and DR-learner CATE estimates (paper Table 7).

    Uses the same X-learner as the main analysis, so all estimates share the
    library's propensity-weighted combination.
    """
    print("Running X-learner...")
    xlearner = xlearner_binary_crossfit(X, T, Y, n_folds=5, seed=seed)
    print("Running T-learner...")
    tlearner = tlearner_crossfit(X, T, Y, n_folds=5, seed=seed)
    print("Running S-learner...")
    slearner = slearner_crossfit(X, T, Y, n_folds=5, seed=seed)
    print("Running DR-learner...")
    drlearner = drlearner_crossfit(X, T, Y, n_folds=5, seed=seed)

    tau_x, tau_t = xlearner["tau"], tlearner["tau"]
    tau_s, tau_dr = slearner["tau"], drlearner["tau"]
    correlations = {
        "X-T": spearmanr(tau_x, tau_t)[0],
        "X-S": spearmanr(tau_x, tau_s)[0],
        "X-DR": spearmanr(tau_x, tau_dr)[0],
        "T-S": spearmanr(tau_t, tau_s)[0],
        "T-DR": spearmanr(tau_t, tau_dr)[0],
        "S-DR": spearmanr(tau_s, tau_dr)[0],
    }
    summary = pd.DataFrame({
        "Estimator": ["X-learner", "T-learner", "S-learner", "DR-learner"],
        "Mean CATE": [tau_x.mean(), tau_t.mean(), tau_s.mean(), tau_dr.mean()],
        "Std CATE": [tau_x.std(), tau_t.std(), tau_s.std(), tau_dr.std()],
        "Min CATE": [tau_x.min(), tau_t.min(), tau_s.min(), tau_dr.min()],
        "Max CATE": [tau_x.max(), tau_t.max(), tau_s.max(), tau_dr.max()],
    })
    return {
        "xlearner": xlearner, "tlearner": tlearner,
        "slearner": slearner, "drlearner": drlearner,
        "correlations": correlations, "summary": summary,
    }
