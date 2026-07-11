#!/usr/bin/env python3
"""
run_analysis.py -- reproduces the full analysis for:

    "A Causal Machine Learning Framework for Treatment Personalization in
     Clinical Trials: Application to Ulcerative Colitis"

Runs the complete pipeline on individual-patient data from the UNIFI
maintenance trial of ustekinumab in ulcerative colitis, using the methods in
`causal_utils.py`, and prints/saves every result reported in the paper.

Analyses (and where they appear in the paper)
---------------------------------------------
1. UST pooled (Q8+Q12) vs Placebo        -> Tables 2-3, Fig. 1-2
2. Q8 vs Q12 among active treatment       -> Tables 2-3, Fig. 2
3. Multi-arm policy (Placebo/Q12/Q8)      -> Fig. 2 (incl. nested out-of-fold)
4. Prognostic contribution / incr. Brier  -> Table 4
5. GATES / CLAN / alternative learners    -> Tables 5-7
6. Policy agreement                       -> Table 8
7. Predictive cross-validation summary

Usage
-----
    pip install -r requirements.txt
    python run_analysis.py

Output
------
    - Console output with all analysis results (Tables 2-8)
    - fig1_feature_importance.pdf
    - fig2_policy_forest.pdf
    - fig3_cate_distributions.pdf
    - table_*.csv

Data availability
-----------------
The UNIFI individual-patient data are not public (data-sharing agreement with
the trial sponsor); qualified researchers may request access from the sponsor.
"""

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier, XGBRegressor

from causal_utils import (
    DEFAULT_RANDOM_STATE,
    CATE_MODEL_PARAMS,
    PROGNOSTIC_MODEL_PARAMS,
    # Data loading and preprocessing
    load_unifi_data,
    build_feature_matrices,
    preprocess_all,
    group_summaries,
    # CATE estimation
    xlearner_binary_crossfit,
    permimp_with_ci,
    blp_test_multiplier,
    # Policy evaluation
    dr_value_binary,
    fit_three_arm_models,
    optimal_arm_assignment,
    dr_value_three_arm,
    compute_three_arm_pseudo_outcome,
    nested_three_arm_oof_values_foldsafe,
    bootstrap_delta,
    # Prognostic evaluation
    oof_proba,
    summarize_perf,
    bootstrap_ci_delta_perf,
    incremental_brier,
    bootstrap_ci_incremental_brier,
    policy_agreement_analysis,
    # Extended validation analyses (GATES, CLAN, alternative learners)
    gates_analysis,
    clan_analysis,
    compare_cate_estimators,
    # Visualization
    plot_feature_importance,
    plot_policy_forest,
    plot_cate_distributions,
)

# Set random seed for reproducibility
RNG = DEFAULT_RANDOM_STATE
np.random.seed(RNG)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print()
    print("-" * 80)
    print(title)
    print("-" * 80)


def save_table(df: pd.DataFrame, filename: str) -> None:
    """Write a results table to CSV and report the path."""
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")


def run_ust_vs_placebo_analysis(
    X_all_np: np.ndarray,
    X_histwk_np: np.ndarray,
    X_endo_np: np.ndarray,
    Tbin: np.ndarray,
    Y: np.ndarray,
    all_cols: list,
    endo_idx: list,
    nonendo_idx: list,
) -> dict:
    """
    Run UST pooled (Q8+Q12) vs Placebo analysis.

    Returns dictionary with results for downstream use.
    """
    print_section("UST pooled (Q8+Q12) vs PLACEBO")

    # Fit X-learner models
    res_all = xlearner_binary_crossfit(X_all_np, Tbin, Y, n_folds=5, seed=RNG)
    res_clinical = xlearner_binary_crossfit(X_histwk_np, Tbin, Y, n_folds=5, seed=RNG)

    # Compute permutation importance
    treated = Tbin == 1
    y_target = Y[treated] - res_all["mu0"][treated]
    imp_mean, imp_se = permimp_with_ci(
        res_all["g"], X_all_np[treated], y_target, repeats=10, n_repeats=10, seed=RNG
    )

    # Aggregate importance by feature group
    endo_total = imp_mean[endo_idx].sum()
    endo_se = np.sqrt((imp_se[endo_idx] ** 2).sum())
    nonendo_total = imp_mean[nonendo_idx].sum()
    nonendo_se = np.sqrt((imp_se[nonendo_idx] ** 2).sum())
    ratio = nonendo_total / max(endo_total, 1e-12)

    print(
        f"CATE importance: endoscopic {endo_total:.4f}±{1.96 * endo_se:.4f}, "
        f"non-endoscopic {nonendo_total:.4f}±{1.96 * nonendo_se:.4f}, "
        f"ratio {ratio:.2f}"
    )

    # BLP test
    G_endo = group_summaries(X_endo_np)
    G_clinical = group_summaries(X_histwk_np)
    z, p = blp_test_multiplier(res_all["tau"], G_endo, G_clinical, B=1000, seed=RNG)
    print(f"BLP test: z={z:.2f}, p={p:.4g}")

    # Feature importance dataframe
    imp_df = pd.DataFrame(
        {"feature": all_cols, "importance_mean": imp_mean, "importance_se": imp_se}
    ).sort_values("importance_mean", ascending=False)

    print("\nTop 15 Features:")
    print(imp_df.head(15)[["feature", "importance_mean"]].to_string(index=False))

    # Policy evaluation
    pi_all = (res_all["mu1"] - res_all["mu0"] > 0).astype(int)
    pi_clinical = (res_clinical["mu1"] - res_clinical["mu0"] > 0).astype(int)

    v_all, ci_all_lo, ci_all_hi, dr_all = dr_value_binary(
        Tbin, Y, pi_all, res_all["mu0"], res_all["mu1"], res_all["p_assign"]
    )
    v_clinical, ci_clinical_lo, ci_clinical_hi, dr_clinical = dr_value_binary(
        Tbin, Y, pi_clinical, res_clinical["mu0"], res_clinical["mu1"], res_clinical["p_assign"]
    )
    delta, d_lo, d_hi = bootstrap_delta(dr_all, dr_clinical, n_boot=5000, seed=RNG)

    print(f"\nPolicy value ALL: {v_all:.3f} CI ({ci_all_lo:.3f}, {ci_all_hi:.3f})")
    print(f"Policy value Clinical: {v_clinical:.3f} CI ({ci_clinical_lo:.3f}, {ci_clinical_hi:.3f})")
    print(f"Difference: {v_all - v_clinical:.3f} bootstrap CI ({d_lo:.3f}, {d_hi:.3f})")

    # Policy agreement
    agreement = policy_agreement_analysis(pi_all, pi_clinical, "UST vs PBO")

    return {
        "res_all": res_all,
        "res_clinical": res_clinical,
        "imp_df": imp_df,
        "v_all": v_all,
        "ci_all": (ci_all_lo, ci_all_hi),
        "v_clinical": v_clinical,
        "ci_clinical": (ci_clinical_lo, ci_clinical_hi),
        "pi_all": pi_all,
        "pi_clinical": pi_clinical,
        "agreement": agreement,
        "endo_total": endo_total,
        "endo_se": endo_se,
        "nonendo_total": nonendo_total,
        "nonendo_se": nonendo_se,
        "blp_z": z,
        "blp_p": p,
    }


def run_q8_vs_q12_analysis(
    X_all_np: np.ndarray,
    X_histwk_np: np.ndarray,
    X_endo_np: np.ndarray,
    T3: np.ndarray,
    Y: np.ndarray,
    all_cols: list,
    endo_idx: list,
    nonendo_idx: list,
) -> dict:
    """
    Run Q8 vs Q12 analysis among active treatment patients.

    Returns dictionary with results for downstream use.
    """
    print_section("Q8 vs Q12 among active treatment")

    # Filter to active treatment only
    active_mask = (T3 == 1) | (T3 == 2)
    T_active = T3[active_mask]
    T_q8 = (T_active == 2).astype(int)
    Y_active = Y[active_mask]
    X_all_active = X_all_np[active_mask]
    X_histwk_active = X_histwk_np[active_mask]
    X_endo_active = X_endo_np[active_mask]

    # Fit X-learner models
    res_all = xlearner_binary_crossfit(X_all_active, T_q8, Y_active, n_folds=5, seed=RNG)
    res_clinical = xlearner_binary_crossfit(
        X_histwk_active, T_q8, Y_active, n_folds=5, seed=RNG
    )

    # Compute permutation importance
    treated = T_q8 == 1
    y_target = Y_active[treated] - res_all["mu0"][treated]
    imp_mean, imp_se = permimp_with_ci(
        res_all["g"],
        X_all_active[treated],
        y_target,
        repeats=10,
        n_repeats=10,
        seed=RNG,
    )

    # Aggregate importance
    endo_total = imp_mean[endo_idx].sum()
    endo_se = np.sqrt((imp_se[endo_idx] ** 2).sum())
    nonendo_total = imp_mean[nonendo_idx].sum()
    nonendo_se = np.sqrt((imp_se[nonendo_idx] ** 2).sum())
    ratio = nonendo_total / max(endo_total, 1e-12)

    print(
        f"CATE importance: endoscopic {endo_total:.4f}±{1.96 * endo_se:.4f}, "
        f"non-endoscopic {nonendo_total:.4f}±{1.96 * nonendo_se:.4f}, "
        f"ratio {ratio:.2f}"
    )

    # BLP test
    G_endo = group_summaries(X_endo_active)
    G_clinical = group_summaries(X_histwk_active)
    z, p = blp_test_multiplier(res_all["tau"], G_endo, G_clinical, B=1000, seed=RNG)
    print(f"BLP test: z={z:.2f}, p={p:.4g}")

    # Feature importance dataframe
    imp_df = pd.DataFrame(
        {"feature": all_cols, "importance_mean": imp_mean, "importance_se": imp_se}
    ).sort_values("importance_mean", ascending=False)

    print("\nTop 15 Features:")
    print(imp_df.head(15)[["feature", "importance_mean"]].to_string(index=False))

    # Policy evaluation
    pi_all = (res_all["mu1"] - res_all["mu0"] > 0).astype(int)
    pi_clinical = (res_clinical["mu1"] - res_clinical["mu0"] > 0).astype(int)

    v_all, ci_all_lo, ci_all_hi, dr_all = dr_value_binary(
        T_q8, Y_active, pi_all, res_all["mu0"], res_all["mu1"], res_all["p_assign"]
    )
    v_clinical, ci_clinical_lo, ci_clinical_hi, dr_clinical = dr_value_binary(
        T_q8, Y_active, pi_clinical, res_clinical["mu0"], res_clinical["mu1"], res_clinical["p_assign"]
    )
    delta, d_lo, d_hi = bootstrap_delta(dr_all, dr_clinical, n_boot=5000, seed=RNG)

    print(f"\nPolicy value ALL: {v_all:.3f} CI ({ci_all_lo:.3f}, {ci_all_hi:.3f})")
    print(f"Policy value Clinical: {v_clinical:.3f} CI ({ci_clinical_lo:.3f}, {ci_clinical_hi:.3f})")
    print(f"Difference: {v_all - v_clinical:.3f} bootstrap CI ({d_lo:.3f}, {d_hi:.3f})")

    # Compare personalized vs always-Q12
    pi_baseline = np.zeros_like(pi_all)
    v_baseline, _, _, dr_baseline = dr_value_binary(
        T_q8, Y_active, pi_baseline, res_all["mu0"], res_all["mu1"], res_all["p_assign"]
    )
    stat, p_val = ttest_rel(dr_all, dr_baseline)
    print(
        f"Personalized vs always-Q12: diff={v_all - v_baseline:.4f}, p={p_val:.4f}"
    )

    # Policy agreement
    agreement = policy_agreement_analysis(pi_all, pi_clinical, "Q8 vs Q12")

    return {
        "res_all": res_all,
        "res_clinical": res_clinical,
        "imp_df": imp_df,
        "v_all": v_all,
        "ci_all": (ci_all_lo, ci_all_hi),
        "v_clinical": v_clinical,
        "ci_clinical": (ci_clinical_lo, ci_clinical_hi),
        "agreement": agreement,
        "endo_total": endo_total,
        "nonendo_total": nonendo_total,
        "blp_z": z,
        "blp_p": p,
    }


def run_multi_arm_analysis(
    X_all_np: np.ndarray,
    X_histwk_np: np.ndarray,
    X_endo_np: np.ndarray,
    T3: np.ndarray,
    Y: np.ndarray,
    all_cols: list,
    endo_idx: list,
    nonendo_idx: list,
    X_all_df=None,
    X_histwk_df=None,
    binary_cols: list = None,
) -> dict:
    """
    Run multi-arm policy analysis (Placebo vs Q12 vs Q8).

    Returns dictionary with results for downstream use.
    """
    print_section("Multi-Arm Policy: Placebo vs Q12 vs Q8")

    # Fit outcome models
    models_all = fit_three_arm_models(X_all_np, T3, Y, seed=RNG)
    models_clinical = fit_three_arm_models(X_histwk_np, T3, Y, seed=RNG)

    # Predict outcomes
    muP_all = models_all[0].predict(X_all_np)
    mu12_all = models_all[1].predict(X_all_np)
    mu8_all = models_all[2].predict(X_all_np)

    muP_clinical = models_clinical[0].predict(X_histwk_np)
    mu12_clinical = models_clinical[1].predict(X_histwk_np)
    mu8_clinical = models_clinical[2].predict(X_histwk_np)

    # Compute propensity scores
    pP = np.mean(T3 == 0)
    pQ12 = np.mean(T3 == 1)
    pQ8 = np.mean(T3 == 2)
    p_assign3 = np.tile([pP, pQ12, pQ8], (len(Y), 1))
    print(f"Empirical arm probabilities: P={pP:.3f}, Q12={pQ12:.3f}, Q8={pQ8:.3f}")

    # In-sample policy evaluation
    pi_all = optimal_arm_assignment(muP_all, mu12_all, mu8_all)
    v_all, ci_all_lo, ci_all_hi, dr_all = dr_value_three_arm(
        T3, Y, pi_all, muP_all, mu12_all, mu8_all, p_assign3
    )
    print(f"\nIn-sample ALL: {v_all:.3f} CI ({ci_all_lo:.3f}, {ci_all_hi:.3f})")

    pi_clinical = optimal_arm_assignment(muP_clinical, mu12_clinical, mu8_clinical)
    v_clinical, ci_clinical_lo, ci_clinical_hi, dr_clinical = dr_value_three_arm(
        T3, Y, pi_clinical, muP_clinical, mu12_clinical, mu8_clinical, p_assign3
    )
    print(f"In-sample Clinical: {v_clinical:.3f} CI ({ci_clinical_lo:.3f}, {ci_clinical_hi:.3f})")
    print(f"Difference: {v_all - v_clinical:.3f}")

    # Policy agreement
    agreement = policy_agreement_analysis(pi_all, pi_clinical, "Multi-Arm")

    # Compute optimality gap and importance
    tau_gap = compute_three_arm_pseudo_outcome(T3, muP_all, mu12_all, mu8_all)
    g3 = XGBRegressor(**CATE_MODEL_PARAMS, random_state=RNG)
    g3.fit(X_all_np, tau_gap)

    imp_mean, imp_se = permimp_with_ci(
        g3, X_all_np, tau_gap, repeats=10, n_repeats=10, seed=RNG
    )

    endo_total = imp_mean[endo_idx].sum()
    endo_se = np.sqrt((imp_se[endo_idx] ** 2).sum())
    nonendo_total = imp_mean[nonendo_idx].sum()
    nonendo_se = np.sqrt((imp_se[nonendo_idx] ** 2).sum())
    ratio = nonendo_total / max(endo_total, 1e-12)

    print(
        f"\nImportance: endoscopic {endo_total:.4f}±{1.96 * endo_se:.4f}, "
        f"non-endoscopic {nonendo_total:.4f}±{1.96 * nonendo_se:.4f}, "
        f"ratio {ratio:.2f}"
    )

    # Feature importance dataframe
    imp_df = pd.DataFrame(
        {"feature": all_cols, "importance_mean": imp_mean, "importance_se": imp_se}
    ).sort_values("importance_mean", ascending=False)

    print("\nTop 15 Features:")
    print(imp_df.head(15)[["feature", "importance_mean"]].to_string(index=False))

    # BLP test
    G_endo = group_summaries(X_endo_np)
    G_clinical = group_summaries(X_histwk_np)
    z, p = blp_test_multiplier(tau_gap, G_endo, G_clinical, B=1000, seed=RNG)
    print(f"BLP test: z={z:.2f}, p={p:.4g}")

    # Compare to always-placebo baseline
    pi_baseline = np.zeros_like(pi_all)
    v_baseline, _, _, _ = dr_value_three_arm(
        T3, Y, pi_baseline, muP_all, mu12_all, mu8_all, p_assign3
    )
    print(f"Improvement vs always-placebo: {v_all - v_baseline:.4f}")

    # Out-of-fold evaluation.
    print("\nRunning nested CV for out-of-fold policy values...")
    oof_res = nested_three_arm_oof_values_foldsafe(
        X_all_df, X_histwk_df, T3, Y, binary_cols, p_assign3, n_splits=5, seed=RNG
    )
    print(f"OOF ALL: {oof_res['value_all']:.3f} CI {oof_res['ci_all']}")
    print(f"OOF Clinical: {oof_res['value_clinical']:.3f} CI {oof_res['ci_clinical']}")
    print(
        f"OOF Difference: {oof_res['delta']:.3f} "
        f"CI ({oof_res['delta_ci'][0]:.3f}, {oof_res['delta_ci'][1]:.3f})"
    )

    return {
        "v_all": v_all,
        "ci_all": (ci_all_lo, ci_all_hi),
        "v_clinical": v_clinical,
        "ci_clinical": (ci_clinical_lo, ci_clinical_hi),
        "oof_res": oof_res,
        "tau_gap": tau_gap,
        "agreement": agreement,
        "endo_total": endo_total,
        "endo_se": endo_se,
        "nonendo_total": nonendo_total,
        "nonendo_se": nonendo_se,
        "blp_z": z,
        "blp_p": p,
        "imp_df": imp_df,
    }


def run_extended_analyses(
    X_all_np: np.ndarray,
    Tbin: np.ndarray,
    T3: np.ndarray,
    Y: np.ndarray,
    all_cols: list,
) -> dict:
    """
    Extended validation analyses (paper Tables 5-7):
      - GATES: predicted vs observed treatment effect across CATE quantiles
      - CLAN: features differentiating high vs low predicted CATE
      - Alternative CATE estimators (X-, T-, S-, DR-learner) consistency

    All use the same propensity-weighted X-learner as the main analysis.
    """
    # --- GATES (Table 5) ---
    print_section("GATES: Sorted Group Average Treatment Effects")

    res_ust = xlearner_binary_crossfit(X_all_np, Tbin, Y, n_folds=5, seed=RNG)
    gates_ust = gates_analysis(res_ust["tau"], Tbin, Y, n_groups=5, seed=RNG)
    print("\n--- UST vs Placebo ---")
    print(f"Group sizes: {gates_ust['group_ns']}")
    print(f"Predicted CATE means: {[f'{x:.3f}' for x in gates_ust['group_tau_means']]}")
    print(f"Observed ATEs:        "
          f"{[f'{x:.3f}' if not np.isnan(x) else 'NA' for x in gates_ust['group_ates']]}")
    print(f"Spearman r (predicted vs observed) = {gates_ust['correlation']:.3f}, "
          f"p = {gates_ust['correlation_p']:.4f}")
    print(f"Heterogeneity test p = {gates_ust['p_heterogeneity']:.4f}")

    active_mask = (T3 == 1) | (T3 == 2)
    T_q8 = (T3[active_mask] == 2).astype(int)
    Y_active = Y[active_mask]
    X_all_active = X_all_np[active_mask]
    res_q8 = xlearner_binary_crossfit(X_all_active, T_q8, Y_active, n_folds=5, seed=RNG)
    gates_q8 = gates_analysis(res_q8["tau"], T_q8, Y_active, n_groups=5, seed=RNG)
    print("\n--- Q8 vs Q12 (among active treatment) ---")
    print(f"Spearman r (predicted vs observed) = {gates_q8['correlation']:.3f}, "
          f"p = {gates_q8['correlation_p']:.4f}")
    print(f"Heterogeneity test p = {gates_q8['p_heterogeneity']:.4f}")

    # --- CLAN (Table 6) ---
    print_section("CLAN: Characteristics of Affected Subgroups (UST vs Placebo)")
    clan_ust = clan_analysis(res_ust["tau"], X_all_np, all_cols, n_groups=3, seed=RNG)
    print("\nTop 10 features differentiating high vs low CATE patients:")
    print(clan_ust.head(10)[["feature", "mean_low_cate", "mean_high_cate",
                             "difference", "p_value"]].to_string(index=False))

    # --- Alternative CATE estimators (Table 7) ---
    print_section("Alternative CATE Estimator Comparison (UST vs Placebo)")
    comparison = compare_cate_estimators(X_all_np, Tbin, Y, seed=RNG)
    print("\nEstimator summary:")
    print(comparison["summary"].to_string(index=False))
    print("\nPairwise Spearman correlations:")
    for pair, corr in comparison["correlations"].items():
        print(f"  {pair}: rho = {corr:.3f}")

    return {"gates_ust": gates_ust, "gates_q8": gates_q8,
            "clan_ust": clan_ust, "estimator_comparison": comparison}


def run_predictive_cv(X_all_np: np.ndarray, Y: np.ndarray) -> dict:
    """Run 5-fold CV for predictive performance on full cohort."""
    print_section("PREDICTIVE PERFORMANCE: 5-fold CV on full cohort")

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    aucs, auprcs, briers = [], [], []

    for tr, te in kf.split(X_all_np, Y):
        clf = CalibratedClassifierCV(
            XGBClassifier(**PROGNOSTIC_MODEL_PARAMS, random_state=RNG),
            cv=3,
            method="isotonic",
        )
        clf.fit(X_all_np[tr], Y[tr])
        prob = clf.predict_proba(X_all_np[te])[:, 1]

        aucs.append(roc_auc_score(Y[te], prob))
        auprcs.append(average_precision_score(Y[te], prob))
        briers.append(brier_score_loss(Y[te], prob))

    print(f"AUROC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"AUPRC: {np.mean(auprcs):.3f} ± {np.std(auprcs):.3f}")
    print(f"Brier: {np.mean(briers):.3f} ± {np.std(briers):.3f}")

    return {
        "auroc_mean": float(np.mean(aucs)), "auroc_std": float(np.std(aucs)),
        "auprc_mean": float(np.mean(auprcs)), "auprc_std": float(np.std(auprcs)),
        "brier_mean": float(np.mean(briers)), "brier_std": float(np.std(briers)),
    }


def run_prognostic_analysis(
    X_all_np: np.ndarray,
    X_histwk_np: np.ndarray,
    Y: np.ndarray,
    T3: np.ndarray,
) -> list:
    """Run prognostic contribution analysis (pooled and by arm).

    Returns one row of metrics per cohort (for saving to CSV).
    """

    def run_block(name: str, X_all: np.ndarray, X_clinical: np.ndarray, y: np.ndarray) -> dict:
        """Run prognostic block for one cohort; returns its metrics as a row."""
        print_section(f"PROGNOSTIC CONTRIBUTION: {name}")

        p_all = oof_proba(X_all, y, n_splits=5, seed=RNG)
        p_clinical = oof_proba(X_clinical, y, n_splits=5, seed=RNG)

        b_all, ll_all, auc_all, aupr_all = summarize_perf(y, p_all, "All")
        b_clin, ll_clin, auc_clin, aupr_clin = summarize_perf(y, p_clinical, "Clinical")

        d_brier, ci_b_lo, ci_b_hi = bootstrap_ci_delta_perf(
            y, p_all, p_clinical, metric="brier", n_boot=5000, seed=RNG
        )
        d_log, ci_l_lo, ci_l_hi = bootstrap_ci_delta_perf(
            y, p_all, p_clinical, metric="log", n_boot=5000, seed=RNG
        )
        d_auc, ci_a_lo, ci_a_hi = bootstrap_ci_delta_perf(
            y, p_all, p_clinical, metric="auc", n_boot=3000, seed=RNG
        )
        pr2, ib_lo, ib_hi, ib_frac = bootstrap_ci_incremental_brier(
            y, p_all, p_clinical, n_boot=5000, seed=RNG
        )
        pr2_str = "N/A" if np.isnan(pr2) else f"{pr2 * 100:.1f}%"
        ib_ci_str = (
            "N/A" if np.isnan(ib_lo) else f"({ib_lo * 100:.1f}%, {ib_hi * 100:.1f}%)"
        )

        print(f"\nBrier Clinical - All: {d_brier:.4f} CI ({ci_b_lo:.4f}, {ci_b_hi:.4f})")
        print(f"LogLoss Clinical - All: {d_log:.4f} CI ({ci_l_lo:.4f}, {ci_l_hi:.4f})")
        print(f"AUROC All - Clinical: {d_auc:.4f} CI ({ci_a_lo:.4f}, {ci_a_hi:.4f})")
        print(
            f"Incremental Brier (relative IPA gain, Eq. 5/7): {pr2_str} "
            f"95% CI {ib_ci_str} [{ib_frac * 100:.0f}% of resamples defined]"
        )

        return {
            "cohort": name,
            "n": int(len(y)),
            "brier_all": b_all, "logloss_all": ll_all,
            "auroc_all": auc_all, "auprc_all": aupr_all,
            "brier_clinical": b_clin, "logloss_clinical": ll_clin,
            "auroc_clinical": auc_clin, "auprc_clinical": aupr_clin,
            "d_brier_clin_minus_all": d_brier,
            "d_brier_ci_lo": ci_b_lo, "d_brier_ci_hi": ci_b_hi,
            "d_logloss_clin_minus_all": d_log,
            "d_logloss_ci_lo": ci_l_lo, "d_logloss_ci_hi": ci_l_hi,
            "d_auroc_all_minus_clin": d_auc,
            "d_auroc_ci_lo": ci_a_lo, "d_auroc_ci_hi": ci_a_hi,
            "incremental_brier_rel_ipa": pr2,
            "ib_ci_lo": ib_lo, "ib_ci_hi": ib_hi, "ib_frac_defined": ib_frac,
        }

    # Pooled analysis
    rows = [run_block("Pooled", X_all_np, X_histwk_np, Y)]

    # By treatment arm
    for arm, label in [(0, "Placebo"), (1, "Q12"), (2, "Q8")]:
        mask = T3 == arm
        rows.append(run_block(label, X_all_np[mask], X_histwk_np[mask], Y[mask]))

    return rows


def generate_figures(
    ust_results: dict,
    q8_results: dict,
    multi_arm_results: dict,
    endo_cols: list,
) -> None:
    """Generate all analysis figures."""
    print_section("GENERATING FIGURES")

    # Figure 1: Feature Importance (UST vs Placebo)
    plot_feature_importance(
        ust_results["imp_df"],
        endo_cols,
        "Feature Importance for Treatment Effect Heterogeneity\n(UST vs Placebo)",
        "fig1_feature_importance.pdf",
        top_n=15,
    )

    # Figure 2: Policy Value Forest Plot
    policy_results = [
        {
            "label": "UST vs PBO: All Features",
            "value": ust_results["v_all"],
            "ci_lo": ust_results["ci_all"][0],
            "ci_hi": ust_results["ci_all"][1],
            "color": "#2E86AB",
        },
        {
            "label": "UST vs PBO: Clinical Only",
            "value": ust_results["v_clinical"],
            "ci_lo": ust_results["ci_clinical"][0],
            "ci_hi": ust_results["ci_clinical"][1],
            "color": "#A23B72",
        },
        {
            "label": "Q8 vs Q12: All Features",
            "value": q8_results["v_all"],
            "ci_lo": q8_results["ci_all"][0],
            "ci_hi": q8_results["ci_all"][1],
            "color": "#2E86AB",
        },
        {
            "label": "Q8 vs Q12: Clinical Only",
            "value": q8_results["v_clinical"],
            "ci_lo": q8_results["ci_clinical"][0],
            "ci_hi": q8_results["ci_clinical"][1],
            "color": "#A23B72",
        },
        {
            "label": "Multi-Arm: All Features",
            "value": multi_arm_results["v_all"],
            "ci_lo": multi_arm_results["ci_all"][0],
            "ci_hi": multi_arm_results["ci_all"][1],
            "color": "#2E86AB",
        },
        {
            "label": "Multi-Arm: Clinical Only",
            "value": multi_arm_results["v_clinical"],
            "ci_lo": multi_arm_results["ci_clinical"][0],
            "ci_hi": multi_arm_results["ci_clinical"][1],
            "color": "#A23B72",
        },
        {
            "label": "Multi-Arm OOF: All Features",
            "value": multi_arm_results["oof_res"]["value_all"],
            "ci_lo": multi_arm_results["oof_res"]["ci_all"][0],
            "ci_hi": multi_arm_results["oof_res"]["ci_all"][1],
            "color": "#2E86AB",
        },
        {
            "label": "Multi-Arm OOF: Clinical Only",
            "value": multi_arm_results["oof_res"]["value_clinical"],
            "ci_lo": multi_arm_results["oof_res"]["ci_clinical"][0],
            "ci_hi": multi_arm_results["oof_res"]["ci_clinical"][1],
            "color": "#A23B72",
        },
    ]
    plot_policy_forest(policy_results, "fig2_policy_forest.pdf")

    # Figure 3: CATE Distributions (UST vs Placebo, Q8 vs Q12, three-arm gap)
    plot_cate_distributions(
        ust_results["res_all"]["tau"],
        q8_results["res_all"]["tau"],
        multi_arm_results["tau_gap"],
        "fig3_cate_distributions.pdf",
    )

    print("Figures saved successfully.")


def save_result_tables(
    ust_results: dict,
    q8_results: dict,
    multi_arm_results: dict,
    extended_results: dict,
    prognostic_rows: list,
    predictive_cv: dict,
) -> None:
    """Write every numeric results table to CSV.
    """
    print_section("SAVING RESULT TABLES")

    # Feature importance behind Fig. 1 (one file per contrast).
    save_table(ust_results["imp_df"], "table_feature_importance_ust_vs_pbo.csv")
    save_table(q8_results["imp_df"], "table_feature_importance_q8_vs_q12.csv")
    save_table(multi_arm_results["imp_df"], "table_feature_importance_multiarm.csv")

    # Policy value / BLP / feature-group importance / agreement (Tables 2-3, 8).
    policy_rows = []
    for label, r in [
        ("UST vs PBO", ust_results),
        ("Q8 vs Q12", q8_results),
        ("Multi-Arm", multi_arm_results),
    ]:
        endo, nonendo = r["endo_total"], r["nonendo_total"]
        policy_rows.append({
            "contrast": label,
            "policy_value_all": r["v_all"],
            "ci_all_lo": r["ci_all"][0], "ci_all_hi": r["ci_all"][1],
            "policy_value_clinical": r["v_clinical"],
            "ci_clinical_lo": r["ci_clinical"][0], "ci_clinical_hi": r["ci_clinical"][1],
            "value_difference": r["v_all"] - r["v_clinical"],
            "blp_z": r["blp_z"], "blp_p": r["blp_p"],
            "endo_importance": endo, "nonendo_importance": nonendo,
            "importance_ratio": nonendo / max(endo, 1e-12),
            "n_agree": int(r["agreement"]["n_agree"]),
            "n_disagree": int(r["agreement"]["n_disagree"]),
            "agreement_rate": r["agreement"]["agreement_rate"],
        })
    save_table(pd.DataFrame(policy_rows), "table_policy_summary.csv")

    # Multi-arm nested out-of-fold policy values (Fig. 2 OOF rows).
    oof = multi_arm_results["oof_res"]
    save_table(pd.DataFrame([{
        "oof_value_all": oof["value_all"],
        "oof_ci_all_lo": oof["ci_all"][0], "oof_ci_all_hi": oof["ci_all"][1],
        "oof_value_clinical": oof["value_clinical"],
        "oof_ci_clinical_lo": oof["ci_clinical"][0], "oof_ci_clinical_hi": oof["ci_clinical"][1],
        "oof_delta": oof["delta"],
        "oof_delta_ci_lo": oof["delta_ci"][0], "oof_delta_ci_hi": oof["delta_ci"][1],
    }]), "table_multiarm_oof.csv")

    # Prognostic contribution, pooled + per arm (Table 4).
    save_table(pd.DataFrame(prognostic_rows), "table_prognostic_contribution.csv")

    # GATES: per-group predicted vs observed + group-level stats (Table 5).
    gates_rows = []
    for label, g in [("UST vs PBO", extended_results["gates_ust"]),
                     ("Q8 vs Q12", extended_results["gates_q8"])]:
        for i in range(g["n_groups"]):
            ci_lo, ci_hi = g["group_cis"][i]
            gates_rows.append({
                "contrast": label,
                "group": i + 1,
                "n": g["group_ns"][i],
                "predicted_cate_mean": g["group_tau_means"][i],
                "observed_ate": g["group_ates"][i],
                "observed_ate_ci_lo": ci_lo,
                "observed_ate_ci_hi": ci_hi,
                "spearman_r": g["correlation"],
                "spearman_p": g["correlation_p"],
                "p_heterogeneity": g["p_heterogeneity"],
            })
    save_table(pd.DataFrame(gates_rows), "table_gates.csv")

    # CLAN: features differentiating high vs low CATE (Table 6).
    save_table(extended_results["clan_ust"], "table_clan_ust.csv")

    # Alternative CATE estimators: summary + pairwise correlations (Table 7).
    save_table(extended_results["estimator_comparison"]["summary"],
               "table_estimator_comparison.csv")
    corr = extended_results["estimator_comparison"]["correlations"]
    save_table(
        pd.DataFrame({"pair": list(corr.keys()), "spearman_rho": list(corr.values())}),
        "table_estimator_correlations.csv",
    )

    # Predictive cross-validation summary.
    save_table(pd.DataFrame([predictive_cv]), "table_predictive_cv.csv")


def print_summary(
    ust_results: dict,
    q8_results: dict,
    multi_arm_results: dict,
    extended_results: dict = None,
) -> None:
    """Print final analysis summary."""
    print_section("ANALYSIS SUMMARY")

    print("\nFeature Importance Ratios (Non-endoscopic / Endoscopic):")
    print(
        f"  UST vs PBO:    {ust_results['nonendo_total']:.2f} / "
        f"{ust_results['endo_total']:.2f} = "
        f"{ust_results['nonendo_total']/max(ust_results['endo_total'], 1e-12):.2f}x"
    )
    print(
        f"  Q8 vs Q12:     {q8_results['nonendo_total']:.2f} / "
        f"{q8_results['endo_total']:.2f} = "
        f"{q8_results['nonendo_total']/max(q8_results['endo_total'], 1e-12):.2f}x"
    )
    print(
        f"  Multi-Arm:     {multi_arm_results['nonendo_total']:.2f} / "
        f"{multi_arm_results['endo_total']:.2f} = "
        f"{multi_arm_results['nonendo_total']/max(multi_arm_results['endo_total'], 1e-12):.2f}x"
    )

    print("\nBLP Test Results (Endoscopic contribution beyond clinical):")
    print(f"  UST vs PBO:    z={ust_results['blp_z']:.2f}, p={ust_results['blp_p']:.4g}")
    print(f"  Q8 vs Q12:     z={q8_results['blp_z']:.2f}, p={q8_results['blp_p']:.4g}")
    print(
        f"  Multi-Arm:     z={multi_arm_results['blp_z']:.2f}, "
        f"p={multi_arm_results['blp_p']:.4g}"
    )

    print("\nPolicy Agreement Rates:")
    print(
        f"  UST vs PBO:    {ust_results['agreement']['n_agree']} agree, "
        f"{ust_results['agreement']['n_disagree']} disagree "
        f"({ust_results['agreement']['agreement_rate']:.1%})"
    )
    print(
        f"  Q8 vs Q12:     {q8_results['agreement']['n_agree']} agree, "
        f"{q8_results['agreement']['n_disagree']} disagree "
        f"({q8_results['agreement']['agreement_rate']:.1%})"
    )
    print(
        f"  Multi-Arm:     {multi_arm_results['agreement']['n_agree']} agree, "
        f"{multi_arm_results['agreement']['n_disagree']} disagree "
        f"({multi_arm_results['agreement']['agreement_rate']:.1%})"
    )

    # Extended validation summary
    if extended_results is not None:
        g_ust = extended_results["gates_ust"]
        g_q8 = extended_results["gates_q8"]
        print("\nGATES (predicted vs observed group ATEs):")
        print(f"  UST vs PBO:  Spearman r = {g_ust['correlation']:.2f} "
              f"(p = {g_ust['correlation_p']:.3f}); "
              f"heterogeneity p = {g_ust['p_heterogeneity']:.3f}")
        print(f"  Q8 vs Q12:   Spearman r = {g_q8['correlation']:.2f} "
              f"(p = {g_q8['correlation_p']:.3f}); "
              f"heterogeneity p = {g_q8['p_heterogeneity']:.3f}")
        corr = extended_results["estimator_comparison"]["correlations"]
        print("\nAlternative CATE estimators (Spearman correlation):")
        print(f"  X-T = {corr['X-T']:.2f}, X-S = {corr['X-S']:.2f}, "
              f"X-DR = {corr['X-DR']:.2f}, T-S = {corr['T-S']:.2f}, "
              f"T-DR = {corr['T-DR']:.2f}, S-DR = {corr['S-DR']:.2f}")

    print_section("ANALYSIS COMPLETE")


def main():
    """Run complete causal analysis pipeline."""
    # Load and preprocess data
    print_section("COMPREHENSIVE CAUSAL ANALYSIS FOR UC MAINTENANCE TRIAL")

    df, T3, Tbin, Y = load_unifi_data()
    X_all, X_histwk, X_endo, all_cols, endo_cols, binary_cols = build_feature_matrices(
        df
    )
    X_all_np, X_histwk_np, X_endo_np, all_cols, endo_idx, nonendo_idx = preprocess_all(
        X_all, X_histwk, X_endo, binary_cols
    )

    print(f"\nDimensions: All {X_all_np.shape}, Clinical {X_histwk_np.shape}, "
          f"Endoscopic {X_endo_np.shape}")

    print("\nSample sizes:")
    print(f"  Placebo: {np.sum(T3 == 0)}")
    print(f"  Q12:     {np.sum(T3 == 1)}")
    print(f"  Q8:      {np.sum(T3 == 2)}")
    print(f"  UST pooled: {np.sum(Tbin == 1)}")

    # Run main analyses
    ust_results = run_ust_vs_placebo_analysis(
        X_all_np, X_histwk_np, X_endo_np, Tbin, Y, all_cols, endo_idx, nonendo_idx
    )

    q8_results = run_q8_vs_q12_analysis(
        X_all_np, X_histwk_np, X_endo_np, T3, Y, all_cols, endo_idx, nonendo_idx
    )

    multi_arm_results = run_multi_arm_analysis(
        X_all_np, X_histwk_np, X_endo_np, T3, Y, all_cols, endo_idx, nonendo_idx,
        X_all_df=X_all, X_histwk_df=X_histwk, binary_cols=binary_cols,
    )

    # Predictive performance
    predictive_cv = run_predictive_cv(X_all_np, Y)

    # Extended validation analyses (GATES, CLAN, alternative learners)
    extended_results = run_extended_analyses(
        X_all_np, Tbin, T3, Y, all_cols
    )

    # Generate figures
    generate_figures(ust_results, q8_results, multi_arm_results, endo_cols)

    # Prognostic contribution analysis
    prognostic_rows = run_prognostic_analysis(X_all_np, X_histwk_np, Y, T3)

    # Save all numeric result tables to CSV (companions to the figures)
    save_result_tables(
        ust_results, q8_results, multi_arm_results,
        extended_results, prognostic_rows, predictive_cv,
    )

    # Print summary
    print_summary(ust_results, q8_results, multi_arm_results, extended_results)


if __name__ == "__main__":
    main()
