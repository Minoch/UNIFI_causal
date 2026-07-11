#!/usr/bin/env python3
"""
reproducibility_multiseed.py

Supplementary and reproducibility harness for:

    "A Causal Machine Learning Framework for Treatment Personalization in
     Clinical Trials: Application to Ulcerative Colitis"

It imports methods only from `causal_utils.py`.

WHAT IT PRODUCES
----------------
Supplementary sections:
    A  Missing-data report, overall and by arm
    B  Hyperparameter sensitivity of headline results
    C  Stability across seeds (in-sample vs OOF)          (see note)
    D  Multiplicity-adjusted BLP p-values
    E  Pre-randomization-only sensitivity

Multi-seed reproducibility:
    - supp_reproducibility_multiseed.csv       (one row per seed)
    - supp_reproducibility_prognostic.csv      (per-arm AUROC/Brier per seed)
    - a summary block (printed to stdout)

USAGE
-----
    python reproducibility_multiseed.py                 # supplementary A/B/D + multiseed sweep (default)
    python reproducibility_multiseed.py --seeds 50      # multiseed over seeds 0..49
    python reproducibility_multiseed.py --seeds 0-9     # explicit range
    python reproducibility_multiseed.py --boot 2000     # bootstrap reps per seed
    python reproducibility_multiseed.py --skip-supplementary  # only the multiseed sweep
    python reproducibility_multiseed.py --skip-multiseed# only the supplementary sections
    python reproducibility_multiseed.py --sections A,B,D,E   # choose supplementary sections explicitly
    python reproducibility_multiseed.py --no-prognostic # skip the per-arm prognostic block

The canonical single-seed run is `run_analysis.py` (seed
causal_utils.DEFAULT_RANDOM_STATE); this script characterizes the sampling
distribution around it.
"""

import argparse
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd

import causal_utils as cu

warnings.filterwarnings("ignore")


# ===========================================================================
# CONFIG
# ===========================================================================
EXCEL_PATH = "Maintenance_ITT_all_ehr_and_video_allmed.xlsx"
SHEET = "Sheet1"
SEED = cu.DEFAULT_RANDOM_STATE

# Supplementary-section toggles (used when --sections is not passed on the CLI).
RUN_A_MISSING = True
RUN_B_HYPERPARAM = True
RUN_C_STABILITY = False   
RUN_D_MULTIPLICITY = True
RUN_E_BASELINE = False     

# Section B/C knobs (lower for a quick pass)
STABILITY_SEEDS = list(range(20))   # only used by Section C when RUN_C_STABILITY=True
BOOT_DELTA = 2000                    # bootstrap reps for policy-value deltas inside B/C


# ===========================================================================
# Helpers
# ===========================================================================
def banner(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


@contextmanager
def override_xgb_params(outcome_params, cate_params):

    old_o, old_c = cu.OUTCOME_MODEL_PARAMS, cu.CATE_MODEL_PARAMS
    cu.OUTCOME_MODEL_PARAMS = outcome_params
    cu.CATE_MODEL_PARAMS = cate_params
    try:
        yield
    finally:
        cu.OUTCOME_MODEL_PARAMS = old_o
        cu.CATE_MODEL_PARAMS = old_c


def load_everything():
    df, T3, Tbin, Y = cu.load_unifi_data(EXCEL_PATH, SHEET)
    X_all, X_histwk, X_endo, all_cols, endo_cols, binary_cols = cu.build_feature_matrices(df)
    X_all_np, X_histwk_np, X_endo_np, tcols, endo_idx, nonendo_idx = cu.preprocess_all(
        X_all, X_histwk, X_endo, binary_cols
    )
    return dict(
        df=df, T3=T3, Tbin=Tbin, Y=Y,
        X_all=X_all, X_histwk=X_histwk, X_endo=X_endo,
        all_cols=all_cols, endo_cols=endo_cols, binary_cols=binary_cols,
        X_all_np=X_all_np, X_histwk_np=X_histwk_np, X_endo_np=X_endo_np,
        endo_idx=endo_idx, nonendo_idx=nonendo_idx,
    )


def headline_quantities(D, seed=SEED, n_boot=BOOT_DELTA):
    """Recompute the headline results used throughout the paper:
      - BLP z for UST vs placebo
      - binary policy-value delta (all - clinical) for UST vs placebo (pp)
      - multi-arm out-of-fold policy values (all, clinical) and delta (pp)
      - multi-arm in-sample values (for the in-sample vs OOF gap)
      - GATES predicted-vs-observed correlation and heterogeneity p
    Returns a dict of scalars.
    """
    Tbin, Y = D["Tbin"], D["Y"]
    X_all_np, X_histwk_np, X_endo_np = D["X_all_np"], D["X_histwk_np"], D["X_endo_np"]
    T3 = D["T3"]

    # --- UST vs placebo: X-learner, BLP, binary policy delta ---
    res_all = cu.xlearner_binary_crossfit(X_all_np, Tbin, Y, n_folds=5, seed=seed)
    res_clin = cu.xlearner_binary_crossfit(X_histwk_np, Tbin, Y, n_folds=5, seed=seed)

    blp_z, blp_p = cu.blp_test_multiplier(
        res_all["tau"], cu.group_summaries(X_endo_np), cu.group_summaries(X_histwk_np),
        B=1000, seed=seed,
    )

    # --- GATES (UST vs placebo): predicted-vs-observed correlation + het test ---
    gates = cu.gates_analysis(res_all["tau"], Tbin, Y, n_groups=5, seed=seed)

    pi_all = (res_all["mu1"] - res_all["mu0"] > 0).astype(int)
    pi_clin = (res_clin["mu1"] - res_clin["mu0"] > 0).astype(int)
    v_all, _, _, dr_all = cu.dr_value_binary(
        Tbin, Y, pi_all, res_all["mu0"], res_all["mu1"], res_all["p_assign"]
    )
    v_clin, _, _, dr_clin = cu.dr_value_binary(
        Tbin, Y, pi_clin, res_clin["mu0"], res_clin["mu1"], res_clin["p_assign"]
    )
    d_bin, d_bin_lo, d_bin_hi = cu.bootstrap_delta(dr_all, dr_clin, n_boot=n_boot, seed=seed)

    # --- Multi-arm out-of-fold ---
    pP, pQ12, pQ8 = (T3 == 0).mean(), (T3 == 1).mean(), (T3 == 2).mean()
    p_assign3 = np.tile([pP, pQ12, pQ8], (len(Y), 1))
    oof = cu.nested_three_arm_oof_values_foldsafe(
        D["X_all"], D["X_histwk"], T3, Y, D["binary_cols"], p_assign3,
        n_splits=5, seed=seed,
    )

    # --- Multi-arm in-sample (for the in-sample vs OOF gap) ---
    models_all = cu.fit_three_arm_models(X_all_np, T3, Y, seed=seed)
    muP, mu12, mu8 = [m.predict(X_all_np) for m in models_all]
    pi_is = cu.optimal_arm_assignment(muP, mu12, mu8)
    v_is_all, _, _, _ = cu.dr_value_three_arm(T3, Y, pi_is, muP, mu12, mu8, p_assign3)

    models_clin = cu.fit_three_arm_models(X_histwk_np, T3, Y, seed=seed)
    muPc, mu12c, mu8c = [m.predict(X_histwk_np) for m in models_clin]
    pi_isc = cu.optimal_arm_assignment(muPc, mu12c, mu8c)
    v_is_clin, _, _, _ = cu.dr_value_three_arm(T3, Y, pi_isc, muPc, mu12c, mu8c, p_assign3)

    return dict(
        blp_z=blp_z, blp_p=blp_p,
        v_bin_all=100 * v_all, v_bin_clin=100 * v_clin,
        delta_bin_pp=100 * d_bin, delta_bin_lo=100 * d_bin_lo, delta_bin_hi=100 * d_bin_hi,
        oof_all=100 * oof["value_all"], oof_clin=100 * oof["value_clinical"],
        oof_delta_pp=100 * oof["delta"],
        oof_delta_lo=100 * oof["delta_ci"][0], oof_delta_hi=100 * oof["delta_ci"][1],
        insample_all=100 * v_is_all, insample_clin=100 * v_is_clin,
        gap_all_pp=100 * (v_is_all - oof["value_all"]),
        gap_clin_pp=100 * (v_is_clin - oof["value_clinical"]),
        gates_corr_ust=gates["correlation"],
        gates_het_ust=gates["p_heterogeneity"],
    )


# ===========================================================================
# Section A -- Missing-data report
# ===========================================================================
def section_A(D):
    banner("SECTION A -- Missing data (availability by arm; proportions)")

    X_all, T3 = D["X_all"], D["T3"]
    arm_names = {0: "Placebo", 1: "Q12", 2: "Q8"}

    rows = []
    for col in X_all.columns:
        r = {"feature": col, "overall_%miss": round(100 * X_all[col].isna().mean(), 2)}
        for a in (0, 1, 2):
            mask = T3 == a
            r[f"{arm_names[a]}_%miss"] = round(100 * X_all.loc[mask, col].isna().mean(), 2)
        rows.append(r)
    miss_df = pd.DataFrame(rows).sort_values("overall_%miss", ascending=False)

    print("\nPer-feature missingness (%), overall and by maintenance arm:\n")
    print(miss_df.to_string(index=False))
    print(f"\nMax single-feature missingness: {miss_df['overall_%miss'].max():.2f}%")
    print(f"Features with >10% missing: "
          f"{(miss_df['overall_%miss'] > 10).sum()} of {len(miss_df)}")

    # Are week-8 measures present in every arm?
    groups = {
        "week-8 labs": ["CALPL_wk8", "CRPL_wk8"],
        "week-8 symptom/pMayo": ["ABSSTOOL_wk8", "PGSCORE_wk8", "PMAYO_wk8",
                                 "RBSCORE_wk8", "SFSCORE_wk8"],
        "week-8 endoscopy": D["endo_cols"],
    }
    print("\nAvailability (% NON-missing) by arm, by measurement group:\n")
    for gname, cols in groups.items():
        cols = [c for c in cols if c in X_all.columns]
        line = [f"{gname:<24}"]
        for a in (0, 1, 2):
            mask = T3 == a
            avail = 100 * (1 - X_all.loc[mask, cols].isna().mean().mean())
            line.append(f"{arm_names[a]}={avail:5.1f}%")
        print("  " + "  ".join(line))

    miss_df.to_csv("supp_missingness.csv", index=False)
    print("\nSaved: supp_missingness.csv")


# ===========================================================================
# Section B -- Hyperparameter sensitivity
# ===========================================================================
def section_B(D):
    banner("SECTION B -- Hyperparameter sensitivity of headline results")

    grid = {
        "manuscript (depth4, lr0.05, 800)": (
            dict(n_estimators=800, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                 learning_rate=0.05, reg_lambda=1.0),
            dict(n_estimators=400, max_depth=4, subsample=0.8, colsample_bytree=0.8),
        ),
        "shallower (depth3)": (
            dict(n_estimators=800, max_depth=3, subsample=0.8, colsample_bytree=0.8,
                 learning_rate=0.05, reg_lambda=1.0),
            dict(n_estimators=400, max_depth=3, subsample=0.8, colsample_bytree=0.8),
        ),
        "deeper (depth6)": (
            dict(n_estimators=800, max_depth=6, subsample=0.8, colsample_bytree=0.8,
                 learning_rate=0.05, reg_lambda=1.0),
            dict(n_estimators=400, max_depth=6, subsample=0.8, colsample_bytree=0.8),
        ),
        "slower (lr0.03)": (
            dict(n_estimators=1200, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                 learning_rate=0.03, reg_lambda=1.0),
            dict(n_estimators=600, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                 learning_rate=0.03),
        ),
        "faster (lr0.1)": (
            dict(n_estimators=400, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                 learning_rate=0.1, reg_lambda=1.0),
            dict(n_estimators=200, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                 learning_rate=0.1),
        ),
        "stronger L2 (reg5)": (
            dict(n_estimators=800, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                 learning_rate=0.05, reg_lambda=5.0),
            dict(n_estimators=400, max_depth=4, subsample=0.8, colsample_bytree=0.8,
                 reg_lambda=5.0),
        ),
    }

    # Hold the tree-construction method fixed to the deterministic setting used
    # in the main analysis, so the grid varies ONLY the hyperparameters.
    for op, cp in grid.values():
        op.update(tree_method="exact", n_jobs=1)
        cp.update(tree_method="exact", n_jobs=1)

    rows = []
    for name, (op, cp) in grid.items():
        print(f"  running config: {name} ...")
        with override_xgb_params(op, cp):
            h = headline_quantities(D, seed=SEED)
        rows.append({
            "config": name,
            "BLP_z (UST vs PBO)": round(h["blp_z"], 2),
            "binary dV pp": round(h["delta_bin_pp"], 1),
            "binary dV 95%CI": f"({h['delta_bin_lo']:.1f}, {h['delta_bin_hi']:.1f})",
            "OOF all %": round(h["oof_all"], 1),
            "OOF clinical %": round(h["oof_clin"], 1),
            "OOF dV pp": round(h["oof_delta_pp"], 1),
        })

    tab = pd.DataFrame(rows)
    print("\nHeadline results across hyperparameter settings:\n")
    print(tab.to_string(index=False))
    tab.to_csv("supp_hyperparam_sensitivity.csv", index=False)
    print("\nSaved: supp_hyperparam_sensitivity.csv")


# ===========================================================================
# Section C -- Stability across seeds; in-sample vs OOF gap
# ===========================================================================
def section_C(D):
    banner("SECTION C -- Stability across random seeds, in-sample vs OOF gap")

    recs = []
    for s in STABILITY_SEEDS:
        print(f"  seed {s} ...")
        h = headline_quantities(D, seed=s, n_boot=500)  # small boot; we summarise across seeds
        recs.append(h)
    R = pd.DataFrame(recs)

    def summ(col):
        x = R[col]
        return f"mean={x.mean():7.2f}  sd={x.std(ddof=1):5.2f}  min={x.min():7.2f}  max={x.max():7.2f}"

    print(f"\nAcross {len(STABILITY_SEEDS)} seeds:\n")
    print(f"  BLP z (UST vs PBO)            : {summ('blp_z')}")
    print(f"  GATES UST Spearman r          : {summ('gates_corr_ust')}")
    print(f"  GATES UST heterogeneity p     : {summ('gates_het_ust')}")
    print(f"  Binary policy dV (pp)         : {summ('delta_bin_pp')}")
    print(f"  Multi-arm OOF, all (%)        : {summ('oof_all')}")
    print(f"  Multi-arm OOF, clinical (%)   : {summ('oof_clin')}")
    print(f"  Multi-arm OOF dV (pp)         : {summ('oof_delta_pp')}")
    print(f"  In-sample minus OOF, all (pp) : {summ('gap_all_pp')}")
    print(f"  In-sample minus OOF, clin (pp): {summ('gap_clin_pp')}")

    R.to_csv("supp_stability_by_seed.csv", index=False)
    print("\nSaved: supp_stability_by_seed.csv")


# ===========================================================================
# Section D -- Multiplicity-adjusted BLP p-values
# ===========================================================================
def _bh_adjust(pvals):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    # enforce monotonicity
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def section_D(D):
    banner("SECTION D -- Multiplicity adjustment for the BLP test family")

    Tbin, Y, T3 = D["Tbin"], D["Y"], D["T3"]
    X_all_np, X_histwk_np, X_endo_np = D["X_all_np"], D["X_histwk_np"], D["X_endo_np"]

    # 1) UST vs placebo
    res = cu.xlearner_binary_crossfit(X_all_np, Tbin, Y, n_folds=5, seed=SEED)
    z1, p1 = cu.blp_test_multiplier(
        res["tau"], cu.group_summaries(X_endo_np), cu.group_summaries(X_histwk_np), seed=SEED
    )
    # 2) Q8 vs Q12 (active only)
    active = (T3 == 1) | (T3 == 2)
    Tq8 = (T3[active] == 2).astype(int)
    res2 = cu.xlearner_binary_crossfit(X_all_np[active], Tq8, Y[active], n_folds=5, seed=SEED)
    z2, p2 = cu.blp_test_multiplier(
        res2["tau"], cu.group_summaries(X_endo_np[active]),
        cu.group_summaries(X_histwk_np[active]), seed=SEED
    )
    # 3) Multi-arm optimality gap
    models = cu.fit_three_arm_models(X_all_np, T3, Y, seed=SEED)
    muP, mu12, mu8 = [m.predict(X_all_np) for m in models]
    tau_gap = cu.compute_three_arm_pseudo_outcome(T3, muP, mu12, mu8)
    z3, p3 = cu.blp_test_multiplier(
        tau_gap, cu.group_summaries(X_endo_np), cu.group_summaries(X_histwk_np), seed=SEED
    )

    names = ["UST vs Placebo", "Q8 vs Q12", "Multi-arm policy"]
    zs = [z1, z2, z3]
    ps = [p1, p2, p3]
    bonf = np.clip(np.array(ps) * len(ps), 0, 1)
    bh = _bh_adjust(ps)

    tab = pd.DataFrame({
        "comparison": names,
        "BLP_z": np.round(zs, 2),
        "p_raw": [f"{p:.4g}" for p in ps],
        "p_Bonferroni": [f"{p:.4g}" for p in bonf],
        "p_BH": [f"{p:.4g}" for p in bh],
    })
    print("\nBLP family (3 tests), raw and adjusted:\n")
    print(tab.to_string(index=False))
    tab.to_csv("supp_blp_multiplicity.csv", index=False)
    print("\nSaved: supp_blp_multiplicity.csv")


# ===========================================================================
# Section E -- OPTIONAL pre-randomization-only sensitivity
# ===========================================================================
def section_E(D):
    banner("SECTION E -- OPTIONAL: pre-randomization-only conditioning")

    PRE_RAND_COLS = [
        "AGE", "SEX_M", "BIONAIVE", "BIMM", "RDIALL", "RDICORT", "InductionMed_UST_SC",
    ]

    BASELINE_ENDO_COLS = []

    df = D["df"]
    Tbin, Y = D["Tbin"], D["Y"]
    binary_cols = D["binary_cols"]

    missing = [c for c in PRE_RAND_COLS if c not in df.columns]
    if missing:
        print(f"  Skipping: columns not found in data: {missing}")
        return

    X_pre = df[PRE_RAND_COLS].copy()
    pre_bin = [c for c in binary_cols if c in PRE_RAND_COLS]
    pre = cu.build_preprocessor(X_pre, pre_bin)
    X_pre_np = pre.fit_transform(X_pre)

    res_pre = cu.xlearner_binary_crossfit(X_pre_np, Tbin, Y, n_folds=5, seed=SEED)
    pi_pre = (res_pre["mu1"] - res_pre["mu0"] > 0).astype(int)
    v_pre, lo, hi, _ = cu.dr_value_binary(
        Tbin, Y, pi_pre, res_pre["mu0"], res_pre["mu1"], res_pre["p_assign"]
    )
    print(f"\nPre-randomization-only policy value (UST vs PBO): "
          f"{100*v_pre:.1f}% (95% CI {100*lo:.1f}, {100*hi:.1f})")

    if BASELINE_ENDO_COLS and all(c in df.columns for c in BASELINE_ENDO_COLS):
        X_pre2 = df[PRE_RAND_COLS + BASELINE_ENDO_COLS].copy()
        pre2 = cu.build_preprocessor(X_pre2, pre_bin)
        X_pre2_np = pre2.fit_transform(X_pre2)
        z, p = cu.blp_test_multiplier(
            res_pre["tau"],
            cu.group_summaries(pre2.fit_transform(df[BASELINE_ENDO_COLS].copy())),
            cu.group_summaries(X_pre_np), seed=SEED,
        )
        print(f"Baseline-endoscopy BLP (vs baseline clinical): z={z:.2f}, p={p:.4g}")
    else:
        print("  (No week-0 baseline endoscopy columns supplied; "
              "filled only the demographics/history-only arm of the sensitivity.)")


# ===========================================================================
# Multi-seed reproducibility sweep (+ per-arm prognostic block)
# ===========================================================================

SUMMARY_FIELDS = [
    ("blp_z", "BLP z (UST vs PBO)", ""),
    ("gates_corr_ust", "GATES UST Spearman r", ""),
    ("gates_het_ust", "GATES UST heterogeneity p", ""),
    ("delta_bin_pp", "Binary policy dV (pp)", "pp"),
    ("oof_all", "Multi-arm OOF, all (%)", "%"),
    ("oof_clin", "Multi-arm OOF, clinical (%)", "%"),
    ("oof_delta_pp", "Multi-arm OOF dV (pp)", "pp"),
    ("gap_all_pp", "In-sample minus OOF, all (pp)", "pp"),
    ("gap_clin_pp", "In-sample minus OOF, clin (pp)", "pp"),
    ("v_bin_all", "Binary policy value, all (%)", "%"),
    ("v_bin_clin", "Binary policy value, clinical (%)", "%"),
    ("insample_all", "Multi-arm in-sample, all (%)", "%"),
    ("insample_clin", "Multi-arm in-sample, clinical (%)", "%"),
]


def parse_seeds(spec: str):
    """Accept an int count ('20' -> 0..19) or a range ('0-9')."""
    if "-" in spec:
        lo, hi = spec.split("-")
        return list(range(int(lo), int(hi) + 1))
    return list(range(int(spec)))


# Cohorts for the prognostic-contribution analysis: pooled + one per arm.
# (arm code in T3: 0=Placebo, 1=Q12, 2=Q8; None -> pooled/all patients)
PROGNOSTIC_COHORTS = [
    ("Pooled", None),
    ("Placebo", 0),
    ("Q12", 1),
    ("Q8", 2),
]
PROGNOSTIC_COHORT_KEYS = ["pooled", "placebo", "q12", "q8"]
PROGNOSTIC_COHORT_LABELS = {
    "pooled": "Pooled", "placebo": "Placebo", "q12": "Q12", "q8": "Q8",
}


def prognostic_quantities(D, seed):
    """Recompute the prognostic-contribution quantities (run_analysis.py's
    prognostic block) for a single seed, for the pooled cohort and each arm.
    """
    Y, T3 = D["Y"], D["T3"]
    X_all_np, X_histwk_np = D["X_all_np"], D["X_histwk_np"]

    out = {}
    for name, arm in PROGNOSTIC_COHORTS:
        if arm is None:
            Xa, Xh, y = X_all_np, X_histwk_np, Y
        else:
            mask = T3 == arm
            Xa, Xh, y = X_all_np[mask], X_histwk_np[mask], Y[mask]

        p_all = cu.oof_proba(Xa, y, n_splits=5, seed=seed)
        p_clin = cu.oof_proba(Xh, y, n_splits=5, seed=seed)

        auc_all = cu.roc_auc_score(y, p_all)
        auc_clin = cu.roc_auc_score(y, p_clin)
        b_all = cu.brier_score_loss(y, p_all)
        b_clin = cu.brier_score_loss(y, p_clin)

        ib = cu.incremental_brier(y, p_all, p_clin)  # np.nan when undefined

        key = name.lower()
        out[f"auc_all_{key}"] = auc_all
        out[f"auc_clin_{key}"] = auc_clin
        out[f"auc_diff_{key}"] = auc_all - auc_clin          # all - clinical
        out[f"brier_diff_{key}"] = b_clin - b_all            # clinical - all (>0 favours all)
        out[f"ib_defined_{key}"] = 0 if np.isnan(ib) else 1  # relative-IPA ratio defined?
        out[f"ib_{key}"] = ib                                # relative incremental Brier (nan if undefined)
    return out


def run_multiseed(D, args):
    """Multi-seed reproducibility sweep and paste-ready summary block."""
    seeds = parse_seeds(args.seeds)
    banner(f"MULTISEED REPRODUCIBILITY SWEEP  ({len(seeds)} seeds: {seeds[0]}..{seeds[-1]})")
    print(f"Canonical single-seed run uses seed = {cu.DEFAULT_RANDOM_STATE} "
          f"(see run_analysis.py).")
    print(f"Characterizing the distribution across {len(seeds)} seeds ...\n")

    recs = []
    prog_recs = []
    for s in seeds:
        print(f"  seed {s} ...")
        h = headline_quantities(D, seed=s, n_boot=args.boot)
        h["seed"] = s
        recs.append(h)
        if not args.no_prognostic:
            pq = prognostic_quantities(D, seed=s)
            pq["seed"] = s
            prog_recs.append(pq)
    R = pd.DataFrame(recs)
    P = pd.DataFrame(prog_recs) if prog_recs else None

    # Persist the full per-seed table(s).
    ordered = ["seed"] + [c for c in R.columns if c != "seed"]
    R[ordered].to_csv("supp_reproducibility_multiseed.csv", index=False)
    if P is not None:
        pordered = ["seed"] + [c for c in P.columns if c != "seed"]
        P[pordered].to_csv("supp_reproducibility_prognostic.csv", index=False)

    # Paste-ready summary block.
    print("\n" + "=" * 78)
    print(f"REPRODUCIBILITY ACROSS {len(seeds)} SEEDS "
          f"(canonical seed = {cu.DEFAULT_RANDOM_STATE})")
    print("=" * 78 + "\n")

    def fmt(x):
        return f"{x:7.2f}"

    canonical = R.loc[R["seed"] == cu.DEFAULT_RANDOM_STATE]
    for key, label, unit in SUMMARY_FIELDS:
        if key not in R:
            continue
        x = R[key].astype(float)
        line = (f"  {label:<32}: mean={fmt(x.mean())}  sd={fmt(x.std(ddof=1))}  "
                f"min={fmt(x.min())}  max={fmt(x.max())}")
        if len(canonical):
            line += f"  [seed {cu.DEFAULT_RANDOM_STATE}: {fmt(float(canonical[key].iloc[0]))}]"
        print(line)

    # --- Prognostic contribution across seeds (per arm) ---
    if P is not None:
        pcanon = P.loc[P["seed"] == cu.DEFAULT_RANDOM_STATE]

        def fmt3(x):
            return f"{x:6.3f}"

        print("\n" + "=" * 78)
        print(f"PROGNOSTIC CONTRIBUTION ACROSS {len(seeds)} SEEDS "
              f"(out-of-fold; canonical seed = {cu.DEFAULT_RANDOM_STATE})")
        print("=" * 78)
        print("Positive AUROC diff or Brier diff = all-features better than clinical-only.\n")

        for key in PROGNOSTIC_COHORT_KEYS:
            label = PROGNOSTIC_COHORT_LABELS[key]
            print(f"  [{label}]")
            for metric, mlabel in [
                (f"auc_all_{key}",   "AUROC all"),
                (f"auc_clin_{key}",  "AUROC clinical"),
                (f"auc_diff_{key}",  "AUROC diff (all-clin)"),
                (f"brier_diff_{key}", "Brier diff (clin-all)"),
            ]:
                x = P[metric].astype(float)
                line = (f"    {mlabel:<24}: mean={fmt3(x.mean())}  sd={fmt3(x.std(ddof=1))}  "
                        f"min={fmt3(x.min())}  max={fmt3(x.max())}")
                if len(pcanon):
                    line += f"  [seed {cu.DEFAULT_RANDOM_STATE}: {fmt3(float(pcanon[metric].iloc[0]))}]"
                print(line)
            ndef = int(P[f"ib_defined_{key}"].sum())
            print(f"    {'incremental Brier':<24}: DEFINED in {ndef}/{len(seeds)} seeds "
                  f"(relative-IPA ratio; not summarized as a point estimate)")
            print()

    print("\nSaved: supp_reproducibility_multiseed.csv"
          + ("" if P is None else " and supp_reproducibility_prognostic.csv"))


# ===========================================================================
# Supplementary-section dispatch
# ===========================================================================
def run_supplementary_sections(D, sections_arg):
    """Run the requested supplementary sections.
    """
    if sections_arg:
        want = {s.strip().upper() for s in sections_arg.split(",") if s.strip()}
        run = {k: (k in want) for k in ["A", "B", "C", "D", "E"]}
    else:
        run = {
            "A": RUN_A_MISSING, "B": RUN_B_HYPERPARAM, "C": RUN_C_STABILITY,
            "D": RUN_D_MULTIPLICITY, "E": RUN_E_BASELINE,
        }

    if run["A"]:
        section_A(D)
    if run["B"]:
        section_B(D)
    if run["C"]:
        section_C(D)
    if run["D"]:
        section_D(D)
    if run["E"]:
        section_E(D)


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Supplementary analyses (A-E) + multi-seed "
                    "reproducibility sweep, in one script.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--seeds", default="20",
                    help="multiseed: int count (N -> 0..N-1) or range 'lo-hi' (default 20)")
    ap.add_argument("--boot", type=int, default=500,
                    help="multiseed: bootstrap reps per seed for policy-value deltas (default 500)")
    ap.add_argument("--no-prognostic", action="store_true",
                    help="multiseed: skip the per-arm prognostic (AUROC/Brier) block")
    ap.add_argument("--skip-supplementary", action="store_true",
                    help="skip the supplementary sections (A-E)")
    ap.add_argument("--skip-multiseed", action="store_true",
                    help="skip the multi-seed reproducibility sweep")
    ap.add_argument("--sections", default=None,
                    help="comma-separated supplementary sections to run (e.g. 'A,B,D,E'); "
                         "overrides the RUN_* defaults")
    args = ap.parse_args()

    print("Loading data and building feature matrices ...")
    D = load_everything()
    print(f"Cohort: n={len(D['Y'])}  "
          f"(Placebo={int((D['T3'] == 0).sum())}, "
          f"Q12={int((D['T3'] == 1).sum())}, "
          f"Q8={int((D['T3'] == 2).sum())})")

    if not args.skip_supplementary:
        run_supplementary_sections(D, args.sections)

    if not args.skip_multiseed:
        run_multiseed(D, args)

    banner("DONE")


if __name__ == "__main__":
    main()
