# =============================================================================
# src/modeling.py
# Stage 5: Model Building — Final 6 Models (M0–M5)
#
# Models:
#   M0: Dummy Classifier          — stratified lower bound
#   M1: LR Baseline               — log_churn + log_ncloc + Project FE (no TD)
#   M2: LR + TDR (RQ1)            — M1 + SQALE_DEBT_RATIO + Dup density
#   M3: LR + TDR + Ref (RQ2)      — M2 + HAS_REFACTORING + TDR_x_REF
#   M4: Random Forest              — all 6 features + Project FE
#   M5: XGBoost                   — all 6 features + Project FE (best ML + SHAP)
#
# All LR/RF models use class_weight='balanced'.
# XGBoost uses scale_pos_weight = neg/pos ratio.
# All models use per-project chronological 80/20 split.
# Project fixed effects implemented via 30 one-hot dummies (drop_first=True).
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model   import LogisticRegression
from sklearn.preprocessing  import StandardScaler
from sklearn.pipeline       import Pipeline
from sklearn.dummy          import DummyClassifier
from sklearn.ensemble       import RandomForestClassifier
from xgboost                import XGBClassifier

from config import (
    PROCESSED_FILES, MODELS, FIGURES, TABLES,
    TARGET, RANDOM_STATE, CLASS_WEIGHT,
    LR_MAX_ITER, LR_SOLVER, FIGURE_DPI
)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "figure.dpi": FIGURE_DPI, 
    
})


# ---------------------------------------------------------------------------
# Chronological train/test split
# ---------------------------------------------------------------------------
def chronological_split(df: pd.DataFrame,
                         test_ratio: float = 0.20) -> tuple:
    """
    Split data by time (per project): earliest 80% = train, latest 20% = test.
    This prevents temporal data leakage in model evaluation.

    We use per-project chronological ordering so that each project contributes
    to both train and test sets proportionally.
    """
    print(f"\n[MODEL] Chronological train/test split ({1-test_ratio:.0%} / {test_ratio:.0%})...")

    df = df.sort_values(["PROJECT_ID", "AUTHOR_DATE"]).copy()

    train_idx = []
    test_idx  = []

    for proj, grp in df.groupby("PROJECT_ID"):
        n    = len(grp)
        cut  = int(n * (1 - test_ratio))
        train_idx.extend(grp.index[:cut].tolist())
        test_idx.extend( grp.index[cut:].tolist())

    train = df.loc[train_idx].copy()
    test  = df.loc[test_idx ].copy()

    print(f"  Train: {len(train):,} commits "
          f"({train['BIC'].mean()*100:.2f}% BIC)")
    print(f"  Test : {len(test):,} commits  "
          f"({test['BIC'].mean()*100:.2f}% BIC)")

    return train, test


# ---------------------------------------------------------------------------
# Build project fixed-effect dummies
# ---------------------------------------------------------------------------
def get_project_dummies(train: pd.DataFrame,
                        test : pd.DataFrame,
                        drop_first: bool = True):
    """
    One-hot encode PROJECT_ID (fixed effects).
    Uses train set categories to avoid unseen categories in test.
    """
    proj_dummies_train = pd.get_dummies(
        train["PROJECT_ID"], prefix="proj", drop_first=drop_first
    )
    proj_dummies_test  = pd.get_dummies(
        test["PROJECT_ID"],  prefix="proj", drop_first=drop_first
    )

    # Align columns (test may have fewer projects)
    proj_dummies_test = proj_dummies_test.reindex(
        columns=proj_dummies_train.columns, fill_value=0
    )

    return proj_dummies_train, proj_dummies_test


# ---------------------------------------------------------------------------
# Build X, y matrices for a given feature set
# ---------------------------------------------------------------------------
def build_Xy(train: pd.DataFrame, test: pd.DataFrame,
             features: list, add_project_fe: bool = True):
    """Prepare feature matrices with optional project fixed effects."""

    proj_train, proj_test = get_project_dummies(train, test)

    if add_project_fe:
        X_train = pd.concat([train[features].reset_index(drop=True),
                              proj_train.reset_index(drop=True)], axis=1)
        X_test  = pd.concat([test[features].reset_index(drop=True),
                              proj_test.reset_index(drop=True)],  axis=1)
    else:
        X_train = train[features].copy().reset_index(drop=True)
        X_test  = test[features].copy().reset_index(drop=True)

    y_train = train[TARGET].values
    y_test  = test[TARGET].values

    # Fill any remaining NaN with column median
    X_train = X_train.fillna(X_train.median())
    X_test  = X_test.fillna(X_train.median())

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Build and fit a Logistic Regression model
# ---------------------------------------------------------------------------
def build_logistic(X_train, y_train,
                   model_name: str = "model",
                   class_weight: str = CLASS_WEIGHT) -> Pipeline:
    """
    Fit a Logistic Regression with:
    - StandardScaler (so coefficients are on same scale)
    - class_weight='balanced' (to handle 8.7% BIC imbalance)
    - L2 regularization (default in scikit-learn)
    """
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            class_weight = class_weight,
            max_iter     = LR_MAX_ITER,
            solver       = LR_SOLVER,
            random_state = RANDOM_STATE,
        ))
    ])
    pipe.fit(X_train, y_train)
    print(f"  ✅ {model_name} fitted ({X_train.shape[1]} features, "
          f"{len(y_train):,} train samples)")
    return pipe


# ---------------------------------------------------------------------------
# Feature sets for the 6 models
# ---------------------------------------------------------------------------
FEATURES_M1 = ["log_churn", "log_ncloc"]
FEATURES_M2 = ["SQALE_DEBT_RATIO", "log_churn", "log_ncloc",
                "DUPLICATED_LINES_DENSITY"]
FEATURES_M3 = ["SQALE_DEBT_RATIO", "log_churn", "log_ncloc",
                "DUPLICATED_LINES_DENSITY", "HAS_REFACTORING", "TDR_x_REF"]
FEATURES_ALL = FEATURES_M3   # M4 and M5 use the full feature set


# ---------------------------------------------------------------------------
# Run all 6 models
# ---------------------------------------------------------------------------
def run_all_models(df: pd.DataFrame) -> dict:
    """
    Train the 6 final models:
      M0 Dummy, M1 LR Baseline, M2 LR+TDR (RQ1),
      M3 LR+TDR+Ref (RQ2), M4 Random Forest, M5 XGBoost.

    Returns dict of {model_id: (model, X_test, y_test, feature_list)}.
    """
    print("\n[MODEL] Training 6 final models (M0–M5)...")

    train, test = chronological_split(df)

    # ── project fixed-effect dummies (shared across all models) ──────────────
    proj_tr = pd.get_dummies(train["PROJECT_ID"], prefix="proj", drop_first=True)
    proj_te = pd.get_dummies(test["PROJECT_ID"],  prefix="proj", drop_first=True)
    proj_te = proj_te.reindex(columns=proj_tr.columns, fill_value=0)

    y_train = train[TARGET].values
    y_test  = test[TARGET].values

    # Scale-pos-weight for XGBoost (ratio neg/pos in training set)
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    spw = round(neg / pos, 2)

    def Xy(feats):
        """Build X matrices for given feature list + project FE."""
        X_tr = pd.concat([train[feats].reset_index(drop=True),
                          proj_tr.reset_index(drop=True)], axis=1).fillna(0)
        X_te = pd.concat([test[feats].reset_index(drop=True),
                          proj_te.reset_index(drop=True)], axis=1).fillna(0)
        return X_tr, X_te

    results = {}

    # ── M0: Dummy classifier (stratified) ────────────────────────────────────
    X_tr0 = train[["log_churn"]].reset_index(drop=True).fillna(0)
    X_te0 = test[["log_churn"]].reset_index(drop=True).fillna(0)
    m0 = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)
    m0.fit(X_tr0, y_train)
    results["M0_dummy"] = (m0, X_te0, y_test, ["stratified_dummy"])
    print(f"  ✅ M0 Dummy")

    # ── M1: LR Baseline (no TD) ───────────────────────────────────────────────
    X_tr1, X_te1 = Xy(FEATURES_M1)
    m1 = Pipeline([
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(class_weight=CLASS_WEIGHT,
                                   max_iter=LR_MAX_ITER,
                                   solver=LR_SOLVER,
                                   random_state=RANDOM_STATE)),
    ])
    m1.fit(X_tr1, y_train)
    results["M1_lr_baseline"] = (m1, X_te1, y_test, FEATURES_M1)
    print(f"  ✅ M1 LR Baseline  ({X_tr1.shape[1]} features)")

    # ── M2: LR + TDR (RQ1) ───────────────────────────────────────────────────
    X_tr2, X_te2 = Xy(FEATURES_M2)
    m2 = Pipeline([
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(class_weight=CLASS_WEIGHT,
                                   max_iter=LR_MAX_ITER,
                                   solver=LR_SOLVER,
                                   random_state=RANDOM_STATE)),
    ])
    m2.fit(X_tr2, y_train)
    results["M2_lr_tdr_rq1"] = (m2, X_te2, y_test, FEATURES_M2)
    print(f"  ✅ M2 LR+TDR (RQ1) ({X_tr2.shape[1]} features)")

    # ── M3: LR + TDR + Refactoring (RQ2) ─────────────────────────────────────
    X_tr3, X_te3 = Xy(FEATURES_M3)
    m3 = Pipeline([
        ("sc",  StandardScaler()),
        ("clf", LogisticRegression(class_weight=CLASS_WEIGHT,
                                   max_iter=LR_MAX_ITER,
                                   solver=LR_SOLVER,
                                   random_state=RANDOM_STATE)),
    ])
    m3.fit(X_tr3, y_train)
    results["M3_lr_tdr_ref_rq2"] = (m3, X_te3, y_test, FEATURES_M3)
    print(f"  ✅ M3 LR+TDR+Ref (RQ2) ({X_tr3.shape[1]} features)")

    # ── M4: Random Forest ─────────────────────────────────────────────────────
    import time
    X_tr4, X_te4 = Xy(FEATURES_ALL)
    t0 = time.time()
    m4 = RandomForestClassifier(
        n_estimators=300, max_depth=15, min_samples_leaf=50,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    m4.fit(X_tr4, y_train)
    results["M4_random_forest"] = (m4, X_te4, y_test, FEATURES_ALL)
    print(f"  ✅ M4 Random Forest ({time.time()-t0:.1f}s)")

    # ── M5: XGBoost ───────────────────────────────────────────────────────────
    X_tr5, X_te5 = Xy(FEATURES_ALL)
    t0 = time.time()
    m5 = XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw,
        eval_metric="aucpr",
        random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
    )
    m5.fit(X_tr5, y_train,
           eval_set=[(X_te5, y_test)], verbose=False)
    results["M5_xgboost"] = (m5, X_te5, y_test, FEATURES_ALL)
    print(f"  ✅ M5 XGBoost      ({time.time()-t0:.1f}s, spw={spw})")

    print(f"\n  ✅ {len(results)} models trained | "
          f"Train: {len(y_train):,} | Test: {len(y_test):,} | "
          f"BIC rate test: {y_test.mean()*100:.2f}%")

    # ── Save all models ───────────────────────────────────────────────────────
    for name, (model, *_) in results.items():
        out = MODELS / f"final_{name}.pkl"
        with open(out, "wb") as f:
            pickle.dump(model, f)
    print(f"  ✅ Models saved to {MODELS}/final_M*.pkl")

    return results, train, test


# ---------------------------------------------------------------------------
# Feature importance / coefficients
# ---------------------------------------------------------------------------
def extract_coefficients(results: dict, df: pd.DataFrame):
    """
    Extract and interpret logistic regression coefficients for M2 (RQ1) and M3 (RQ2).
    Computes: Beta, Odds Ratio, and OR%.
    Saves coefficient table for the paper.
    """
    print("\n[MODEL] Extracting feature coefficients (M2 RQ1 / M3 RQ2)...")

    all_coef_rows = []

    for model_name, (pipe, X_te, y_te, feats) in results.items():
        if not hasattr(pipe, "named_steps"):
            continue
        clf = pipe.named_steps.get("clf")
        if not isinstance(clf, LogisticRegression):
            continue

        coef = clf.coef_[0]
        for i, feat in enumerate(feats):
            if i < len(coef):
                beta = coef[i]
                OR   = np.exp(beta)
                all_coef_rows.append({
                    "Model"  : model_name,
                    "Feature": feat,
                    "Beta"   : round(beta, 4),
                    "OR"     : round(OR, 4),
                    "OR_pct" : round((OR - 1) * 100, 2),
                })

    coef_df = pd.DataFrame(all_coef_rows)
    out_path = TABLES / "model_coefficients.csv"
    coef_df.to_csv(out_path, index=False)
    print(f"  ✅ Coefficients saved: {out_path}")

    # Print M2 (RQ1) coefficients
    rq1 = coef_df[coef_df["Model"] == "M2_lr_tdr_rq1"]
    if not rq1.empty:
        print("\n  M2 (RQ1) core coefficients:")
        print(rq1[rq1["Feature"].isin(
            ["SQALE_DEBT_RATIO", "log_churn", "log_ncloc", "DUPLICATED_LINES_DENSITY"]
        )][["Feature", "Beta", "OR", "OR_pct"]].to_string(index=False))

    # Print M3 (RQ2) interaction coefficient
    rq2 = coef_df[coef_df["Model"] == "M3_lr_tdr_ref_rq2"]
    if not rq2.empty:
        print("\n  M3 (RQ2) interaction coefficient:")
        int_row = rq2[rq2["Feature"] == "TDR_x_REF"]
        if not int_row.empty:
            print(int_row[["Feature", "Beta", "OR", "OR_pct"]].to_string(index=False))

    return coef_df


# ---------------------------------------------------------------------------
# Figure: Coefficient plot (Forest plot style)
# ---------------------------------------------------------------------------
def fig_coefficient_plot(coef_df: pd.DataFrame):
    """Forest-plot style OR visualization for M2 (RQ1) and M3 (RQ2)."""
    print("\n[MODEL] Figure: Coefficient forest plot (M2/M3)...")

    main_models = ["M2_lr_tdr_rq1", "M3_lr_tdr_ref_rq2"]
    df_plot = coef_df[coef_df["Model"].isin(main_models)].copy()
    df_plot = df_plot[~df_plot["Feature"].str.startswith("proj_")]

    if df_plot.empty:
        print("  ⚠️  No coefficient data available for plotting")
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(df_plot) * 0.4 + 1)))
    colors  = {"M2_lr_tdr_rq1": "#3498DB", "M3_lr_tdr_ref_rq2": "#C0392B"}
    markers = {"M2_lr_tdr_rq1": "o",       "M3_lr_tdr_ref_rq2": "s"}
    labels  = {"M2_lr_tdr_rq1": "M2: LR+TDR (RQ1)",
               "M3_lr_tdr_ref_rq2": "M3: LR+TDR+Ref (RQ2)"}

    for model in main_models:
        sub   = df_plot[df_plot["Model"] == model]
        color = colors[model]
        mkr   = markers[model]
        ax.scatter(sub["OR"], sub["Feature"],
                   color=color, marker=mkr, s=80,
                   label=labels[model], zorder=5)

    ax.axvline(1.0, ls="--", color="gray", alpha=0.6, label="OR = 1.0 (no effect)")
    ax.set_xlabel("Odds Ratio (OR)")
    ax.set_title("Odds Ratios — M2 (RQ1) and M3 (RQ2)\n"
                 "(OR > 1 = higher BIC probability)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = FIGURES / "fig_coefficient_plot.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Coefficient plot saved: {out}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run_modeling(df: pd.DataFrame = None) -> dict:
    """Full modeling pipeline — trains M0 through M5."""
    print("\n" + "=" * 65)
    print("MODELING PIPELINE — TDInsight (M0–M5)")
    print("=" * 65)

    if df is None:
        df = pd.read_parquet(PROCESSED_FILES["modeling_ready"])
        print(f"  Loaded: {len(df):,} rows")

    results, train, test = run_all_models(df)
    coef_df = extract_coefficients(results, df)
    fig_coefficient_plot(coef_df)

    print("\n" + "=" * 65)
    print("✅ MODELING COMPLETE — 6 models (M0–M5) saved")
    print("=" * 65)

    return results, coef_df


if __name__ == "__main__":
    run_modeling()
