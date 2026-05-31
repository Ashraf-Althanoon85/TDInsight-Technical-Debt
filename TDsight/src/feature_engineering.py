# =============================================================================
# src/feature_engineering.py
# Stage 3: Feature engineering and modeling-ready dataset construction
#
# Steps:
#   1. Log-transform skewed variables (Churn, NCLOC)
#   2. Handle missing values
#   3. Multicollinearity check (VIF)
#   4. Build final modeling-ready dataset
#   5. Build RQ3 panel dataset (project-month with ΔTDR)
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from config import (
    INTERIM_FILES, PROCESSED_FILES, TD_FEATURES,
    TD_PRIMARY, CONTROLS, TARGET, COLLINEAR_GROUPS, RQ3_WINDOWS_DAYS
)


# ---------------------------------------------------------------------------
# STEP 1: Log-transform skewed variables
# ---------------------------------------------------------------------------
def apply_log_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply log1p transformation to heavy-tailed variables.

    Justification (from EDA):
      - CHURN: median=24, max=416,424 → extreme right skew
      - NCLOC: large projects vs small → right skew
      - SQALE_INDEX: raw minutes, correlated with size → right skew

    log1p(x) = log(1+x) handles zeros gracefully.
    """
    print("\n[FEATURE ENG] Applying log1p transforms to skewed variables...")

    df = df.copy()

    # Churn transform (primary control)
    df["log_churn"] = np.log1p(df["CHURN"])
    print(f"  → log_churn: mean={df['log_churn'].mean():.2f}, "
          f"std={df['log_churn'].std():.2f}")

    # NCLOC transform (size proxy)
    if "NCLOC" in df.columns:
        df["log_ncloc"] = np.log1p(df["NCLOC"])
        print(f"  → log_ncloc: mean={df['log_ncloc'].mean():.2f}, "
              f"std={df['log_ncloc'].std():.2f}")

    # SQALE_INDEX transform (raw debt in minutes)
    if "SQALE_INDEX" in df.columns:
        df["log_sqale_index"] = np.log1p(df["SQALE_INDEX"])

    # REF_COUNT transform (heavy-tailed: median=3, max=3516)
    if "REF_COUNT" in df.columns:
        df["log_ref_count"] = np.log1p(df["REF_COUNT"])

    print("  ✅ Log transforms applied")
    return df


# ---------------------------------------------------------------------------
# STEP 2: Handle missing values
# ---------------------------------------------------------------------------
def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strategy for remaining missing values:
    - TD metrics: impute with project-level median (missing = project not yet analyzed)
    - CHURN: fill with 0 (no recorded changes = 0 lines changed)
    - After imputation: report remaining NAs
    """
    print("\n[FEATURE ENG] Handling missing values...")

    df = df.copy()

    # Report before
    td_cols = [c for c in TD_FEATURES if c in df.columns]
    missing_before = df[td_cols].isnull().sum()
    if missing_before.sum() > 0:
        print(f"  Missing values BEFORE imputation:")
        print(f"  {missing_before[missing_before > 0].to_dict()}")

    # Project-level median imputation for TD metrics
    for col in td_cols:
        if df[col].isnull().sum() > 0:
            project_medians = df.groupby("PROJECT_ID")[col].transform("median")
            global_median   = df[col].median()
            df[col] = df[col].fillna(project_medians).fillna(global_median)

    # Log-transformed features
    for log_col in ["log_churn", "log_ncloc", "log_sqale_index", "log_ref_count"]:
        if log_col in df.columns:
            df[log_col] = df[log_col].fillna(0)

    # Verify
    missing_after = df[td_cols + ["log_churn", "log_ncloc"]].isnull().sum()
    print(f"  Missing values AFTER imputation: {missing_after.sum()} total")
    print("  ✅ Missing value handling complete")
    return df


# ---------------------------------------------------------------------------
# STEP 3: Multicollinearity check (VIF)
# ---------------------------------------------------------------------------
def check_multicollinearity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Variance Inflation Factor (VIF) for all candidate predictors.
    VIF > 10 signals severe multicollinearity → that variable should not be
    used together with correlated variables in the same model.

    This confirms our model design choice:
      - Use TDR alone (not with SQALE_INDEX, CODE_SMELLS, etc.)
      - Report the full correlation matrix for the paper
    """
    print("\n[FEATURE ENG] Multicollinearity check (VIF)...")

    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        from statsmodels.tools.tools import add_constant

        # Variables to check
        check_vars = [
            "SQALE_DEBT_RATIO", "log_sqale_index", "CODE_SMELLS",
            "COMPLEXITY", "COGNITIVE_COMPLEXITY",
            "DUPLICATED_LINES_DENSITY", "log_ncloc", "log_churn"
        ]
        check_vars = [v for v in check_vars if v in df.columns]

        # Drop rows with NA for VIF calculation
        vif_data = df[check_vars].dropna()

        # Sample for speed if large
        if len(vif_data) > 20000:
            vif_data = vif_data.sample(20000, random_state=42)

        X = add_constant(vif_data)
        vif_results = pd.DataFrame({
            "Variable": check_vars,
            "VIF": [variance_inflation_factor(X.values, i + 1)
                    for i in range(len(check_vars))]
        }).sort_values("VIF", ascending=False)

        print("\n  VIF Results (>10 = problematic multicollinearity):")
        print(vif_results.to_string(index=False))
        print("\n  ⚠️  Variables with VIF > 10 should NOT be used together in one model")
        print("  ✅ Model A uses TDR only (VIF-safe)")

        return vif_results

    except ImportError:
        print("  ⚠️  statsmodels not available — skipping VIF (install with: pip install statsmodels)")

    # Fallback: Pearson correlation matrix
    check_vars = [v for v in [
        "SQALE_DEBT_RATIO", "SQALE_INDEX", "CODE_SMELLS",
        "COMPLEXITY", "COGNITIVE_COMPLEXITY",
        "DUPLICATED_LINES_DENSITY", "NCLOC"
    ] if v in df.columns]
    corr = df[check_vars].corr().round(2)
    print("  Correlation Matrix (|r| > 0.8 indicates collinearity):")
    print(corr.to_string())
    return corr


# ---------------------------------------------------------------------------
# STEP 4: Build final modeling-ready dataset
# ---------------------------------------------------------------------------
def build_modeling_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct the final modeling-ready DataFrame with:
      - All features in clean, named columns
      - Project dummy encoding (for fixed effects in statsmodels)
      - Interaction term: TDR × HAS_REFACTORING (for RQ2)
      - Time index for chronological splitting

    Final columns:
      TARGET: BIC
      PRIMARY: SQALE_DEBT_RATIO
      CONTROLS: log_churn, log_ncloc, DUPLICATED_LINES_DENSITY
      RQ2: HAS_REFACTORING, REF_COUNT, log_ref_count, TDR_x_REF
      METADATA: PROJECT_ID, COMMIT_HASH, AUTHOR_DATE
    """
    print("\n[FEATURE ENG] Building final modeling-ready dataset...")

    df = df.copy()

    # --- Interaction term for RQ2 ---
    df["TDR_x_REF"] = df["SQALE_DEBT_RATIO"] * df["HAS_REFACTORING"]
    print("  → Interaction term TDR × HAS_REFACTORING created")

    # --- Also: TDR × log_ref_count (continuous interaction for RQ2 extension) ---
    if "log_ref_count" in df.columns:
        df["TDR_x_REFCOUNT"] = df["SQALE_DEBT_RATIO"] * df["log_ref_count"]

    # --- Project index (for fixed effects) ---
    proj_cats = pd.Categorical(df["PROJECT_ID"])
    df["PROJECT_IDX"] = proj_cats.codes
    print(f"  → Project codes: {df['PROJECT_IDX'].nunique()} unique projects")

    # --- Time index (for chronological train/test split) ---
    df = df.sort_values(["PROJECT_ID", "AUTHOR_DATE"]).reset_index(drop=True)
    df["TIME_IDX"] = df.groupby("PROJECT_ID").cumcount()

    # --- Define final feature sets ---
    # Model A: RQ1 baseline (TDR + controls + project FE)
    model_a_features = [
        "SQALE_DEBT_RATIO",        # Primary TD predictor
        "log_churn",               # Size of change
        "log_ncloc",               # Size of system
        "DUPLICATED_LINES_DENSITY" # TD proxy (low collinearity with TDR)
    ]

    # Model B: RQ2 (Model A + refactoring + interaction)
    model_b_features = model_a_features + [
        "HAS_REFACTORING",
        "TDR_x_REF"
    ]

    # Model C: Baseline (churn + size only — no TD)
    model_c_features = [
        "log_churn",
        "log_ncloc",
    ]

    # Model D: Sensitivity (one TD metric at a time — for robustness)
    sensitivity_features = {
        "CODE_SMELLS"          : ["CODE_SMELLS",           "log_churn", "log_ncloc"],
        "COMPLEXITY"           : ["COMPLEXITY",             "log_churn", "log_ncloc"],
        "COGNITIVE_COMPLEXITY" : ["COGNITIVE_COMPLEXITY",   "log_churn", "log_ncloc"],
        "log_sqale_index"      : ["log_sqale_index",        "log_churn", "log_ncloc"],
    }

    print(f"\n  Feature sets defined:")
    print(f"     Model A (RQ1)     : {model_a_features}")
    print(f"     Model B (RQ2)     : {model_b_features}")
    print(f"     Model C (baseline): {model_c_features}")
    print(f"     Sensitivity models: {list(sensitivity_features.keys())}")

    # Verify all features are present
    all_feats = list(set(
        model_a_features + model_b_features + model_c_features
    ))
    missing_feats = [f for f in all_feats if f not in df.columns]
    if missing_feats:
        print(f"\n  ⚠️  Missing features: {missing_feats}")
    else:
        print(f"\n  ✅ All features present in dataset")

    # Final check
    print(f"\n  MODELING DATASET SUMMARY:")
    print(f"     Rows    : {len(df):,}")
    print(f"     BIC=1   : {df['BIC'].sum():,} ({df['BIC'].mean()*100:.2f}%)")
    print(f"     BIC=0   : {(df['BIC']==0).sum():,}")
    print(f"     Projects: {df['PROJECT_ID'].nunique()}")
    print(f"     Date range: {df['AUTHOR_DATE'].min().date()} → "
          f"{df['AUTHOR_DATE'].max().date()}")

    return df


# ---------------------------------------------------------------------------
# STEP 5: Build RQ3 panel dataset (project × month with ΔTDR)
# ---------------------------------------------------------------------------
def build_rq3_panel(cohort: pd.DataFrame) -> pd.DataFrame:
    """
    RQ3: Does accumulation of TD (ΔTDR) predict future BIC rate?

    Construction:
      1. Group commits into project × calendar-month bins
      2. For each bin: compute mean TDR, mean churn, BIC count
      3. Compute ΔTDR = TDR_t - TDR_{t-1} (within-project lag)
      4. For each window τ ∈ {30, 60, 90 days}:
         - Compute forward BIC rate in next τ days
         - This is the dependent variable for RQ3

    Returns a panel DataFrame indexed by (PROJECT_ID, PERIOD).
    """
    print("\n[FEATURE ENG] Building RQ3 panel dataset (ΔTDR analysis)...")

    df = cohort.copy()
    df["PERIOD"] = df["AUTHOR_DATE"].dt.to_period("M")  # Month-level bins

    # Aggregate to project-month level
    panel = (
        df.groupby(["PROJECT_ID", "PERIOD"])
        .agg(
            TDR_mean        = ("SQALE_DEBT_RATIO", "mean"),
            TDR_max         = ("SQALE_DEBT_RATIO", "max"),
            CHURN_sum       = ("CHURN", "sum"),
            NCLOC_mean      = ("NCLOC", "mean"),
            N_COMMITS       = ("COMMIT_HASH", "count"),
            BIC_COUNT       = ("BIC", "sum"),
            REF_COUNT_sum   = ("REF_COUNT", "sum"),
        )
        .reset_index()
    )

    panel["BIC_RATE"] = panel["BIC_COUNT"] / panel["N_COMMITS"]

    # Sort for lag computation
    panel = panel.sort_values(["PROJECT_ID", "PERIOD"])

    # Compute ΔTDR (within-project lag-1 difference)
    panel["TDR_LAG1"]  = panel.groupby("PROJECT_ID")["TDR_mean"].shift(1)
    panel["DELTA_TDR"] = panel["TDR_mean"] - panel["TDR_LAG1"]

    # Add log_churn at panel level
    panel["log_churn_sum"] = np.log1p(panel["CHURN_sum"])
    panel["log_ncloc"]     = np.log1p(panel["NCLOC_mean"])

    # Compute forward BIC rate for each window (τ days)
    # We convert Period to timestamp for window computation
    panel["PERIOD_START"] = panel["PERIOD"].dt.to_timestamp()

    for tau in RQ3_WINDOWS_DAYS:
        col_name = f"BIC_RATE_NEXT_{tau}D"
        forward_bic = []

        for _, row in panel.iterrows():
            proj = row["PROJECT_ID"]
            t0   = row["PERIOD_START"]
            t_end = t0 + pd.Timedelta(days=tau)

            # Count BICs in project in [t0+1day, t0+tau days]
            future = df[
                (df["PROJECT_ID"] == proj) &
                (df["AUTHOR_DATE"] > t0) &
                (df["AUTHOR_DATE"] <= t_end)
            ]
            rate = future["BIC"].mean() if len(future) > 0 else np.nan
            forward_bic.append(rate)

        panel[col_name] = forward_bic
        print(f"  → Forward BIC rate ({tau}d): mean={panel[col_name].mean():.3f}")

    # Drop rows with missing ΔTDR (first observation per project)
    panel = panel.dropna(subset=["DELTA_TDR"])

    print(f"  ✅ RQ3 panel: {len(panel):,} project-months across "
          f"{panel['PROJECT_ID'].nunique()} projects")
    return panel


# ---------------------------------------------------------------------------
# MAIN: Run full feature engineering pipeline
# ---------------------------------------------------------------------------
def run_feature_engineering(cohort: pd.DataFrame = None) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    Loads cohort from file if not passed directly.
    """
    print("\n" + "=" * 65)
    print("FEATURE ENGINEERING PIPELINE — TD → BIC Study")
    print("=" * 65)

    # Load cohort if not provided
    if cohort is None:
        print("  Loading cohort from disk...")
        cohort = pd.read_parquet(PROCESSED_FILES["cohort"])
        print(f"  Loaded: {len(cohort):,} rows")

    # Step 1: Log transforms
    cohort = apply_log_transforms(cohort)

    # Step 2: Missing values
    cohort = handle_missing_values(cohort)

    # Step 3: Multicollinearity check (report only)
    vif = check_multicollinearity(cohort)

    # Step 4: Build modeling dataset
    modeling_df = build_modeling_dataset(cohort)

    # Step 5: Build RQ3 panel
    # Note: RQ3 panel can be slow on large datasets — runs after main modeling
    print("\n  Building RQ3 panel (may take a few minutes on full dataset)...")
    try:
        panel_df = build_rq3_panel(cohort)
        panel_df.to_parquet(PROCESSED_FILES["panel_data"], index=False)
        print(f"  ✅ RQ3 panel saved: {PROCESSED_FILES['panel_data']}")
    except Exception as e:
        print(f"  ⚠️  RQ3 panel skipped (error: {e})")

    # Save modeling dataset
    modeling_df.to_parquet(PROCESSED_FILES["modeling_ready"], index=False)
    print(f"\n  ✅ Modeling dataset saved: {PROCESSED_FILES['modeling_ready']}")

    print("\n" + "=" * 65)
    print("✅ FEATURE ENGINEERING COMPLETE")
    print(f"   {len(modeling_df):,} commits × {modeling_df.shape[1]} features")
    print("=" * 65)

    return modeling_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    modeling_df = run_feature_engineering()
