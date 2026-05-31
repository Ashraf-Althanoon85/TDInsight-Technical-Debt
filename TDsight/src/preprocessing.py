# =============================================================================
# src/preprocessing.py
# Stage 1 & 2: Load, clean, and link all raw data sources
#
# Steps covered:
#   1. Load all CSV files with proper dtypes and date parsing
#   2. Dataset inventory (rows, cols, missing values)
#   3. Clean column names and standardize data types
#   4. Build commit-level base from GIT_COMMITS
#   5. Compute Churn from GIT_COMMITS_CHANGES
#   6. Create BIC label from SZZ_FAULT_INDUCING_COMMITS
#   7. Create binary Refactoring variable from REFACTORING_MINER
#   8. Time-safe linking with SONAR_ANALYSIS + SONAR_MEASURES (merge_asof)
#   9. Save all interim outputs
# =============================================================================

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

from config import (
    RAW_FILES, INTERIM_FILES, TD_FEATURES,
    MIN_COMMITS_PER_PROJECT, MIN_SONAR_SNAPSHOTS
)

# ---------------------------------------------------------------------------
# HELPER: Dataset inventory report
# ---------------------------------------------------------------------------
def dataset_inventory(dfs: dict) -> pd.DataFrame:
    """
    Print and return a summary table for all loaded DataFrames.
    Covers: rows, columns, missing values (count + %).
    """
    rows = []
    for name, df in dfs.items():
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        rows.append({
            "File"            : name,
            "Rows"            : df.shape[0],
            "Cols"            : df.shape[1],
            "Missing Cells"   : missing_cells,
            "Missing %"       : round(missing_cells / total_cells * 100, 2)
                                if total_cells > 0 else 0,
        })
    inv = pd.DataFrame(rows)
    print("\n" + "=" * 65)
    print("DATASET INVENTORY")
    print("=" * 65)
    print(inv.to_string(index=False))
    print("=" * 65)
    return inv


# ---------------------------------------------------------------------------
# STEP 1: Load all raw CSV files
# ---------------------------------------------------------------------------
def load_raw_data() -> dict:
    """
    Load all 10 CSV files into DataFrames.
    Applies minimal parsing (dates as strings first — cleaned later).
    Returns a dict of {name: DataFrame}.
    """
    print("\n[STEP 1] Loading raw CSV files...")

    dfs = {}

    # -- GIT_COMMITS --
    print("  → GIT_COMMITS.csv")
    dfs["commits"] = pd.read_csv(
        RAW_FILES["commits"],
        usecols=["PROJECT_ID", "COMMIT_HASH", "AUTHOR_DATE",
                 "IN_MAIN_BRANCH", "MERGE"],
        dtype={"PROJECT_ID": str, "COMMIT_HASH": str,
               "IN_MAIN_BRANCH": bool, "MERGE": bool},
        low_memory=False,
    )

    # -- GIT_COMMITS_CHANGES (large: 1.1M rows) --
    print("  → GIT_COMMITS_CHANGES.csv  (large file — be patient)")
    dfs["changes"] = pd.read_csv(
        RAW_FILES["changes"],
        usecols=["PROJECT_ID", "COMMIT_HASH", "LINES_ADDED", "LINES_REMOVED"],
        dtype={"PROJECT_ID": str, "COMMIT_HASH": str,
               "LINES_ADDED": "Int64", "LINES_REMOVED": "Int64"},
        low_memory=False,
    )

    # -- PROJECTS --
    print("  → PROJECTS.csv")
    dfs["projects"] = pd.read_csv(RAW_FILES["projects"])

    # -- SZZ_FAULT_INDUCING_COMMITS --
    print("  → SZZ_FAULT_INDUCING_COMMITS.csv")
    dfs["szz"] = pd.read_csv(
        RAW_FILES["szz"],
        dtype={"PROJECT_ID": str,
               "FAULT_FIXING_COMMIT_HASH": str,
               "FAULT_INDUCING_COMMIT_HASH": str},
    )

    # -- REFACTORING_MINER --
    print("  → REFACTORING_MINER.csv")
    dfs["refactoring"] = pd.read_csv(
        RAW_FILES["refactoring"],
        usecols=["PROJECT_ID", "COMMIT_HASH", "REFACTORING_TYPE"],
        dtype={"PROJECT_ID": str, "COMMIT_HASH": str},
        low_memory=False,
    )

    # -- SONAR_ANALYSIS --
    print("  → SONAR_ANALYSIS.csv")
    dfs["sonar_analysis"] = pd.read_csv(
        RAW_FILES["sonar_analysis"],
        dtype={"PROJECT_ID": str, "ANALYSIS_KEY": str, "REVISION": str},
        low_memory=False,
    )

    # -- SONAR_MEASURES (240 cols — select only what we need) --
    print("  → SONAR_MEASURES.csv  (selecting key columns only)")
    measures_cols = ["PROJECT_ID", "ANALYSIS_KEY"] + TD_FEATURES
    dfs["sonar_measures"] = pd.read_csv(
        RAW_FILES["sonar_measures"],
        usecols=measures_cols,
        dtype={"PROJECT_ID": str, "ANALYSIS_KEY": str},
        low_memory=False,
    )

    print(f"  ✅ Loaded {len(dfs)} datasets")

    # Print sizes
    for name, df in dfs.items():
        print(f"     {name:20s}: {len(df):>10,} rows × {df.shape[1]} cols")

    return dfs


# ---------------------------------------------------------------------------
# STEP 2: Dataset inventory
# ---------------------------------------------------------------------------
def run_inventory(dfs: dict) -> pd.DataFrame:
    """Run and save the dataset inventory report."""
    print("\n[STEP 2] Running dataset inventory...")
    inv = dataset_inventory(dfs)
    return inv


# ---------------------------------------------------------------------------
# STEP 3: Clean and standardize
# ---------------------------------------------------------------------------
def clean_data(dfs: dict) -> dict:
    """
    - Parse date columns to timezone-aware UTC datetime
    - Strip whitespace from string columns
    - Drop duplicates where applicable
    Returns cleaned dict of DataFrames.
    """
    print("\n[STEP 3] Cleaning and standardizing data...")

    # --- commits: parse AUTHOR_DATE ---
    print("  → Parsing AUTHOR_DATE in GIT_COMMITS...")
    dfs["commits"]["AUTHOR_DATE"] = pd.to_datetime(
        dfs["commits"]["AUTHOR_DATE"], format="mixed", utc=True
    )
    dfs["commits"] = dfs["commits"].drop_duplicates(
        subset=["PROJECT_ID", "COMMIT_HASH"]
    )
    print(f"     commits after dedup: {len(dfs['commits']):,}")

    # --- changes: fill NA with 0 ---
    dfs["changes"]["LINES_ADDED"]   = dfs["changes"]["LINES_ADDED"].fillna(0)
    dfs["changes"]["LINES_REMOVED"] = dfs["changes"]["LINES_REMOVED"].fillna(0)

    # --- sonar_analysis: parse DATE ---
    print("  → Parsing DATE in SONAR_ANALYSIS...")
    dfs["sonar_analysis"]["DATE"] = pd.to_datetime(
        dfs["sonar_analysis"]["DATE"], format="mixed", utc=True
    )
    dfs["sonar_analysis"] = dfs["sonar_analysis"].drop_duplicates(
        subset=["PROJECT_ID", "ANALYSIS_KEY"]
    )

    # --- sonar_measures: clip negative values (data quality) ---
    for col in TD_FEATURES:
        if col in dfs["sonar_measures"].columns:
            dfs["sonar_measures"][col] = dfs["sonar_measures"][col].clip(lower=0)

    # --- szz: drop duplicates ---
    dfs["szz"] = dfs["szz"].drop_duplicates()

    # --- refactoring: drop duplicates ---
    dfs["refactoring"] = dfs["refactoring"].drop_duplicates()

    print("  ✅ Data cleaning complete")
    return dfs


# ---------------------------------------------------------------------------
# STEP 4: Build commit-level base dataset
# ---------------------------------------------------------------------------
def build_commit_base(dfs: dict) -> pd.DataFrame:
    """
    Start from GIT_COMMITS:
    - Keep only main-branch, non-merge commits
    - Drop projects with fewer than MIN_COMMITS_PER_PROJECT commits
    """
    print("\n[STEP 4] Building commit-level base dataset...")

    commits = dfs["commits"].copy()

    # Filter: main branch only, no merge commits
    before = len(commits)
    commits = commits[
        (commits["IN_MAIN_BRANCH"] == True) &
        (commits["MERGE"] == False)
    ].copy()
    print(f"  → After main-branch + no-merge filter: {len(commits):,} "
          f"(removed {before - len(commits):,})")

    # Drop projects below minimum commit threshold
    proj_counts = commits.groupby("PROJECT_ID").size()
    valid_projects = proj_counts[proj_counts >= MIN_COMMITS_PER_PROJECT].index
    before = len(commits)
    commits = commits[commits["PROJECT_ID"].isin(valid_projects)].copy()
    print(f"  → After min-commits filter ({MIN_COMMITS_PER_PROJECT}): "
          f"{len(commits):,} commits across {commits['PROJECT_ID'].nunique()} projects")

    # Sort chronologically per project
    commits = commits.sort_values(["PROJECT_ID", "AUTHOR_DATE"]).reset_index(drop=True)

    print(f"  ✅ Commit base: {len(commits):,} commits, "
          f"{commits['PROJECT_ID'].nunique()} projects")
    return commits


# ---------------------------------------------------------------------------
# STEP 5: Compute Churn
# ---------------------------------------------------------------------------
def compute_churn(dfs: dict) -> pd.DataFrame:
    """
    Aggregate GIT_COMMITS_CHANGES to commit level.
    CHURN = LINES_ADDED + LINES_REMOVED (per commit, summed across files)
    Returns DataFrame with [PROJECT_ID, COMMIT_HASH, LINES_ADDED, LINES_REMOVED, CHURN]
    """
    print("\n[STEP 5] Computing Churn from GIT_COMMITS_CHANGES...")

    changes = dfs["changes"].copy()

    churn = (
        changes
        .groupby(["PROJECT_ID", "COMMIT_HASH"], as_index=False)
        .agg(
            LINES_ADDED=("LINES_ADDED", "sum"),
            LINES_REMOVED=("LINES_REMOVED", "sum"),
        )
    )
    churn["CHURN"] = churn["LINES_ADDED"] + churn["LINES_REMOVED"]

    print(f"  → Churn computed for {len(churn):,} commits")
    print(f"     CHURN — median: {churn['CHURN'].median():.0f}, "
          f"max: {churn['CHURN'].max():,.0f}, "
          f"mean: {churn['CHURN'].mean():.0f}")
    print(f"  ✅ Churn table ready — log1p transform will be applied in feature engineering")

    return churn


# ---------------------------------------------------------------------------
# STEP 6: Create BIC label
# ---------------------------------------------------------------------------
def create_bic_labels(dfs: dict) -> pd.DataFrame:
    """
    From SZZ_FAULT_INDUCING_COMMITS:
    - Extract unique (PROJECT_ID, FAULT_INDUCING_COMMIT_HASH) pairs
    - Create BIC = 1 for those commits, BIC = 0 otherwise
    Returns DataFrame with [PROJECT_ID, COMMIT_HASH, BIC]
    """
    print("\n[STEP 6] Creating BIC labels from SZZ...")

    szz = dfs["szz"].copy()

    # Each row is a (bug-fixing commit → fault-inducing commit) mapping
    # We care about the FAULT_INDUCING_COMMIT_HASH (the commits that caused bugs)
    bic_df = (
        szz[["PROJECT_ID", "FAULT_INDUCING_COMMIT_HASH"]]
        .drop_duplicates()
        .rename(columns={"FAULT_INDUCING_COMMIT_HASH": "COMMIT_HASH"})
        .copy()
    )
    bic_df["BIC"] = 1

    print(f"  → Unique fault-inducing commits: {len(bic_df):,}")
    print(f"  → Projects covered by SZZ: {bic_df['PROJECT_ID'].nunique()}")
    print(f"  ✅ BIC labels ready")

    return bic_df


# ---------------------------------------------------------------------------
# STEP 7: Create Refactoring variable
# ---------------------------------------------------------------------------
def create_refactoring_variable(dfs: dict) -> pd.DataFrame:
    """
    From REFACTORING_MINER:
    - HAS_REFACTORING = 1 if commit has at least one refactoring
    - REF_COUNT = total number of refactoring operations (quantitative)
    Returns DataFrame with [PROJECT_ID, COMMIT_HASH, HAS_REFACTORING, REF_COUNT]
    """
    print("\n[STEP 7] Creating Refactoring variable from REFACTORING_MINER...")

    ref = dfs["refactoring"].copy()

    ref_agg = (
        ref
        .groupby(["PROJECT_ID", "COMMIT_HASH"], as_index=False)
        .agg(REF_COUNT=("REFACTORING_TYPE", "count"))
    )
    ref_agg["HAS_REFACTORING"] = 1  # All rows here have at least 1

    print(f"  → Commits with refactoring: {len(ref_agg):,}")
    print(f"  → REF_COUNT — median: {ref_agg['REF_COUNT'].median():.0f}, "
          f"mean: {ref_agg['REF_COUNT'].mean():.1f}, "
          f"max: {ref_agg['REF_COUNT'].max()}")
    print(f"  ✅ Refactoring variable ready")

    return ref_agg


# ---------------------------------------------------------------------------
# STEP 8: Time-safe Sonar linking (KEY methodological step)
# ---------------------------------------------------------------------------
def link_sonar_time_safe(commits: pd.DataFrame, dfs: dict) -> pd.DataFrame:
    """
    TIME-SAFE LINKING STRATEGY (prevents data leakage):

    For each commit C at time T in project P:
      → Find the LAST Sonar analysis snapshot with DATE < T in project P
      → Assign those Sonar metrics to commit C

    This ensures we only use information available BEFORE the commit was made.
    Using the snapshot AT or AFTER the commit would constitute data leakage.

    Implementation: pd.merge_asof with direction='backward'
      - Sorts both tables by date
      - For each commit, matches the most recent Sonar snapshot that is <= commit date

    Returns commits DataFrame enriched with Sonar measures.
    """
    print("\n[STEP 8] Time-safe Sonar linking (anti-leakage merge_asof)...")
    print("  ⚠️  KEY METHODOLOGICAL NOTE:")
    print("      Each commit is linked to the LAST Sonar snapshot BEFORE it.")
    print("      This prevents future information leakage into predictors.")

    # Merge SONAR_ANALYSIS with SONAR_MEASURES to get metrics + dates
    sa = dfs["sonar_analysis"].copy()
    sm = dfs["sonar_measures"].copy()

    sonar = sa.merge(
        sm,
        on=["PROJECT_ID", "ANALYSIS_KEY"],
        how="inner"
    )
    sonar = sonar.sort_values(["PROJECT_ID", "DATE"])

    # Drop rows with missing TDR (primary predictor — must not be null)
    before = len(sonar)
    sonar = sonar.dropna(subset=["SQALE_DEBT_RATIO"])
    print(f"  → Sonar snapshots with valid TDR: {len(sonar):,} / {before:,}")

    # Check projects with enough snapshots
    snap_counts = sonar.groupby("PROJECT_ID").size()
    valid_projs = snap_counts[snap_counts >= MIN_SONAR_SNAPSHOTS].index
    sonar = sonar[sonar["PROJECT_ID"].isin(valid_projs)]
    print(f"  → After min-snapshots filter ({MIN_SONAR_SNAPSHOTS}): "
          f"{len(sonar):,} snapshots, {sonar['PROJECT_ID'].nunique()} projects")

    # Perform merge_asof per project
    all_results = []
    projects = commits["PROJECT_ID"].unique()
    total_linked = 0
    total_unlinked = 0

    for proj in sorted(projects):
        commits_p = commits[commits["PROJECT_ID"] == proj].copy()
        sonar_p   = sonar[sonar["PROJECT_ID"] == proj].copy()

        if len(sonar_p) == 0:
            print(f"     ⚠️  {proj}: No Sonar data — skipping")
            total_unlinked += len(commits_p)
            continue

        # Sort both by date (required by merge_asof)
        commits_p = commits_p.sort_values("AUTHOR_DATE")
        sonar_p   = sonar_p.sort_values("DATE")

        # Select Sonar columns to carry over
        sonar_cols = ["DATE"] + TD_FEATURES
        sonar_cols = [c for c in sonar_cols if c in sonar_p.columns]

        # TIME-SAFE MERGE: direction='backward' = last snapshot <= commit date
        merged = pd.merge_asof(
            commits_p,
            sonar_p[sonar_cols],
            left_on="AUTHOR_DATE",
            right_on="DATE",
            direction="backward",
        )

        # Track linking success
        linked = merged["SQALE_DEBT_RATIO"].notna().sum()
        total_linked   += linked
        total_unlinked += (len(merged) - linked)

        all_results.append(merged)

    result = pd.concat(all_results, ignore_index=True)

    print(f"\n  Linking results:")
    print(f"     Total commits    : {len(result):,}")
    print(f"     Linked (TDR ≠ NA): {total_linked:,} "
          f"({total_linked / len(result) * 100:.1f}%)")
    print(f"     Unlinked         : {total_unlinked:,} "
          f"({total_unlinked / len(result) * 100:.1f}%)")

    # Keep only commits that were successfully linked to a Sonar snapshot
    result = result.dropna(subset=["SQALE_DEBT_RATIO"]).copy()
    print(f"     Final cohort (linked): {len(result):,} commits")
    print(f"  ✅ Time-safe linking complete")

    return result


# ---------------------------------------------------------------------------
# STEP 9: Assemble final cohort
# ---------------------------------------------------------------------------
def assemble_cohort(
    commits_sonar: pd.DataFrame,
    churn: pd.DataFrame,
    bic_labels: pd.DataFrame,
    refactoring: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join all computed tables into one commit-level cohort:
      commits_sonar + churn + BIC labels + refactoring

    BIC=0 for commits NOT in SZZ (left join → fill NA with 0).
    HAS_REFACTORING=0 for commits without refactoring operations.
    CHURN=0 for commits with no recorded file changes.
    """
    print("\n[STEP 9] Assembling final cohort...")

    cohort = commits_sonar.copy()

    # --- Join Churn ---
    cohort = cohort.merge(
        churn[["PROJECT_ID", "COMMIT_HASH", "LINES_ADDED",
               "LINES_REMOVED", "CHURN"]],
        on=["PROJECT_ID", "COMMIT_HASH"],
        how="left",
    )
    cohort["CHURN"]         = cohort["CHURN"].fillna(0).astype(float)
    cohort["LINES_ADDED"]   = cohort["LINES_ADDED"].fillna(0).astype(float)
    cohort["LINES_REMOVED"] = cohort["LINES_REMOVED"].fillna(0).astype(float)

    # --- Join BIC labels ---
    cohort = cohort.merge(
        bic_labels[["PROJECT_ID", "COMMIT_HASH", "BIC"]],
        on=["PROJECT_ID", "COMMIT_HASH"],
        how="left",
    )
    cohort["BIC"] = cohort["BIC"].fillna(0).astype(int)

    # --- Join Refactoring ---
    cohort = cohort.merge(
        refactoring[["PROJECT_ID", "COMMIT_HASH",
                     "HAS_REFACTORING", "REF_COUNT"]],
        on=["PROJECT_ID", "COMMIT_HASH"],
        how="left",
    )
    cohort["HAS_REFACTORING"] = cohort["HAS_REFACTORING"].fillna(0).astype(int)
    cohort["REF_COUNT"]       = cohort["REF_COUNT"].fillna(0).astype(float)

    # --- Summary stats ---
    print(f"\n  COHORT SUMMARY:")
    print(f"     Total commits  : {len(cohort):,}")
    print(f"     Projects       : {cohort['PROJECT_ID'].nunique()}")
    print(f"     BIC = 1        : {cohort['BIC'].sum():,} "
          f"({cohort['BIC'].mean()*100:.2f}%)")
    print(f"     BIC = 0        : {(cohort['BIC']==0).sum():,}")
    print(f"     Has refactoring: {cohort['HAS_REFACTORING'].sum():,} "
          f"({cohort['HAS_REFACTORING'].mean()*100:.1f}%)")
    print(f"     Date range     : {cohort['AUTHOR_DATE'].min().date()} → "
          f"{cohort['AUTHOR_DATE'].max().date()}")

    print(f"\n  BIC rate per project:")
    bic_by_proj = (
        cohort.groupby("PROJECT_ID")["BIC"]
        .agg(["sum", "mean", "count"])
        .rename(columns={"sum": "BIC_count", "mean": "BIC_rate", "count": "N"})
        .sort_values("BIC_rate", ascending=False)
    )
    bic_by_proj["BIC_rate_%"] = (bic_by_proj["BIC_rate"] * 100).round(1)
    print(bic_by_proj[["N", "BIC_count", "BIC_rate_%"]].to_string())

    return cohort


# ---------------------------------------------------------------------------
# MAIN: Run full preprocessing pipeline
# ---------------------------------------------------------------------------
def run_preprocessing(data_dir: Path = None) -> pd.DataFrame:
    """
    Full preprocessing pipeline.
    Returns the assembled cohort DataFrame and saves all interim files.
    """
    print("\n" + "=" * 65)
    print("PREPROCESSING PIPELINE — TD → BIC Study")
    print("=" * 65)

    # Step 1: Load
    dfs = load_raw_data()

    # Step 2: Inventory
    run_inventory(dfs)

    # Step 3: Clean
    dfs = clean_data(dfs)

    # Step 4: Commit base
    commits = build_commit_base(dfs)

    # Step 5: Churn
    churn = compute_churn(dfs)

    # Step 6: BIC labels
    bic_labels = create_bic_labels(dfs)

    # Step 7: Refactoring
    refactoring = create_refactoring_variable(dfs)

    # Step 8: Time-safe Sonar linking
    commits_sonar = link_sonar_time_safe(commits, dfs)

    # Step 9: Assemble cohort
    cohort = assemble_cohort(commits_sonar, churn, bic_labels, refactoring)

    # Save all interim outputs
    print("\n[SAVING] Saving interim and processed files...")

    churn.to_parquet(INTERIM_FILES["churn"], index=False)
    bic_labels.to_parquet(INTERIM_FILES["bic_labels"], index=False)
    refactoring.to_parquet(INTERIM_FILES["refactoring_bin"], index=False)
    cohort.to_parquet(INTERIM_FILES["sonar_linked"], index=False)

    from config import PROCESSED_FILES
    cohort.to_parquet(PROCESSED_FILES["cohort"], index=False)
    print(f"  ✅ Cohort saved: {PROCESSED_FILES['cohort']}")

    print("\n" + "=" * 65)
    print("✅ PREPROCESSING COMPLETE")
    print(f"   Final cohort: {len(cohort):,} commits ready for feature engineering")
    print("=" * 65)

    return cohort


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cohort = run_preprocessing()
