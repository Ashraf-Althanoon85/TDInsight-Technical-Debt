# =============================================================================
# config.py
# Central configuration — TDInsight
# "Does Technical Debt Increase the Probability of Bug-Inducing Commits?"
# Target: Q1 journal (EMSE / IEEE TSE / ACM TOSEM)
# =============================================================================

from pathlib import Path

# ---------------------------------------------------------------------------
# 1. ROOT PATHS
# ---------------------------------------------------------------------------
ROOT_DIR   = Path(__file__).parent
DATA_RAW   = ROOT_DIR / "data" / "raw"
DATA_INTER = ROOT_DIR / "data" / "interim"
DATA_PROC  = ROOT_DIR / "data" / "processed"
RESULTS    = ROOT_DIR / "results"
FIGURES    = RESULTS / "figures"
TABLES     = RESULTS / "tables"
MODELS     = RESULTS / "models"
SRC        = ROOT_DIR / "src"

for d in [DATA_RAW, DATA_INTER, DATA_PROC, FIGURES, TABLES, MODELS]:
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. RAW DATA FILES
# ---------------------------------------------------------------------------
RAW_FILES = {
    "commits"        : DATA_RAW / "GIT_COMMITS.csv",
    "changes"        : DATA_RAW / "GIT_COMMITS_CHANGES.csv",
    "jira"           : DATA_RAW / "JIRA_ISSUES.csv",
    "projects"       : DATA_RAW / "PROJECTS.csv",
    "refactoring"    : DATA_RAW / "REFACTORING_MINER.csv",
    "sonar_analysis" : DATA_RAW / "SONAR_ANALYSIS.csv",
    "sonar_issues"   : DATA_RAW / "SONAR_ISSUES.csv",
    "sonar_measures" : DATA_RAW / "SONAR_MEASURES.csv",
    "sonar_rules"    : DATA_RAW / "SONAR_RULES.csv",
    "szz"            : DATA_RAW / "SZZ_FAULT_INDUCING_COMMITS.csv",
}

# ---------------------------------------------------------------------------
# 3. PROCESSED OUTPUT FILES
# ---------------------------------------------------------------------------
INTERIM_FILES = {
    "commits_clean"  : DATA_INTER / "commits_clean.parquet",
    "churn"          : DATA_INTER / "churn_per_commit.parquet",
    "bic_labels"     : DATA_INTER / "bic_labels.parquet",
    "refactoring_bin": DATA_INTER / "refactoring_binary.parquet",
    "sonar_linked"   : DATA_INTER / "sonar_linked.parquet",
}

PROCESSED_FILES = {
    "cohort"        : DATA_PROC / "cohort_final.parquet",
    "modeling_ready": DATA_PROC / "modeling_ready.parquet",
}

# ---------------------------------------------------------------------------
# 4. TD VARIABLES
# ---------------------------------------------------------------------------
TD_PRIMARY  = "SQALE_DEBT_RATIO"    # Primary predictor (RQ1)

TD_FEATURES = [
    "SQALE_DEBT_RATIO",          # Primary: size-normalized debt ratio (%)
    "SQALE_INDEX",               # Raw remediation cost (minutes)
    "CODE_SMELLS",               # Code smells count
    "COMPLEXITY",                # Cyclomatic complexity
    "COGNITIVE_COMPLEXITY",      # Cognitive complexity
    "DUPLICATED_LINES_DENSITY",  # % duplicated lines  [VIF=1.3 — safe]
    "NCLOC",                     # Non-commented LOC (size proxy)
]

# VIF results (justifies feature selection):
#   SQALE_INDEX/CODE_SMELLS/COMPLEXITY/NCLOC  VIF > 70 → EXCLUDED from models
#   SQALE_DEBT_RATIO                          VIF = 7.6 → PRIMARY predictor
#   DUPLICATED_LINES_DENSITY                  VIF = 1.3 → CONTROL variable
COLLINEAR_EXCLUDE = [
    "SQALE_INDEX", "CODE_SMELLS", "COMPLEXITY", "COGNITIVE_COMPLEXITY",
]

# ---------------------------------------------------------------------------
# 5. FEATURE SETS — per model (M0–M5)
#    Project FE dummies are added automatically in modeling.py
# ---------------------------------------------------------------------------
FEATURES_M0 = []
FEATURES_M1 = ["log_churn", "log_ncloc"]
FEATURES_M2 = ["SQALE_DEBT_RATIO", "log_churn", "log_ncloc",
                "DUPLICATED_LINES_DENSITY"]
FEATURES_M3 = ["SQALE_DEBT_RATIO", "log_churn", "log_ncloc",
                "DUPLICATED_LINES_DENSITY", "HAS_REFACTORING", "TDR_x_REF"]
FEATURES_M4 = FEATURES_M3
FEATURES_M5 = FEATURES_M3

# ---------------------------------------------------------------------------
# 6. MODELING HYPERPARAMETERS
# ---------------------------------------------------------------------------
TARGET          = "BIC"
TEST_SIZE_RATIO = 0.20
RANDOM_STATE    = 42

# Logistic Regression (M1, M2, M3)
CLASS_WEIGHT    = "balanced"
LR_MAX_ITER     = 1000
LR_SOLVER       = "lbfgs"

# Random Forest (M4)
RF_N_ESTIMATORS     = 300
RF_MAX_DEPTH        = 15
RF_MIN_SAMPLES_LEAF = 50
RF_CLASS_WEIGHT     = "balanced"

# XGBoost (M5) — scale_pos_weight computed at runtime (neg/pos ≈ 7.3)
XGB_N_ESTIMATORS  = 300
XGB_MAX_DEPTH     = 6
XGB_LEARNING_RATE = 0.05
XGB_SUBSAMPLE     = 0.8
XGB_COLSAMPLE     = 0.8
XGB_EVAL_METRIC   = "aucpr"

# ---------------------------------------------------------------------------
# 7. EVALUATION METRICS
# ---------------------------------------------------------------------------
PRIMARY_METRIC = "AUC_PR"          # Main metric (imbalanced data)
EVAL_METRICS   = ["AUC_ROC", "AUC_PR", "F1", "Precision", "Recall", "Brier"]
GLOBAL_BIC_RATE = 0.1068           # Random baseline for AUC-PR

# ---------------------------------------------------------------------------
# 8. STUDY CONSTANTS
# ---------------------------------------------------------------------------
MIN_COMMITS_PER_PROJECT = 50
MIN_SONAR_SNAPSHOTS     = 5
EXPECTED_PROJECTS       = 31
SZZ_PROJECTS            = 28

# ---------------------------------------------------------------------------
# 9. FIGURE SETTINGS + CONSISTENT COLOR SCHEME (M0–M5)
# ---------------------------------------------------------------------------
FIGURE_DPI    = 300
FIGURE_FORMAT = "pdf"
FIGURE_SIZE   = (7.0, 3.2)      # Single-column width

MODEL_COLORS = {
    "M0_dummy"          : "#AAAAAA",
    "M1_lr_baseline"    : "#AED6F1",
    "M2_lr_tdr_rq1"     : "#2E86C1",
    "M3_lr_tdr_ref_rq2" : "#1A5276",
    "M4_random_forest"  : "#E67E22",
    "M5_xgboost"        : "#C0392B",
}

MODEL_DISPLAY_NAMES = {
    "M0_dummy"          : "M0: Dummy",
    "M1_lr_baseline"    : "M1: LR Baseline",
    "M2_lr_tdr_rq1"     : "M2: LR + TDR (RQ1)",
    "M3_lr_tdr_ref_rq2" : "M3: LR + TDR + Ref (RQ2)",
    "M4_random_forest"  : "M4: Random Forest",
    "M5_xgboost"        : "M5: XGBoost ★",
}

print("✅ config.py loaded")
print(f"   ROOT={ROOT_DIR} | Models: M0–M5 | Primary metric: {PRIMARY_METRIC}")
