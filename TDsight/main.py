# =============================================================================
# main.py — Master pipeline runner — TDInsight
# "Does Technical Debt Increase the Probability of Bug-Inducing Commits?"
# Target: Q1 journal (EMSE / IEEE TSE / ACM TOSEM)
#
# Usage:
#   python main.py                    # Run full pipeline (all stages)
#   python main.py --stage preprocess
#   python main.py --stage features
#   python main.py --stage eda
#   python main.py --stage model
#   python main.py --stage evaluate
#   python main.py --stage all
#
# Output:
#   results/models/  → final_M0.pkl … final_M5.pkl
#   results/figures/ → fig1_pipeline, fig2_descriptive, fig3_performance,
#                       fig4_shap, fig5_rq2_interaction
#   results/tables/  → table1_descriptive_stats, table2_bic_per_project,
#                       table3_model_comparison, model_coefficients, shap_importance_M5
# =============================================================================

import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


def run_stage_preprocess():
    from src.preprocessing import run_preprocessing
    return run_preprocessing()


def run_stage_features(cohort=None):
    from src.feature_engineering import run_feature_engineering
    return run_feature_engineering(cohort)


def run_stage_eda(df=None):
    from src.eda import run_eda
    return run_eda(df)


def run_stage_model(df=None):
    from src.modeling import run_modeling
    return run_modeling(df)


def run_stage_evaluate(results=None):
    from src.evaluation import run_evaluation
    return run_evaluation(results)


def main(stage: str = "all"):
    print("\n" + "=" * 65)
    print("TDInsight — FULL EMPIRICAL PIPELINE")
    print("6 Models (M0–M5)  |  31 Apache Projects  |  150,372 Commits")
    print("Target: Q1 Journal (EMSE / IEEE TSE / ACM TOSEM)")
    print("=" * 65)

    t0 = time.time()

    cohort      = None
    modeling_df = None
    results     = None

    # ── Stage 1: Preprocessing ────────────────────────────────────────────────
    if stage in ("preprocess", "all"):
        t = time.time()
        cohort = run_stage_preprocess()
        print(f"  ⏱  Preprocessing     : {time.time()-t:.1f}s")

    # ── Stage 2: Feature Engineering ─────────────────────────────────────────
    if stage in ("features", "all"):
        t = time.time()
        modeling_df = run_stage_features(cohort)
        print(f"  ⏱  Feature engineering: {time.time()-t:.1f}s")

    # ── Stage 3: EDA — Figures 1, 2, 5 + Tables 1, 2 ────────────────────────
    if stage in ("eda", "all"):
        t = time.time()
        run_stage_eda(modeling_df)
        print(f"  ⏱  EDA               : {time.time()-t:.1f}s")

    # ── Stage 4: Modeling — M0 through M5 ────────────────────────────────────
    if stage in ("model", "all"):
        t = time.time()
        results, coef_df = run_stage_model(modeling_df)
        print(f"  ⏱  Modeling          : {time.time()-t:.1f}s")

    # ── Stage 5: Evaluation — Figures 3, 4 + Table 3 + SHAP ─────────────────
    if stage in ("evaluate", "all"):
        t = time.time()
        metrics_df = run_stage_evaluate(results)
        print(f"  ⏱  Evaluation        : {time.time()-t:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"✅ PIPELINE COMPLETE — {elapsed:.1f}s")
    print(f"{'='*65}")
    print("\nOutputs:")
    print("  Models  → results/models/final_M0.pkl … final_M5.pkl")
    print("  Figures → results/figures/fig1_pipeline, fig2_descriptive,")
    print("                            fig3_performance, fig4_shap, fig5_rq2")
    print("  Tables  → results/tables/table1_descriptive_stats,")
    print("                           table2_bic_per_project,")
    print("                           table3_model_comparison,")
    print("                           model_coefficients, shap_importance_M5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TDInsight Empirical Pipeline")
    parser.add_argument(
        "--stage",
        choices=["preprocess", "features", "eda", "model", "evaluate", "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    args = parser.parse_args()
    main(args.stage)
