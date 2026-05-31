# =============================================================================
# src/evaluation.py
# Stage 6: Model Evaluation — Final 6 Models (M0–M5)
#
# Evaluation:
#   - AUC-ROC, AUC-PR, F1, Precision, Recall, Brier per model
#   - ROC curves + PR curves (paper figures)
#   - Confusion Matrix for M5 (XGBoost)
#   - SHAP feature importance for M5
#   - Statistical inference: coefficients from M2 (RQ1) and M3 (RQ2)
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    f1_score, precision_score, recall_score,
    confusion_matrix, roc_curve, precision_recall_curve,
    brier_score_loss, classification_report, ConfusionMatrixDisplay
)
from sklearn.calibration import calibration_curve

from config import (
    PROCESSED_FILES, FIGURES, TABLES,
    TARGET, RANDOM_STATE, FIGURE_DPI
)

plt.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "figure.dpi": FIGURE_DPI,
    
    
})


# ---------------------------------------------------------------------------
# Core evaluation metrics for one model
# ---------------------------------------------------------------------------
def evaluate_model(pipe, X_test, y_test,
                   model_name: str, threshold: float = 0.5) -> dict:
    """
    Compute all evaluation metrics for one model.
    Returns a dict of metric values.
    """
    y_prob = pipe.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "Model"       : model_name,
        "AUC_ROC"     : round(roc_auc_score(y_test, y_prob), 4),
        "AUC_PR"      : round(average_precision_score(y_test, y_prob), 4),
        "F1"          : round(f1_score(y_test, y_pred, zero_division=0), 4),
        "Precision"   : round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall"      : round(recall_score(y_test, y_pred, zero_division=0), 4),
        "Brier"       : round(brier_score_loss(y_test, y_prob), 4),
        "N_test"      : len(y_test),
        "BIC_rate_test": round(y_test.mean(), 4),
    }
    return metrics, y_prob, y_pred


# ---------------------------------------------------------------------------
# Evaluate all models and build comparison table
# ---------------------------------------------------------------------------
def evaluate_all_models(results: dict) -> pd.DataFrame:
    """
    Run evaluation for all 6 models (M0–M5) and create the comparison table.
    Saves Table 3 (model comparison) for the paper.
    Primary sort metric: AUC-PR (appropriate for imbalanced BIC data).
    """
    print("\n[EVAL] Evaluating all 6 models (M0–M5)...")

    # Canonical display names aligned with paper Table 3
    DISPLAY = {
        "M0_dummy"          : "M0: Dummy",
        "M1_lr_baseline"    : "M1: LR Baseline",
        "M2_lr_tdr_rq1"     : "M2: LR + TDR (RQ1)",
        "M3_lr_tdr_ref_rq2" : "M3: LR + TDR + Ref (RQ2)",
        "M4_random_forest"  : "M4: Random Forest",
        "M5_xgboost"        : "M5: XGBoost ★",
    }

    all_metrics = []
    all_probs   = {}

    for model_name, (pipe, X_te, y_te, feats) in results.items():
        if not hasattr(pipe, "predict_proba"):
            continue
        metrics, y_prob, y_pred = evaluate_model(pipe, X_te, y_te, model_name)
        metrics["Display"] = DISPLAY.get(model_name, model_name)
        all_metrics.append(metrics)
        all_probs[model_name] = (y_prob, y_te)
        print(f"  {DISPLAY.get(model_name, model_name):30s}: "
              f"AUC-ROC={metrics['AUC_ROC']:.4f}  "
              f"AUC-PR={metrics['AUC_PR']:.4f}  "
              f"F1={metrics['F1']:.4f}  "
              f"Recall={metrics['Recall']:.4f}  "
              f"Brier={metrics['Brier']:.4f}")

    metrics_df = pd.DataFrame(all_metrics)
    # Sort by model ID order (M0 first, M5 last)
    order = list(DISPLAY.keys())
    metrics_df["sort_key"] = metrics_df["Model"].map(
        {k: i for i, k in enumerate(order)}
    )
    metrics_df = metrics_df.sort_values("sort_key").drop(columns="sort_key")

    out_path = TABLES / "table3_model_comparison.csv"
    metrics_df.to_csv(out_path, index=False)
    print(f"\n  ✅ Table 3 saved: {out_path}")

    # Key comparison: M1 (no TD) vs M5 (best ML)
    if "M1_lr_baseline" in results and "M5_xgboost" in results:
        m1_pr = metrics_df.loc[metrics_df["Model"]=="M1_lr_baseline", "AUC_PR"].values[0]
        m5_pr = metrics_df.loc[metrics_df["Model"]=="M5_xgboost",     "AUC_PR"].values[0]
        delta_pct = (m5_pr - m1_pr) / m1_pr * 100
        print(f"\n  ⭐ M5 (XGBoost) improvement over M1 (no TD): "
              f"ΔAUC-PR = {m5_pr - m1_pr:+.4f} ({delta_pct:+.1f}%)")

    return metrics_df, all_probs


# ---------------------------------------------------------------------------
# FIGURE: ROC curves (all models)
# ---------------------------------------------------------------------------
def fig_roc_curves(results: dict):
    """Multi-model ROC curve — 6 models M0–M5."""
    print("\n[EVAL] Figure: ROC + PR curves...")

    COLORS = {
        "M0_dummy"          : "#AAAAAA",
        "M1_lr_baseline"    : "#AED6F1",
        "M2_lr_tdr_rq1"     : "#2E86C1",
        "M3_lr_tdr_ref_rq2" : "#1A5276",
        "M4_random_forest"  : "#E67E22",
        "M5_xgboost"        : "#C0392B",
    }
    LABELS = {
        "M0_dummy"          : "M0: Dummy",
        "M1_lr_baseline"    : "M1: LR Baseline",
        "M2_lr_tdr_rq1"     : "M2: LR+TDR (RQ1)",
        "M3_lr_tdr_ref_rq2" : "M3: LR+TDR+Ref (RQ2)",
        "M4_random_forest"  : "M4: Random Forest",
        "M5_xgboost"        : "M5: XGBoost ★",
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for model_name, (pipe, X_te, y_te, feats) in results.items():
        if not hasattr(pipe, "predict_proba"):
            continue
        y_prob = pipe.predict_proba(X_te)[:, 1]
        color  = COLORS.get(model_name, "gray")
        label  = LABELS.get(model_name, model_name)
        lw     = 2.2 if model_name in ["M5_xgboost", "M3_lr_tdr_ref_rq2"] else 1.3
        ls     = "--" if model_name == "M0_dummy" else "-"

        # ROC
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        auc = roc_auc_score(y_te, y_prob)
        axes[0].plot(fpr, tpr, color=color, lw=lw, ls=ls,
                     label=f"{label} ({auc:.3f})")

        # PR
        prec, rec, _ = precision_recall_curve(y_te, y_prob)
        ap = average_precision_score(y_te, y_prob)
        axes[1].plot(rec, prec, color=color, lw=lw, ls=ls,
                     label=f"{label} ({ap:.3f})")

    axes[0].plot([0,1],[0,1],"k--",lw=0.8,alpha=0.3,label="Random (0.500)")
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("(a) ROC Curves", fontweight="bold")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[0].grid(alpha=0.25)

    axes[1].axhline(y_te.mean(), ls=":", color="#555", lw=1, alpha=0.7,
                    label=f"Random ({y_te.mean():.3f})")
    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("(b) Precision-Recall Curves\n(Primary metric — imbalanced data)",
                      fontweight="bold")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[1].grid(alpha=0.25)

    fig.suptitle("Figure 3 — Model Performance: ROC and PR Curves (M0–M5)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        out = FIGURES / f"fig3_performance.{ext}"
        plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ ROC + PR curves saved to {FIGURES}/fig3_performance.*")


# ---------------------------------------------------------------------------
# FIGURE: Precision-Recall curves (most important for imbalanced data)
# ---------------------------------------------------------------------------
def fig_pr_curves(results: dict):
    """Multi-model Precision-Recall curve."""
    print("\n[EVAL] Figure: PR curves (key figure for imbalanced data)...")

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.tab10.colors

    for i, (model_name, (pipe, X_te, y_te, feats)) in enumerate(results.items()):
        if not hasattr(pipe, "predict_proba"):
            continue
        y_prob = pipe.predict_proba(X_te)[:, 1]
        prec, rec, _ = precision_recall_curve(y_te, y_prob)
        ap = average_precision_score(y_te, y_prob)

        style = "-" if model_name in ["A_rq1", "B_rq2"] else "--"
        lw    = 2.5 if model_name in ["A_rq1", "B_rq2"] else 1.5
        ax.plot(rec, prec, color=colors[i % 10],
                linestyle=style, linewidth=lw,
                label=f"{model_name} (AP={ap:.3f})")

    # Random baseline line (BIC rate)
    bic_rate = y_te.mean()
    ax.axhline(bic_rate, ls=":", color="black", alpha=0.5,
               label=f"Random ({bic_rate:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models\n"
                 "(AUC-PR preferred for imbalanced BIC data)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = FIGURES / "fig_pr_curves.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  ✅ PR curves saved: {out}")


# ---------------------------------------------------------------------------
# FIGURE: Confusion matrix for Model A
# ---------------------------------------------------------------------------
def fig_confusion_matrix(results: dict):
    """Confusion matrix for M5 (XGBoost) — best model."""
    print("\n[EVAL] Figure: Confusion matrix (M5: XGBoost)...")

    if "M5_xgboost" not in results:
        print("  ⚠️  M5 XGBoost not found")
        return

    pipe, X_te, y_te, feats = results["M5_xgboost"]
    y_prob = pipe.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    cm   = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not BIC", "BIC"]
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix — M5: XGBoost (threshold = 0.50)")

    plt.tight_layout()
    out = FIGURES / "fig_confusion_matrix.pdf"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Confusion matrix saved: {out}")
    print(f"\n  Classification Report (M5 XGBoost):")
    print(classification_report(y_te, y_pred,
                                target_names=["Not BIC", "BIC"], digits=4))


# ---------------------------------------------------------------------------
# FIGURE: Calibration curves
# ---------------------------------------------------------------------------
def fig_calibration(results: dict):
    """Reliability diagram (calibration curves) for main models."""
    print("\n[EVAL] Figure: Calibration curves...")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated", alpha=0.7)

    colors = plt.cm.tab10.colors
    main_models = ["B0_baseline", "B1_no_tdr", "A_rq1", "B_rq2"]

    for i, mname in enumerate(main_models):
        if mname not in results:
            continue
        pipe, X_te, y_te, feats = results[mname]
        if not hasattr(pipe, "predict_proba"):
            continue
        y_prob = pipe.predict_proba(X_te)[:, 1]
        fraction_pos, mean_pred = calibration_curve(
            y_te, y_prob, n_bins=10, strategy="uniform"
        )
        brier = brier_score_loss(y_te, y_prob)
        ax.plot(mean_pred, fraction_pos, marker="o", color=colors[i],
                label=f"{mname} (Brier={brier:.4f})")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives (BIC)")
    ax.set_title("Calibration Curves\n(closer to diagonal = better calibrated)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = FIGURES / "fig_calibration.pdf"
    plt.savefig(out)
    plt.close()
    print(f"  ✅ Calibration curves saved: {out}")


# ---------------------------------------------------------------------------
# ROBUSTNESS CHECK R3: sensitivity to TDR metric choice
# ---------------------------------------------------------------------------
def fig_shap_importance(results: dict):
    """
    SHAP feature importance for M5 (XGBoost) — Figure 4 in paper.
    Shows mean |SHAP value| for the 6 core features.
    Red bars = increases BIC risk; Blue bars = decreases BIC risk.
    """
    print("\n[EVAL] Figure: SHAP feature importance (M5: XGBoost)...")

    if "M5_xgboost" not in results:
        print("  ⚠️  M5 XGBoost not found")
        return

    try:
        import shap
        import matplotlib.patches as mpatches
    except ImportError:
        print("  ⚠️  shap not installed — skipping SHAP figure")
        return

    pipe, X_te, y_te, feats = results["M5_xgboost"]
    model = pipe  # XGBClassifier directly (not a Pipeline)

    # Sample 3000 test rows for speed
    np.random.seed(42)
    samp = np.random.choice(len(X_te), min(3000, len(X_te)), replace=False)
    Xs   = X_te.iloc[samp]

    ex = shap.TreeExplainer(model)
    sv = np.array(ex.shap_values(Xs))

    core_feats = ["SQALE_DEBT_RATIO", "log_churn", "log_ncloc",
                  "DUPLICATED_LINES_DENSITY", "HAS_REFACTORING", "TDR_x_REF"]
    core_labs  = ["TDR (SQALE Debt Ratio)", "log(1 + Churn)", "log(1 + NCLOC)",
                  "Duplicated Lines Density", "Has Refactoring", "TDR × Refactoring"]

    fl_list = list(X_te.columns)
    ci      = [fl_list.index(f) for f in core_feats if f in fl_list]
    ma      = np.abs(sv[:, ci]).mean(axis=0)
    md      = sv[:, ci].mean(axis=0)
    si      = np.argsort(ma)
    clr     = ["#C0392B" if md[i] > 0 else "#2E86C1" for i in si]

    fig, ax = plt.subplots(figsize=(6, 3.2))
    bars = ax.barh([core_labs[i] for i in si], ma[si],
                   color=clr, edgecolor="white", height=0.6)
    for bar, val in zip(bars, ma[si]):
        ax.text(val + 0.004, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8.5)

    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title("Figure 4 — SHAP Feature Importance (M5: XGBoost)\n"
                 "Red = increases BIC risk  ·  Blue = decreases BIC risk",
                 fontweight="bold", fontsize=9)
    leg = [mpatches.Patch(color="#C0392B", label="Increases BIC risk"),
           mpatches.Patch(color="#2E86C1",  label="Decreases BIC risk")]
    ax.legend(handles=leg, fontsize=8, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", alpha=0.25)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        out = FIGURES / f"fig4_shap.{ext}"
        plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  ✅ SHAP figure saved to {FIGURES}/fig4_shap.*")

    # Save SHAP importance table
    shap_df = pd.DataFrame({
        "Feature"      : [core_feats[i] for i in si],
        "Feature_Label": [core_labs[i] for i in si],
        "Mean_SHAP"    : ma[si].round(4),
        "Mean_SHAP_dir": ["positive" if md[i] > 0 else "negative" for i in si],
    })
    out_tbl = TABLES / "shap_importance_M5.csv"
    shap_df.to_csv(out_tbl, index=False)
    print(f"  ✅ SHAP table saved: {out_tbl}")
    print(shap_df.to_string(index=False))


# ---------------------------------------------------------------------------
# MAIN: Full evaluation pipeline
# ---------------------------------------------------------------------------
def run_evaluation(results: dict = None) -> pd.DataFrame:
    """Full evaluation pipeline for M0–M5."""
    print("\n" + "=" * 65)
    print("EVALUATION PIPELINE — TDInsight (M0–M5)")
    print("=" * 65)

    if results is None:
        print("  ⚠️  No results passed — please run modeling.py first")
        return None

    metrics_df, all_probs = evaluate_all_models(results)
    fig_roc_curves(results)         # Figure 3: ROC + PR
    fig_confusion_matrix(results)   # Confusion matrix (M5)
    fig_shap_importance(results)    # Figure 4: SHAP (M5)

    print("\n" + "=" * 65)
    print("✅ EVALUATION COMPLETE")
    print(f"   Figures  → {FIGURES}")
    print(f"   Tables   → {TABLES}")
    print("=" * 65)

    return metrics_df


if __name__ == "__main__":
    print("Run main.py to execute the full pipeline.")
