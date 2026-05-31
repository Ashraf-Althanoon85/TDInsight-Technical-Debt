# =============================================================================
# src/eda.py
# Stage 4: Exploratory Data Analysis — TDInsight
#
# Produces the 5 paper figures and 2 tables:
#   Fig 1: Study pipeline diagram  (in figures/)
#   Fig 2: Descriptive — BIC per project + BIC by TDR quartile
#   Fig 3: Model performance — ROC + PR curves   (produced by evaluation.py)
#   Fig 4: SHAP importance                       (produced by evaluation.py)
#   Fig 5: RQ2 — BIC rate by TDR × Refactoring
#
#   Tab 1: Descriptive statistics
#   Tab 2: BIC rate per project
# =============================================================================

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

from config import (
    PROCESSED_FILES, FIGURES, TABLES,
    TD_FEATURES, TARGET, FIGURE_DPI, FIGURE_SIZE,
    MODEL_COLORS, MODEL_DISPLAY_NAMES
)

plt.rcParams.update({
    "font.family"    : "DejaVu Serif",
    "font.size"      : 8.5,
    "axes.titlesize" : 9.5,
    "axes.labelsize" : 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.spines.top"   : "False",
    "axes.spines.right" : "False",
    "axes.grid"      : True,
    "grid.alpha"     : 0.25,
    "grid.linestyle" : "--",
    "figure.dpi"     : FIGURE_DPI,
})


# ---------------------------------------------------------------------------
# TABLE 1: Descriptive statistics
# ---------------------------------------------------------------------------
def table_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Table 1 — Mean/Median/SD/Min/Max for all key variables, split by BIC."""
    print("\n[EDA] Table 1: Descriptive statistics...")

    cols = [
        "SQALE_DEBT_RATIO", "DUPLICATED_LINES_DENSITY",
        "NCLOC", "CHURN", "HAS_REFACTORING",
        "log_churn", "log_ncloc", "TDR_x_REF",
    ]
    cols = [c for c in cols if c in df.columns]

    rows = []
    for col in cols:
        for bic_val in [0, 1, "all"]:
            sub = df if bic_val == "all" else df[df["BIC"] == bic_val]
            s   = sub[col].dropna()
            rows.append({
                "Variable" : col,
                "Group"    : "All" if bic_val == "all" else f"BIC={bic_val}",
                "N"        : len(s),
                "Mean"     : round(s.mean(), 4),
                "Median"   : round(s.median(), 4),
                "SD"       : round(s.std(), 4),
                "Min"      : round(s.min(), 4),
                "Max"      : round(s.max(), 4),
                "Missing%" : round(sub[col].isnull().mean() * 100, 2),
            })

    stats_df = pd.DataFrame(rows)
    out = TABLES / "table1_descriptive_stats.csv"
    stats_df.to_csv(out, index=False)
    print(f"  ✅ Table 1 saved: {out}")
    print(stats_df[stats_df["Group"] == "All"][
        ["Variable", "Mean", "Median", "SD", "Missing%"]
    ].to_string(index=False))
    return stats_df


# ---------------------------------------------------------------------------
# TABLE 2: BIC rate per project
# ---------------------------------------------------------------------------
def table_bic_per_project(df: pd.DataFrame) -> pd.DataFrame:
    """Table 2 — Per-project BIC rates, TDR medians, commit counts."""
    print("\n[EDA] Table 2: BIC rate per project...")

    tbl = (
        df.groupby("PROJECT_ID")
        .agg(
            N_Commits     = ("COMMIT_HASH", "count"),
            N_BIC         = ("BIC", "sum"),
            BIC_Rate_pct  = ("BIC", lambda x: round(x.mean() * 100, 2)),
            TDR_Median    = ("SQALE_DEBT_RATIO", "median"),
            Churn_Median  = ("CHURN", "median"),
            N_Refactoring = ("HAS_REFACTORING", "sum"),
        )
        .sort_values("BIC_Rate_pct", ascending=False)
        .reset_index()
    )
    tbl["PROJECT_ID"] = tbl["PROJECT_ID"].str.replace("org.apache:", "", regex=False)

    out = TABLES / "table2_bic_per_project.csv"
    tbl.to_csv(out, index=False)
    print(f"  ✅ Table 2 saved: {out}")
    print(tbl.to_string(index=False))
    return tbl


# ---------------------------------------------------------------------------
# FIGURE 1: Study Pipeline
# ---------------------------------------------------------------------------
def fig_pipeline():
    """Figure 1 — TDInsight study design and data processing pipeline."""
    print("\n[EDA] Figure 1: Study pipeline...")

    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    steps = [
        (0.02, "#D6EAF8", "#1A5276", "Raw Data\n(10 CSV)"),
        (0.21, "#D5F5E3", "#1E8449", "Preprocessing\n& Cleaning"),
        (0.40, "#FCF3CF", "#7D6608", "Feature\nEngineering"),
        (0.59, "#FADBD8", "#922B21", "Model\nTraining"),
        (0.78, "#E8DAEF", "#6C3483", "Evaluation\n& SHAP"),
    ]
    subs = [
        "GIT·SZZ·SONAR\nREF·JIRA",
        "Time-safe\nmerge_asof\nAnti-leakage",
        "log1p(Churn)\nVIF analysis\nTDR×Ref",
        "Chrono 80/20\n6 models\nProject FE",
        "AUC-PR\nROC·F1\nSHAP",
    ]

    for (x, fc, tc, txt), sub in zip(steps, subs):
        ax.add_patch(plt.Rectangle(
            (x, 0.38), 0.17, 0.40, facecolor=fc,
            edgecolor="#555", linewidth=0.9, transform=ax.transAxes, clip_on=False
        ))
        ax.text(x + 0.085, 0.58, txt, ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=tc, transform=ax.transAxes)
        ax.add_patch(plt.Rectangle(
            (x, 0.05), 0.17, 0.28, facecolor=fc,
            edgecolor="#BBB", linewidth=0.5, alpha=0.6,
            transform=ax.transAxes, clip_on=False
        ))
        ax.text(x + 0.085, 0.19, sub, ha="center", va="center",
                fontsize=6.2, color="#333", transform=ax.transAxes)

    for xa in [0.19, 0.38, 0.57, 0.76]:
        ax.annotate("", xy=(xa + 0.02, 0.58), xytext=(xa, 0.58),
                    xycoords="axes fraction", textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", lw=1.3, color="#333"))

    ax.set_title(
        "Figure 1 — TDInsight Study Design and Data Processing Pipeline\n"
        "31 Apache Projects  ·  150,372 Commits  ·  2000–2021",
        fontsize=8.5, fontweight="bold", pad=5
    )
    fig.tight_layout(pad=0.3)

    for ext in ["png", "pdf"]:
        out = FIGURES / f"fig1_pipeline.{ext}"
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Figure 1 saved → {FIGURES}/fig1_pipeline.*")


# ---------------------------------------------------------------------------
# FIGURE 2: Descriptive Statistics (2 panels — no histogram)
# ---------------------------------------------------------------------------
def fig_descriptive(df: pd.DataFrame):
    """
    Figure 2 — (a) BIC rate per project  (b) BIC rate by TDR quartile.
    Histogram removed as it adds no analytical value for the paper.
    """
    print("\n[EDA] Figure 2: Descriptive statistics...")

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))

    # ── (a) BIC rate per project ──────────────────────────────────────────────
    ps = (df.groupby("PROJECT_ID")["BIC"]
          .agg(["mean", "count"]).reset_index())
    ps["pct"] = ps["mean"] * 100
    ps["sh"]  = ps["PROJECT_ID"].str.replace("org.apache:", "", regex=False)
    ps = ps.sort_values("pct")

    clr = ["#C0392B" if v > 15 else "#2E86C1" if v > 8 else "#AED6F1"
           for v in ps["pct"]]
    axes[0].barh(ps["sh"], ps["pct"], color=clr, edgecolor="white", height=0.75)
    axes[0].axvline(ps["pct"].mean(), ls="--", color="#8E44AD", lw=1.2,
                    label=f"Mean = {ps['pct'].mean():.1f}%")
    axes[0].set_xlabel("BIC Rate (%)")
    axes[0].set_title("(a) BIC Rate per Project", fontsize=9, fontweight="bold")
    axes[0].tick_params(axis="y", labelsize=5.5)
    axes[0].legend(fontsize=7)

    # ── (b) BIC rate by TDR quartile ─────────────────────────────────────────
    dc = df.copy()
    dc["Q"] = pd.qcut(dc["SQALE_DEBT_RATIO"], 4,
                      labels=["Q1\n(Low TD)", "Q2", "Q3", "Q4\n(High TD)"])
    bq = dc.groupby("Q")["BIC"].agg(["mean", "count"]).reset_index()
    bq["pct"] = bq["mean"] * 100

    bars = axes[1].bar(
        bq["Q"].astype(str), bq["pct"],
        color=["#AED6F1", "#6BAED6", "#2171B5", "#C0392B"],
        edgecolor="white", width=0.62
    )
    for bar, (_, row) in zip(bars, bq.iterrows()):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.18,
            f"{row['pct']:.1f}%\n(n={int(row['count']):,})",
            ha="center", fontsize=7
        )
    axes[1].set_xlabel("TDR Quartile")
    axes[1].set_ylabel("BIC Rate (%)")
    axes[1].set_title("(b) BIC Rate by TDR Quartile\n(RQ1 Preliminary Evidence)",
                      fontsize=9, fontweight="bold")

    fig.suptitle("Figure 2 — Dataset Descriptive Statistics",
                 fontsize=9, fontweight="bold", y=1.02)
    fig.tight_layout(pad=0.8)

    for ext in ["png", "pdf"]:
        out = FIGURES / f"fig2_descriptive.{ext}"
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Figure 2 saved → {FIGURES}/fig2_descriptive.*")


# ---------------------------------------------------------------------------
# FIGURE 5: RQ2 — TDR × Refactoring interaction
# ---------------------------------------------------------------------------
def fig_rq2_interaction(df: pd.DataFrame):
    """
    Figure 5 — BIC rate by TDR quartile × refactoring presence.
    Provides visual confirmation of the RQ2 positive interaction term.
    """
    print("\n[EDA] Figure 5: RQ2 TDR × Refactoring interaction...")

    dc = df.copy()
    dc["Q"] = pd.qcut(dc["SQALE_DEBT_RATIO"], 4,
                      labels=["Q1 (Low TD)", "Q2", "Q3", "Q4 (High TD)"])
    g = (dc.groupby(["Q", "HAS_REFACTORING"])["BIC"]
         .mean().reset_index())
    g["pct"] = g["BIC"] * 100

    qs = ["Q1 (Low TD)", "Q2", "Q3", "Q4 (High TD)"]
    xv = np.arange(4)
    wi = 0.38

    def get_val(q, ref):
        sub = g[(g["Q"] == q) & (g["HAS_REFACTORING"] == ref)]
        return sub["pct"].values[0] if len(sub) > 0 else 0.0

    no_r  = [get_val(q, 0) for q in qs]
    has_r = [get_val(q, 1) for q in qs]

    fig, ax = plt.subplots(figsize=(5.5, 3.3))
    b1 = ax.bar(xv - wi / 2, no_r,  wi, label="No Refactoring",
                color="#2E86C1", alpha=0.88, edgecolor="white")
    b2 = ax.bar(xv + wi / 2, has_r, wi, label="Has Refactoring",
                color="#C0392B", alpha=0.88, edgecolor="white")

    for bar, val in zip(list(b1) + list(b2), no_r + has_r):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.12, f"{val:.1f}%",
                ha="center", fontsize=7.5)

    ax.set_xticks(xv)
    ax.set_xticklabels(["Q1\n(Low TD)", "Q2", "Q3", "Q4\n(High TD)"])
    ax.set_xlabel("TDR Quartile")
    ax.set_ylabel("BIC Rate (%)")
    ax.set_title(
        "Figure 5 — TDR × Refactoring Interaction (RQ2 Evidence)\n"
        "Refactoring amplifies fault-induction risk under high-TD conditions",
        fontsize=8.5, fontweight="bold"
    )
    ax.legend(fontsize=8)
    fig.tight_layout(pad=0.5)

    for ext in ["png", "pdf"]:
        out = FIGURES / f"fig5_rq2_interaction.{ext}"
        plt.savefig(out, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  ✅ Figure 5 saved → {FIGURES}/fig5_rq2_interaction.*")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame = None):
    """Run all EDA steps — tables + figures 1, 2, 5."""
    print("\n" + "=" * 65)
    print("EDA PIPELINE — TDInsight")
    print("=" * 65)

    if df is None:
        df = pd.read_parquet(PROCESSED_FILES["modeling_ready"])
        print(f"  Loaded: {len(df):,} rows | BIC rate: {df['BIC'].mean()*100:.2f}%")

    # Tables
    table_descriptive_stats(df)
    table_bic_per_project(df)

    # Figures produced by EDA
    fig_pipeline()
    fig_descriptive(df)
    fig_rq2_interaction(df)

    # NOTE: Fig 3 (ROC+PR) and Fig 4 (SHAP) are produced by evaluation.py

    print("\n" + "=" * 65)
    print("✅ EDA COMPLETE")
    print(f"   Figures → {FIGURES}")
    print(f"   Tables  → {TABLES}")
    print("   Note: Fig 3 + Fig 4 generated by evaluation.py")
    print("=" * 65)


if __name__ == "__main__":
    run_eda()
