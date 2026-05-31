# TDInsight
## A Reproducible Framework for Mining Technical Debt and Predicting Bug-Inducing Commits

**Paper:** "TDInsight: A Reproducible Framework for Mining Technical Debt and Predicting Bug-Inducing Commits in Software Repositories"  
**Target:** Empirical Software Engineering (Q1)  
**Dataset:** 31 Apache Projects · 150,372 Commits · 2000–2021

---

## Project Structure

```
TDInsight/
├── config.py                    # Central configuration (paths, hyperparameters, feature sets)
├── main.py                      # Master pipeline runner
├── src/
│   ├── preprocessing.py         # Stage 1–2: Data loading, cleaning, anti-leakage Sonar linkage
│   ├── feature_engineering.py   # Stage 3: Feature construction (log transforms, VIF, TDR×Ref)
│   ├── eda.py                   # Stage 4: EDA — Fig 1, Fig 2, Fig 5 + Tables 1–2
│   ├── modeling.py              # Stage 5: Train M0–M5
│   └── evaluation.py            # Stage 6: Evaluate M0–M5, Fig 3, Fig 4, Table 3
│
├── data/
│   ├── raw/                     # ← Place raw CSVs here (not included)
│   ├── interim/                 # Intermediate parquet files
│   └── processed/               # Final cohort (cohort_final.parquet, modeling_ready.parquet)
│
└── results/
    ├── figures/                 # 5 paper figures (PNG + PDF, 300 DPI)
    ├── tables/                  # 7 CSV tables
    └── models/                  # 6 trained models (M0–M5)
```

---

## Quick Start

### Install dependencies
```bash
pip install pandas numpy scikit-learn xgboost shap imbalanced-learn matplotlib pyarrow
```

### Run full pipeline (requires raw CSVs in data/raw/)
```bash
python main.py
```

### Run individual stages
```bash
python main.py --stage preprocess    # Load & clean data
python main.py --stage features      # Feature engineering
python main.py --stage eda           # EDA figures & tables
python main.py --stage model         # Train M0–M5
python main.py --stage evaluate      # Evaluate + SHAP
```

### Use pre-trained models directly
```python
import pickle, pandas as pd, numpy as np

# Load test data and M5
X_test = pd.read_parquet("results/models/final_X_test.parquet")
y_test = np.load("results/models/final_y_test.npy")

with open("results/models/final_M5_xgboost.pkl", "rb") as f:
    m5 = pickle.load(f)

probs = m5.predict_proba(X_test)[:, 1]
```

---

## Models (M0–M5)

| ID | Name | Features | Purpose |
|---|---|---|---|
| M0 | Dummy (Stratified) | — | Lower bound |
| M1 | Logistic Regression | log_churn + log_ncloc | H₀: no TD baseline |
| M2 | Logistic Regression | M1 + TDR + Dup | RQ1: TDR effect |
| M3 | Logistic Regression | M2 + HAS_REF + TDR×Ref | RQ2: moderation |
| M4 | Random Forest (n=300) | All 6 + Project FE | ML comparison |
| M5 | XGBoost (n=300) ★ | All 6 + Project FE | Best ML + SHAP |

---

## Key Results

| Model | AUC-ROC | AUC-PR | F1 | Recall |
|---|---|---|---|---|
| M0: Dummy | 0.503 | 0.053 | 0.076 | 0.124 |
| M1: LR Baseline | 0.826 | 0.248 | 0.252 | 0.723 |
| M2: LR + TDR (RQ1) | 0.827 | 0.249 | 0.255 | 0.721 |
| M3: LR + TDR + Ref (RQ2) | 0.830 | 0.259 | 0.258 | 0.716 |
| M4: Random Forest | 0.841 | 0.265 | 0.241 | 0.738 |
| **M5: XGBoost ★** | **0.852** | **0.274** | 0.251 | **0.767** |

**RQ1:** TDR β = 0.019, OR = 1.020, p < 0.01  
**RQ2:** TDR × Refactoring β = 0.031, p < 0.05  
**SHAP:** log(Churn) dominant (1.654), TDR 4th (0.112)

---

## Paper Figures

| File | Description |
|---|---|
| `fig1_pipeline.png/pdf` | Study design & data processing pipeline |
| `fig2_descriptive.png/pdf` | BIC per project + BIC by TDR quartile |
| `fig3_performance.png/pdf` | ROC + PR curves (M0–M5) |
| `fig4_shap.png/pdf` | SHAP feature importance (M5: XGBoost) |
| `fig5_rq2_interaction.png/pdf` | TDR × Refactoring interaction (RQ2) |

---

## Methodological Highlights

- **Anti-leakage linkage:** `pd.merge_asof(direction="backward")` prevents look-ahead bias
- **VIF analysis:** TDR (VIF=7.6) and Dup.Density (VIF=1.3) safe; others excluded
- **Class imbalance:** BIC rate = 10.68%; AUC-PR is primary metric (random baseline = 0.107)
- **Project FE:** 30 one-hot project dummies in all models
- **Chronological split:** 80/20 per-project temporal split

---

## Citation

If you use TDInsight, please cite:

```
[Author(s)]. (2025). TDInsight: A Reproducible Framework for Mining Technical Debt
and Predicting Bug-Inducing Commits in Software Repositories.
Empirical Software Engineering.
```
