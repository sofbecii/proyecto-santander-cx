# proyecto-santander-cx
Behavioral churn propensity model and Customer Experience Score for Santander México
<div align="center">

<img src="https://upload.wikimedia.org/wikipedia/commons/thumb/4/4c/Banco_Santander_Logotipo.svg/320px-Banco_Santander_Logotipo.svg.png" height="48" alt="Santander" />

# Santander CX — Churn Propensity & Customer Experience Score

**A behavioral machine learning pipeline that identifies banking customers at risk of switching — before they decide.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org)
[![License](https://img.shields.io/badge/License-Academic-lightgrey?style=flat-square)](#license)

*TC2004B · Tecnológico de Monterrey · June 2026*

</div>

---

## Overview

Traditional banking churn models rely on static demographics and lagging transaction signals — by the time they fire, the customer has already decided to leave. This project takes a different approach.

We construct a **Customer Experience Score (CES)**: a composite behavioral risk metric anchored to two industry-standard frameworks — the **NPS loyalty model** (Reichheld, 2003) and the **Kano satisfaction model** (Kano et al., 1984) — and train a machine learning classifier to predict that score from 21 behavioral signals that exist *before* any defection event.

The pipeline delivers three actionable outputs:
1. **A risk score** per customer, ranked by churn propensity
2. **Explainable drivers** via SHAP global interpretability
3. **Three behavioral archetypes** via K-Means segmentation, each with a differentiated retention playbook

---

## Methodology

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PROJECT PIPELINE                                 │
│                                                                          │
│  [Survey Data]  →  [CES Target]  →  [Feature Eng.]  →  [Modeling]       │
│    N = 300          NPS + Kano        21 signals         LogReg          │
│    21 variables     threshold 2.75    anti-leakage       K-Means         │
│                     Y ∈ {0, 1}        Spearman filter    K = 3           │
│                     50/50 balance                                         │
│                                              ↓                           │
│                                       [Outputs]                          │
│                                   CES risk scores                        │
│                                   SHAP feature importance                │
│                                   Behavioral archetypes                  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Target Variable (Y)

The binary churn risk label is constructed as a composite score:

```
risk_score = NPS_Trust_Points + Kano_Friction_Index
Y = 1  if  risk_score ≥ 2.75
Y = 0  otherwise
```

| Trust Level | NPS Band | Points Added |
|:-----------:|:--------:|:------------:|
| 1–2 | Detractor | +2.5 |
| 3 | Borderline | +1.0 |
| 4 | Passive | 0.0 |
| 5 | Promoter | −1.0 |

Kano friction weights range from **2.0** (Must-Be ruptures) down to **0.0** (loyalty / indifferent). No single reason alone crosses the 2.75 threshold — churn risk always requires more than one source of friction.

### Anti-Leakage Architecture

All variables that define Y (`trust_level`, `switch_reasons`, `risk_score`, `IF`) are strictly excluded from the feature matrix X. The model learns to predict risk from **indirect behavioral signals only** — the kind observable before any defection decision is made.

---

## Key Results

| Metric | Value | Context |
|:-------|:-----:|:--------|
| ROC-AUC (holdout) | **0.661** | Correctly ranks higher-risk client 2 out of 3 times |
| F1-Score (threshold 0.44) | **0.613** | Optimal operating point for retention cost structure |
| Recall (threshold 0.44) | **0.633** | Captures 63% of true behavioral risk signals |
| Cross-validated AUC (5-fold) | **0.70 ± 0.09** | Stable generalization on training set |
| Bootstrap 95% CI | [0.59, 0.73] | True population AUC range |

### SHAP — Top Behavioral Drivers

```
Fintech Awareness      ████████████████████  0.124   ← #1 driver
Security Consciousness ████████████████████  0.124
App Frustrations       ████████████████      0.091
Benefits Sought        ██████████████        0.075
Operational Friction   ██████████            0.055
Transaction Difficulty ██████████            0.051

Age & Income           ██                    < 0.05  ← Demographics barely register
```

**Churn risk is behavioral, not demographic.** Age and income have near-zero Spearman correlation with Y (−0.04 and 0.02 respectively, p > 0.05).

### K-Means Behavioral Archetypes (K=3)

| Archetype | n | Churn Risk | Profile |
|:----------|:-:|:----------:|:--------|
|  **Hybrid Demanding** | 73 | **74%** | Highest fintech awareness + app frustrations. Uses both traditional banking and fintech simultaneously. Actively compares. Concentrates **36% of projected churn** in 24% of the base. |
|  **Young Mainstream** | 117 | 44% | Loyal but fintech-curious. Has touched one alternative. Window to build digital habit is open — and closing. |
|  **Established Adult** | 110 | 40% | High income, low friction, multi-app. Lowest risk but highest cost to replace if lost. |

---

## Repository Structure

```
proyecto-santander/
│
├── 📓 pipeline_churn_v2.ipynb      ← Full ML pipeline (annotated)
│
├── 📁 data/
│   └── survey_data.csv             ← Master poll (N=300, 21 variables)
│
├── 📁 presentation/
│   ├── executive_dashboard.html    ← Interactive executive presentation
│   └── appendix_slides.html        ← Technical Q&A appendix slides
│
├── 📄 README.md
├── 📄 requirements.txt
└── 📄 .gitignore
```

---

## How to Run

### Option 1 — Google Colab (recommended)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leoroldanp/Proyecto-Santander/blob/main/pipeline_churn_v2.ipynb)

1. Open the notebook in Colab
2. Upload `data/survey_data.csv` to the Colab session storage
3. Run all cells top to bottom (`Runtime → Run all`)

### Option 2 — Local

```bash
# Clone the repository
git clone https://github.com/leoroldanp/Proyecto-Santander.git
cd Proyecto-Santander

# Install dependencies
pip install -r requirements.txt

# Launch notebook
jupyter notebook pipeline_churn_v2.ipynb
```

> **Note:** The notebook expects the data at `data/survey_data.csv` relative to the project root.

---

## Dependencies

```
pandas >= 2.0
numpy >= 1.24
scikit-learn >= 1.3
xgboost >= 1.7
shap >= 0.42
matplotlib >= 3.7
seaborn >= 0.12
scipy >= 1.10
jupyter >= 1.0
```

Full pinned versions in `requirements.txt`.

---

## Team

| Name | Institution |
|:-----|:-----------|
| Fernando Aguirre Higuera | Tecnológico de Monterrey |
| Axel Rodrigo Vélez Robles | Tecnológico de Monterrey |
| Sofía Becerra Pedraza | Tecnológico de Monterrey |
| Leonardo Roldán Pérez | Tecnológico de Monterrey |
| Diana Gabriela Rivera Reyes | Tecnológico de Monterrey |

*Course: TC2004B — Análisis de Ciencia de Datos · Group 641 · June 2026*

---

## References

- Reichheld, F. F. (2003). The one number you need to grow. *Harvard Business Review, 81*(12), 46–54.
- Kano, N., Seraku, N., Takahashi, F., & Tsuji, S. (1984). Attractive quality and must-be quality. *Journal of the Japanese Society for Quality Control, 14*(2), 39–48.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*, 4765–4774.
- Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.
- MacQueen, J. (1967). Some methods for classification and analysis of multivariate observations. *Proceedings of the Fifth Berkeley Symposium on Mathematical Statistics and Probability, 1*(14), 281–297.
- CNBV & INEGI. (2024). *Encuesta Nacional de Inclusión Financiera (ENIF) 2024*. Gobierno de México.
- Deloitte México. (2024). *Estudio del Mercado Crediticio e Inclusión Financiera en México*. Deloitte Financial Services.

---

## License

This project was developed for academic purposes as part of the TC2004B curriculum at Tecnológico de Monterrey. The survey data and methodology are original work by the project team. No proprietary Santander data was used.
