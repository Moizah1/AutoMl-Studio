# 🤖 Data God — AutoML Studio

A Streamlit app that takes you from a raw CSV to a trained, downloadable machine learning model — no code required.

Upload a dataset → clean it → explore it → train & compare models → predict on new data.

Built entirely on `streamlit`, `pandas`, `numpy`, `scikit-learn`, and `plotly` — **fully compatible with Python 3.13** (deliberately avoids `pycaret` and `ydata-profiling`, which don't yet support 3.13).

---

## Features

| Page | What it does |
|---|---|
| **Home** | Overview of the workflow |
| **Upload** | Upload a CSV, preview it, see shape/dtypes/missing-value summary |
| **Data Cleaning** | Drop columns, handle missing values (drop/mean/median/mode/constant), remove duplicates |
| **Profile Report** | Summary statistics, missing-value chart, histograms, correlation heatmap, categorical breakdowns |
| **Model Training** | Pick a target + task type (Regression/Classification), auto-trains and compares several scikit-learn models, shows a leaderboard, picks the best one, plots feature importance, and lets you download the trained pipeline |
| **Predict** | Upload new data and get predictions from your trained model, downloadable as CSV |

---

## Installation

Requires **Python 3.10 – 3.13**.

```bash
# 1. (Recommended) create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt
```

Dependencies are lightweight — this should install in well under a minute, with no compiled build steps.

## Running the app

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).


<img width="1653" height="665" alt="image" src="https://github.com/user-attachments/assets/a9d414e0-e004-455e-ac4b-b574f335ea10" />

## Project structure

```
.
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

At runtime the app also creates:
- `sourcedata.csv` — your currently loaded/cleaned dataset (persisted so it survives a page refresh)
- `best_model.pkl` — the trained model pipeline (preprocessing + model bundled together via `joblib`)

Both are safe to delete; the app regenerates them as needed.

---

## How model training works

For each task type, a small set of proven scikit-learn models is trained and compared on a held-out test split:

- **Regression:** Linear Regression, Ridge Regression, Random Forest Regressor, Gradient Boosting Regressor — scored on R², MAE, RMSE
- **Classification:** Logistic Regression, Random Forest Classifier, Gradient Boosting Classifier — scored on Accuracy, F1, Precision, Recall (macro-averaged)

Numeric features are median-imputed (and optionally standardized); categorical features are most-frequent-imputed and one-hot encoded. The whole preprocessing + model combo is saved as a single scikit-learn `Pipeline`, so the Predict page just needs the same raw columns you trained on — no manual encoding required.

---

## Known limitations

- Designed for tabular CSV data; no image/text-specific pipelines.
- Model selection is based on a fixed shortlist of well-rounded scikit-learn models rather than exhaustive hyperparameter search — good for a fast baseline, not a substitute for deeper tuning on high-stakes projects.
- Local file-based persistence (`sourcedata.csv`, `best_model.pkl`) is intended for single-user local use, not multi-user concurrent deployments.

---
