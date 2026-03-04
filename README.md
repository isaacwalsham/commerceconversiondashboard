# commerceconversiondashboard

A production-style data science project for predicting e-commerce conversion and turning model output into commercial actions.

## Project Snapshot
- Objective: predict whether a session converts (`Revenue`) and optimize targeting decisions.
- Dataset: UCI Online Shoppers Purchasing Intention.
- Primary metrics: PR-AUC, ROC-AUC, Brier score, decile lift, expected value by threshold.
- Delivery: training pipeline, evaluation reports, and an interactive Streamlit dashboard.

## Portfolio Case Study Summary
Built an end-to-end conversion intelligence system in Python using `pandas`, `numpy`, and `scikit-learn`. The pipeline includes robust preprocessing, model selection with stratified cross-validation, calibration diagnostics, and value-based threshold optimization. Results are surfaced in an interactive dashboard with performance diagnostics, decile lift analytics, and a scenario simulator for stakeholder decision support.

## Results (latest run)
- Selected model: `random_forest`
- PR-AUC: `0.724`
- ROC-AUC: `0.926`
- Optimal threshold: `0.09`
- Expected value at optimal threshold: `21,775.5` (relative units)

## CV Bullets
- Developed a production-style conversion prediction pipeline in Python (`pandas`, `numpy`, `scikit-learn`) with stratified CV model selection and calibration diagnostics.
- Built a business-facing analytics layer with threshold optimization, decile lift analysis, and expected-value targeting metrics.
- Delivered an interactive Streamlit dashboard with scenario simulation and downloadable model governance artifacts (executive summary and model card).

## GitHub Profile Copy
- Repo name: `commerceconversiondashboard`
- Short description: `End-to-end e-commerce conversion prediction with Streamlit dashboard, lift analysis, calibration, and threshold economics.`
- Suggested topics: `data-science`, `machine-learning`, `python`, `streamlit`, `pandas`, `scikit-learn`, `classification`, `analytics`, `ecommerce`, `portfolio-project`

## Stack
- Python 3.10+
- pandas, numpy
- scikit-learn
- plotly, streamlit
- pytest, ruff

## Project Structure
```text
.
├── app/
│   ├── app.py
│   ├── lite_app.py
│   └── style.css
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── scripts/
│   ├── download_data.py
│   └── train_model.py
├── src/
│   └── commerceconversiondashboard/
│       ├── data.py
│       ├── features.py
│       ├── inference.py
│       ├── metrics.py
│       ├── modeling.py
│       ├── paths.py
│       └── train.py
├── tests/
├── Makefile
├── pyproject.toml
└── README.md
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
```

## Train Model
```bash
python scripts/train_model.py --data-path data/raw/online_shoppers_intention.csv
```

If internet access is available, you can also auto-download:
```bash
python scripts/train_model.py
```

## Run Dashboard
```bash
streamlit run app/app.py
```

If you hit browser rendering issues, use the minimal fallback:
```bash
streamlit run app/lite_app.py
```

## Generated Artifacts
- `models/conversion_model.joblib`
- `models/model_metadata.json`
- `reports/evaluation_summary.json`
- `reports/cv_results.csv`
- `reports/threshold_value_table.csv`
- `reports/precision_recall_curve.csv`
- `reports/roc_curve.csv`
- `reports/calibration_curve.csv`
- `reports/decile_lift_table.csv`
- `reports/feature_importance.csv`
- `reports/test_predictions.csv`
- `reports/model_card.md`
- `reports/executive_summary.md`

## Dashboard Sections
- Overview
- Performance
- Value & Lift
- Scenario Lab
- Report Center

## Quality Checks
```bash
pytest
ruff check .
```
