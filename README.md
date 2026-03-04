# commerceconversiondashboard

A side project where I built a model to predict whether an e-commerce session ends in a purchase.

## What This Project Does
- Trains a classifier on the UCI Online Shoppers dataset.
- Compares models using cross-validation.
- Shows ROC/PR, calibration, lift, and threshold trade-offs.
- Runs an interactive Streamlit app so you can explore the results.

## Latest Results (full dataset)
- Selected model: `random_forest`
- PR-AUC: `0.724`
- ROC-AUC: `0.926`
- Best threshold: `0.09`

## Built-in Datasets
These are included in `data/bundled/` so the app works without manual uploads:
- `online_shoppers_full.csv`
- `online_shoppers_returning_focus.csv`
- `online_shoppers_seasonal_focus.csv`

The app has a dataset selector, so you can switch and retrain from the UI.

## Stack
- Python
- pandas, numpy
- scikit-learn
- plotly, streamlit
- pytest

## Project Structure
```text
.
├── app/
│   ├── app.py
│   ├── lite_app.py
│   └── style.css
├── data/
│   ├── bundled/
│   ├── raw/
│   └── processed/
├── models/
├── reports/
├── scripts/
├── src/
│   └── commerceconversiondashboard/
├── tests/
├── Makefile
└── README.md
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e '.[dev]'
```

## Train From CLI
```bash
python scripts/train_model.py --data-path data/bundled/online_shoppers_full.csv
```

## Run App
```bash
streamlit run app/app.py
```

If your browser has issues with the full app, use:
```bash
streamlit run app/lite_app.py
```

## Output Files
Training writes these files:
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

## Tests
```bash
pytest
```
