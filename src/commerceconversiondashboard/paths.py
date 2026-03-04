"""Project path helpers."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
DATA_BUNDLED_DIR = PROJECT_ROOT / "data" / "bundled"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = PROJECT_ROOT / "figures"

RAW_DATA_FILE = DATA_RAW_DIR / "online_shoppers_intention.csv"
DATASET_MANIFEST_FILE = DATA_BUNDLED_DIR / "datasets_manifest.json"
TRAIN_DATA_FILE = DATA_PROCESSED_DIR / "train.csv"
TEST_DATA_FILE = DATA_PROCESSED_DIR / "test.csv"
MODEL_FILE = MODELS_DIR / "conversion_model.joblib"
METADATA_FILE = MODELS_DIR / "model_metadata.json"
EVAL_FILE = REPORTS_DIR / "evaluation_summary.json"
PREDICTIONS_FILE = REPORTS_DIR / "test_predictions.csv"
FEATURE_IMPORTANCE_FILE = REPORTS_DIR / "feature_importance.csv"
CALIBRATION_FILE = REPORTS_DIR / "calibration_curve.csv"
LIFT_TABLE_FILE = REPORTS_DIR / "decile_lift_table.csv"
ROC_CURVE_FILE = REPORTS_DIR / "roc_curve.csv"
MODEL_CARD_FILE = REPORTS_DIR / "model_card.md"
EXECUTIVE_SUMMARY_FILE = REPORTS_DIR / "executive_summary.md"
