from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent          # backend/app
BACKEND_DIR = BASE_DIR.parent                        # backend
DATA_DIR = BACKEND_DIR / "data"
CSV_PATH = DATA_DIR / "capstone_smartgrid_20000.csv"

FRONTEND_DIR = BACKEND_DIR / "frontend"

RANDOM_STATE = 42
TEST_SIZE = 0.20

POSSIBLE_TARGETS = [
    "next_hour_consumption_kwh",
    "high_usage_flag",
    "anomaly_flag",
    "outage_risk_score",
]

CATEGORICAL = ["region", "building_type", "tariff_tier"]

MODEL_TASKS = {
    "logistic": {"kind": "classification", "target": "anomaly_flag", "label": "Logistic Regression"},
    "random_forest": {"kind": "classification", "target": "anomaly_flag", "label": "Random Forest"},
    "xgboost": {"kind": "classification", "target": "high_usage_flag", "label": "XGBoost"},
    "ridge": {"kind": "regression", "target": "next_hour_consumption_kwh", "label": "Ridge Regression"},
}