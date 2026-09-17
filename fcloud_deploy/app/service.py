from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import CSV_PATH, POSSIBLE_TARGETS
from .data_loader import load_dataset
from .models import train_logistic, train_ridge, train_random_forest, train_xgboost
from .preprocessing import build_feature_frame, engineer_time_features, fill_missing
from .serialize import data_fingerprint, is_binary, to_python
from .validation import clean_features

# Positive-class label per classification target.
POSITIVE_LABELS = {"anomaly_flag": "Anomaly", "high_usage_flag": "High usage"}


@dataclass
class Project:
    raw_df: pd.DataFrame
    features: dict
    models: dict
    fingerprint: str


class ProjectService:
    """Builds the full pipeline + trains the four models once, then serves it."""

    def __init__(self):
        self._project: Project | None = None

    def load(self, source: str = "csv", n_rows: int | None = None, seed: int = 42) -> Project:
        """Load the dataset and train all four models.

        For `source="csv"` the ENTIRE CSV (20,000 rows) is loaded — `n_rows`
        only applies to the `synthetic` generator.
        """
        df = load_dataset(source, n_rows, seed)
        fp = data_fingerprint(df)
        if self._project is None or self._project.fingerprint != fp:
            features = build_feature_frame(df)
            models = {
                "logistic": train_logistic(features),
                "ridge": train_ridge(features),
                "random_forest": train_random_forest(features),
                "xgboost": train_xgboost(features),
            }
            self._project = Project(raw_df=df, features=features, models=models, fingerprint=fp)
        return self._project

    def project(self) -> Project:
        if self._project is None:
            raise RuntimeError("Project not loaded - call load() first.")
        return self._project

    # ------------------------------------------------------------------
    # Prediction helpers
    # ------------------------------------------------------------------
    def encode_row_as_training(self, raw_values: dict) -> pd.DataFrame:
        """Turn raw CSV-style values into a 1-row df_encoded aligned to training columns."""
        feats = self.project().features
        raw = self.project().raw_df

        row = pd.DataFrame([raw_values])
        for col in raw.columns:
            if col not in row.columns:
                row[col] = np.nan
        row = row[raw.columns]

        df = engineer_time_features(row)
        df = fill_missing(df, feats)
        df_model = df.drop(columns=[c for c in feats["drop_for_model"] if c in df.columns])
        cat_cols = [c for c in feats["categorical_cols_model"] if c in df_model.columns]
        encoded = pd.get_dummies(df_model, columns=cat_cols, drop_first=True, dtype=int)
        encoded = encoded.reindex(columns=feats["df_encoded"].columns, fill_value=0)
        return encoded

    def predict(self, model_key: str, raw_values: dict) -> dict:
        """Run a model on raw CSV-style feature values and return a JSON-safe result."""
        proj = self.project()
        spec = proj.models[model_key]
        raw_values = clean_features(raw_values)

        if model_key == "ridge":
            X_raw = pd.DataFrame([raw_values]).reindex(columns=spec["feature_columns"])
            X = spec["prep"].transform(X_raw)
            value = float(spec["model"].predict(X)[0])
            return {
                "model_key": model_key,
                "model": spec["name"],
                "task": spec["target"],
                "prediction": round(value, 4),
                "kind": "regression",
            }

        encoded = self.encode_row_as_training(raw_values)
        if model_key == "xgboost":
            X = encoded.copy()
            scaler = proj.features["scaler"]
            scale_cols = [c for c in proj.features["scale_cols"] if c in X.columns]
            if scale_cols:
                X[scale_cols] = scaler.transform(X[scale_cols])
            X = X.reindex(columns=spec["feature_columns"])
        else:
            X = encoded.reindex(columns=spec["feature_columns"])

        proba = spec["model"].predict_proba(X)[0, 1]
        pred = int((proba >= 0.5))
        positive = POSITIVE_LABELS.get(spec["target"], "Positive")
        return {
            "model_key": model_key,
            "model": spec["name"],
            "task": spec["target"],
            "prediction": pred,
            "label": positive if pred == 1 else "Normal",
            "probability": round(float(proba), 4),
            "kind": "classification",
        }


service = ProjectService()