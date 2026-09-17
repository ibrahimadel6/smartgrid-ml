import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import POSSIBLE_TARGETS
from .serialize import is_binary


def engineer_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cell 24: convert timestamp and extract month/day/year + sin/cos cyclic features."""
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")

    out["month"] = out["timestamp"].dt.month
    out["day"] = out["timestamp"].dt.day
    out["year"] = out["timestamp"].dt.year

    if "hour" in out.columns:
        out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
        out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    if "day_of_week" in out.columns:
        out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
        out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    return out


def fill_missing(df: pd.DataFrame, ref: dict | None = None) -> pd.DataFrame:
    """Cell 26: numeric -> median, categorical -> mode.

    When `ref` (the training feature frame) is provided, missing values in a
    new row are filled with the statistics learned from training data.
    """
    out = df.copy()

    def _groups(base):
        return (
            base.select_dtypes(include=np.number).columns,
            base.select_dtypes(include=["str", "object", "category"]).columns,
        )

    if ref is not None:
        num_cols, cat_cols = _groups(ref["df"])
    else:
        num_cols, cat_cols = _groups(out)
        num_cols = [c for c in num_cols if c in out.columns]
        cat_cols = [c for c in cat_cols if c in out.columns]

    for col in num_cols:
        if col in out.columns and out[col].isnull().any():
            if ref is not None:
                out[col] = out[col].fillna(ref["df"][col].median())
            else:
                out[col] = out[col].fillna(out[col].median())
    for col in cat_cols:
        if col in out.columns and out[col].isnull().any():
            if ref is not None:
                out[col] = out[col].fillna(ref["df"][col].mode()[0])
            else:
                out[col] = out[col].fillna(out[col].mode()[0])
    return out


def build_feature_frame(df: pd.DataFrame) -> dict:
    """
    Post-engineering pipeline (cells 24-36):
    returns df (cleaned), df_model, df_encoded, X_prepared and metadata.
    """
    df = engineer_time_features(df)
    df = fill_missing(df)
    remaining_missing = int(df.isnull().sum().sum())

    drop_for_model = [c for c in ["timestamp", "meter_id"] if c in df.columns]
    df_model = df.drop(columns=drop_for_model).copy()

    categorical_cols_model = df_model.select_dtypes(include=["str", "object", "category"]).columns.tolist()
    df_encoded = pd.get_dummies(df_model, columns=categorical_cols_model, drop_first=True, dtype=int)

    X_prepared = df_encoded.copy()
    binary_cols = [c for c in X_prepared.columns if is_binary(X_prepared[c])]
    possible_target_cols = [c for c in POSSIBLE_TARGETS if c in X_prepared.columns]
    scale_cols = [
        c
        for c in X_prepared.select_dtypes(include=np.number).columns
        if c not in binary_cols and c not in possible_target_cols
    ]

    scaler = StandardScaler()
    if scale_cols:
        X_prepared[scale_cols] = scaler.fit_transform(X_prepared[scale_cols])

    return {
        "df": df,
        "df_model": df_model,
        "df_encoded": df_encoded,
        "X_prepared": X_prepared,
        "categorical_cols_model": categorical_cols_model,
        "drop_for_model": drop_for_model,
        "remaining_missing": remaining_missing,
        "scale_cols": scale_cols,
        "binary_cols": binary_cols,
        "possible_target_cols": possible_target_cols,
        "possible_targets": [c for c in POSSIBLE_TARGETS if c in df_encoded.columns],
        "scaler": scaler,
    }


def make_time_split(df: pd.DataFrame, test_size: float = 0.20):
    order = np.argsort(df["timestamp"].values)
    cut = int(len(order) * (1 - test_size))
    return order[:cut], order[cut:]