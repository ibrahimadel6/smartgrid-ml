import hashlib
import math

import numpy as np
import pandas as pd


def is_binary(series: pd.Series) -> bool:
    values = series.dropna()
    if values.empty:
        return False
    return set(np.unique(values)).issubset({0, 1, 0.0, 1.0, True, False})


def data_fingerprint(df: pd.DataFrame) -> str:
    raw = pd.util.hash_pandas_object(df.fillna(-999), index=True)
    return hashlib.md5(raw.values.tobytes()).hexdigest()


def _is_missing(value) -> bool:
    if value is None:
        return True
    if value is pd.NA:
        return True
    if isinstance(value, float):
        return math.isnan(value)
    if isinstance(value, np.floating):
        return bool(np.isnan(value))
    return False


def to_python(value):
    """Recursively convert numpy/pandas primitives to JSON-safe Python types."""
    if _is_missing(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return [to_python(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {k: to_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_python(v) for v in value]
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return str(value)
    if isinstance(value, pd.Series):
        return value.map(to_python).tolist()
    if isinstance(value, pd.DataFrame):
        return [to_python(r) for r in value.to_dict("records")]
    return value


def rows_with_index(df: pd.DataFrame, n: int) -> list[dict]:
    return to_python(df.head(n).reset_index().rename(columns={"index": "row"}).to_dict("records"))