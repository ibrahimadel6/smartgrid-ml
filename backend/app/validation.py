"""Shared boundary rules for raw prediction payloads.

Defense-in-depth: the frontend restricts what the user can type, and the API
enforces the same rules again here. Bounds come from the actual dataset so the
models never see values they were not trained on (linear models extrapolate
wildly outside the training range - e.g. Logistic Regression reporting a
~100% anomaly probability for an impossible input).

These rules are the single source of truth for both the backend check and the
frontend form constraints (kept in sync by hand; see frontend/js/predict-form.js).
"""

from __future__ import annotations

import math
from typing import Any

# Keys that control routing/model selection - never model features.
CONTROL_KEYS = {"model"}

# Inclusive (min, max) numeric bounds, mirroring the training data ranges.
NUMERIC_BOUNDS: dict[str, tuple[float, float]] = {
    "consumption_kwh": (0.0, 40.0),
    "hour": (0.0, 23.0),
    "day_of_week": (0.0, 6.0),  # dataset encodes Monday=0 .. Sunday=6
    "temp_c": (-20.0, 60.0),
    "humidity_pct": (0.0, 100.0),
    "grid_price_usd_per_kwh": (0.0, 10.0),
    "next_hour_consumption_kwh": (0.0, 40.0),
    "outage_risk_score": (0.0, 1.0),
}

# Numeric fields that must be whole numbers.
INTEGER_FIELDS = {"hour", "day_of_week"}

# Categorical fields must be one of these exact dataset values.
CATEGORICAL_ALLOWED: dict[str, set[str]] = {
    "region": {"MW", "NE", "SE", "SW", "W"},
    "building_type": {"residential", "commercial", "industrial"},
    "tariff_tier": {"mid_peak", "off_peak", "on_peak"},
}


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_features(features: dict[str, Any]) -> list[str]:
    """Return a list of human-readable errors. Empty list == features are valid.

    Control keys (e.g. "model") are simply skipped - they are consumed by the
    router for routing and must never be treated as model features.
    """
    errors: list[str] = []

    for key, value in features.items():
        if key in CONTROL_KEYS or value is None:
            continue

        if key in NUMERIC_BOUNDS:
            lo, hi = NUMERIC_BOUNDS[key]
            if not is_number(value):
                errors.append(f"'{key}' must be a number, got {type(value).__name__}.")
                continue
            num = float(value)
            if not math.isfinite(num):
                errors.append(f"'{key}' must be a finite number.")
                continue
            if key in INTEGER_FIELDS and num != int(num):
                errors.append(f"'{key}' must be a whole number, got {num:g}.")
                continue
            if not (lo <= num <= hi):
                errors.append(f"'{key}' must be between {lo:g} and {hi:g}, got {num:g}.")
            continue

        if key in CATEGORICAL_ALLOWED:
            allowed = CATEGORICAL_ALLOWED[key]
            if not isinstance(value, str) or value not in allowed:
                shown = ", ".join(sorted(allowed))
                errors.append(f"'{key}' must be one of: {shown} (got {value!r}).")

    return errors


def clean_features(features: dict[str, Any]) -> dict[str, Any]:
    """Drop routing/control keys and None values so only real features reach the ML layer."""
    cleaned = {k: v for k, v in features.items() if k not in CONTROL_KEYS and v is not None}
    return {k: (int(v) if k in INTEGER_FIELDS and is_number(v) else v) for k, v in cleaned.items()}


__all__ = [
    "CONTROL_KEYS",
    "NUMERIC_BOUNDS",
    "INTEGER_FIELDS",
    "CATEGORICAL_ALLOWED",
    "is_number",
    "validate_features",
    "clean_features",
]
