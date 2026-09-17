from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    source: str
    shape: list[int]


class DataFrameInfo(BaseModel):
    shape: list[int]
    columns: list[str]
    dtypes: dict[str, str]
    head: list[dict[str, Any]]


class ModelDetails(BaseModel):
    key: str
    name: str
    target: str
    kind: str
    metrics: dict[str, float]
    info: dict[str, Any]
    cm: list[list[int]] | None = None
    samples: list[dict[str, Any]]


class ModelListResponse(BaseModel):
    models: list[dict[str, Any]]


class PredictRequest(BaseModel):
    model: Literal["logistic", "ridge", "random_forest", "xgboost"]
    features: dict[str, Any] = Field(..., description="Raw CSV-style feature values, e.g. {'region': 'SW', ...}")


class PredictResponse(BaseModel):
    model_key: str
    model: str
    task: str
    kind: str
    prediction: Any
    label: str | None = None
    probability: float | None = None