import numpy as np
import pandas as pd
from fastapi import APIRouter

from ..serialize import to_python
from ..service import service

router = APIRouter(prefix="/api/eda", tags=["eda"])


@router.get("/basic")
def basic_inspection():
    proj = service.project()
    df = proj.raw_df
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    categorical = df.select_dtypes(include=["str", "object", "category"]).columns.tolist()
    return {
        "shape": list(df.shape),
        "numeric_cols": numeric,
        "categorical_cols": categorical,
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "describe": to_python(df.describe(include="all").T.round(4).to_dict("index")),
    }


@router.get("/missing")
def missing_values():
    proj = service.project()
    df = proj.raw_df
    missing = df.isnull().sum().sort_values(ascending=False)
    pct = (df.isnull().mean() * 100).sort_values(ascending=False)
    rows = [
        {"column": str(c), "missing": int(m), "percent": round(float(p), 2)}
        for c, m, p in zip(missing.index, missing, pct)
        if m > 0
    ]
    return {"total": int(missing.sum()), "rows": rows}


@router.get("/duplicates")
def duplicate_rows():
    proj = service.project()
    return {"duplicate_rows": int(proj.raw_df.duplicated().sum())}


@router.get("/unique")
def unique_values():
    proj = service.project()
    df = proj.raw_df
    return {"nunique": to_python(df.nunique().sort_values().to_dict())}


@router.get("/suspicious")
def suspicious_values():
    proj = service.project()
    df = proj.raw_df
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    negative = to_python((df[numeric] < 0).sum().sort_values(ascending=False).to_dict())
    zeros = to_python((df[numeric] == 0).sum().sort_values(ascending=False).to_dict())
    return {"negative": negative, "zeros": zeros}


@router.get("/outliers")
def outlier_analysis():
    proj = service.project()
    df = proj.raw_df
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    rows = []
    for col in numeric:
        s = df[col].dropna()
        if s.empty:
            continue
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((s < lower) | (s > upper)).sum())
        rows.append(
            {
                "column": col,
                "q1": round(float(q1), 4),
                "q3": round(float(q3), 4),
                "iqr": round(float(iqr), 4),
                "lower": round(float(lower), 4),
                "upper": round(float(upper), 4),
                "outliers": count,
                "outlier_pct": round(count / len(s) * 100, 2),
            }
        )
    rows.sort(key=lambda r: r["outliers"], reverse=True)
    return {"rows": rows}


@router.get("/correlation")
def correlation_matrix():
    proj = service.project()
    df = proj.raw_df
    numeric = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[numeric].corr()

    corr_pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high = (
        corr_pairs.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "correlation"})
    )
    high["abs"] = high["correlation"].abs()
    high = high[high["abs"] >= 0.80].sort_values("abs", ascending=False)
    high = high.drop(columns="abs")

    return {
        "columns": numeric,
        "matrix": to_python(corr.round(4).values),
        "high_pairs": to_python(high.reset_index(drop=True)),
    }