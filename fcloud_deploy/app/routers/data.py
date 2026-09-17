from fastapi import APIRouter
from fastapi.responses import FileResponse

import numpy
import pandas as pd

from ..config import CSV_PATH
from ..serialize import rows_with_index, to_python
from ..service import service

router = APIRouter(prefix="/api/data", tags=["data"])


@router.get("/info")
def data_info():
    proj = service.project()
    df = proj.raw_df
    return {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "head": rows_with_index(df, 50),
        "n_unique": to_python(df.nunique().sort_values()),
    }


@router.get("/targets")
def target_counts():
    proj = service.project()
    df = proj.raw_df
    out = {}
    discrete = ("high_usage_flag", "anomaly_flag")
    continuous = ("outage_risk_score", "next_hour_consumption_kwh")
    for col in discrete:
        if col in df.columns:
            counts = df[col].value_counts(dropna=False, sort=False)
            items = [
                {
                    "label": str(int(k)) if float(k).is_integer() else str(k),
                    "value": int(v),
                    "percent": round(float(v) / len(df) * 100, 2),
                }
                for k, v in counts.items()
            ]
            out[col] = {
                "kind": "discrete",
                "items": items,
                "nunique": int(df[col].nunique(dropna=False)),
            }
    for col in continuous:
        if col in df.columns:
            s = df[col].dropna()
            counts, edges = numpy.histogram(s.to_numpy(), bins=30)
            out[col] = {
                "kind": "continuous",
                "describe": to_python(s.describe().round(4).to_dict()),
                "nunique": int(df[col].nunique(dropna=False)),
                "histogram": {
                    "counts": counts.astype(int).tolist(),
                    "bin_edges": numpy.round(edges, 4).tolist(),
                },
            }
    return out


@router.get("/overview")
def full_overview():
    """Aggregations over the FULL dataset (not a sample) for the dashboard."""
    proj = service.project()
    df = proj.raw_df.copy()
    out = {"shape": list(df.shape)}

    ts = None
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")

    def _count(col):
        if col not in df.columns:
            return None
        n = int((df[col] == 1).sum())
        return {"count": n, "percent": round(n / len(df) * 100, 2)}

    out["anomaly"] = _count("anomaly_flag")
    out["high_usage"] = _count("high_usage_flag")
    out["avg_consumption"] = round(float(df["consumption_kwh"].mean()), 4)
    if "next_hour_consumption_kwh" in df.columns:
        out["avg_next_hour"] = round(float(df["next_hour_consumption_kwh"].mean()), 4)

    if ts is not None and ts.notna().any():
        out["period"] = {
            "start": str(ts.min()),
            "end": str(ts.max()),
            "days": int((ts.max() - ts.min()).days) + 1,
            "hours": int(len(ts)),
        }

    mix = {}
    for col in ("building_type", "region", "tariff_tier"):
        if col in df.columns:
            vc = df[col].value_counts(dropna=False)
            mix[col] = [{"label": str(k), "value": int(v)} for k, v in vc.items()]
    out["mix"] = mix

    if ts is not None and ts.notna().any():
        work = df.assign(_ts=ts)
        if "hour" in work.columns:
            hourly = work.groupby("hour")["consumption_kwh"].mean()
            out["hourly_profile"] = {
                "labels": [int(h) for h in hourly.index],
                "values": [round(float(v), 4) for v in hourly.values],
            }
        daily = work.set_index("_ts")["consumption_kwh"].resample("D").mean().dropna().iloc[-90:]
        out["daily_trend"] = {
            "labels": [str(d.date()) for d in daily.index],
            "values": [round(float(v), 4) for v in daily.values],
        }
    return out


@router.get("/download")
def download_csv():
    return FileResponse(CSV_PATH, media_type="text/csv", filename="capstone_smartgrid_20000.csv")