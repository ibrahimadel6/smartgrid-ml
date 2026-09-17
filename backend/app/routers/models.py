from fastapi import APIRouter, HTTPException

from ..serialize import to_python
from ..service import service

router = APIRouter(prefix="/api/models", tags=["models"])

SUMMARY_KEYS = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
REG_KEYS = ["R2", "MAE", "MSE", "RMSE"]
PRIMARY = {"classification": ("accuracy", "Accuracy"), "regression": ("R2", "R²")}


@router.get("")
def list_models():
    proj = service.project()
    out = []
    for key, spec in proj.models.items():
        m = spec["metrics"]
        primary, primary_label = PRIMARY.get(spec["kind"], ("accuracy", "Accuracy"))
        metrics = {}
        keys = REG_KEYS if spec["kind"] == "regression" else SUMMARY_KEYS
        for mk in keys:
            if mk in m and m[mk] is not None:
                metrics[mk] = round(float(m[mk]), 4)
        score = metrics.get(primary, 0.0)
        row = {
            "key": key,
            "name": spec["name"],
            "target": spec["target"],
            "kind": spec["kind"],
            "score": round(float(score), 6),
            "score_label": primary_label,
            "metrics": metrics,
        }
        baseline = spec["info"].get("dummy_acc") if not spec["kind"] == "regression" else None
        if baseline is not None:
            row["baseline_acc"] = round(float(baseline), 4)
        out.append(row)
    return {"models": out}


@router.get("/{key}")
def get_model(key: str):
    proj = service.project()
    if key not in proj.models:
        raise HTTPException(404, f"Unknown model '{key}'")
    spec = proj.models[key]
    payload = {
        "key": key,
        "name": spec["name"],
        "target": spec["target"],
        "kind": spec["kind"],
        "metrics": spec["metrics"],
        "info": spec["info"],
        "cm": spec.get("cm"),
        "cm_flat": spec.get("cm_flat"),
        "best_thr": spec.get("best_thr"),
        "best_prec": spec.get("best_prec"),
        "best_rec": spec.get("best_rec"),
        "best_f1": spec.get("best_f1"),
        "baseline_pr": spec.get("baseline_pr"),
        "roc_fpr": spec.get("roc_fpr"),
        "roc_tpr": spec.get("roc_tpr"),
        "pr_rec": spec.get("pr_rec"),
        "pr_prec": spec.get("pr_prec"),
        "report": spec.get("report"),
        "verdict": spec.get("verdict"),
        "r2_gap": spec.get("r2_gap"),
        "rmse_gap": spec.get("rmse_gap"),
        "train_r2": spec.get("train_r2"),
        "test_r2": spec.get("test_r2"),
        "train_rmse": spec.get("train_rmse"),
        "test_rmse": spec.get("test_rmse"),
        "train_metrics": spec.get("train_metrics"),
        "results": spec.get("results"),
        "top10": spec.get("top10"),
        "importance_full": spec.get("importance_full"),
        "y_test": spec.get("y_test"),
        "y_pred": spec.get("y_pred"),
        "errors": spec.get("errors"),
        "mean_error": spec.get("mean_error"),
        "median_error": spec.get("median_error"),
        "samples": spec.get("samples"),
        "feature_columns": spec.get("feature_columns"),
    }
    return to_python(payload)