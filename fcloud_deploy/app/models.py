import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_curve,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from .config import POSSIBLE_TARGETS, RANDOM_STATE
from .preprocessing import make_time_split
from .serialize import is_binary, to_python

TEST_SIZE = 0.20


def _reg_metrics(y_true, y_pred):
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


def train_logistic(feats: dict):
    df = feats["df"]
    df_encoded = feats["df_encoded"]
    target_col = "anomaly_flag"
    y = df_encoded[target_col].astype(int)

    drop_cols = [c for c in POSSIBLE_TARGETS if c in df_encoded.columns]
    X = df_encoded.drop(columns=drop_cols)
    const_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
    if const_cols:
        X = X.drop(columns=const_cols)

    tr_idx, te_idx = make_time_split(df, TEST_SIZE)
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

    binary_cols = [c for c in X_tr.columns if is_binary(X_tr[c])]
    cont_cols = [c for c in X_tr.columns if c not in binary_cols]

    prep = ColumnTransformer(
        [("scale", StandardScaler(), cont_cols), ("pass", "passthrough", binary_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    model = Pipeline(
        [
            ("prep", prep),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear", C=1.0, class_weight="balanced",
                    max_iter=2000, random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)

    dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE).fit(X_tr, y_tr)
    dummy_acc = float(accuracy_score(y_te, dummy.predict(X_te)))

    metrics = {
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "pr_auc": float(average_precision_score(y_te, proba)),
    }

    prec, rec, thr = precision_recall_curve(y_te, proba)
    f1s = 2 * prec * rec / np.clip(prec + rec, 1e-12, None)
    best_i = int(np.nanargmax(f1s[:-1]))

    cm = confusion_matrix(y_te, pred, labels=[0, 1]).tolist()
    tn, fp, fn, tp = np.ravel(np.array(cm))

    return {
        "model": model,
        "name": "Logistic Regression",
        "target": "anomaly_flag",
        "kind": "classification",
        "feature_columns": X.columns.tolist(),
        "info": {
            "X_shape": list(X.shape),
            "train_shape": list(X_tr.shape),
            "test_shape": list(X_te.shape),
            "train_rate": float(y_tr.mean()),
            "test_rate": float(y_te.mean()),
            "n_cont": len(cont_cols),
            "n_binary": len(binary_cols),
            "n_iter": int(model.named_steps["clf"].n_iter_[0]),
            "dummy_acc": dummy_acc,
            "drop_cols": drop_cols,
            "const_cols": const_cols,
        },
        "metrics": metrics,
        "cm": cm,
        "cm_flat": {"TN": tn, "FP": fp, "FN": fn, "TP": tp},
        "best_thr": float(thr[best_i]),
        "best_prec": float(prec[best_i]),
        "best_rec": float(rec[best_i]),
        "best_f1": float(f1s[best_i]),
        "baseline_pr": float(y_te.mean()),
        "roc_fpr": roc_curve(y_te, proba)[0].tolist(),
        "roc_tpr": roc_curve(y_te, proba)[1].tolist(),
        "pr_rec": rec.tolist(),
        "pr_prec": prec.tolist(),
        "report": classification_report(y_te, pred, labels=[0, 1],
                                        target_names=["Normal (0)", "Anomaly (1)"], zero_division=0),
        "samples": _sample_rows(y_te.values, pred, proba),
    }


def train_ridge(feats: dict):
    df = feats["df"].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp").reset_index(drop=True)
    target = "next_hour_consumption_kwh"
    df = df.dropna(subset=[target, "timestamp"]).reset_index(drop=True)

    drop_cols = [c for c in ["timestamp", "meter_id", "high_usage_flag", "anomaly_flag",
                             "outage_risk_score", target] if c in df.columns]
    X_raw = df.drop(columns=drop_cols)
    y = df[target]

    cat_cols = X_raw.select_dtypes(include=["str", "object", "category"]).columns.tolist()
    num_cols = X_raw.select_dtypes(include=["number", "bool"]).columns.tolist()

    split = int(len(X_raw) * 0.80)
    X_tr_raw, X_te_raw = X_raw.iloc[:split], X_raw.iloc[split:]
    y_tr, y_te = y.iloc[:split], y.iloc[split:]

    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", num_pipe, num_cols), ("cat", cat_pipe, cat_cols)])
    X_tr = prep.fit_transform(X_tr_raw)
    X_te = prep.transform(X_te_raw)

    grid = GridSearchCV(
        estimator=Pipeline([("ridge", Ridge())]),
        param_grid={"ridge__alpha": [0.01, 0.1, 1, 10, 100]},
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    grid.fit(X_tr, y_tr)

    best_alpha = float(grid.best_params_["ridge__alpha"])
    best_cv_rmse = float(-grid.best_score_)
    y_tr_pred = grid.best_estimator_.predict(X_tr)
    y_te_pred = grid.best_estimator_.predict(X_te)

    tr_m = _reg_metrics(y_tr, y_tr_pred)
    te_m = _reg_metrics(y_te, y_te_pred)

    results = pd.DataFrame({
        "Metric": ["MAE", "MSE", "RMSE", "R2"],
        "Training": [tr_m["MAE"], tr_m["MSE"], tr_m["RMSE"], tr_m["R2"]],
        "Testing": [te_m["MAE"], te_m["MSE"], te_m["RMSE"], te_m["R2"]],
    })

    r2_gap = tr_m["R2"] - te_m["R2"]
    if r2_gap < 0.05:
        verdict = "No strong evidence of overfitting."
    elif r2_gap < 0.10:
        verdict = "Mild possible overfitting."
    else:
        verdict = "Possible overfitting - investigate features and regularization."

    errors = (y_te.values - y_te_pred).tolist()

    return {
        "model": grid.best_estimator_,
        "prep": prep,
        "name": "Ridge Regression",
        "target": "next_hour_consumption_kwh",
        "kind": "regression",
        "feature_columns": X_raw.columns.tolist(),
        "info": {
            "shape": list(df.shape),
            "cat_feats": cat_cols,
            "n_numeric": len(num_cols),
            "n_raw_features": int(X_raw.shape[1]),
            "train_rows": int(len(X_tr_raw)),
            "test_rows": int(len(X_te_raw)),
            "best_alpha": best_alpha,
            "best_cv_rmse": best_cv_rmse,
        },
        "metrics": te_m,
        "results": results.where(pd.notna(results), None),
        "train_rmse": tr_m["RMSE"],
        "test_rmse": te_m["RMSE"],
        "rmse_gap": te_m["RMSE"] - tr_m["RMSE"],
        "train_r2": tr_m["R2"],
        "test_r2": te_m["R2"],
        "r2_gap": r2_gap,
        "verdict": verdict,
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "y_test": y_te.values.tolist(),
        "y_pred": y_te_pred.tolist(),
        "errors": errors,
        "samples": _sample_rows(y_te.values, y_te_pred, None),
    }


def train_random_forest(feats: dict):
    df_encoded = feats["df_encoded"]
    target = "anomaly_flag"
    drop_cols = [c for c in POSSIBLE_TARGETS if c in df_encoded.columns]
    X = df_encoded.drop(columns=drop_cols)
    y = df_encoded[target]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_leaf=2,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "f1": float(f1_score(y_te, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "pr_auc": float(average_precision_score(y_te, proba)),
    }

    train_f1 = float(f1_score(y_tr, model.predict(X_tr), zero_division=0))
    test_f1 = metrics["f1"]
    f1_gap = train_f1 - test_f1
    if f1_gap <= 0.05:
        verdict = "No strong sign of overfitting."
    elif f1_gap <= 0.10:
        verdict = "Possible mild overfitting."
    else:
        verdict = "Strong sign of overfitting."

    imp = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_})
    imp = imp.sort_values("Importance", ascending=False).reset_index(drop=True)

    return {
        "model": model,
        "name": "Random Forest",
        "target": "anomaly_flag",
        "kind": "classification",
        "feature_columns": X.columns.tolist(),
        "info": {
            "X_shape": list(X.shape),
            "class_counts": {int(k): int(v) for k, v in y.value_counts().items()},
            "train_shape": list(X_tr.shape),
            "test_shape": list(X_te.shape),
        },
        "metrics": metrics,
        "cm": confusion_matrix(y_te, pred).tolist(),
        "train_f1": train_f1,
        "test_f1": test_f1,
        "f1_gap": f1_gap,
        "verdict": verdict,
        "top10": to_python(imp.head(10)),
        "importance_full": to_python(imp),
        "roc_fpr": roc_curve(y_te, proba)[0].tolist(),
        "roc_tpr": roc_curve(y_te, proba)[1].tolist(),
        "pr_rec": precision_recall_curve(y_te, proba)[0].tolist(),
        "pr_prec": precision_recall_curve(y_te, proba)[1].tolist(),
        "report": classification_report(y_te, pred, target_names=["Normal (0)", "Anomaly (1)"], zero_division=0),
        "samples": _sample_rows(y_te.values, pred, proba),
    }


def train_xgboost(feats: dict):
    X_prepared = feats["X_prepared"]
    X = X_prepared.drop(columns=[c for c in ["high_usage_flag", "consumption_kwh"] if c in X_prepared.columns])
    y = X_prepared["high_usage_flag"].astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        random_state=RANDOM_STATE, eval_metric="logloss",
    )
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)
    proba = model.predict_proba(X_te)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_te, pred)),
        "precision": float(precision_score(y_te, pred)),
        "recall": float(recall_score(y_te, pred)),
        "f1": float(f1_score(y_te, pred)),
        "roc_auc": float(roc_auc_score(y_te, proba)),
        "pr_auc": float(average_precision_score(y_te, proba)),
    }

    return {
        "model": model,
        "name": "XGBoost",
        "target": "high_usage_flag",
        "kind": "classification",
        "feature_columns": X.columns.tolist(),
        "info": {
            "X_shape": list(X.shape),
            "class_counts": {int(k): int(v) for k, v in y.value_counts().items()},
            "train_shape": list(X_tr.shape),
            "test_shape": list(X_te.shape),
            "drop_cols": ["high_usage_flag", "consumption_kwh"],
        },
        "metrics": metrics,
        "cm": confusion_matrix(y_te, pred).tolist(),
        "roc_fpr": roc_curve(y_te, proba)[0].tolist(),
        "roc_tpr": roc_curve(y_te, proba)[1].tolist(),
        "pr_rec": precision_recall_curve(y_te, proba)[0].tolist(),
        "pr_prec": precision_recall_curve(y_te, proba)[1].tolist(),
        "report": classification_report(y_te, pred),
        "samples": _sample_rows(y_te.values, pred, proba),
    }


def _sample_rows(y_test, y_pred, proba, n=20):
    if proba is not None:
        rows = [
            {"Actual": int(a), "Predicted": int(p), "Probability": round(float(pb), 4)}
            for a, p, pb in zip(y_test[:n], y_pred[:n], proba[:n])
        ]
    else:
        rows = [
            {"Actual": round(float(a), 4), "Predicted": round(float(p), 4),
             "Absolute Error": round(float(abs(a - p)), 4)}
            for a, p in zip(y_test[:n], y_pred[:n])
        ]
    return rows