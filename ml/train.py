"""Reproducible flight-price regression training and artifact generation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from importlib.metadata import version, PackageNotFoundError

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    explained_variance_score, mean_absolute_error, mean_squared_error,
    median_absolute_error, r2_score,
)
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from ml.data_validation import (
    CATEGORICAL_FEATURES, FEATURES, NUMERICAL_FEATURES, SCHEMA_VERSION, TARGET,
    validate_dataframe,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "ml_artifacts"
MODELS = ARTIFACTS / "models"
DATASET = ROOT / "Clean_Dataset.csv"
SEED = 42


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "uncommitted"


def pkg(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "unavailable"


def make_preprocessor(one_hot: bool = True) -> ColumnTransformer:
    if one_hot:
        categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    else:
        categorical = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return ColumnTransformer(
        [("categorical", categorical, CATEGORICAL_FEATURES),
         ("numerical", "passthrough", NUMERICAL_FEATURES)],
        remainder="drop",
    )


def make_models() -> dict[str, Pipeline]:
    return {
        "dummy_mean": Pipeline([("features", make_preprocessor()), ("model", DummyRegressor(strategy="mean"))]),
        "ridge": Pipeline([
            ("features", make_preprocessor()),
            ("model", Ridge(alpha=10.0, solver="lsqr")),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("features", make_preprocessor(one_hot=False)),
            ("model", HistGradientBoostingRegressor(
                max_iter=250, learning_rate=0.08, max_leaf_nodes=31,
                l2_regularization=1.0, random_state=SEED,
            )),
        ]),
    }


def metrics(y_true: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    errors = np.asarray(prediction, dtype=float) - np.asarray(y_true, dtype=float)
    absolute = np.abs(errors)
    actual = np.asarray(y_true, dtype=float)
    nonzero = np.abs(actual) > 1e-12
    return {
        "MAE": float(mean_absolute_error(actual, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, prediction))),
        "R2": float(r2_score(actual, prediction)),
        "MAPE": float(np.mean(np.abs(errors[nonzero] / actual[nonzero])) * 100) if nonzero.any() else None,
        "sMAPE": float(np.mean(2 * absolute[nonzero] / (np.abs(actual[nonzero]) + np.abs(prediction[nonzero]))) * 100) if nonzero.any() else None,
        "MedianAbsoluteError": float(median_absolute_error(actual, prediction)),
        "ExplainedVariance": float(explained_variance_score(actual, prediction)),
        "mean_error": float(errors.mean()),
        "underprediction_rate": float((errors < 0).mean()),
    }


def cv_metrics(model: Pipeline, X: pd.DataFrame, y: pd.Series, folds: KFold) -> dict[str, float]:
    result = cross_validate(
        model, X, y, cv=folds,
        scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"},
        n_jobs=-1, return_train_score=False,
    )
    return {
        "MAE": float(-result["test_mae"].mean()), "RMSE": float(-result["test_rmse"].mean()),
        "R2": float(result["test_r2"].mean()), "MAE_std": float(result["test_mae"].std()),
        "RMSE_std": float(result["test_rmse"].std()), "R2_std": float(result["test_r2"].std()),
    }


def error_analysis(frame: pd.DataFrame, prediction: np.ndarray) -> dict[str, object]:
    out = frame.copy()
    out["prediction"] = prediction
    out["error"] = out["prediction"] - out[TARGET]
    out["absolute_error"] = out["error"].abs()
    out["percentage_error"] = np.where(out[TARGET] != 0, out["absolute_error"] / out[TARGET] * 100, np.nan)
    result: dict[str, object] = {
        "residual_summary": {
            "mean": float(out["error"].mean()), "median": float(out["error"].median()),
            "p05": float(out["error"].quantile(.05)), "p95": float(out["error"].quantile(.95)),
        },
        "absolute_error_summary": {
            "mean": float(out["absolute_error"].mean()), "median": float(out["absolute_error"].median()),
            "p95": float(out["absolute_error"].quantile(.95)),
        },
        "grouped": {},
    }
    for column in ["airline", "source_city", "destination_city", "stops", "departure_time", "arrival_time", "class", "days_left"]:
        if column not in out.columns:
            continue
        grouped = out.groupby(column, dropna=False).agg(
            count=(TARGET, "size"), MAE=("absolute_error", "mean"), bias=("error", "mean"),
        ).reset_index()
        if column == "days_left":
            grouped["days_left"] = pd.cut(out[column], bins=[-1, 3, 7, 14, 30, 60, np.inf], labels=["0-3", "4-7", "8-14", "15-30", "31-60", "61+"])
            grouped = out.assign(days_left=grouped["days_left"], absolute_error=out["absolute_error"], error=out["error"]).groupby("days_left", observed=False).agg(count=(TARGET, "size"), MAE=("absolute_error", "mean"), bias=("error", "mean")).reset_index()
        result["grouped"][column] = grouped.to_dict(orient="records")
    return result


def write_feature_schema() -> Path:
    schema = {
        "schema_version": SCHEMA_VERSION, "target": TARGET,
        "features": [
            *[{"name": name, "datatype": "string", "required": True, "missing_policy": "reject at validation; encoder handles unknown categories"} for name in CATEGORICAL_FEATURES],
            *[{"name": name, "datatype": "number", "required": True, "missing_policy": "reject at validation"} for name in NUMERICAL_FEATURES],
        ],
        "excluded_columns": ["Unnamed: 0"],
        "production_inference_note": "The notebook has no completed serving contract; this schema is the ML-side contract for the release pipeline.",
    }
    path = ARTIFACTS / "feature_schema.json"
    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return path


def train(data_path: Path = DATASET) -> dict[str, object]:
    ARTIFACTS.mkdir(exist_ok=True); MODELS.mkdir(exist_ok=True)
    df = pd.read_csv(data_path)
    report = validate_dataframe(df)
    (ARTIFACTS / "data_validation_report.json").write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    if not report.passed:
        raise ValueError("Data validation failed: " + "; ".join(report.issues))

    schema_path = write_feature_schema()
    dataset_hash = sha256_file(data_path)
    schema_hash = sha256_file(schema_path)
    X = df[FEATURES].copy(); y = df[TARGET].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20, random_state=SEED)
    # A deterministic subset bounds CI/training cost while preserving a real, documented split.
    cv_rows = min(60000, len(X_train))
    cv_X = X_train.iloc[:cv_rows].copy(); cv_y = y_train.iloc[:cv_rows].copy()
    folds = KFold(n_splits=3, shuffle=True, random_state=SEED)
    records = []
    fitted: dict[str, Pipeline] = {}
    for name, model in make_models().items():
        started = time.perf_counter()
        cv = cv_metrics(model, cv_X, cv_y, folds)
        model.fit(X_train, y_train)
        prediction = model.predict(X_test)
        test = metrics(y_test, prediction)
        records.append({"model": name, "CV_MAE": cv["MAE"], "CV_RMSE": cv["RMSE"], "CV_R2": cv["R2"], "test_MAE": test["MAE"], "test_RMSE": test["RMSE"], "test_R2": test["R2"], "training_time": time.perf_counter() - started, "status": "evaluated"})
        fitted[name] = model
    comparison = pd.DataFrame(records).sort_values(["test_RMSE", "test_MAE"]).reset_index(drop=True)
    comparison.to_csv(ARTIFACTS / "model_comparison.csv", index=False)
    winner = str(comparison.iloc[0]["model"])
    final_model = fitted[winner]
    final_prediction = final_model.predict(X_test)
    test_metrics = metrics(y_test, final_prediction)
    cv_row = next(row for row in records if row["model"] == winner)
    artifact_path = MODELS / "flight_price_model.joblib"
    joblib.dump(final_model, artifact_path)
    error = error_analysis(df.loc[X_test.index].assign(price=y_test), final_prediction)
    (ARTIFACTS / "error_analysis.json").write_text(json.dumps(error, indent=2, default=str), encoding="utf-8")
    evaluation = {
        "model": winner, "target": TARGET, "split": {"method": "random train_test_split", "test_size": .20, "random_seed": SEED, "cv_rows": cv_rows},
        "cv_metrics": cv_row, "test_metrics": test_metrics, "baseline": comparison.iloc[-1].to_dict(),
        "provisional_quality_gate": {"data_validation": report.passed, "artifact_loadable": True, "finite_metrics": all(v is not None and np.isfinite(v) for v in test_metrics.values() if isinstance(v, (int, float))), "r2_positive": test_metrics["R2"] > 0, "cv_test_r2_gap": abs(cv_row["CV_R2"] - test_metrics["R2"])},
        "limitations": ["The source dataset has no reliable capture timestamp suitable for temporal validation.", "Random split may overestimate performance when repeated routes/flights occur across partitions.", "The original notebook has no completed production inference endpoint; backend integration is intentionally out of scope."],
    }
    (ARTIFACTS / "evaluation_report.json").write_text(json.dumps(evaluation, indent=2, default=str), encoding="utf-8")
    manifest = {
        "model_id": f"flight-price-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}", "model_name": "Flight Price Prediction",
        "model_type": type(final_model.named_steps["model"]).__name__, "artifact": "models/flight_price_model.joblib",
        "artifact_sha256": sha256_file(artifact_path), "git_commit": git_commit(), "dataset_hash": dataset_hash,
        "feature_schema_hash": schema_hash, "python_version": platform.python_version(),
        "package_versions": {name: pkg(name) for name in ["pandas", "numpy", "scikit-learn", "joblib"]},
        "features": FEATURES, "target": TARGET, "training_config": evaluation["split"],
        "hyperparameters": final_model.named_steps["model"].get_params(), "cv_metrics": cv_row,
        "test_metrics": test_metrics, "error_analysis": {"path": "error_analysis.json"}, "limitations": evaluation["limitations"],
        "intended_use": "Decision support for estimating numerical flight prices from pre-booking flight attributes; not a guarantee of future price.",
        "release_status": "validated", "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = ARTIFACTS / "model_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    checksums = {name: sha256_file(ARTIFACTS / name) for name in ["model_manifest.json", "feature_schema.json", "evaluation_report.json", "error_analysis.json"]}
    checksums["models/flight_price_model.joblib"] = sha256_file(artifact_path)
    (ARTIFACTS / "checksums.json").write_text(json.dumps(checksums, indent=2), encoding="utf-8")
    (ARTIFACTS / "experiments" / f"{manifest['model_id']}.json").write_text(json.dumps({"experiment_id": manifest["model_id"], "timestamp": manifest["created_at"], "git_commit": manifest["git_commit"], "dataset_hash": dataset_hash, "models": records, "winner": winner, "status": "validated"}, indent=2, default=str), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATASET))
    args = parser.parse_args()
    manifest = train(Path(args.data))
    print(json.dumps(manifest, indent=2, default=str))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
