"""Validate the complete ML release bundle without changing application code."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from ml.data_validation import FEATURES, TARGET, validate_dataframe

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "ml_artifacts"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_release(data_path: Path | None = None) -> tuple[bool, list[str]]:
    failures: list[str] = []
    manifest_path = ARTIFACTS / "model_manifest.json"
    schema_path = ARTIFACTS / "feature_schema.json"
    evaluation_path = ARTIFACTS / "evaluation_report.json"
    checksums_path = ARTIFACTS / "checksums.json"
    artifact_path = ARTIFACTS / "models/flight_price_model.joblib"
    for path in [manifest_path, schema_path, evaluation_path, checksums_path, artifact_path]:
        if not path.exists():
            failures.append(f"missing: {path.relative_to(ROOT)}")
    if failures:
        return False, failures
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        checksums = json.loads(checksums_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return False, [f"invalid JSON: {exc}"]
    required_manifest = ["model_id", "model_type", "artifact", "artifact_sha256", "dataset_hash", "feature_schema_hash", "features", "target", "cv_metrics", "test_metrics", "release_status"]
    failures.extend(f"manifest missing field: {field}" for field in required_manifest if field not in manifest)
    if manifest.get("features") != FEATURES:
        failures.append("manifest feature order does not match canonical contract")
    if manifest.get("target") != TARGET:
        failures.append("manifest target does not match canonical target")
    if schema.get("features") is None or schema.get("target") != TARGET:
        failures.append("feature schema is invalid")
    for name, expected in checksums.items():
        path = ARTIFACTS / name
        if not path.exists():
            failures.append(f"checksum target missing: {name}")
        elif sha256(path) != expected:
            failures.append(f"checksum mismatch: {name}")
    if manifest.get("artifact_sha256") != sha256(artifact_path):
        failures.append("manifest artifact checksum mismatch")
    try:
        model = joblib.load(artifact_path)
        sample = pd.read_csv(data_path or (ROOT / "Clean_Dataset.csv")).head(3)[FEATURES]
        predictions = np.asarray(model.predict(sample), dtype=float)
        if predictions.shape != (3,) or not np.isfinite(predictions).all():
            failures.append("inference compatibility failed: non-finite or malformed predictions")
    except Exception as exc:
        failures.append(f"model loading/inference failed: {exc}")
    test_metrics = evaluation.get("test_metrics", {})
    for key in ["MAE", "RMSE", "R2"]:
        if key not in test_metrics or not np.isfinite(test_metrics[key]):
            failures.append(f"evaluation metric invalid: {key}")
    if test_metrics.get("R2", -np.inf) <= 0:
        failures.append("provisional quality gate failed: test R2 must be positive")
    if abs(evaluation.get("cv_metrics", {}).get("CV_R2", 0) - test_metrics.get("R2", 0)) > .25:
        failures.append("provisional quality gate failed: CV/test R2 gap exceeds 0.25")
    if data_path is not None:
        data_report = validate_dataframe(pd.read_csv(data_path))
        if not data_report.passed:
            failures.append("data validation failed: " + "; ".join(data_report.issues))
    return not failures, failures


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=None)
    args = parser.parse_args()
    passed, failures = validate_release(Path(args.data) if args.data else None)
    print("=" * 48)
    print("FLIGHT PRICE PREDICTION ML RELEASE")
    print("=" * 48)
    print("RELEASE STATUS:", "PASS" if passed else "FAIL")
    if failures:
        print("REASONS:")
        print("\n".join(f"- {item}" for item in failures))
    return 0 if passed else 1

if __name__ == "__main__":
    raise SystemExit(main())
