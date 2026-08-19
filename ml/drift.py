"""Offline drift statistics; alert thresholds must be selected by operators."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    expected_counts, _ = np.histogram(expected, bins=edges)
    actual_counts, _ = np.histogram(actual, bins=edges)
    expected_rate = np.clip(expected_counts / max(len(expected), 1), 1e-6, None)
    actual_rate = np.clip(actual_counts / max(len(actual), 1), 1e-6, None)
    return float(np.sum((actual_rate - expected_rate) * np.log(actual_rate / expected_rate)))


def categorical_shift(expected: pd.Series, actual: pd.Series) -> float:
    categories = sorted(set(expected.dropna().astype(str)) | set(actual.dropna().astype(str)))
    p = expected.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    q = actual.astype(str).value_counts(normalize=True).reindex(categories, fill_value=0).to_numpy()
    return float(0.5 * np.abs(p - q).sum())


def compare(training: pd.DataFrame, production: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {"numerical": {}, "categorical": {}}
    for column in training.columns.intersection(production.columns):
        if pd.api.types.is_numeric_dtype(training[column]) and pd.api.types.is_numeric_dtype(production[column]):
            left = training[column].dropna().to_numpy(dtype=float); right = production[column].dropna().to_numpy(dtype=float)
            if len(left) and len(right):
                statistic, pvalue = ks_2samp(left, right)
                result["numerical"][column] = {"psi": psi(left, right), "ks_statistic": float(statistic), "ks_pvalue": float(pvalue), "training_count": len(left), "production_count": len(right)}
        else:
            result["categorical"][column] = {"total_variation_distance": categorical_shift(training[column], production[column])}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("training_csv")
    parser.add_argument("production_csv")
    parser.add_argument("--output", default="ml_artifacts/drift_report.json")
    args = parser.parse_args()
    report = compare(pd.read_csv(args.training_csv), pd.read_csv(args.production_csv))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
