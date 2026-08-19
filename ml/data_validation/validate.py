"""CLI for validating the approved flight-price dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd

from .checks import validate_dataframe


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Clean_Dataset.csv")
    parser.add_argument("--report", default="ml_artifacts/data_validation_report.json")
    args = parser.parse_args()
    df = pd.read_csv(args.data)
    report = validate_dataframe(df)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    print(json.dumps(report.to_dict(), indent=2))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
