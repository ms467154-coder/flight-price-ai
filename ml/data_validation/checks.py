"""Data validation checks. Checks report issues without silently changing data."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import pandas as pd

from .schema import (
    CATEGORY_FIELDS, DOMAIN_CONSTRAINTS, EXPECTED_COLUMNS, FEATURES, NUMERICAL_FEATURES,
    TARGET,
)

@dataclass
class ValidationReport:
    passed: bool
    row_count: int
    column_count: int
    missing_total: int
    duplicate_rows: int
    duplicate_identifier_values: int
    issues: list[str]
    warnings: list[str]
    missing_by_column: dict[str, int]
    unexpected_categories: dict[str, list[str]]
    target_summary: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_dataframe(df: pd.DataFrame, strict: bool = True) -> ValidationReport:
    issues: list[str] = []
    warnings: list[str] = []
    missing_by_column = {str(k): int(v) for k, v in df.isna().sum().items()}
    unexpected_categories: dict[str, list[str]] = {}

    missing_columns = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra_columns = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")
    if extra_columns:
        warnings.append(f"Unexpected columns will not be used: {extra_columns}")

    for column in FEATURES:
        if column in df.columns and df[column].dtype == "object":
            null_rate = float(df[column].isna().mean())
            if null_rate > 0.20:
                warnings.append(f"High missingness in {column}: {null_rate:.2%}")
            if column in CATEGORY_FIELDS:
                observed = set(df[column].dropna().astype(str).unique())
                unexpected = sorted(observed.difference(CATEGORY_FIELDS[column]))
                if unexpected:
                    unexpected_categories[column] = unexpected
                    warnings.append(f"Observed categories outside reference vocabulary in {column}: {unexpected}")

    if TARGET not in df.columns:
        issues.append("Target column 'price' is missing")
    else:
        if not pd.api.types.is_numeric_dtype(df[TARGET]):
            issues.append("Target column 'price' must be numeric")
        target_missing = int(df[TARGET].isna().sum())
        if target_missing:
            issues.append(f"Target contains {target_missing} missing values")
        if pd.api.types.is_numeric_dtype(df[TARGET]):
            negative = int((df[TARGET] < DOMAIN_CONSTRAINTS[TARGET]["min"]).sum())
            if negative:
                issues.append(f"Target contains {negative} negative prices")

    for column in NUMERICAL_FEATURES:
        if column in df.columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                issues.append(f"Numerical feature {column} is not numeric")
            elif df[column].isna().mean() > 0.20:
                warnings.append(f"High missingness in numerical feature {column}: {df[column].isna().mean():.2%}")

    for column, constraint in DOMAIN_CONSTRAINTS.items():
        if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
            invalid = int((df[column] < constraint["min"]).sum())
            if invalid:
                issues.append(f"{column} has {invalid} values below {constraint['min']}")

    duplicate_rows = int(df.duplicated().sum())
    if duplicate_rows:
        warnings.append(f"Found {duplicate_rows} duplicate rows; no rows were deleted automatically")
    duplicate_identifier_values = 0
    if "Unnamed: 0" in df.columns:
        duplicate_identifier_values = int(df["Unnamed: 0"].duplicated().sum())
        if duplicate_identifier_values:
            warnings.append(f"Identifier column contains {duplicate_identifier_values} duplicate values")

    target_summary: dict[str, Any] = {}
    if TARGET in df.columns and pd.api.types.is_numeric_dtype(df[TARGET]) and len(df):
        target_summary = {
            "min": float(df[TARGET].min()), "max": float(df[TARGET].max()),
            "mean": float(df[TARGET].mean()), "median": float(df[TARGET].median()),
            "std": float(df[TARGET].std()), "p01": float(df[TARGET].quantile(.01)),
            "p99": float(df[TARGET].quantile(.99)),
        }
        if target_summary["max"] > target_summary["p99"] * 5:
            warnings.append("Target has extreme upper-tail values; inspect before any outlier policy")

    if strict and missing_columns:
        passed = False
    else:
        passed = not issues
    return ValidationReport(
        passed=passed, row_count=len(df), column_count=len(df.columns),
        missing_total=int(df.isna().sum().sum()), duplicate_rows=duplicate_rows,
        duplicate_identifier_values=duplicate_identifier_values, issues=issues,
        warnings=warnings, missing_by_column=missing_by_column,
        unexpected_categories=unexpected_categories, target_summary=target_summary,
    )
