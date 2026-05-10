"""Reusable debugging helpers for the fraud lakehouse.

The functions are intentionally small and opinionated so operators can call them from a
notebook cell while troubleshooting without reading the entire pipeline implementation.
"""

from __future__ import annotations

import os
from typing import Iterable


def validate_environment_widget(environment: str, allowed: Iterable[str] = ("dev", "test", "prod")) -> dict:
    """Confirm that notebook code is isolated per environment.

    Environment scoping is critical because Bronze/Silver/Gold paths use the environment in
    their S3 prefix. A typo here can leak test data into production paths.
    """

    allowed_values = tuple(allowed)
    return {
        "environment": environment,
        "is_valid": environment in allowed_values,
        "allowed_values": allowed_values,
        "default": "test",
    }


def describe_stream_paths(environment: str) -> dict:
    """Return the canonical landing, schema, and checkpoint locations.

    Streaming incidents are often path incidents. Showing these paths together makes
    checkpoint collisions and schema-location reuse obvious.
    """

    base = f"s3://fraud-transection-detection/{environment}"
    return {
        "landing": f"{base}/raw-landing/",
        "bronze_checkpoint": f"{base}/_checkpoints/bronze_autoloader_stream/",
        "silver_checkpoint": f"{base}/_checkpoints/silver_cleaning_stream/",
        "schema_location": f"{base}/_schemas/bronze_autoloader/",
    }


def preview_quarantine_logic(rules: list[str]) -> str:
    """Render CASE WHEN logic so stewards can validate LLM output before deployment."""

    if not rules:
        rules = ["amount IS NULL", "transaction_date IS NULL", "is_fraud NOT IN (0, 1)"]
    return "CASE " + " ".join(f"WHEN {rule} THEN '{rule}'" for rule in rules) + " ELSE NULL END"


def summarize_feature_nulls(rows: list[dict], monitored_columns: list[str]) -> dict:
    """Count feature nulls before ML training.

    The model step is expensive compared with this summary, so we fail fast by making data
    quality gaps visible before LightGBM is fit.
    """

    summary = {}
    for column in monitored_columns:
        total = len(rows)
        null_count = sum(1 for row in rows if row.get(column) is None)
        summary[column] = {
            "null_count": null_count,
            "null_ratio": (null_count / total) if total else 0.0,
        }
    return summary


def explain_openai_failure(exc: Exception | str) -> dict:
    """Translate a raw LLM error into operator-friendly guidance."""

    message = str(exc)
    return {
        "error": message,
        "has_api_key": bool(os.getenv("OPENAI_API_KEY")),
        "suggested_action": (
            "Retry with OPENAI_API_KEY configured, or use Ollama/deterministic fallback "
            "until network or secret access is restored."
        ),
    }
