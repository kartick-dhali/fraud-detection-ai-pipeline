"""Convert the Kaggle CSV into daily JSONL landing files.

This notebook-style script groups records by transaction date and writes one file per day.
Unique file names are used because checkpointed file ingestion works best when landed objects
are immutable and never overwritten.
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pandas as pd

BUCKET_NAME = "fraud-transection-detection"
LOCAL_OUTPUT_ROOT = Path("/tmp/fraud-jsonl-preview")
CONFIG_PATH = Path("config/llm_pipeline_config.json")

try:
    dbutils.widgets.dropdown("environment", "test", ["dev", "test", "prod"])
    ENVIRONMENT = dbutils.widgets.get("environment")
except Exception:
    ENVIRONMENT = os.getenv("ENVIRONMENT", "test")


def load_config(config_path: Path = CONFIG_PATH) -> dict:
    if not config_path.exists():
        raise FileNotFoundError(
            f"Missing {config_path}. Run 01_landing/llm_data_profiler.py before landing JSONL files."
        )
    return json.loads(config_path.read_text(encoding="utf-8"))


def normalize_dataframe(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    mapped = config.get("mapped_columns", {})
    df = df.rename(columns=mapped)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["landing_partition"] = df["transaction_date"].dt.strftime("%Y-%m-%d")
    return df


def inject_bad_records(df: pd.DataFrame) -> pd.DataFrame:
    """Add a few intentionally bad rows so rescue and quarantine paths are testable."""

    broken = df.head(2).copy()
    if not broken.empty:
        broken.loc[:, "amount"] = [None, "bad-amount"][: len(broken)]
        broken.loc[:, "transaction_date"] = [pd.NaT] * len(broken)
    return pd.concat([df, broken], ignore_index=True)


def output_path_for(day: str) -> str:
    unique_name = f"transactions_{day}_{uuid.uuid4().hex}.jsonl"
    return f"s3://{BUCKET_NAME}/{ENVIRONMENT}/raw-landing/{day}/{unique_name}"


def persist_jsonl(target_path: str, payload: str) -> None:
    """Write through dbutils.fs.put in Databricks, or fall back to a local preview file."""

    try:
        dbutils.fs.put(target_path, payload, overwrite=False)
        print(f"Wrote {target_path}")
    except Exception:
        local_path = LOCAL_OUTPUT_ROOT / target_path.replace("s3://", "")
        local_path.parent.mkdir(parents=True, exist_ok=True)
        local_path.write_text(payload, encoding="utf-8")
        print(f"Previewed {local_path}")


def land_jsonl(dataset_path: str) -> None:
    config = load_config()
    df = pd.read_csv(dataset_path)
    df = normalize_dataframe(df, config)
    df = inject_bad_records(df)

    for day, partition_df in df.groupby("landing_partition", dropna=False):
        safe_day = day if isinstance(day, str) and day else "unknown-date"
        target_path = output_path_for(safe_day)
        payload = partition_df.drop(columns=["landing_partition"]).to_json(
            orient="records",
            lines=True,
            date_format="iso",
        )
        persist_jsonl(target_path, payload)


if __name__ == "__main__":
    dataset_path = os.getenv("FRAUD_DATASET_PATH", "data/raw_csv/bank_transactions.csv")
    land_jsonl(dataset_path)
