"""Auto Loader Bronze ingestion for the fraud landing zone.

Bronze is intentionally permissive: it captures every raw record, preserves unexpected
fields in a rescue column, and records schema history so later layers can tighten quality.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

BUCKET_NAME = "fraud-transection-detection"
CONFIG_PATH = Path("config/llm_pipeline_config.json")

try:
    dbutils.widgets.dropdown("environment", "test", ["dev", "test", "prod"])
    ENVIRONMENT = dbutils.widgets.get("environment")
except Exception:
    ENVIRONMENT = os.getenv("ENVIRONMENT", "test")


def load_schema_hints(config_path: Path = CONFIG_PATH) -> str:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    hints = config.get("schema_hints", {})
    return ", ".join(f"{column} {dtype}" for column, dtype in hints.items())


def build_paths() -> dict[str, str]:
    base = f"s3://{BUCKET_NAME}/{ENVIRONMENT}"
    return {
        "landing": f"{base}/raw-landing/",
        "schema": f"{base}/_schemas/bronze_autoloader/",
        "checkpoint": f"{base}/_checkpoints/bronze_autoloader_stream/",
        "table_path": f"{base}/bronze/table/",
    }


def build_stream_reader(spark):
    paths = build_paths()
    return (
        spark.readStream.format("cloudFiles")
        .option("cloudFiles.format", "json")
        .option("cloudFiles.schemaHints", load_schema_hints())
        .option("cloudFiles.schemaLocation", paths["schema"])
        .option("cloudFiles.inferColumnTypes", "true")
        .option("cloudFiles.schemaEvolutionMode", "addNewColumns")
        .option("rescuedDataColumn", "_rescued_data")
        .option("mergeSchema", "true")
        .load(paths["landing"])
    )


def run_bronze_ingestion(spark) -> None:
    paths = build_paths()
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS fraud_{ENVIRONMENT}")

    (
        build_stream_reader(spark)
        .writeStream.format("delta")
        .option("checkpointLocation", paths["checkpoint"])
        .option("path", paths["table_path"])
        .outputMode("append")
        .trigger(availableNow=True)
        .toTable(f"fraud_{ENVIRONMENT}.bronze")
    )


if __name__ == "__main__":
    print("Run this script inside Databricks with an active Spark session.")
    print(build_paths())
