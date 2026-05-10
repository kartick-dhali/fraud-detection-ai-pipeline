"""Silver cleaning and quarantine split for the fraud lakehouse.

Silver is where raw ingest becomes analytically trustworthy. The code uses foreachBatch so
good records, quarantine records, and operator summaries can all be produced from one stream.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from openai import OpenAI

BUCKET_NAME = "fraud-transection-detection"
CONFIG_PATH = Path("config/llm_pipeline_config.json")

try:
    dbutils.widgets.dropdown("environment", "test", ["dev", "test", "prod"])
    ENVIRONMENT = dbutils.widgets.get("environment")
except Exception:
    ENVIRONMENT = os.getenv("ENVIRONMENT", "test")


def call_llm(prompt: str) -> str:
    """Generate operator-facing text, but never let an API outage stop the stream."""

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            response = client.responses.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                input=prompt,
                temperature=0,
            )
            return response.output_text
        except Exception as exc:
            print(f"OpenAI call failed, using fallback: {exc}")
    return "Manual fallback: quarantine rules were generated from deterministic validations."


def load_rules() -> list[str]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    return config.get("quarantine_rules", [])


def build_quarantine_case_when(rules: list[str]) -> str:
    """Build CASE WHEN SQL that explains why records were quarantined."""

    if not rules:
        rules = ["amount IS NULL", "transaction_date IS NULL", "is_fraud NOT IN (0, 1)"]
    clauses = [f"WHEN {rule} THEN '{rule}'" for rule in rules]
    return "CASE " + " ".join(clauses) + " ELSE NULL END AS quarantine_reason"


def silver_paths() -> dict[str, str]:
    base = f"s3://{BUCKET_NAME}/{ENVIRONMENT}"
    return {
        "checkpoint": f"{base}/_checkpoints/silver_cleaning_stream/",
        "silver_data": f"{base}/silver/data/",
        "silver_quarantine": f"{base}/silver/quarantine/",
    }


def process_micro_batch(batch_df, batch_id: int) -> None:
    """Split clean and quarantined data in a single cached micro-batch.

    Caching is important here because the same input is written to two outputs and summarized
    for operators. Without cache/unpersist, Spark could recompute the batch multiple times.
    """

    rules = load_rules()
    quarantine_sql = build_quarantine_case_when(rules)
    batch_df.createOrReplaceTempView("bronze_batch")
    transformed = batch_df.sparkSession.sql(
        f"""
        SELECT
            sha2(CAST(AccountID AS STRING), 256) AS account_hash,
            sha2(CAST(MerchantID AS STRING), 256) AS merchant_hash,
            CAST(TransactionID AS STRING) AS tr_id,
            try_cast(TransactionDate AS TIMESTAMP) AS tr_timestamp,
            date(try_cast(TransactionDate AS TIMESTAMP)) AS tr_date,
            try_cast(Amount AS DOUBLE) AS amount,
            CAST(TransactionType AS STRING) AS transaction_type,
            CAST(Location AS STRING) AS location,
            try_cast(IsFraud AS INT) AS is_fraud,
            {quarantine_sql}
        FROM bronze_batch
        """
    ).dropDuplicates(["tr_id"])

    transformed.cache()
    paths = silver_paths()
    good_records = transformed.filter("quarantine_reason IS NULL")
    bad_records = transformed.filter("quarantine_reason IS NOT NULL")

    (
        good_records.write.format("delta")
        .mode("append")
        .partitionBy("tr_date")
        .save(paths["silver_data"])
    )
    bad_records.write.format("delta").mode("append").save(paths["silver_quarantine"])

    summary_prompt = (
        "Explain this quarantine summary in plain English for a data steward: "
        f"batch_id={batch_id}, bad_count={bad_records.count()}, rules={rules}"
    )
    print(call_llm(summary_prompt))
    transformed.unpersist()


def run_silver_stream(spark) -> None:
    paths = silver_paths()
    (
        spark.readStream.table(f"fraud_{ENVIRONMENT}.bronze")
        .writeStream.foreachBatch(process_micro_batch)
        .option("checkpointLocation", paths["checkpoint"])
        .trigger(availableNow=True)
        .start()
        .awaitTermination()
    )


if __name__ == "__main__":
    print(build_quarantine_case_when(load_rules() if CONFIG_PATH.exists() else []))
