"""Profile the bank transactions CSV from S3 and generate a reusable pipeline config.

Runs in Databricks. Uses OpenAI when OPENAI_API_KEY is set, otherwise falls back
to a deterministic profiler — so Stage 1 never blocks the rest of the pipeline.

Usage in Databricks notebook:
    %run ./01_landing/llm_data_profiler
"""

from __future__ import annotations

import json
import os
from typing import Any

# ── CONFIG ──────────────────────────────────────────────────────────────────
BUCKET        = "s3://fraud-transection-detection-nabo"
ENVIRONMENT   = "test"
CSV_PATH      = f"{BUCKET}/{ENVIRONMENT}/raw-landing/banking_transaction/bank_transactions_data_2.csv"
CONFIG_PATH   = f"{BUCKET}/{ENVIRONMENT}/raw-landing/llm_pipeline_config.json"

# All 16 columns in your dataset + their correct types
SCHEMA_HINTS: dict[str, str] = {
    "TransactionID"          : "string",
    "AccountID"              : "string",
    "TransactionAmount"      : "double",     # fix: was string
    "TransactionDate"        : "timestamp",  # fix: was string
    "TransactionType"        : "string",
    "Location"               : "string",
    "DeviceID"               : "string",
    "IP Address"             : "string",
    "MerchantID"             : "string",
    "Channel"                : "string",
    "CustomerAge"            : "integer",    # fix: was string
    "CustomerOccupation"     : "string",
    "TransactionDuration"    : "integer",    # fix: was string
    "LoginAttempts"          : "integer",    # fix: was string
    "AccountBalance"         : "double",     # fix: was string
    "PreviousTransactionDate": "timestamp",  # fix: was string
}

# Columns that contain private/sensitive customer data
PII_COLUMNS = ["AccountID", "MerchantID", "Location", "DeviceID", "IP Address"]

# Rows matching any of these rules will be quarantined in Silver layer
QUARANTINE_RULES = [
    "TransactionID IS NULL",
    "TransactionAmount IS NULL",
    "TransactionDate IS NULL",
    "AccountID IS NULL",
    "CAST(TransactionAmount AS DOUBLE) < 0",
    "CAST(LoginAttempts AS INT) < 0",
]


# ── LLM CALL (optional) ─────────────────────────────────────────────────────
def call_openai(prompt: str) -> str:
    """Call OpenAI if API key is available, otherwise return empty string."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return ""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""
    except Exception as exc:
        print(f"⚠️ OpenAI unavailable: {exc} — using deterministic fallback")
        return ""


# ── DETERMINISTIC PROFILER ───────────────────────────────────────────────────
def deterministic_profile(columns: list[str], record_count: int) -> dict[str, Any]:
    """Generate safe defaults without any LLM — always works."""
    return {
        "dataset_path"    : CSV_PATH,
        "record_count"    : record_count,
        "column_count"    : len(columns),
        "columns"         : columns,
        "mapped_columns"  : {c: c.lower().replace(" ", "_") for c in columns},
        "pii_columns"     : [c for c in columns if c in PII_COLUMNS],
        "schema_hints"    : {c: SCHEMA_HINTS.get(c, "string") for c in columns},
        "quarantine_rules": QUARANTINE_RULES,
        "generated_by"    : "deterministic_fallback",
        "environment"     : ENVIRONMENT,
    }


# ── MAIN ─────────────────────────────────────────────────────────────────────
def run_profiler(spark_session) -> dict[str, Any]:
    """Read CSV from S3, profile it, save config back to S3."""

    print(f"📖 Reading CSV from: {CSV_PATH}")
    df_spark = (
        spark_session.read
        .option("header", "true")
        .option("inferSchema", "false")
        .csv(CSV_PATH)
    )
    record_count = df_spark.count()
    columns      = df_spark.columns
    print(f"✅ Loaded {record_count:,} rows × {len(columns)} columns")

    # Try LLM enrichment (optional)
    df_pandas  = df_spark.limit(5).toPandas()
    prompt     = (
        "You are profiling a bank transaction fraud dataset. "
        "Return JSON with mapped_columns, pii_columns, schema_hints, quarantine_rules. "
        f"Columns: {columns}. Sample: {df_pandas.to_dict(orient='records')}"
    )
    llm_response = call_openai(prompt)

    profile = deterministic_profile(columns, record_count)

    if llm_response:
        try:
            candidate = json.loads(llm_response)
            profile.update({k: v for k, v in candidate.items() if v})
            profile["generated_by"] = "openai"
            print("✅ LLM enrichment applied!")
        except Exception as exc:
            print(f"⚠️ LLM JSON parse failed: {exc} — using deterministic profile")

    # Save config to S3
    dbutils.fs.put(CONFIG_PATH, json.dumps(profile, indent=2), overwrite=True)  # noqa: F821
    print(f"✅ Config saved to: {CONFIG_PATH}")

    return profile


# ── PRINT SUMMARY ────────────────────────────────────────────────────────────
def print_summary(profile: dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("🎉 LLM DATA PROFILER COMPLETE!")
    print("=" * 60)
    print(f"📊 Rows        : {profile['record_count']:,}")
    print(f"📋 Columns     : {profile['column_count']}")
    print(f"🔒 PII columns : {profile['pii_columns']}")
    print(f"🤖 Generated by: {profile['generated_by']}")
    print("\n⚠️  Type fixes (string → correct type):")
    for col, dtype in profile["schema_hints"].items():
        if dtype != "string":
            print(f"   → {col:30} : string → {dtype}")
    print("\n🚫 Quarantine rules:")
    for rule in profile["quarantine_rules"]:
        print(f"   → {rule}")
    print(f"\n💾 Config saved → {CONFIG_PATH}")
    print("⏭️  NEXT → 02_bronze_ingestion notebook!")


if __name__ == "__main__":
    profile = run_profiler(spark)  # noqa: F821  (spark is injected by Databricks)
    print_summary(profile)
