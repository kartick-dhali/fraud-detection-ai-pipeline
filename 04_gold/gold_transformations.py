"""Gold transformations for fraud analytics, ML features, and executive summaries.

Gold tables are curated for consumption. This layer focuses on repeatable business metrics,
idempotent Delta writes, and historical risk tracking with SCD Type 2 semantics.
"""

from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI

BUCKET_NAME = "fraud-transection-detection"
SUMMARY_OUTPUT = Path("/tmp/fraud_gold_summary.txt")

try:
    dbutils.widgets.dropdown("environment", "test", ["dev", "test", "prod"])
    ENVIRONMENT = dbutils.widgets.get("environment")
except Exception:
    ENVIRONMENT = os.getenv("ENVIRONMENT", "test")


def call_llm(prompt: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            print(f"OpenAI summary failed, using fallback: {exc}")
    return "Fallback summary: review merchant spikes, high-risk users, and rolling velocity anomalies."


def gold_paths() -> dict[str, str]:
    base = f"s3://{BUCKET_NAME}/{ENVIRONMENT}/gold"
    return {
        "ml_features": f"{base}/ml_features/",
        "user_summary": f"{base}/user_summary/",
        "merchant_summary": f"{base}/merchant_summary/",
        "user_risk_scd2": f"{base}/user_risk_scd2/",
    }


def build_feature_sql() -> str:
    return """
    SELECT
        tr_id,
        tr_date,
        account_hash,
        merchant_hash,
        amount,
        is_fraud,
        transaction_type,
        location,
        lag(amount) OVER (PARTITION BY account_hash ORDER BY tr_timestamp) AS prev_amount,
        avg(amount) OVER (
            PARTITION BY account_hash
            ORDER BY tr_timestamp
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW
        ) AS rolling_avg_amount_7,
        rank() OVER (PARTITION BY tr_date ORDER BY amount DESC) AS daily_amount_rank,
        percent_rank() OVER (PARTITION BY tr_date ORDER BY amount) AS daily_amount_percent_rank
    FROM delta.`{silver_path}`
    """


def create_gold_tables(spark) -> None:
    paths = gold_paths()
    silver_path = f"s3://{BUCKET_NAME}/{ENVIRONMENT}/silver/data/"
    feature_sql = build_feature_sql().format(silver_path=silver_path)
    features_df = spark.sql(feature_sql)

    features_df.write.format("delta").mode("overwrite").save(paths["ml_features"])

    spark.sql(
        f"""
        CREATE OR REPLACE TABLE fraud_{ENVIRONMENT}.ml_features
        USING DELTA
        LOCATION '{paths['ml_features']}'
        """
    )

    user_summary = spark.sql(
        f"""
        SELECT account_hash,
               count(*) AS transaction_count,
               sum(amount) AS total_amount,
               avg(amount) AS avg_amount,
               sum(is_fraud) AS fraud_count
        FROM delta.`{silver_path}`
        GROUP BY account_hash
        """
    )
    user_summary.write.format("delta").mode("overwrite").save(paths["user_summary"])

    merchant_summary = spark.sql(
        f"""
        SELECT merchant_hash,
               count(*) AS transaction_count,
               sum(amount) AS total_amount,
               sum(is_fraud) AS fraud_count,
               rank() OVER (ORDER BY sum(is_fraud) DESC, sum(amount) DESC) AS fraud_rank
        FROM delta.`{silver_path}`
        GROUP BY merchant_hash
        """
    )
    merchant_summary.write.format("delta").mode("overwrite").save(paths["merchant_summary"])

    spark.sql(
        f"""
        CREATE TABLE IF NOT EXISTS fraud_{ENVIRONMENT}.user_risk_scd2 (
            account_hash STRING,
            risk_score DOUBLE,
            valid_from TIMESTAMP,
            valid_to TIMESTAMP,
            is_current BOOLEAN
        ) USING DELTA LOCATION '{paths['user_risk_scd2']}'
        """
    )

    spark.sql(
        f"""
        MERGE INTO fraud_{ENVIRONMENT}.user_risk_scd2 AS target
        USING (
            SELECT account_hash,
                   cast(fraud_count / nullif(transaction_count, 0) AS DOUBLE) AS risk_score,
                   current_timestamp() AS valid_from
            FROM delta.`{paths['user_summary']}`
        ) AS source
        ON target.account_hash = source.account_hash AND target.is_current = TRUE
        WHEN MATCHED AND target.risk_score <> source.risk_score THEN
          UPDATE SET valid_to = current_timestamp(), is_current = FALSE
        WHEN NOT MATCHED THEN
          INSERT (account_hash, risk_score, valid_from, valid_to, is_current)
          VALUES (source.account_hash, source.risk_score, source.valid_from, TIMESTAMP '9999-12-31 00:00:00', TRUE)
        """
    )

    summary_prompt = (
        "Summarize fraud patterns from these Gold tables for banking executives: "
        f"ml_features={paths['ml_features']}, user_summary={paths['user_summary']}, merchant_summary={paths['merchant_summary']}"
    )
    summary_text = call_llm(summary_prompt)
    SUMMARY_OUTPUT.write_text(summary_text, encoding="utf-8")
    print(summary_text)


if __name__ == "__main__":
    print(gold_paths())
