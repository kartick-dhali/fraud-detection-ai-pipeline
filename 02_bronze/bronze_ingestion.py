"""Bronze ingestion — reads raw CSV from S3, adds metadata, saves as Delta table.

Bronze layer rules:
  - NEVER modify original data
  - Add metadata columns only (_ingestion_timestamp, _source_file, etc.)
  - Keep ALL rows (even bad ones — Silver layer will clean)
  - Save as Delta format for versioning + fast reads
  - Register in Unity Catalog (fraud.test.bronze_transactions)

Usage in Databricks notebook:
    %run ./02_bronze/bronze_ingestion
    run_bronze(spark)
"""

from __future__ import annotations

import json
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

# ── CONFIG ───────────────────────────────────────────────────────────────────
BUCKET      = "s3://fraud-transection-detection-nabo"
ENVIRONMENT = "test"
CATALOG     = "fraud"

# Paths
CSV_PATH      = f"{BUCKET}/{ENVIRONMENT}/raw-landing/banking_transaction/bank_transactions_data_2.csv"
CONFIG_PATH   = f"{BUCKET}/{ENVIRONMENT}/raw-landing/llm_pipeline_config.json"
BRONZE_PATH   = f"{BUCKET}/{ENVIRONMENT}/bronze/transactions/"
TABLE_NAME    = f"{CATALOG}.{ENVIRONMENT}.bronze_transactions"


# ── STEP 1: Load pipeline config saved by llm_data_profiler ─────────────────
def load_config(spark_session: SparkSession) -> dict:
    """Read llm_pipeline_config.json from S3."""
    try:
        raw = dbutils.fs.head(CONFIG_PATH, 100_000)  # noqa: F821
        config = json.loads(raw)
        print(f"✅ Config loaded from: {CONFIG_PATH}")
        return config
    except Exception as exc:
        print(f"⚠️  Could not load config: {exc} — using defaults")
        return {}


# ── STEP 2: Read raw CSV from S3 ─────────────────────────────────────────────
def read_raw_csv(spark_session: SparkSession) -> object:
    """Read CSV exactly as-is. inferSchema=false keeps everything as string (Bronze rule)."""
    print(f"\n📖 Reading CSV from:\n   {CSV_PATH}")
    df = (
        spark_session.read
        .option("header", "true")
        .option("inferSchema", "false")   # ← keep all as string in Bronze
        .option("encoding", "UTF-8")
        .csv(CSV_PATH)
    )
    count = df.count()
    print(f"✅ Raw rows loaded : {count:,}")
    print(f"✅ Columns found   : {len(df.columns)}")
    return df


# ── STEP 3: Add Bronze metadata columns ──────────────────────────────────────
def add_metadata(df: object) -> object:
    """Add tracking columns. Never touch original columns."""
    print("\n➕ Adding metadata columns...")
    df_bronze = (
        df
        .withColumn("_ingestion_timestamp", F.current_timestamp())
        .withColumn("_ingestion_date",      F.current_date())
        .withColumn("_source_file",         F.lit("bank_transactions_data_2.csv"))
        .withColumn("_source_path",         F.lit(CSV_PATH))
        .withColumn("_environment",         F.lit(ENVIRONMENT))
        .withColumn("_pipeline_version",    F.lit("1.0"))
        .withColumn("_catalog",             F.lit(CATALOG))
    )
    print("✅ Metadata columns added:")
    print("   → _ingestion_timestamp  : when data was ingested")
    print("   → _ingestion_date       : date of ingestion")
    print("   → _source_file          : original CSV filename")
    print("   → _source_path          : full S3 path of source")
    print("   → _environment          : test/prod")
    print("   → _pipeline_version     : version tracking")
    print("   → _catalog              : Unity Catalog name")
    return df_bronze


# ── STEP 4: Write as Delta table to S3 ───────────────────────────────────────
def write_bronze_delta(df: object) -> None:
    """Write to S3 as Delta format. overwrite = safe to re-run."""
    print(f"\n💾 Writing Bronze Delta table to:\n   {BRONZE_PATH}")
    (
        df.write
        .format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .save(BRONZE_PATH)
    )
    print("✅ Delta table written successfully!")


# ── STEP 5: Register in Unity Catalog ────────────────────────────────────────
def register_in_catalog(spark_session: SparkSession) -> None:
    """Register Delta table in Unity Catalog so it appears in Databricks UI."""
    print(f"\n📚 Registering in Unity Catalog as: {TABLE_NAME}")

    # Make sure schema exists
    spark_session.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{ENVIRONMENT}")

    # Register table (or refresh if exists)
    spark_session.sql(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME}
        USING DELTA
        LOCATION '{BRONZE_PATH}'
    """)
    print(f"✅ Table registered: {TABLE_NAME}")


# ── STEP 6: Validate ─────────────────────────────────────────────────────────
def validate_bronze(spark_session: SparkSession) -> None:
    """Quick sanity check — show row count + sample."""
    print(f"\n🔍 Validating Bronze table...")

    count = spark_session.sql(f"SELECT COUNT(*) as total FROM {TABLE_NAME}").collect()[0][0]
    print(f"✅ Total rows in Bronze : {count:,}")

    print("\n📋 Sample rows (5):")
    spark_session.sql(f"""
        SELECT
            TransactionID,
            AccountID,
            TransactionAmount,
            TransactionDate,
            TransactionType,
            _ingestion_timestamp,
            _source_file
        FROM {TABLE_NAME}
        LIMIT 5
    """).show(truncate=False)

    print("📋 Full Schema:")
    spark_session.sql(f"DESCRIBE TABLE {TABLE_NAME}").show(100, truncate=False)


# ── MAIN: run_bronze() ────────────────────────────────────────────────────────
def run_bronze(spark_session: SparkSession) -> None:
    """Run full Bronze ingestion pipeline. Call this from your Databricks notebook."""

    print("=" * 60)
    print("🥉 BRONZE INGESTION PIPELINE STARTED")
    print("=" * 60)

    # Step 1: Load config
    config = load_config(spark_session)

    # Step 2: Read raw CSV
    df_raw = read_raw_csv(spark_session)

    # Step 3: Add metadata
    df_bronze = add_metadata(df_raw)

    # Step 4: Write Delta to S3
    write_bronze_delta(df_bronze)

    # Step 5: Register in Unity Catalog
    register_in_catalog(spark_session)

    # Step 6: Validate
    validate_bronze(spark_session)

    print("\n" + "=" * 60)
    print("🎉 BRONZE INGESTION COMPLETE!")
    print("=" * 60)
    print(f"""
📍 Bronze Delta table location:
   {BRONZE_PATH}

📚 Unity Catalog table:
   {TABLE_NAME}

📊 What Bronze contains:
   → ALL {config.get('record_count', '2,512')} original rows (nothing removed)
   → ALL 16 original columns (nothing changed)
   → 7 new metadata columns added

⏭️  NEXT → 03_silver_cleaning notebook!
   → Fix data types (string → double/timestamp/integer)
   → Remove bad rows using quarantine rules
   → Mask PII columns
    """)


if __name__ == "__main__":
    run_bronze(spark)  # noqa: F821  (spark injected by Databricks)
