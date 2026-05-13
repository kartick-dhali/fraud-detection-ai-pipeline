"""Silver Cleaning — foreachBatch pattern (study notes style).

Pattern:
  readStream.table(bronze)
  → foreachBatch → micro_processing()
      → try_cast  (defensive casting)
      → filter    (good rows only)
      → quarantine (bad rows saved separately)
      → write good → silver_data    (partitioned by date)
      → write bad  → silver_quarantine
  → writeStream with checkpointLocation

Usage in Databricks notebook:
    %run ./03_silver/silver_cleaning
"""

from pyspark.sql import functions as F
from pyspark.sql.functions import col, expr, current_date

# ── STEP 1: Environment Widget ────────────────────────────────────────────────
dbutils.widgets.dropdown("environment", "test", ["test", "prod"])  # noqa: F821
env = dbutils.widgets.get("environment")                           # noqa: F821

print(f"🌍 Environment : {env}")

# ── STEP 2: All Paths ────────────────────────────────────────────────────────
base_path    = f"s3://fraud-transection-detection-nabo/{env}"
silver_path  = f"{base_path}/silver"
bronze_db    = f"fraud_{env}"
silver_db    = f"fraud_{env}"

print(f"📂 Silver path       : {silver_path}")
print(f"📂 Silver data       : {silver_path}/silver_data")
print(f"📂 Silver quarantine : {silver_path}/silver_quarantine")

# ── STEP 3: Read from Bronze using readStream ─────────────────────────────────
# IMPORTANT: readStream.table() already has schema → cannot redefine using .schema()
# We read directly from the registered Bronze Delta table!
print(f"\n📖 Reading from Bronze table: {bronze_db}.bronze")

bronze_df = spark.readStream.table(f"{bronze_db}.bronze")          # noqa: F821


# ── STEP 4: foreachBatch micro_processing() ───────────────────────────────────
# foreachBatch = processes each micro-batch as a normal DataFrame
# Why foreachBatch?
#   → Can split one batch into MULTIPLE outputs (silver + quarantine)
#   → Can cache/unpersist for performance
#   → More control than plain writeStream
def micro_processing(batch_df, batch_id):
    print(f"\n⚙️  Processing batch_id: {batch_id}")
    print(f"   Incoming rows: {batch_df.count()}")

    # Cache for performance (read batch_df twice: good + bad)
    batch_df.cache()

    # ── GOOD ROWS: Defensive casting + filtering ──────────────────────────────
    # try_cast = safe conversion → returns NULL instead of crashing!
    # alias() required after try_cast
    cleaned_df = batch_df.select(
        expr("try_cast(TransactionAmount  as double)" ).alias("TransactionAmount"),
        expr("try_cast(TransactionDate    as date)"   ).alias("TransactionDate"),
        expr("try_cast(CustomerAge        as int)"    ).alias("CustomerAge"),
        expr("try_cast(LoginAttempts      as int)"    ).alias("LoginAttempts"),
        expr("try_cast(AccountBalance     as double)" ).alias("AccountBalance"),
        expr("try_cast(TransactionDuration as int)"   ).alias("TransactionDuration"),
        expr("try_cast(PreviousTransactionDate as date)").alias("PreviousTransactionDate"),
        col("TransactionID"),
        col("AccountID"),
        col("TransactionType"),
        col("Location"),
        col("DeviceID"),
        col("IP_Address"),
        col("MerchantID"),
        col("Channel"),
        col("CustomerOccupation"),
        col("_rescued_data"),
        col("_ingestion_timestamp"),
        col("_ingestion_date"),
        col("_source_file"),
        col("_environment"),
    ).filter(
        # Good rows: rescued_data NULL + valid amount + valid date + no future dates
        (col("_rescued_data").isNull()) &
        (col("TransactionAmount").isNotNull()) &
        (col("TransactionAmount") > 0) &
        (col("TransactionDate").isNotNull()) &
        (col("TransactionDate") <= current_date())
    ).dropDuplicates(["TransactionID", "AccountID"])

    # ── BAD ROWS: Quarantine ──────────────────────────────────────────────────
    # Bad rows = rescued_data NOT NULL OR amount NULL OR date NULL
    # We KEEP bad rows in quarantine (never delete!)
    # Why? → Investigate later, audit trail, compliance
    bad_df = batch_df.filter(
        (col("_rescued_data").isNotNull()) |
        (col("TransactionAmount").isNull()) |
        (col("TransactionDate").isNull()) |
        (col("TransactionID").isNull()) |
        (col("AccountID").isNull())
    )

    good_count = cleaned_df.count()
    bad_count  = bad_df.count()
    print(f"   ✅ Good rows      : {good_count:,}")
    print(f"   🚫 Quarantined    : {bad_count:,}")

    # ── Write good rows → Silver (partitioned by date) ────────────────────────
    # partitionBy("TransactionDate") → creates folders like:
    #   silver_data/TransactionDate=2026-01-15/part-xxx.parquet
    # Makes date-range queries 10-100x faster!
    if good_count > 0:
        (
            cleaned_df.write
            .format("delta")
            .mode("append")
            .partitionBy("TransactionDate")
            .save(f"{silver_path}/silver_data")
        )
        print(f"   💾 Good rows written → {silver_path}/silver_data")

    # ── Write bad rows → Quarantine ───────────────────────────────────────────
    if bad_count > 0:
        (
            bad_df.write
            .format("delta")
            .mode("append")
            .save(f"{silver_path}/silver_quarantine")
        )
        print(f"   🚫 Bad rows written  → {silver_path}/silver_quarantine")

    # Unpersist cache after use
    batch_df.unpersist()
    print(f"   ✅ Batch {batch_id} done!")


# ── STEP 5: writeStream with foreachBatch ────────────────────────────────────
# trigger(availableNow=True) = process all available data then stop
# checkpointLocation = bookmark — never reprocess same rows!
# foreachBatch = call micro_processing() for each micro-batch
print("\n🚀 Starting Silver writeStream...")

query = (
    bronze_df.writeStream
    .format("delta")
    .trigger(availableNow=True)
    .option("checkpointLocation", f"{silver_path}/checkpoint/")
    .foreachBatch(micro_processing)
    .queryName("silver_processing")
    .start()
)

# Wait for stream to finish
query.awaitTermination()

print("\n" + "=" * 60)
print("🎉 SILVER CLEANING COMPLETE!")
print("=" * 60)

# ── STEP 6: Register Silver tables in Unity Catalog ──────────────────────────
spark.sql(f"""                                                     # noqa: F821
    CREATE TABLE IF NOT EXISTS {silver_db}.silver_transactions
    USING DELTA
    LOCATION '{silver_path}/silver_data'
""")

spark.sql(f"""                                                     # noqa: F821
    CREATE TABLE IF NOT EXISTS {silver_db}.silver_quarantine
    USING DELTA
    LOCATION '{silver_path}/silver_quarantine'
""")

# ── STEP 7: Validate ─────────────────────────────────────────────────────────
good_total = spark.sql(f"SELECT COUNT(*) FROM {silver_db}.silver_transactions").collect()[0][0]  # noqa: F821
bad_total  = spark.sql(f"SELECT COUNT(*) FROM {silver_db}.silver_quarantine").collect()[0][0]   # noqa: F821

print(f"""
📊 Silver Results:
   ✅ Good rows   → {silver_db}.silver_transactions : {good_total:,}
   🚫 Bad rows    → {silver_db}.silver_quarantine   : {bad_total:,}
   📦 Total       : {good_total + bad_total:,}

📂 Silver Locations:
   → {silver_path}/silver_data        (partitioned by TransactionDate)
   → {silver_path}/silver_quarantine  (bad rows for investigation)

🔑 Key Concepts Used:
   → readStream.table()  : read Bronze as stream
   → foreachBatch()      : split into good + bad
   → try_cast()          : safe type conversion
   → partitionBy()       : fast date-range queries
   → checkpoint          : never reprocess rows

⏭️  NEXT → 04_gold/gold_aggregation notebook!
""")

print("\n📋 Silver sample (5 good rows):")
spark.sql(f"""                                                     # noqa: F821
    SELECT
        TransactionID,
        AccountID,
        TransactionAmount,
        TransactionDate,
        CustomerAge,
        LoginAttempts,
        AccountBalance
    FROM {silver_db}.silver_transactions
    LIMIT 5
""").show(truncate=False)
