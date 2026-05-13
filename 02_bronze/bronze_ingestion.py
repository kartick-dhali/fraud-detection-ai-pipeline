"""Bronze Ingestion — Auto Loader pattern (study notes style).

Pattern:
  S3 CSV
  → Auto Loader (cloudFiles) readStream
  → schemaHints + rescuedDataColumn + schemaLocation
  → writeStream → Delta table (fraud_{env}.bronze)

Usage in Databricks notebook:
    %run ./02_bronze/bronze_ingestion
"""

from pyspark.sql import functions as F

# ── STEP 1: Environment Widget ────────────────────────────────────────────────
dbutils.widgets.dropdown("environment", "test", ["test", "prod"])  # noqa: F821
env = dbutils.widgets.get("environment")                           # noqa: F821

print(f"🌍 Environment : {env}")

# ── STEP 2: All Paths ─────────────────────────────────────────────────────────
base_path       = f"s3://fraud-transection-detection-nabo/{env}"
landing_path    = f"{base_path}/raw-landing/banking_transaction/"
schema_path     = f"{base_path}/_schemas/bronze_autoloader/"
checkpoint_path = f"{base_path}/_checkpoints/bronze_autoloader/"
bronze_db       = f"fraud_{env}"

print(f"📂 Landing path  : {landing_path}")
print(f"📂 Schema path   : {schema_path}")
print(f"📂 Checkpoint    : {checkpoint_path}")
print(f"📂 Bronze DB     : {bronze_db}")

# ── STEP 3: Create Schema (Database) in Unity Catalog ─────────────────────────
# CREATE SCHEMA = just a folder in the 'brain' (Unity Catalog)
# No data moves until writeStream!
spark.sql(f"""
    CREATE SCHEMA IF NOT EXISTS {bronze_db}
    MANAGED LOCATION '{base_path}/bronze/'
""")                                                               # noqa: F821
print(f"\n✅ Schema ready : {bronze_db}")

# ── STEP 4: Auto Loader readStream ────────────────────────────────────────────
# cloudFiles = Auto Loader (Databricks feature)
# → Watches S3 folder for new/changed files automatically
# → schemaHints         = tell Spark the correct types for key columns
# → rescuedDataColumn   = safety net — bad rows go here, not lost!
# → schemaLocation      = schema memory — locked after first run
# → schemaEvolutionMode = addNewColumns — if CSV gets new column, auto add it
#
# NOTE: input_file_name() is NOT supported in Unity Catalog!
#       Use _metadata.file_path instead ✅
print("\n📖 Setting up Auto Loader readStream...")

df_stream = (
    spark.readStream                                               # noqa: F821
    .format("cloudFiles")
    .option("cloudFiles.format",           "csv")
    .option("header",                      "true")
    .option("cloudFiles.schemaHints",
            "TransactionAmount double, "
            "CustomerAge int, "
            "AccountBalance double, "
            "LoginAttempts int, "
            "TransactionDuration int")
    .option("cloudFiles.rescuedDataColumn",   "_rescued_data")    # safety net
    .option("cloudFiles.schemaLocation",      schema_path)        # schema memory
    .option("cloudFiles.schemaEvolutionMode", "addNewColumns")    # evolution
    .option("cloudFiles.inferColumnTypes",    "true")
    .load(landing_path)
    # Add metadata columns
    .withColumn("_ingestion_timestamp", F.current_timestamp())
    .withColumn("_ingestion_date",      F.current_date())
    .withColumn("_source_file",         F.col("_metadata.file_path"))  # ← Unity Catalog safe!
    .withColumn("_environment",         F.lit(env))
    .withColumn("_pipeline_version",    F.lit("1.0"))
    # Fix column name with space → Delta doesn't allow spaces!
    .withColumnRenamed("IP Address", "IP_Address")
)

print("✅ Auto Loader stream configured!")

# ── STEP 5: writeStream → Bronze Delta Table ──────────────────────────────────
# trigger(availableNow=True) = process all available files then stop
#   (like a batch run but using streaming engine)
# checkpointLocation = bookmark — remembers which files already processed
#   Rule: every stream MUST have its own unique checkpoint folder!
# mergeSchema = True → if new columns appear, add them automatically
# toTable → registers in Unity Catalog automatically!
print("\n💾 Starting writeStream → Bronze Delta table...")

query = (
    df_stream.writeStream
    .format("delta")
    .option("checkpointLocation", checkpoint_path)
    .option("mergeSchema",        "true")
    .outputMode("append")
    .trigger(availableNow=True)
    .toTable(f"{bronze_db}.bronze")
)

# Wait for stream to finish
query.awaitTermination()

print("\n" + "=" * 60)
print("🎉 BRONZE INGESTION COMPLETE!")
print("=" * 60)

# ── STEP 6: Validate ──────────────────────────────────────────────────────────
count = spark.sql(f"SELECT COUNT(*) as total FROM {bronze_db}.bronze").collect()[0][0]  # noqa: F821
print(f"\n✅ Total rows in Bronze : {count:,}")

print("\n📋 Sample rows (5):")
spark.sql(f"""                                                     # noqa: F821
    SELECT
        TransactionID,
        AccountID,
        TransactionAmount,
        TransactionDate,
        TransactionType,
        _rescued_data,
        _ingestion_timestamp,
        _source_file
    FROM {bronze_db}.bronze
    LIMIT 5
""").show(truncate=False)

print(f"""
📍 Bronze Delta location : {base_path}/bronze/
📚 Unity Catalog table   : {bronze_db}.bronze
📊 Total rows            : {count:,}
🛟 Rescued data column   : _rescued_data (bad rows flagged, not lost!)

⏭️  NEXT → 03_silver/silver_cleaning notebook!
""")
