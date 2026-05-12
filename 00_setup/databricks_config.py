"""Databricks bootstrap notebook for storage credential and external location setup.

Run this ONCE per environment (dev/test/prod) before running any pipeline notebook.

Your real values:
  - AWS Account ID  : 346720668814
  - S3 Bucket       : fraud-transection-detection-nabo
  - IAM Role        : Databricks_S3_Access_Role
  - External ID     : 2ced7911-e55a-4495-9893-37c4eda9f023
  - Storage Cred    : databricks_s3_access (already created in Databricks UI)
"""

import os
from dataclasses import dataclass

# ── YOUR REAL BUCKET NAME ──
BUCKET_NAME = "fraud-transection-detection-nabo"

try:
    dbutils.widgets.dropdown("environment", "test", ["dev", "test", "prod"])
    ENVIRONMENT = dbutils.widgets.get("environment")
except Exception:
    ENVIRONMENT = os.getenv("ENVIRONMENT", "test")


@dataclass
class SetupConfig:
    """Container for the values that operators need to update once per environment."""

    catalog: str = "fraud"
    schema: str = ENVIRONMENT
    role_arn: str = "arn:aws:iam::346720668814:role/Databricks_S3_Access_Role"
    external_id: str = "2ced7911-e55a-4495-9893-37c4eda9f023"
    storage_credential_name: str = "databricks_s3_access"   # already exists!
    external_location_name: str = f"fraud_{ENVIRONMENT}_location"
    bucket_url: str = f"s3://{BUCKET_NAME}/{ENVIRONMENT}"


def build_sql(config: SetupConfig) -> list[str]:
    """Return SQL statements in execution order. Safe to re-run (IF NOT EXISTS)."""

    return [
        # Step 1: Create Unity Catalog
        f"CREATE CATALOG IF NOT EXISTS {config.catalog}",

        # Step 2: Create schema inside catalog (fraud.test)
        f"CREATE SCHEMA IF NOT EXISTS {config.catalog}.{config.schema}",

        # Step 3: Create External Location using existing storage credential
        (
            "CREATE EXTERNAL LOCATION IF NOT EXISTS "
            f"{config.external_location_name} URL '{config.bucket_url}' "
            f"WITH (STORAGE CREDENTIAL {config.storage_credential_name})"
        ),

        # Step 4: Grant access — READ FILES + WRITE FILES works with Unity Catalog v1.0
        # NOTE: GRANT USAGE does NOT work with metastore v1.0 → use READ/WRITE FILES
        (
            f"GRANT READ FILES, WRITE FILES ON EXTERNAL LOCATION "
            f"{config.external_location_name} TO `account users`"
        ),

        # Step 5: Validate
        f"DESCRIBE STORAGE CREDENTIAL {config.storage_credential_name}",
        f"DESCRIBE EXTERNAL LOCATION {config.external_location_name}",
    ]


def validate_paths(config: SetupConfig) -> dict[str, str]:
    """Produce the canonical S3 paths used throughout the pipeline."""

    return {
        "landing":    f"{config.bucket_url}/raw-landing/",
        "bronze":     f"{config.bucket_url}/bronze/",
        "silver":     f"{config.bucket_url}/silver/",
        "gold":       f"{config.bucket_url}/gold/",
        "checkpoint": f"{config.bucket_url}/checkpoint/",
        "monitoring": f"{config.bucket_url}/monitoring/",
    }


def run_setup(spark_session, config: SetupConfig) -> None:
    """Execute the SQL sequence in Databricks. Run in a notebook cell by cell."""

    for statement in build_sql(config):
        print(f"\n▶ Executing:\n  {statement[:120]}")
        try:
            spark_session.sql(statement).show(truncate=False)
            print("  ✅ Done!")
        except Exception as e:
            err = str(e)
            # Skip "already exists" errors — safe to ignore on re-runs
            if "already exists" in err.lower():
                print(f"  ⚠️  Already exists — skipping (safe)")
            else:
                print(f"  ❌ Error: {err[:200]}")
                raise

    print("\n✅ All setup complete! Validated S3 paths:")
    for name, value in validate_paths(config).items():
        print(f"  {name:12} → {value}")


def test_s3_access() -> None:
    """Run in a Databricks notebook cell to confirm S3 access works."""
    try:
        files = dbutils.fs.ls(f"s3://{BUCKET_NAME}/")
        print(f"✅ S3 access confirmed! Folders found:")
        for f in files:
            print(f"  {f.path}")
    except Exception as e:
        print(f"❌ S3 access failed: {e}")
        print("Check: IAM trust policy has self-assume + correct External ID")


if __name__ == "__main__":
    config = SetupConfig()
    print(f"Setup config for environment: {ENVIRONMENT}")
    print(f"Bucket: {BUCKET_NAME}")
    for statement in build_sql(config):
        print(statement)
    print(validate_paths(config))
