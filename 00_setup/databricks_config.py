"""
Databricks bootstrap notebook for storage credential and external location setup.

Run this ONCE per environment (dev/test/prod) before running any pipeline notebook.
This file is intentionally verbose - setup code is where operators need most context.

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
    schema: str = ENVIRONMENT                         # fraud.test / fraud.prod
    role_arn: str = "arn:aws:iam::346720668814:role/Databricks_S3_Access_Role"
    external_id: str = "2ced7911-e55a-4495-9893-37c4eda9f023"
    storage_credential_name: str = "databricks_s3_access"   # already exists!
    external_location_name: str = f"fraud_{ENVIRONMENT}_location"
    bucket_url: str = f"s3://{BUCKET_NAME}/{ENVIRONMENT}"


def build_sql(config: SetupConfig) -> list[str]:
    """Return SQL statements in execution order.

    Safe to re-run - all statements use IF NOT EXISTS.
    """
    return [
        # Step 1: Create Unity Catalog (top-level namespace)
        f"CREATE CATALOG IF NOT EXISTS {config.catalog}",

        # Step 2: Create schema inside catalog (fraud.test)
        f"CREATE SCHEMA IF NOT EXISTS {config.catalog}.{config.schema}",

        # Step 3: Create External Location using EXISTING storage credential
        # Storage credential 'databricks_s3_access' already created in Databricks UI
        (
            "CREATE EXTERNAL LOCATION IF NOT EXISTS "
            f"{config.external_location_name} URL '{config.bucket_url}' "
            f"WITH (STORAGE CREDENTIAL {config.storage_credential_name})"
        ),

        # Step 4: Grant access to all account users
        f"GRANT USAGE ON EXTERNAL LOCATION {config.external_location_name} TO `account users`",

        # Step 5: Validate everything is correct
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
    """Execute the SQL sequence in Databricks. Run cell by cell in a notebook."""
    for statement in build_sql(config):
        print(f"\n▶ Executing:\n  {statement}")
        spark_session.sql(statement).show(truncate=False)

    print("\n✅ All setup complete! Validated S3 paths:")
    for name, value in validate_paths(config).items():
        print(f"  {name:12} → {value}")


# ── VALIDATE S3 ACCESS ──
def test_s3_access():
    """
    Run this in a Databricks notebook cell to confirm S3 access works.
    If it lists folders → setup is complete!
    """
    try:
        files = dbutils.fs.ls(f"s3://{BUCKET_NAME}/")
        print(f"✅ S3 access confirmed! Folders found:")
        for f in files:
            print(f"  {f.path}")
    except Exception as e:
        print(f"❌ S3 access failed: {e}")
        print("Check: IAM trust policy has correct External ID and your S3 policy is attached")


if __name__ == "__main__":
    config = SetupConfig()
    print(f"Setup config for environment: {ENVIRONMENT}")
    print(f"Bucket: {BUCKET_NAME}")
    for statement in build_sql(config):
        print(statement)
    print(validate_paths(config))
