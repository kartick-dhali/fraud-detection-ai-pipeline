"""Databricks bootstrap notebook for storage credential and external location setup.

This file is intentionally verbose because setup code is where operators need the most
context. The goal is to make the "why" obvious for every step before any data lands.
"""

import os
from dataclasses import dataclass

BUCKET_NAME = "fraud-transection-detection"

try:
    dbutils.widgets.dropdown("environment", "test", ["dev", "test", "prod"])
    ENVIRONMENT = dbutils.widgets.get("environment")
except Exception:
    ENVIRONMENT = os.getenv("ENVIRONMENT", "test")


@dataclass
class SetupConfig:
    """Container for the values that operators need to update once per environment."""

    catalog: str = os.getenv("DATABRICKS_CATALOG", "fraud")
    schema: str = os.getenv("DATABRICKS_SCHEMA", ENVIRONMENT)
    role_arn: str = os.getenv("AWS_ROLE_ARN", "arn:aws:iam::077740385275:role/fraud-databricks-role")
    external_id: str = os.getenv("AWS_EXTERNAL_ID", "REPLACE_WITH_DATABRICKS_EXTERNAL_ID")
    storage_credential_name: str = f"fraud_{ENVIRONMENT}_credential"
    external_location_name: str = f"fraud_{ENVIRONMENT}_location"
    bucket_url: str = f"s3://{BUCKET_NAME}/{ENVIRONMENT}"


def build_sql(config: SetupConfig) -> list[str]:
    """Return SQL statements in execution order.

    The statements are separated to make reruns safe: metadata objects are created
    idempotently and validation happens only after the security boundary exists.
    """

    return [
        f"CREATE CATALOG IF NOT EXISTS {config.catalog}",
        f"CREATE SCHEMA IF NOT EXISTS {config.catalog}.{config.schema}",
        (
            "CREATE STORAGE CREDENTIAL IF NOT EXISTS "
            f"{config.storage_credential_name} WITH IAM_ROLE '{config.role_arn}'"
        ),
        (
            "ALTER STORAGE CREDENTIAL "
            f"{config.storage_credential_name} SET COMMENT 'ExternalId={config.external_id}'"
        ),
        (
            "CREATE EXTERNAL LOCATION IF NOT EXISTS "
            f"{config.external_location_name} URL '{config.bucket_url}' "
            f"WITH (STORAGE CREDENTIAL {config.storage_credential_name})"
        ),
        f"GRANT USAGE ON EXTERNAL LOCATION {config.external_location_name} TO `account users`",
        f"DESCRIBE STORAGE CREDENTIAL {config.storage_credential_name}",
        f"DESCRIBE EXTERNAL LOCATION {config.external_location_name}",
    ]


def validate_paths(config: SetupConfig) -> dict[str, str]:
    """Produce the canonical S3 paths used throughout the pipeline.

    Keeping path derivation in one place reduces the chance of subtle typos between
    Bronze, Silver, Gold, ML, and RAG jobs.
    """

    return {
        "landing": f"{config.bucket_url}/raw-landing/",
        "bronze": f"{config.bucket_url}/bronze/",
        "silver": f"{config.bucket_url}/silver/",
        "gold": f"{config.bucket_url}/gold/",
        "monitoring": f"{config.bucket_url}/monitoring/",
    }


def run_setup(spark_session, config: SetupConfig) -> None:
    """Execute the SQL sequence in Databricks.

    The notebook prints each command before execution so platform teams can audit the
    exact security objects that were created.
    """

    for statement in build_sql(config):
        print(f"Executing: {statement}")
        spark_session.sql(statement).show(truncate=False)

    print("Validated paths:")
    for name, value in validate_paths(config).items():
        print(f"  - {name}: {value}")


if __name__ == "__main__":
    config = SetupConfig()
    print("Setup configuration prepared for environment:", ENVIRONMENT)
    for statement in build_sql(config):
        print(statement)
    print(validate_paths(config))
