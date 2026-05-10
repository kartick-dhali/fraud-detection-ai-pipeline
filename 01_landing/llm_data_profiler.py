"""Profile the Kaggle CSV and generate a reusable pipeline configuration.

The profiler uses OpenAI when available, falls back to Ollama when configured, and ends
with deterministic defaults so Stage 0 never blocks the rest of the pipeline.
"""

from __future__ import annotations

import json
import os
from urllib import request
from pathlib import Path
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEFAULT_DATASET = Path("data/raw_csv/bank_transactions.csv")
CONFIG_PATH = Path("config/llm_pipeline_config.json")
EXPECTED_COLUMNS = {
    "TransactionID": "transaction_id",
    "TransactionDate": "transaction_date",
    "Amount": "amount",
    "AccountID": "account_id",
    "MerchantID": "merchant_id",
    "TransactionType": "transaction_type",
    "Location": "location",
    "IsFraud": "is_fraud",
}


def call_openai_or_fallback(prompt: str) -> str:
    """Ask OpenAI first, then Ollama, then return an empty response.

    Every network call is wrapped in try/except because infrastructure code must degrade
    gracefully. The deterministic fallback below makes the pipeline reproducible even when
    no model endpoint is reachable.
    """

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
            print(f"OpenAI fallback triggered: {exc}")

    ollama_base = os.getenv("OLLAMA_BASE_URL")
    if ollama_base:
        try:
            payload = json.dumps(
                {
                    "model": os.getenv("OLLAMA_MODEL", "llama3.1"),
                    "prompt": prompt,
                    "stream": False,
                }
            ).encode("utf-8")
            http_request = request.Request(
                f"{ollama_base.rstrip('/')}/api/generate",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with request.urlopen(http_request, timeout=30) as response:
                body = json.loads(response.read().decode("utf-8"))
            return body.get("response", "")
        except Exception as exc:
            print(f"Ollama fallback triggered: {exc}")

    return ""


def deterministic_profile(columns: list[str]) -> dict[str, Any]:
    """Generate safe defaults when no LLM response is available."""

    mapped_columns = {column: EXPECTED_COLUMNS.get(column, column.lower()) for column in columns}
    pii_columns = [column for column in columns if column in {"AccountID", "MerchantID", "Location"}]
    schema_hints = {
        "TransactionID": "string",
        "TransactionDate": "timestamp",
        "Amount": "double",
        "AccountID": "string",
        "MerchantID": "string",
        "TransactionType": "string",
        "Location": "string",
        "IsFraud": "int",
    }
    return {
        "mapped_columns": mapped_columns,
        "pii_columns": pii_columns,
        "schema_hints": schema_hints,
        "quarantine_rules": [
            "Amount IS NULL",
            "TransactionID IS NULL",
            "TransactionDate IS NULL",
            "IsFraud NOT IN (0, 1)",
        ],
    }


def build_prompt(sample: pd.DataFrame) -> str:
    """Provide enough business context for the LLM to generate useful hints."""

    return (
        "You are profiling a bank transaction fraud dataset. "
        "Return JSON with mapped_columns, pii_columns, schema_hints, and quarantine_rules. "
        f"Columns: {list(sample.columns)}. "
        f"Sample rows: {sample.head(5).to_dict(orient='records')}"
    )


def profile_dataset(dataset_path: Path = DEFAULT_DATASET) -> dict[str, Any]:
    df = pd.read_csv(dataset_path)
    prompt = build_prompt(df)
    llm_response = call_openai_or_fallback(prompt)
    profile = deterministic_profile(df.columns.tolist())

    if llm_response:
        try:
            candidate = json.loads(llm_response)
            profile.update({k: v for k, v in candidate.items() if v})
        except Exception as exc:
            print(f"Using deterministic profile because JSON parsing failed: {exc}")

    profile["dataset_path"] = str(dataset_path)
    profile["record_count"] = int(len(df))
    profile["generated_by"] = "OpenAI/Ollama with deterministic fallback"
    return profile


def save_profile(profile: dict[str, Any], config_path: Path = CONFIG_PATH) -> Path:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    return config_path


if __name__ == "__main__":
    dataset_path = Path(os.getenv("FRAUD_DATASET_PATH", DEFAULT_DATASET))
    profile = profile_dataset(dataset_path)
    saved_path = save_profile(profile)
    print(f"Saved pipeline config to {saved_path}")
