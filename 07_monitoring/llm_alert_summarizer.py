"""Create executive fraud alerts from anomaly statistics and LLM narration."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
from openai import OpenAI

INPUT_PATH = Path(os.getenv("MONITORING_INPUT_PATH", "data/monitoring/daily_metrics.csv"))
OUTPUT_PATH = Path("artifacts/monitoring/executive_alert.json")


def detect_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """Flag values above mean + 2*std because it is easy for operators to explain."""

    mean_amount = df["fraud_amount"].mean()
    std_amount = df["fraud_amount"].std()
    threshold = mean_amount + (2 * std_amount)
    return df.assign(alert=df["fraud_amount"] > threshold, threshold=threshold)


def summarize_with_llm(records: list[dict]) -> str:
    prompt = (
        "Write an executive banking alert summary using professional language. "
        f"Input records: {records}"
    )
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
            print(f"OpenAI alert failed, using fallback: {exc}")
    return "Fallback executive alert: fraud volume has breached the statistical control band and requires analyst review."


def build_alert_payload(df: pd.DataFrame) -> dict:
    anomalies = detect_anomalies(df)
    flagged = anomalies[anomalies["alert"]]
    summary = summarize_with_llm(flagged.to_dict(orient="records"))
    return {
        "threshold": float(anomalies["threshold"].iloc[0]),
        "alert_count": int(flagged.shape[0]),
        "flagged_days": flagged.to_dict(orient="records"),
        "executive_summary": summary,
    }


if __name__ == "__main__":
    dataframe = pd.read_csv(INPUT_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = build_alert_payload(dataframe)
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
