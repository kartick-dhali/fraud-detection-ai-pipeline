"""Evaluate the fraud model and export metrics plus a lightweight SVG PR-curve plot."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

DATA_PATH = Path("data/ml_features.parquet")
MODEL_PATH = Path("models/fraud_model.joblib")
OUTPUT_DIR = Path("artifacts/evaluation")


def load_features(data_path: Path = DATA_PATH) -> pd.DataFrame:
    if data_path.suffix == ".parquet":
        return pd.read_parquet(data_path)
    return pd.read_csv(data_path)


def prepare_features(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    y = df["is_fraud"].astype(int)
    X = pd.get_dummies(df.drop(columns=[col for col in ["is_fraud", "tr_id", "tr_date"] if col in df.columns]), dummy_na=True)
    X = X.reindex(columns=columns, fill_value=0)
    return X, y


def save_pr_curve_svg(precision, recall, target_path: Path) -> None:
    """Generate an SVG directly so plotting works without adding a new dependency."""

    width, height, padding = 640, 420, 40
    points = []
    for p, r in zip(precision, recall):
        x = padding + r * (width - 2 * padding)
        y = height - padding - p * (height - 2 * padding)
        points.append(f"{x:.2f},{y:.2f}")
    polyline = " ".join(points)
    svg = f"""<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>
      <rect width='100%' height='100%' fill='white'/>
      <line x1='{padding}' y1='{height-padding}' x2='{width-padding}' y2='{height-padding}' stroke='black'/>
      <line x1='{padding}' y1='{padding}' x2='{padding}' y2='{height-padding}' stroke='black'/>
      <polyline fill='none' stroke='#d62728' stroke-width='2' points='{polyline}'/>
      <text x='{width/2}' y='{height-5}' text-anchor='middle'>Recall</text>
      <text x='18' y='{height/2}' text-anchor='middle' transform='rotate(-90 18 {height/2})'>Precision</text>
    </svg>"""
    target_path.write_text(svg, encoding="utf-8")


def evaluate() -> dict:
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    threshold = float(bundle["threshold"])
    columns = bundle["columns"]
    df = load_features()
    X, y = prepare_features(df, columns)
    scores = model.predict_proba(X)[:, 1]
    predictions = (scores >= threshold).astype(int)

    precision, recall, _ = precision_recall_curve(y, scores)
    pr_auc = average_precision_score(y, scores)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_pr_curve_svg(precision, recall, OUTPUT_DIR / "pr_curve.svg")
    results = {
        "pr_auc": pr_auc,
        "threshold": threshold,
        "confusion_matrix": confusion_matrix(y, predictions).tolist(),
        "classification_report": classification_report(y, predictions, output_dict=True),
    }
    (OUTPUT_DIR / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))
