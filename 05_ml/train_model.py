"""Train a LightGBM fraud classifier with MLflow tracking and threshold tuning."""

from __future__ import annotations

import json
import os
from pathlib import Path

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split

DATA_PATH = Path(os.getenv("ML_FEATURES_PATH", "data/ml_features.parquet"))
MODEL_DIR = Path("models")


def load_features(data_path: Path = DATA_PATH) -> pd.DataFrame:
    if data_path.suffix == ".parquet":
        return pd.read_parquet(data_path)
    return pd.read_csv(data_path)


def best_threshold_by_f1(y_true: pd.Series, y_score: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_scores = (2 * precision * recall) / np.clip(precision + recall, 1e-12, None)
    best_index = int(np.nanargmax(f1_scores[:-1])) if len(thresholds) else 0
    threshold = float(thresholds[best_index]) if len(thresholds) else 0.5
    return threshold, float(f1_scores[best_index])


def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    target = df["is_fraud"].astype(int)
    feature_frame = df.drop(columns=[col for col in ["is_fraud", "tr_id", "tr_date"] if col in df.columns])
    feature_frame = pd.get_dummies(feature_frame, dummy_na=True)
    return feature_frame, target


def train_model(df: pd.DataFrame) -> dict:
    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    negative = int((y_train == 0).sum())
    positive = max(int((y_train == 1).sum()), 1)
    scale_pos_weight = negative / positive

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        objective="binary",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )

    with mlflow.start_run(run_name="fraud-lightgbm"):
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_param("feature_count", X_train.shape[1])
        model.fit(X_train, y_train)
        y_score = model.predict_proba(X_test)[:, 1]
        pr_auc = average_precision_score(y_test, y_score)
        threshold, best_f1 = best_threshold_by_f1(y_test, y_score)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("best_f1", best_f1)
        mlflow.log_metric("best_threshold", threshold)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": X.columns.tolist(), "threshold": threshold}, MODEL_DIR / "fraud_model.joblib")
    metrics = {
        "pr_auc": pr_auc,
        "best_threshold": threshold,
        "best_f1": best_f1,
        "feature_columns": X.columns.tolist(),
    }
    (MODEL_DIR / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    dataframe = load_features()
    print(json.dumps(train_model(dataframe), indent=2))
