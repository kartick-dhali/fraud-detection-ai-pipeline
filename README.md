# 🏦 AI-Powered Fraud Detection Pipeline
### Databricks + AWS S3 + Delta Lake + GenAI + RAG

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PySpark](https://img.shields.io/badge/PySpark-Distributed%20ETL-orange)
![Databricks](https://img.shields.io/badge/Databricks-Lakehouse-red)
![LightGBM](https://img.shields.io/badge/LightGBM-Fraud%20Model-green)
![OpenAI](https://img.shields.io/badge/OpenAI-gpt--4o--mini-black)
![License MIT](https://img.shields.io/badge/License-MIT-yellow)

## 1. Project Title + Badges
This repository delivers a production-style fraud analytics blueprint that starts with a Kaggle banking transaction dataset and expands it into a Databricks lakehouse, a feature engineering pipeline, a supervised fraud model, and a GenAI-powered RAG analyst experience.

## 2. Project Overview
This project shows how a modern fraud platform can combine classic data engineering, machine learning, and GenAI into one operating model:

- **Data Engineering**: Bronze / Silver / Gold lakehouse architecture on Databricks + Delta Lake, with raw JSONL landing in AWS S3 and streaming-style ingestion via Auto Loader.
- **GenAI**: LLM-powered ETL assistance, schema hint generation, quarantine explanation, alert summarization, and natural-language analytics.
- **ML**: LightGBM fraud classifier with imbalance handling, threshold tuning, offline evaluation, and MLflow experiment tracking.
- **RAG**: Gold-table summaries indexed in ChromaDB so analysts can ask plain-English questions and receive contextual answers synthesized by OpenAI (or a local Ollama fallback).

The source dataset is the Kaggle **bank-transaction-dataset-for-fraud-detection** dataset:

- `TransactionID`
- `TransactionDate`
- `Amount`
- `AccountID`
- `MerchantID`
- `TransactionType`
- `Location`
- `IsFraud`

## 3. Full Architecture Diagram (ASCII art, detailed)
```text
┌─────────────────────────────────────────────────────────────────────────┐
│                    FULL PIPELINE ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  📥 Kaggle Dataset (bank-transaction-dataset-for-fraud-detection)     │
│      TransactionID | TransactionDate | Amount | AccountID             │
│      MerchantID | TransactionType | Location | IsFraud                │
│           │                                                           │
│           ▼                                                           │
│  ┌─────────────────────────────────────────────────────┐              │
│  │ STAGE 0: AI-Powered ETL          [GenAI #1]         │              │
│  │  LLM reads CSV → auto-maps columns → detects PII    │              │
│  │  LLM generates schemaHints for Auto Loader          │              │
│  │  Kaggle CSV → JSONL files (one per day) → S3        │              │
│  │  dbutils.fs.put + unique filenames (checkpoint fix) │              │
│  └─────────────────────────────────────────────────────┘              │
│           │                                                           │
│           ▼  s3://fraud-transection-detection/{env}/raw-landing/      │
│  ┌─────────────────────────────────────────────────────┐              │
│  │ STAGE 1: BRONZE LAYER            [Data Engineering] │              │
│  │  Auto Loader (cloudFiles) + schemaHints (from LLM)  │              │
│  │  rescuedDataColumn + schemaLocation + mergeSchema    │              │
│  │  checkpointLocation + trigger(availableNow=True)     │              │
│  │  → Delta Table: fraud_{env}.bronze                   │              │
│  └─────────────────────────────────────────────────────┘              │
│           │                                                           │
│           ▼                                                           │
│  ┌─────────────────────────────────────────────────────┐              │
│  │ STAGE 2: SILVER LAYER        [DE + GenAI #2]        │              │
│  │  DE: foreachBatch + try_cast + cache/unpersist      │              │
│  │  DE: tr_id enrichment + PII hashing + dedup         │              │
│  │  GenAI: LLM generates CASE WHEN quarantine SQL      │              │
│  │  GenAI: LLM explains quarantine summary             │              │
│  │  → silver_data (clean, partitioned by tr_date)      │              │
│  │  → silver_quarantine (bad records + reason)         │              │
│  └─────────────────────────────────────────────────────┘              │
│           │                                                           │
│           ▼                                                           │
│  ┌─────────────────────────────────────────────────────┐              │
│  │ STAGE 3: GOLD LAYER          [DE + GenAI #3]        │              │
│  │  DE: Window Functions (lag, rolling avg, rank)      │              │
│  │  DE: SCD Type 2 (user risk score history)           │              │
│  │  DE: MERGE/UPSERT (idempotent gold writes)          │              │
│  │  GenAI: LLM summarizes fraud patterns → saves to S3 │              │
│  │  → ml_features | user_summary | merchant_summary    │              │
│  │  → user_risk_scd2                                   │              │
│  └─────────────────────────────────────────────────────┘              │
│           │                                                           │
│           ▼                                                           │
│  ┌─────────────────────────────────────────────────────┐              │
│  │ STAGE 4: ML FRAUD MODEL      [ML + GenAI #4]        │              │
│  │  LightGBM on Gold ml_features                       │              │
│  │  scale_pos_weight for class imbalance               │              │
│  │  MLflow tracking + PR-AUC + threshold tuning        │              │
│  │  GenAI: LLM explains predictions in plain English   │              │
│  └─────────────────────────────────────────────────────┘              │
│           │                                                           │
│           ▼                                                           │
│  ┌─────────────────────────────────────────────────────┐              │
│  │ STAGE 5: RAG ANALYTICS       [GenAI #5]             │              │
│  │  ChromaDB indexes Gold table summaries              │              │
│  │  SentenceTransformer embeddings                     │              │
│  │  FastAPI /query endpoint                            │              │
│  │  “Which merchant has most fraud?” → LLM answers     │              │
│  └─────────────────────────────────────────────────────┘              │
│           │                                                           │
│           ▼                                                           │
│  ┌─────────────────────────────────────────────────────┐              │
│  │ STAGE 6: MONITORING          [GenAI #6]             │              │
│  │  Statistical anomaly detection (mean + 2*std)       │              │
│  │  LLM writes executive fraud alert summary           │              │
│  │  Professional banking language output               │              │
│  └─────────────────────────────────────────────────────┘              │
│                                                                       │
└─────────────────────────────────────────────────────────────────────────┘
```

## 4. Data Engineering Concepts Used (table)
| Concept | Stage | Description |
|---|---|---|
| JSONL Format | Stage 0 | Splittable, schema-resilient, streaming-ready |
| dbutils.fs.put | Stage 0 | Lands files in S3 from Databricks |
| Environment Isolation (Widget) | All | dev/test/prod separation via Databricks widgets |
| Auto Loader (cloudFiles) | Stage 1 | Incremental file detection from S3 |
| schemaHints | Stage 1 | Enforces types BEFORE inference (enables rescue) |
| rescuedDataColumn | Stage 1 | Safety net: bad fields captured not crashed |
| schemaLocation | Stage 1 | Schema memory / DNA archive |
| mergeSchema | Stage 1 | Schema evolution: auto-add new columns |
| Checkpoint (bookmark) | Stage 1+2 | Stream state, prevents reprocessing |
| trigger(availableNow) | Stage 1+2 | Process all pending files then stop |
| foreachBatch | Stage 2 | Multi-output stream processor |
| try_cast | Stage 2 | Safe casting: NULL instead of crash |
| cache() / unpersist() | Stage 2 | Memory optimization for multi-output |
| dropDuplicates | Stage 2 | Deduplication by business key |
| Window Functions | Stage 3 | Rolling avg, lag, lead, rank, percent_rank |
| SCD Type 2 | Stage 3 | Slowly Changing Dimension: risk score history |
| MERGE / UPSERT | Stage 3 | Idempotent Gold writes via Delta merge |
| CREATE SCHEMA | Stage 1 | Metadata only — no data moves until writeStream |

## 5. GenAI Concepts Used (table)
| Concept | Stage | What LLM Does |
|---|---|---|
| AI-Powered ETL | Stage 0 | Auto-maps Kaggle columns, generates schemaHints, detects PII |
| Data Quality Automation | Stage 2 | Auto-generates PySpark CASE WHEN quarantine SQL |
| Quarantine Explainer | Stage 2 | Explains WHY records failed in plain English |
| Fraud Pattern Summarizer | Stage 3 | Reads Gold aggregations → professional summary |
| RAG Analytics System | Stage 5 | Analysts query Gold data in plain English |
| LLM Alert Summarizer | Stage 6 | Writes executive fraud alert from anomaly stats |

## 6. Tech Stack
| Technology | Purpose |
|---|---|
| Databricks | Unified analytics platform |
| AWS S3 | Data lake storage (JSONL + Delta) |
| Delta Lake | ACID transactions, time travel, schema evolution |
| Auto Loader | Incremental file ingestion from S3 |
| PySpark | Distributed data processing |
| Unity Catalog | Metadata + access control |
| LightGBM | Fraud classification model |
| MLflow | Model tracking and registry |
| OpenAI gpt-4o-mini | LLM for GenAI features |
| Ollama (optional) | Local LLM alternative (free) |
| ChromaDB | Vector database for RAG |
| SentenceTransformers | Text embeddings |
| FastAPI | RAG query API |

## 7. Project Folder Structure (tree diagram)
```text
fraud-detection-ai-pipeline/
├── 00_setup/
│   ├── databricks_config.py
│   ├── iam_s3_policy.json
│   └── iam_trust_policy.json
├── 01_landing/
│   ├── kaggle_to_jsonl.py
│   └── llm_data_profiler.py
├── 02_bronze/
│   ├── bronze_ingestion.py
│   └── schema_evolution_test.py
├── 03_silver/
│   └── silver_cleaning.py
├── 04_gold/
│   └── gold_transformations.py
├── 05_ml/
│   ├── evaluate_model.py
│   └── train_model.py
├── 06_rag/
│   ├── build_vector_store.py
│   └── rag_query_engine.py
├── 07_monitoring/
│   └── llm_alert_summarizer.py
├── utils/
│   └── debugging_utils.py
├── .gitignore
├── README.md
└── requirements.txt
```

## 8. Setup Instructions
### Step 1: Clone & Install
```bash
git clone https://github.com/kartick-dhali/fraud-detection-ai-pipeline
cd fraud-detection-ai-pipeline
pip install -r requirements.txt
```

### Step 2: Download Kaggle Dataset
```bash
kaggle datasets download -d valakhorasani/bank-transaction-dataset-for-fraud-detection -p data/raw_csv/ --unzip
```

### Step 3: AWS IAM Setup (explain trust policy + S3 policy from 00_setup/)
1. Create an IAM role for Databricks storage access.
2. Apply `00_setup/iam_trust_policy.json` as the trust relationship so the Databricks account **414351767826** can assume the role with an `sts:ExternalId`, while account root **077740385275:root** can self-assume for break-glass validation.
3. Attach `00_setup/iam_s3_policy.json` so the role can list the bucket and read/write objects only inside `arn:aws:s3:::fraud-transection-detection`.
4. Record the role ARN and feed it into `00_setup/databricks_config.py` when creating the storage credential and external location.

### Step 4: Databricks Setup (run 00_setup/databricks_config.py once)
- Open the notebook/script in a Databricks workspace attached to a cluster with Unity Catalog enabled.
- Provide catalog, schema, IAM role ARN, external ID, and workspace storage path.
- Run the script once per environment to create the storage credential, external location, and validation queries.

### Step 5: Set environment variable OPENAI_API_KEY (or use Ollama)
```bash
export OPENAI_API_KEY="your-key"
```
If you prefer local inference, start Ollama and use a model such as `llama3.1` by setting `OLLAMA_MODEL` and `OLLAMA_BASE_URL`.

## 9. Run Order
```text
Stage 0 → 01_landing/llm_data_profiler.py      (GenAI: profile + config)
Stage 0 → 01_landing/kaggle_to_jsonl.py        (DE: JSONL landing)
Stage 1 → 02_bronze/bronze_ingestion.py        (DE: Auto Loader)
Stage 2 → 03_silver/silver_cleaning.py         (DE + GenAI)
Stage 3 → 04_gold/gold_transformations.py      (DE + GenAI)
Stage 4 → 05_ml/train_model.py                 (ML)
Stage 5 → 06_rag/rag_query_engine.py           (GenAI RAG)
Stage 6 → 07_monitoring/llm_alert_summarizer.py (GenAI alerts)
```

## 10. Schema Evolution Scenarios (from 02_bronze/schema_evolution_test.py)
| Scenario | Incoming change | Expected Auto Loader behavior | Why it matters |
|---|---|---|---|
| 1. Additive column | New `DeviceID` column appears | `mergeSchema` adds the field to Bronze | Upstream systems often add optional telemetry fields without notice |
| 2. Type drift | `Amount` arrives as a string like `"120.55"` or `"unknown"` | `schemaHints` keeps the canonical type while bad payload lands in `_rescued_data` | Pipelines stay alive instead of failing the stream |
| 3. Sparse optional attribute | `Channel` only appears in a subset of files | Bronze records missing the field still load with nulls | Optional partner feeds should not require a hard reload |
| 4. Nested payload expansion | New JSON blob `RiskSignals` is appended | Rescue column captures unexpected nested content for later Silver parsing | Lets engineering inspect drift safely before promoting it |

## 11. Debugging Guide (from utils/debugging_utils.py)
| Utility | When to use it | What it checks | Typical fix |
|---|---|---|---|
| `validate_environment_widget` | Notebook starts in wrong environment | Widget value, allowed values, default fallback | Recreate the exact dropdown: `dbutils.widgets.dropdown("environment", "test", ["dev", "test", "prod"])` |
| `describe_stream_paths` | Streaming job reprocesses data | Landing, checkpoint, and schema paths | Ensure every stream has a unique checkpoint path per environment |
| `preview_quarantine_logic` | Too many rows are quarantined | Generated CASE WHEN rules | Tighten the LLM prompt or pin deterministic fallback rules |
| `summarize_feature_nulls` | Model training fails on missing features | Null counts and ratios in Gold features | Backfill defaults or quarantine bad upstream records earlier |
| `explain_openai_failure` | LLM calls fail intermittently | Missing API key, timeout, and fallback routing | Switch to Ollama or rely on deterministic logic until secrets are fixed |

Common issues and fixes:
- **Checkpoint collision**: two jobs pointing at the same checkpoint can re-read or corrupt state. Use one checkpoint path per stream and environment.
- **Schema drift confusion**: if a new field arrives unexpectedly, inspect Bronze `_rescued_data` first instead of forcing a manual cast.
- **Quarantine explosion**: preview the generated quarantine SQL and compare it with sampled records before pushing stricter rules to production.
- **LLM outages**: every OpenAI integration in this repo falls back to Ollama or deterministic defaults, so the platform remains operational.
- **Gold feature gaps**: run the feature null summary before training to catch upstream data regressions early.

## 12. Key Design Decisions
- **Why JSONL over JSON arrays**: JSONL is splittable, stream-friendly, and works naturally with file-based ingestion because each line is one record.
- **Why unique filenames**: unique daily JSONL filenames prevent accidental overwrites and help checkpoint-driven ingestion avoid ambiguity.
- **Why LLM-generated schemaHints**: the model accelerates onboarding by translating raw column names into typed ingestion hints, while deterministic fallbacks keep behavior safe.
- **Why SCD Type 2 for risk scores**: fraud risk changes over time, and SCD2 preserves historical context for investigators and model retraining.
- **Why foreachBatch for multi-output**: Silver needs to split good records, quarantine records, and explain failures in one micro-batch without rereading the same stream repeatedly.
