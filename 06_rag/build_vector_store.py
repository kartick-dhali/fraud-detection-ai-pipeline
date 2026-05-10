"""Build a ChromaDB vector store from Gold-table summaries.

Gold tables are turned into short analytical documents because retrieval works best when
each chunk has a focused business meaning rather than raw denormalized rows.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import chromadb
import pandas as pd
from sentence_transformers import SentenceTransformer

GOLD_DIR = Path(os.getenv("GOLD_EXPORT_DIR", "data/gold_exports"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", "chroma/fraud_rag"))


def load_gold_files(gold_dir: Path = GOLD_DIR) -> list[dict]:
    documents = []
    for path in sorted(gold_dir.glob("*")):
        if path.suffix == ".parquet":
            df = pd.read_parquet(path)
        elif path.suffix == ".csv":
            df = pd.read_csv(path)
        elif path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            df = pd.DataFrame(payload)
        else:
            continue
        documents.append(
            {
                "id": path.stem,
                "text": df.head(50).to_json(orient="records"),
                "metadata": {"source": str(path)},
            }
        )
    return documents


def build_store() -> int:
    documents = load_gold_files()
    if not documents:
        raise FileNotFoundError(f"No Gold exports found in {GOLD_DIR}")

    model = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_or_create_collection("fraud_gold_summaries")

    ids = [doc["id"] for doc in documents]
    texts = [doc["text"] for doc in documents]
    metadatas = [doc["metadata"] for doc in documents]
    embeddings = model.encode(texts).tolist()
    collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    return len(ids)


if __name__ == "__main__":
    print(f"Indexed {build_store()} Gold documents")
