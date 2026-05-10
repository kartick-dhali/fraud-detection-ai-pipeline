"""Expose a FastAPI endpoint for natural-language fraud analytics via RAG."""

from __future__ import annotations

import os
from typing import Any

import chromadb
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma/fraud_rag")
COLLECTION_NAME = "fraud_gold_summaries"
app = FastAPI(title="Fraud Detection RAG API")


class QueryRequest(BaseModel):
    question: str
    top_k: int = 3


def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(COLLECTION_NAME)


def synthesize_answer(question: str, context: list[str]) -> str:
    prompt = (
        "Answer as a fraud analyst using the supplied context only. "
        f"Question: {question}\n"
        f"Context: {context}"
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
            print(f"OpenAI synthesis failed, using fallback: {exc}")

    return (
        "Fallback answer based on retrieved Gold summaries: "
        + " ".join(context[:2])
        + " | Suggested next step: validate the top-risk merchants in the Gold merchant summary."
    )


def query(question: str, top_k: int = 3) -> dict[str, Any]:
    embedder = SentenceTransformer(os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"))
    collection = get_collection()
    vector = embedder.encode([question]).tolist()
    response = collection.query(query_embeddings=vector, n_results=top_k)
    context = response.get("documents", [[]])[0]
    answer = synthesize_answer(question, context)
    return {
        "question": question,
        "context": context,
        "answer": answer,
    }


@app.post("/query")
def query_endpoint(payload: QueryRequest) -> dict[str, Any]:
    return query(payload.question, payload.top_k)


DEMO_QUESTIONS = [
    "Which merchant has the most fraud?",
    "Which users have rising transaction velocity?",
    "Summarize the highest-risk regions from the Gold tables.",
]


if __name__ == "__main__":
    for question in DEMO_QUESTIONS:
        print(query(question))
