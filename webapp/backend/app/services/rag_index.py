from pathlib import Path
from typing import List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

INDEX_PATH = Path(__file__).resolve().parents[2] / "model" / "rag_index.joblib"
DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"

_INDEX_CACHE = None


def _chunk_text(text: str, size: int = 500, overlap: int = 80) -> List[str]:
    cleaned = " ".join(text.split())
    if not cleaned:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = min(len(cleaned), start + size)
        chunks.append(cleaned[start:end])
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def build_index() -> dict:
    docs = []
    for path in DOCS_DIR.glob("*.txt"):
        text = path.read_text(encoding="utf-8")
        for i, chunk in enumerate(_chunk_text(text)):
            docs.append({"source": path.name, "chunk_id": i, "text": chunk})

    if not docs:
        payload = {"vectorizer": None, "matrix": None, "docs": []}
        INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(payload, INDEX_PATH)
        return payload

    corpus = [d["text"] for d in docs]
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), max_features=12000)
    matrix = vectorizer.fit_transform(corpus)

    payload = {"vectorizer": vectorizer, "matrix": matrix, "docs": docs}
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(payload, INDEX_PATH)
    return payload


def load_index() -> dict:
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE

    if not INDEX_PATH.exists():
        _INDEX_CACHE = build_index()
    else:
        _INDEX_CACHE = joblib.load(INDEX_PATH)
    return _INDEX_CACHE


def search(query: str, top_k: int = 3) -> List[Tuple[str, float, str]]:
    payload = load_index()
    vectorizer = payload.get("vectorizer")
    matrix = payload.get("matrix")
    docs = payload.get("docs", [])

    if vectorizer is None or matrix is None or not docs:
        return []

    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, matrix).flatten()
    top_idx = sims.argsort()[::-1][:top_k]

    results: List[Tuple[str, float, str]] = []
    for idx in top_idx:
        doc = docs[int(idx)]
        score = float(sims[int(idx)])
        results.append((doc["source"], score, doc["text"]))
    return results
