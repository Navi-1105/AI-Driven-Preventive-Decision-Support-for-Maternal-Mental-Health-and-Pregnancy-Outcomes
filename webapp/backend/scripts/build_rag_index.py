import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.services.rag_index import build_index


if __name__ == "__main__":
    payload = build_index()
    print(f"RAG index built with {len(payload.get('docs', []))} chunks.")
