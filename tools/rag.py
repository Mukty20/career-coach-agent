"""
Lightweight RAG (Retrieval-Augmented Generation) module.

Uses TF-IDF + cosine similarity to retrieve relevant chunks from a small
curated knowledge base of career resources. No embedding API needed —
keeps the app free to run and easy to deploy on Streamlit Cloud.
"""

import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KNOWLEDGE_BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "knowledge_base")


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return [c.strip() for c in chunks if c.strip()]


class CareerKnowledgeBase:
    def __init__(self, kb_dir: str = KNOWLEDGE_BASE_DIR):
        self.kb_dir = kb_dir
        self.chunks: list[str] = []
        self.sources: list[str] = []
        self._vectorizer = TfidfVectorizer(stop_words="english")
        self._matrix = None
        self._load_and_index()

    def _load_and_index(self):
        filepaths = sorted(glob.glob(os.path.join(self.kb_dir, "*.md")))
        for path in filepaths:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            for chunk in _chunk_text(text):
                self.chunks.append(chunk)
                self.sources.append(os.path.basename(path))

        if self.chunks:
            self._matrix = self._vectorizer.fit_transform(self.chunks)

    def retrieve(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.chunks or self._matrix is None:
            return []

        query_vec = self._vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self._matrix)[0]

        top_indices = similarities.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.05:
                results.append({
                    "text": self.chunks[idx],
                    "source": self.sources[idx],
                    "score": float(similarities[idx]),
                })
        return results

    def format_context(self, query: str, top_k: int = 3) -> str:
        results = self.retrieve(query, top_k=top_k)
        if not results:
            return ""
        return "\n\n".join(f"[Source: {r['source']}]\n{r['text']}" for r in results)
