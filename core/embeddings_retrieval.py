from typing import List, Optional, Tuple, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import streamlit as st

from .processing_utils import chunk_text, clean_text


class EmbedderRetriever:
    def __init__(self, embed_model_name: Optional[str] = None):
        self.embed_model_name = embed_model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self.embedder: SentenceTransformer = SentenceTransformer(self.embed_model_name)
        self.dim = self.embedder.get_sentence_embedding_dimension()
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: List[Dict] = []  # each item: {"text":..., "source": {...}}

    def chunk_texts(self, full_text: str, sources: List[Dict] = None, max_chunk_size: int = 800) -> List[Dict]:
        """Split full text into chunks and store them with optional metadata."""
        cleaned = clean_text(full_text)
        raw_chunks = chunk_text(cleaned, max_chunk_size=max_chunk_size)

        self.chunks = []
        for i, c in enumerate(raw_chunks):
            meta = sources[i] if sources and i < len(sources) else {}
            self.chunks.append({"text": c, "source": meta})

        return self.chunks

    def build_index_from_texts(self, texts: List[Dict]):
        """Build a FAISS index from text chunks."""
        if not texts:
            raise ValueError("No texts provided for index build.")

        docs = [t["text"] for t in texts]
        embs = self.embedder.encode(docs, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(embs)

        index = faiss.IndexFlatIP(embs.shape[1])
        index.add(embs)

        self.index = index

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[float, Dict]]:
        """Retrieve top-k similar chunks for a given query."""
        if self.index is None:
            raise ValueError("Index not built yet.")

        q_emb = self.embedder.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)

        D, I = self.index.search(q_emb, top_k)
        results = []
        for score, idx in zip(D[0], I[0]):
            if idx < len(self.chunks):
                results.append((float(score), self.chunks[idx]))

        return results
