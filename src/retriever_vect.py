import glob
from pathlib import Path
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

class EmbeddingRetriever:
    """Ретривер на эмбеддингах + FAISS (L2)."""
    def __init__(self, pattern: str = "data/faq/*.md", model_name: str = "sentence-transformers/all-MiniLM-L6-v2", chunk_size: int = 600, overlap: int = 150):
        self.chunks: List[str] = []
        self.chunk_meta: List[str] = []
        paths = sorted(glob.glob(pattern))
        for fp in paths:
            text = Path(fp).read_text(encoding="utf-8", errors="ignore")
            for i in range(0, max(1, len(text)), max(1, chunk_size - overlap)):
                chunk = text[i:i+chunk_size]
                self.chunks.append(chunk)
                self.chunk_meta.append(f"{fp}:{i}-{i+len(chunk)}")
        if not self.chunks:
            self.chunks = [""]
            self.chunk_meta = ["<empty>"]
        self.model = SentenceTransformer(model_name)
        self.emb = self.model.encode(self.chunks, convert_to_numpy=True, normalize_embeddings=True)
        d = self.emb.shape[1]
        self.index = faiss.IndexFlatIP(d)  # косинус через скалярное произведение (т.к. нормализовано)
        self.index.add(self.emb)

    def topk(self, query: str, k: int = 3) -> List[Tuple[str,str,float]]:
        q = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        sims, idx = self.index.search(q, k)
        sims = sims[0]
        idx = idx[0]
        out: List[Tuple[str,str,float]] = []
        for i, s in zip(idx, sims):
            if i == -1:  # на всякий случай
                continue
            out.append((self.chunks[i], self.chunk_meta[i], float(s)))
        return out
