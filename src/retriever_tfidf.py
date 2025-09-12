import glob
from pathlib import Path
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class TfidfRetriever:
    """Простой ретривер на TF-IDF."""
    def __init__(self, pattern: str = "data/faq/*.md", chunk_size: int = 600, overlap: int = 150):
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
        self.tfidf = TfidfVectorizer().fit(self.chunks)
        self.mat = self.tfidf.transform(self.chunks)

    def topk(self, query: str, k: int = 3) -> List[Tuple[str,str,float]]:
        qv = self.tfidf.transform([query])
        sims = cosine_similarity(qv, self.mat)[0]
        idx = sims.argsort()[-k:][::-1]
        return [(self.chunks[i], self.chunk_meta[i], float(sims[i])) for i in idx]
