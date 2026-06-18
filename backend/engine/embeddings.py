"""Embedding service with multiple, selectable backends.

Backends (best -> fallback):
  * "openai"          OpenAI text-embedding-3-small  (multilingual, strong; needs key)
  * "multilingual"    sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
                      (free, offline, good on French/Arabic/etc.)
  * "english"         sentence-transformers/all-MiniLM-L6-v2 (English only — original)
  * "tfidf"           TF-IDF + SVD (pure sklearn) — always available

The English MiniLM model is fine for clean English but degrades badly on other
languages, which is why French legal documents scored poorly.  The default is
"openai" when a key is present, else "multilingual".

All backends return L2-normalised float32 vectors so cosine == dot product.
Token counting uses tiktoken (cl100k_base) when available.
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np

ST_MODELS = {
    "multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
    "english": "all-MiniLM-L6-v2",
}
# Selectable OpenAI embedding models (backend id -> API model name).
OPENAI_MODELS = {
    "openai": "text-embedding-3-small",   # 1536-d, fast, cheap (default)
    "openai-large": "text-embedding-3-large",  # 3072-d, strongest quality
}
OPENAI_MAX_TOKENS = 8000   # text-embedding-3 limit is 8191; keep margin
OPENAI_BATCH = 256         # inputs per request (API allows up to 2048)
OPENAI_WORKERS = 6         # parallel request batches (big docs stay fast)


class EmbeddingService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._st_models: Dict[str, object] = {}     # backend -> SentenceTransformer
        self._tfidf_ok = True
        self._tokenizer = None
        self._init_tokenizer()

    # ------------------------------------------------------------------ tokens
    def _init_tokenizer(self) -> None:
        try:
            import tiktoken
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._tokenizer = None

    def count_tokens(self, text: str) -> int:
        if self._tokenizer is not None:
            try:
                return len(self._tokenizer.encode(text))
            except Exception:
                pass
        return max(1, int(round(len(text.split()) / 0.75)))

    def _truncate_tokens(self, text: str, max_tokens: int) -> str:
        if self._tokenizer is None:
            # ~4 chars/token approximation
            return text[: max_tokens * 4]
        toks = self._tokenizer.encode(text)
        if len(toks) <= max_tokens:
            return text
        return self._tokenizer.decode(toks[:max_tokens])

    # -------------------------------------------------------------- discovery
    def default_backend(self) -> str:
        if os.environ.get("OPENAI_API_KEY"):
            return "openai"
        return "multilingual"

    def available_backends(self) -> List[dict]:
        has_key = bool(os.environ.get("OPENAI_API_KEY"))
        return [{
            "id": "openai", "label": "OpenAI text-embedding-3-small · 1536-d (fast, default)",
            "available": has_key,
        }, {
            "id": "openai-large", "label": "OpenAI text-embedding-3-large · 3072-d (best quality)",
            "available": has_key,
        }, {
            "id": "multilingual", "label": "Local multilingual · MiniLM-L12-v2 · 384-d (offline)",
            "available": True,
        }, {
            "id": "english", "label": "Local English · all-MiniLM-L6-v2 · 384-d (offline)",
            "available": True,
        }]

    # -------------------------------------------------------------- embedding
    def resolve(self, backend: Optional[str]) -> str:
        b = backend or self.default_backend()
        if b in OPENAI_MODELS and not os.environ.get("OPENAI_API_KEY"):
            return "multilingual"  # graceful fallback when no key
        return b

    def embed(self, texts: List[str], backend: Optional[str] = None) -> np.ndarray:
        b = self.resolve(backend)
        if not texts:
            return np.zeros((0, 8), dtype=np.float32)
        try:
            if b in OPENAI_MODELS:
                return self._embed_openai(texts, OPENAI_MODELS[b])
            if b in ST_MODELS:
                return self._embed_st(texts, b)
        except Exception as exc:  # network / model load issues -> degrade
            print(f"[embeddings] backend '{b}' failed ({exc}); falling back.")
            if b in OPENAI_MODELS:
                return self.embed(texts, "multilingual")
            if b == "multilingual":
                return self.embed(texts, "english")
        return self._embed_tfidf(texts)

    def _embed_openai(self, texts: List[str], model: str) -> np.ndarray:
        """Embed with OpenAI, splitting into batches that run in PARALLEL so even
        documents with thousands of sentences finish in a few seconds."""
        from openai import OpenAI

        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        cleaned = [self._truncate_tokens(t if t.strip() else " ", OPENAI_MAX_TOKENS)
                   for t in texts]
        batches = [cleaned[i : i + OPENAI_BATCH] for i in range(0, len(cleaned), OPENAI_BATCH)]

        def run(batch):
            resp = client.embeddings.create(model=model, input=batch)
            return [d.embedding for d in resp.data]

        if len(batches) <= 1:
            results = [run(batches[0])] if batches else []
        else:
            results = [None] * len(batches)
            with ThreadPoolExecutor(max_workers=OPENAI_WORKERS) as ex:
                for i, vecs in zip(range(len(batches)),
                                   ex.map(run, batches)):
                    results[i] = vecs
        vecs = [v for batch_vecs in results for v in batch_vecs]
        return _l2(np.asarray(vecs, dtype=np.float32))

    def _embed_st(self, texts: List[str], backend: str) -> np.ndarray:
        model = self._get_st(backend)
        vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    def _get_st(self, backend: str):
        if backend not in self._st_models:
            with self._lock:
                if backend not in self._st_models:
                    from sentence_transformers import SentenceTransformer
                    self._st_models[backend] = SentenceTransformer(ST_MODELS[backend])
        return self._st_models[backend]

    def _embed_tfidf(self, texts: List[str]) -> np.ndarray:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD

        n = len(texts)
        vec = TfidfVectorizer(max_features=4096)
        try:
            X = vec.fit_transform(texts)
        except ValueError:
            return np.tile(np.ones((1, 8), dtype=np.float32) / np.sqrt(8), (n, 1))
        dim = int(min(128, max(2, X.shape[1] - 1)))
        if X.shape[1] <= dim or n <= 2:
            dense = X.toarray().astype(np.float32)
        else:
            dense = TruncatedSVD(n_components=dim, random_state=0).fit_transform(X).astype(np.float32)
        return _l2(dense)


def _l2(a: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(a, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return a / norms


# Module-level singleton (kept name get_embedder for backward compat).
_SERVICE: Optional[EmbeddingService] = None


def get_embedder() -> EmbeddingService:
    global _SERVICE
    if _SERVICE is None:
        _SERVICE = EmbeddingService()
    return _SERVICE


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), dtype=np.float32)
    return np.clip(a @ b.T, -1.0, 1.0)
