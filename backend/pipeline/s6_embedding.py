"""
S6 — Ensemble Embeddings & Domain-Aware Headers
Uses three complementary embedding models with graceful fallbacks and
domain-specific context header templates.
"""

import hashlib
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_model_cache: Dict[str, Any] = {}
_embed_cache_l1: Dict[str, List[List[float]]] = {}
_cache_dir = os.path.join(os.path.dirname(__file__), "..", ".cache", "embeddings")
os.makedirs(_cache_dir, exist_ok=True)

DEFAULT_ENSEMBLE = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "jina-embeddings-v2-base-en",
]

DOMAIN_TEMPLATES = {
    "legal": "Legal context: section intent, obligations, governing terms, and enforceable clauses.",
    "medical": "Medical context: patient condition, clinical findings, interventions, and outcomes.",
    "technical": "Technical context: architecture, implementation details, interfaces, and constraints.",
    "financial": "Financial context: performance indicators, accounting treatment, and risk factors.",
    "academic": "Academic context: hypothesis, methods, evidence, and contribution claims.",
    "narrative": "Narrative context: storyline progression, actors, events, and thematic transitions.",
}


def _get_model(model_name: str):
    if model_name in _model_cache:
        return _model_cache[model_name]
    try:
        from sentence_transformers import SentenceTransformer  # noqa: E402
        model = SentenceTransformer(model_name)
        _model_cache[model_name] = model
        return model
    except Exception:
        _model_cache[model_name] = None
        return None


def embed_chunks(
    chunks: List[Dict],
    full_text: str,
    doc_profile: Dict[str, Any],
    model_name: str,
    config: Dict[str, Any],
) -> Tuple[List[Dict], List[List[float]]]:
    if not chunks:
        return chunks, []

    domain = doc_profile.get("domain", "general")
    doc_type = doc_profile.get("type", "prose")
    length_bucket = doc_profile.get("length_bucket", "medium")
    ensemble_models = config.get("ensemble_models", DEFAULT_ENSEMBLE)
    if not isinstance(ensemble_models, list) or not ensemble_models:
        ensemble_models = [model_name]

    enriched = [dict(c) for c in chunks]
    input_texts = _build_input_texts(enriched, full_text, domain, doc_type, length_bucket)
    embeddings, composition = _ensemble_encode(input_texts, ensemble_models)

    for i, chunk in enumerate(enriched):
        vec = embeddings[i] if i < len(embeddings) else []
        chunk["embedding"] = vec
        chunk["ensemble_embedding"] = {
            "models": composition.get("models", []),
            "weights": composition.get("weights", []),
            "projection_dim": composition.get("projection_dim", 0),
        }
    return enriched, embeddings


def _build_input_texts(
    chunks: List[Dict],
    full_text: str,
    domain: str,
    doc_type: str,
    length_bucket: str,
) -> List[str]:
    inferred_topic = _infer_topic(full_text)
    template = DOMAIN_TEMPLATES.get(domain, "General context: preserve semantics and continuity across chunks.")
    texts = []
    for chunk in chunks:
        first_sentence = _first_sentence(chunk.get("text", ""))
        header = (
            f"{template} Document type: {doc_type}. Topic: {inferred_topic}. "
            f"Chunk focus: {first_sentence}"
        )
        chunk["context_header"] = header
        if length_bucket == "short":
            texts.append(chunk.get("text", ""))
        else:
            texts.append(header + "\n\n" + chunk.get("text", ""))
    return texts


def _ensemble_encode(texts: List[str], models: List[str]) -> Tuple[List[List[float]], Dict[str, Any]]:
    projection_dim = 256
    component_vectors: List[List[np.ndarray]] = []
    available_models: List[str] = []
    for model_name in models:
        vecs = _encode_with_model(texts, model_name)
        if not vecs:
            continue
        projected = [_project_vector(np.array(v, dtype=np.float32), projection_dim, model_name).astype(np.float32) for v in vecs]
        component_vectors.append(projected)
        available_models.append(model_name)

    if not component_vectors:
        fallback = [_bow_embed_text(t, projection_dim) for t in texts]
        return [v.tolist() for v in fallback], {"models": ["bow_fallback"], "weights": [1.0], "projection_dim": projection_dim}

    weights = np.array([1.0] * len(component_vectors), dtype=np.float32)
    weights = weights / max(np.sum(weights), 1.0)
    final: List[List[float]] = []
    for i in range(len(texts)):
        agg = np.zeros(projection_dim, dtype=np.float32)
        for j, vectors in enumerate(component_vectors):
            agg += vectors[i] * weights[j]
        final.append(agg.tolist())

    return final, {
        "models": available_models,
        "weights": [round(float(w), 4) for w in weights.tolist()],
        "projection_dim": projection_dim,
    }


def _encode_with_model(texts: List[str], model_name: str) -> Optional[List[List[float]]]:
    model = _get_model(model_name)
    if model is None:
        return None
    cache_key = _cache_key(model_name, texts)
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached
    try:
        vectors = model.encode(texts, show_progress_bar=False, batch_size=16)
        output = [v.tolist() for v in vectors]
        _cache_set(cache_key, output)
        return output
    except Exception:
        return None


def _cache_key(model_name: str, texts: List[str]) -> str:
    digest = hashlib.sha256((model_name + "||" + "||".join(texts)).encode("utf-8", errors="ignore")).hexdigest()
    return digest


def _cache_get(key: str) -> Optional[List[List[float]]]:
    if key in _embed_cache_l1:
        return _embed_cache_l1[key]
    path = os.path.join(_cache_dir, key + ".json")
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, list):
                return data
        except Exception:
            return None
    return None


def _cache_set(key: str, vectors: List[List[float]]) -> None:
    if not vectors:
        return
    _embed_cache_l1[key] = vectors
    path = os.path.join(_cache_dir, key + ".json")
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(vectors, fh)
    except Exception:
        pass


def _project_vector(vec: np.ndarray, out_dim: int, salt: str) -> np.ndarray:
    if vec.size == out_dim:
        return vec
    rng = np.random.default_rng(abs(hash(salt)) % (2**32))
    proj = rng.normal(0, 1.0 / np.sqrt(out_dim), size=(vec.size, out_dim)).astype(np.float32)
    return vec @ proj


def _bow_embed_text(text: str, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"\b\w+\b", text.lower()):
        vec[hash(tok) % dim] += 1.0
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def _infer_topic(text: str) -> str:
    heading = re.search(r"^#{1,3}\s+(.+)$", text, re.MULTILINE)
    if heading:
        return heading.group(1).strip()[:80]
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for s in sentences[:3]:
        clean = s.strip()
        if len(clean.split()) >= 4:
            return clean[:80]
    return "the provided content"


def _first_sentence(text: str) -> str:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return (parts[0].strip() if parts else text[:120].strip())[:120]
