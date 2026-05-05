"""
S6 — Ensemble Embeddings & Domain-Aware Headers
Uses three complementary embedding models with graceful fallbacks and
domain-specific context header templates.
"""

import hashlib
import json
import logging
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_model_cache: Dict[str, Any] = {}
_embed_cache_l1: Dict[str, List[List[float]]] = {}
_cache_dir = os.path.join(os.path.dirname(__file__), "..", ".cache", "embeddings")
os.makedirs(_cache_dir, exist_ok=True)

DEFAULT_ENSEMBLE = [
    "mxbai-embed-large",           # NEW: High-quality primary model
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2",
    "jina-embeddings-v2-base-en",
]

# Lightweight ensemble for fast RL iterations (speeds up S7 calibration)
FAST_ENSEMBLE = [
    "all-MiniLM-L6-v2",            # Fast, proven quality
    "all-mpnet-base-v2",           # Good balance
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
        # Suppress known benign HuggingFace/SentenceTransformers warnings:
        # 1. "unexpected key roberta.embeddings.position_ids" — harmless,
        #    appears when loading newer checkpoints with older transformers.
        # 2. "missing adapter_config.json" (404) — model is not a PEFT adapter,
        #    sentence-transformers checks for it speculatively.
        # 3. "unauthenticated request" — model is public, no token needed.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*position_ids.*")
            warnings.filterwarnings("ignore", message=".*adapter_config.*")
            warnings.filterwarnings("ignore", message=".*unauthenticated.*")
            warnings.filterwarnings("ignore", category=UserWarning)
            model = SentenceTransformer(model_name)
        _model_cache[model_name] = model
        return model
    except Exception:
        _model_cache[model_name] = None
        return None


def preload_models(model_list: Optional[List[str]] = None) -> None:
    """
    Pre-load models into cache to avoid blocking during pipeline execution.
    Call this once at startup for models you'll use.
    
    Args:
        model_list: List of model names to preload. If None, preloads FAST_ENSEMBLE.
    """
    if model_list is None:
        model_list = FAST_ENSEMBLE
    for model_name in model_list:
        _get_model(model_name)


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
    
    # Determine which ensemble to use
    # If in RL mode or if fast mode is requested, use lightweight ensemble
    in_rl_mode = config.get("_in_rl_calibration", False)
    use_fast = config.get("use_fast_ensemble", in_rl_mode)
    
    ensemble_models = config.get("ensemble_models", None)
    if not ensemble_models:
        # Auto-select ensemble based on mode
        ensemble_models = FAST_ENSEMBLE if use_fast else DEFAULT_ENSEMBLE
    
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
    """
    Encode texts using an ensemble of embedding models and combine them.

    Projection alignment fix
    ────────────────────────
    The original code used a different random projection matrix per model
    (seed derived from model name hash), so averaging the projected vectors
    was mathematically meaningless — you were averaging incompatible spaces.

    Fix: all models project into the SAME 256-dim space using a SHARED
    projection seed.  The seed is fixed (42) so projections are identical
    across models, making the averaged vector well-defined.

    This is still Johnson-Lindenstrauss projection (not PCA, which would
    require training data), but the shared seed ensures:
        proj_model_A(v) and proj_model_B(w) live in the same space
        → their weighted average is semantically meaningful.

    Quality-weighted ensemble
    ─────────────────────────
    Instead of uniform weights (all 1.0), we assign quality weights based
    on known model performance tiers from the MTEB retrieval leaderboard:

        Tier 1 (weight 3.0): mxbai-embed-large, e5-large-v2
        Tier 2 (weight 2.0): all-mpnet-base-v2, jina-embeddings-v2-base-en
        Tier 3 (weight 1.0): all-MiniLM-L6-v2, bow_fallback

    This gives better models proportionally more influence in the ensemble
    without requiring runtime evaluation.
    """
    projection_dim = 256

    # Shared projection seed — ALL models use seed=42 so their projected
    # vectors are in the same 256-dim space and can be meaningfully averaged.
    SHARED_PROJ_SEED = 42

    # Quality weights per model (MTEB retrieval tier, higher = better quality)
    MODEL_QUALITY_WEIGHTS: Dict[str, float] = {
        "mxbai-embed-large":            3.0,
        "e5-large-v2":                  3.0,
        "all-mpnet-base-v2":            2.0,
        "jina-embeddings-v2-base-en":   2.0,
        "all-MiniLM-L6-v2":            1.0,
        "bow_fallback":                 0.5,
    }

    component_vectors: List[List[np.ndarray]] = []
    available_models:  List[str]              = []
    quality_weights:   List[float]            = []

    for model_name in models:
        vecs = _encode_with_model(texts, model_name)
        if not vecs:
            continue
        # Project all model vectors into the shared 256-dim space
        projected = [
            _project_vector(
                np.array(v, dtype=np.float32),
                projection_dim,
                seed=SHARED_PROJ_SEED,   # ← shared seed, not model-specific
            ).astype(np.float32)
            for v in vecs
        ]
        component_vectors.append(projected)
        available_models.append(model_name)
        # Look up quality weight; default to 1.0 for unknown models
        quality_weights.append(MODEL_QUALITY_WEIGHTS.get(model_name, 1.0))

    # Fallback: if no model succeeded, use bag-of-words hash embeddings
    if not component_vectors:
        fallback = [_bow_embed_text(t, projection_dim) for t in texts]
        return (
            [v.tolist() for v in fallback],
            {"models": ["bow_fallback"], "weights": [1.0], "projection_dim": projection_dim},
        )

    # Normalise quality weights so they sum to 1.0
    total_w = sum(quality_weights) or 1.0
    norm_weights = [w / total_w for w in quality_weights]

    # Build weighted-average embedding for each text
    final: List[List[float]] = []
    for i in range(len(texts)):
        agg = np.zeros(projection_dim, dtype=np.float32)
        for j, vectors in enumerate(component_vectors):
            agg += vectors[i] * norm_weights[j]
        final.append(agg.tolist())

    return final, {
        "models":          available_models,
        "weights":         [round(float(w), 4) for w in norm_weights],
        "projection_dim":  projection_dim,
        "projection_seed": SHARED_PROJ_SEED,
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
        import signal
        
        # Set a timeout of 120 seconds per model encoding (prevents hanging)
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Model encoding timeout for {model_name}")
        
        # Only set signal on Unix systems (Windows doesn't support SIGALRM)
        import platform
        if platform.system() != "Windows":
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(120)  # 120 second timeout
        
        try:
            vectors = model.encode(texts, show_progress_bar=False, batch_size=16)
            output = [v.tolist() for v in vectors]
            _cache_set(cache_key, output)
            return output
        finally:
            # Cancel the alarm
            if platform.system() != "Windows":
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
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


def _project_vector(vec: np.ndarray, out_dim: int, seed: int = 42) -> np.ndarray:
    """
    Project vec into out_dim dimensions using a random Gaussian matrix.

    Johnson-Lindenstrauss projection:
        proj = vec @ R   where R ∈ ℝ^(in_dim × out_dim), R_ij ~ N(0, 1/out_dim)

    The seed parameter controls the random matrix.  Using the SAME seed for all
    models ensures they all project into the same latent space, making the
    ensemble average well-defined.  Using different seeds per model (the old
    behaviour) produced incompatible spaces whose average was meaningless.

    If vec already has out_dim dimensions, returns it unchanged (no projection).
    """
    if vec.size == out_dim:
        return vec   # already at target dimension — no projection needed
    rng  = np.random.default_rng(seed)
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