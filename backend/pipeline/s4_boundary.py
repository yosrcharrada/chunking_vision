"""
S4 — Advanced Boundary Scoring
Adds structural continuity, cross-encoder semantic signal, multi-scale analysis,
and weighted boundary decisions.
"""

import re
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

_ALPHA = 0.25
_BETA = 0.20
_GAMMA = 0.15
_DELTA = 0.20
_EPSILON = 0.20
_cross_encoder = None


def filter_boundaries(
    chunks: List[Dict],
    doc_type: str,
    embeddings: Optional[List[List[float]]],
    config: Dict[str, Any],
) -> List[Dict]:
    if not chunks:
        return chunks

    tau_sem = float(config.get("tau_sem", 0.75))
    n_max = int(config.get("n_max", 500))
    merge_weight = float(config.get("boundary_merge_weight", 1.0))
    result: List[Dict] = [dict(chunks[0])]
    result[0]["boundary_score"] = 0.0
    result[0]["icc"] = _compute_icc(result[0]["text"])
    result[0]["boundary_breakdown"] = {}

    for idx in range(1, len(chunks)):
        prev = result[-1]
        curr = dict(chunks[idx])
        lexical = _lexical_boundary_score(prev["text"], curr["text"], doc_type)
        structural = _structural_continuity_score(prev["text"], curr["text"], doc_type)
        semantic = _semantic_score(prev["text"], curr["text"], embeddings, idx)
        multiscale = _multi_scale_boundary_score(chunks, idx, doc_type)
        weighted = (
            _ALPHA * lexical
            + _BETA * _syntactic_overlap(prev["text"], curr["text"], doc_type)
            + _GAMMA * _token_type_match(prev["text"], curr["text"])
            + _DELTA * structural
            + _EPSILON * ((semantic + multiscale) / 2.0)
        )
        decision_score = float(np.clip(weighted * merge_weight, 0.0, 1.0))
        curr["boundary_score"] = round(decision_score, 4)
        curr["icc"] = _compute_icc(curr["text"])
        curr["boundary_breakdown"] = {
            "lexical": round(float(lexical), 4),
            "structural": round(float(structural), 4),
            "semantic": round(float(semantic), 4),
            "multiscale": round(float(multiscale), 4),
            "weighted": round(float(decision_score), 4),
        }

        prev_wc = len(prev["text"].split())
        curr_wc = len(curr["text"].split())
        if decision_score > tau_sem and (prev_wc + curr_wc) <= n_max * 1.5:
            result[-1]["text"] = prev["text"] + "\n\n" + curr["text"]
            result[-1]["end"] = curr.get("end", prev.get("end", 0))
            result[-1]["boundary_score"] = round(decision_score, 4)
            result[-1]["boundary_breakdown"] = curr["boundary_breakdown"]
        else:
            result.append(curr)
    return result


def _lexical_boundary_score(text1: str, text2: str, doc_type: str) -> float:
    bleu = _ngram_precision(text1, text2, n=2)
    syn = _syntactic_overlap(text1, text2, doc_type)
    ttm = _token_type_match(text1, text2)
    return float(0.4 * bleu + 0.3 * syn + 0.3 * ttm)


def _ngram_precision(text1: str, text2: str, n: int = 2) -> float:
    tokens1 = _tokenize(text1)
    tokens2 = _tokenize(text2)
    if len(tokens1) < n or len(tokens2) < n:
        s1, s2 = set(tokens1), set(tokens2)
        union = s1 | s2
        return len(s1 & s2) / len(union) if union else 0.0
    ngrams1 = Counter(_ngrams(tokens1, n))
    ngrams2 = Counter(_ngrams(tokens2, n))
    clipped = sum(min(c, ngrams2[ng]) for ng, c in ngrams1.items())
    total = sum(ngrams1.values())
    return clipped / total if total else 0.0


def _ngrams(tokens: List[str], n: int):
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def _syntactic_overlap(text1: str, text2: str, doc_type: str) -> float:
    if doc_type == "code":
        t1 = set(re.findall(r"[+\-*/%&|^~<>=!;:,.()\[\]{}]|\b\w+\b", text1))
        t2 = set(re.findall(r"[+\-*/%&|^~<>=!;:,.()\[\]{}]|\b\w+\b", text2))
        union = t1 | t2
        return len(t1 & t2) / len(union) if union else 0.0
    func = {
        "the", "a", "an", "is", "was", "are", "were", "be", "been", "have", "has", "had",
        "do", "does", "did", "will", "would", "could", "should", "may", "might", "shall",
        "can", "of", "in", "on", "at", "by", "for", "with", "about", "as", "to",
    }
    t1 = [w for w in _tokenize(text1) if w in func]
    t2 = [w for w in _tokenize(text2) if w in func]
    c1, c2 = Counter(t1), Counter(t2)
    shared = sum(min(c1[w], c2[w]) for w in c1)
    total = max(len(t1), len(t2))
    return shared / total if total else 0.0


def _token_type_match(text1: str, text2: str) -> float:
    t1 = set(_tokenize(text1))
    t2 = set(_tokenize(text2))
    union = t1 | t2
    return len(t1 & t2) / len(union) if union else 0.0


def _structural_continuity_score(text1: str, text2: str, doc_type: str) -> float:
    if doc_type not in {"code", "mixed", "table"}:
        return 0.5
    brace_delta = abs(text1.count("{") - text1.count("}")) + abs(text2.count("{") - text2.count("}"))
    markdown_bridge = 1.0 if re.search(r"^#{1,6}\s", text2, re.MULTILINE) else 0.6
    xml_bridge = 0.8 if ("<" in text1 and ">" in text1 and "<" in text2 and ">" in text2) else 0.4
    cont = 1.0 - min(1.0, brace_delta / 6.0)
    return float(np.clip((cont + markdown_bridge + xml_bridge) / 3.0, 0.0, 1.0))


def _semantic_score(text1: str, text2: str, embeddings: Optional[List[List[float]]], idx: int) -> float:
    if embeddings and (idx - 1) < len(embeddings) and idx < len(embeddings):
        return _cosine_similarity(embeddings[idx - 1], embeddings[idx])
    ce = _cross_encoder_similarity(text1, text2)
    if ce is not None:
        return ce
    return _fallback_semantic(text1, text2)


def _multi_scale_boundary_score(chunks: List[Dict], idx: int, doc_type: str) -> float:
    windows = [1, 2, 3]
    scores = []
    for w in windows:
        left_start = max(0, idx - w)
        left = " ".join(c["text"] for c in chunks[left_start:idx]).strip()
        right = " ".join(c["text"] for c in chunks[idx: min(len(chunks), idx + w)]).strip()
        if not left or not right:
            continue
        scores.append(_lexical_boundary_score(left, right, doc_type))
    return float(np.mean(scores)) if scores else 0.5


def _cross_encoder_similarity(text1: str, text2: str) -> Optional[float]:
    global _cross_encoder
    try:
        if _cross_encoder is None:
            from sentence_transformers import CrossEncoder  # noqa: E402
            _cross_encoder = CrossEncoder("cross-encoder/stsb-distilroberta-base")
        score = float(_cross_encoder.predict([(text1[:800], text2[:800])])[0])
        return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))
    except Exception:
        return None


def _fallback_semantic(text1: str, text2: str) -> float:
    v1 = _hash_embedding(text1)
    v2 = _hash_embedding(text2)
    return _cosine_similarity(v1.tolist(), v2.tolist())


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _compute_icc(text: str) -> float:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if len(sentences) < 2:
        return 0.5
    overlaps: List[float] = []
    for i in range(len(sentences) - 1):
        a = set(_tokenize(sentences[i]))
        b = set(_tokenize(sentences[i + 1]))
        union = a | b
        if union:
            overlaps.append(len(a & b) / len(union))
    return float(np.mean(overlaps)) if overlaps else 0.5


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a = np.array(v1, dtype=np.float32)
    b = np.array(v2, dtype=np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _hash_embedding(text: str, dim: int = 128) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in _tokenize(text):
        vec[hash(tok) % dim] += 1.0
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec
