"""
S1 — Document Profiler
Expanded domain detection, adaptive metric weighting, and uncertainty estimates.
"""

import re
from typing import Dict, Any, List, Tuple

import numpy as np


DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "legal": ["statute", "clause", "agreement", "liability", "jurisdiction", "indemnity", "contract", "hereby"],
    "medical": ["patient", "diagnosis", "treatment", "clinical", "therapy", "symptom", "hospital", "dosage"],
    "academic": ["abstract", "methodology", "citation", "hypothesis", "literature", "peer review", "thesis", "dataset"],
    "financial": ["revenue", "profit", "loss", "equity", "fiscal", "balance sheet", "cash flow", "valuation"],
    "technical": ["algorithm", "api", "repository", "deployment", "protocol", "database", "framework", "runtime"],
    "narrative": ["chapter", "character", "plot", "scene", "dialogue", "protagonist", "story", "novel"],
    "scientific": ["experiment", "variable", "control group", "statistical", "finding", "evidence", "sample", "observation"],
    "regulatory": ["compliance", "regulation", "audit", "governance", "policy", "risk", "control", "obligation"],
    "marketing": ["campaign", "conversion", "audience", "brand", "segmentation", "retention", "funnel", "roi"],
    "education": ["curriculum", "assessment", "learning", "student", "teacher", "pedagogy", "course", "instruction"],
    "cybersecurity": ["vulnerability", "threat", "malware", "encryption", "incident", "firewall", "authentication", "exploit"],
    "product": ["roadmap", "feature", "release", "user story", "backlog", "ux", "adoption", "prioritization"],
    "operations": ["workflow", "throughput", "sla", "capacity", "scheduling", "logistics", "inventory", "downtime"],
    "policy": ["guideline", "directive", "standards", "framework", "mandate", "protocol", "code of conduct", "principle"],
    "research": ["benchmark", "model", "inference", "evaluation", "baseline", "ablation", "metric", "corpus"],
}


DOMAIN_METRIC_WEIGHTS: Dict[str, Dict[str, float]] = {
    "legal": {"RC": 0.30, "ICC": 0.20, "DCC": 0.20, "BI": 0.20, "SC": 0.10},
    "medical": {"RC": 0.20, "ICC": 0.25, "DCC": 0.25, "BI": 0.15, "SC": 0.15},
    "academic": {"RC": 0.30, "ICC": 0.20, "DCC": 0.25, "BI": 0.10, "SC": 0.15},
    "financial": {"RC": 0.20, "ICC": 0.20, "DCC": 0.25, "BI": 0.20, "SC": 0.15},
    "technical": {"RC": 0.15, "ICC": 0.25, "DCC": 0.25, "BI": 0.20, "SC": 0.15},
    "narrative": {"RC": 0.05, "ICC": 0.30, "DCC": 0.30, "BI": 0.20, "SC": 0.15},
}

DEFAULT_WEIGHTS = {"RC": 0.20, "ICC": 0.20, "DCC": 0.20, "BI": 0.20, "SC": 0.20}


def profile_document(text: str, config: Dict[str, Any]) -> Dict[str, Any]:
    tokens = text.split()
    token_count = len(tokens)
    doc_type = _classify_type(text)
    domain, domain_scores = _classify_domain(text, config)

    if token_count < 1000:
        length_bucket = "short"
    elif token_count <= 10000:
        length_bucket = "medium"
    else:
        length_bucket = "long"

    metrics = _compute_metrics(text, tokens, doc_type)
    adaptive_weights = _resolve_metric_weights(domain, config)
    weighted_overall = float(sum(metrics[k] * adaptive_weights[k] for k in ("RC", "ICC", "DCC", "BI", "SC")))
    uncertainty = _bootstrap_uncertainty(text, doc_type, adaptive_weights, int(config.get("bootstrap_samples", 120)))
    metrics["weighted_overall"] = round(weighted_overall, 4)
    suggested = _suggest_hyperparams(token_count, doc_type, metrics, config)

    return {
        "type": doc_type,
        "domain": domain,
        "domain_scores": domain_scores,
        "length_bucket": length_bucket,
        "token_count": token_count,
        "metrics": metrics,
        "metric_weights": adaptive_weights,
        "uncertainty": uncertainty,
        "suggested_config": suggested,
    }


# ---------------------------------------------------------------------------
# Classification helpers
# ---------------------------------------------------------------------------

def _classify_type(text: str) -> str:
    lines = text.split("\n")
    total = max(len(lines), 1)

    code_score = 0
    code_patterns = [
        r"^\s*(def |class |async def )",
        r"^\s*(import |from \w+ import )",
        r"^\s*(public |private |protected |static )",
        r"^\s*(function |const |let |var |=>)",
        r"^\s*(#include|#define|#pragma)",
        r"[{};]\s*$",
    ]
    for line in lines[:200]:
        for pat in code_patterns:
            if re.search(pat, line):
                code_score += 1
                break

    md_header_count = sum(1 for l in lines if re.match(r"^#{1,6}\s+\S", l))
    table_count = sum(1 for l in lines if re.match(r"^\s*\|.+\|", l))

    code_ratio = code_score / total
    if code_ratio > 0.08:
        return "code"
    if table_count / total > 0.08:
        return "table"
    if md_header_count >= 3 or (md_header_count > 0 and table_count > 0):
        return "mixed"
    return "prose"


def _classify_domain(text: str, config: Dict[str, Any]) -> Tuple[str, Dict[str, int]]:
    text_lower = text.lower()
    custom = config.get("domain_keywords", {})
    domains = {**DOMAIN_KEYWORDS, **(custom if isinstance(custom, dict) else {})}
    scores = {}
    for domain, kws in domains.items():
        count = 0
        for kw in kws:
            pattern = r"\b" + re.escape(str(kw).lower()) + r"\b"
            count += len(re.findall(pattern, text_lower))
        scores[domain] = count
    best = max(scores, key=scores.get)
    return (best if scores[best] > 0 else "general"), scores


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _compute_metrics(text: str, tokens: List[str], doc_type: str) -> Dict[str, float]:
    sentences = _split_sentences(text)
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    # RC — References Completeness
    ref_hits = len(re.findall(r"\[\d+\]|\([\w\s\-]+,\s*\d{4}\)", text))
    rc = float(np.clip(ref_hits / max(1, len(paragraphs)) * 1.5, 0.0, 1.0))

    # ICC — Intra-chunk Cohesion (vocabulary Jaccard between consecutive sentences)
    icc = _jaccard_consecutive(sentences) if len(sentences) >= 2 else 0.5

    # DCC — Document Contextual Coherence (Jaccard between consecutive paragraphs)
    dcc = _jaccard_consecutive(paragraphs) if len(paragraphs) >= 2 else 0.5

    # BI — Block Integrity (fraction of sentences that end with terminal punctuation)
    complete = sum(
        1 for s in sentences if s.rstrip().endswith((".", "!", "?", ":", ";"))
    )
    bi = float(complete / max(1, len(sentences)))

    # SC — Size Compliance (how close average sentence length is to ideal 10-50 words)
    lengths = [len(s.split()) for s in sentences]
    avg_len = float(np.mean(lengths)) if lengths else 20.0
    ideal_mid = 30.0
    sc = float(np.clip(1.0 - abs(avg_len - ideal_mid) / ideal_mid, 0.0, 1.0))

    overall = float(np.mean([rc, icc, dcc, bi, sc]))
    return {
        "RC": round(rc, 4),
        "ICC": round(icc, 4),
        "DCC": round(dcc, 4),
        "BI": round(bi, 4),
        "SC": round(sc, 4),
        "overall": round(overall, 4),
    }


def _resolve_metric_weights(domain: str, config: Dict[str, Any]) -> Dict[str, float]:
    user_weights = config.get("metric_weights", {})
    if isinstance(user_weights, dict) and all(k in user_weights for k in ("RC", "ICC", "DCC", "BI", "SC")):
        raw = {k: float(user_weights[k]) for k in ("RC", "ICC", "DCC", "BI", "SC")}
    else:
        raw = DOMAIN_METRIC_WEIGHTS.get(domain, DEFAULT_WEIGHTS)
    total = sum(raw.values()) or 1.0
    return {k: round(float(v / total), 4) for k, v in raw.items()}


def _bootstrap_uncertainty(
    text: str,
    doc_type: str,
    weights: Dict[str, float],
    samples: int,
) -> Dict[str, Any]:
    sentences = _split_sentences(text)
    if len(sentences) < 3:
        return {"samples": 0, "weighted_overall_ci95": [0.0, 0.0], "weighted_overall_std": 0.0}

    rng = np.random.default_rng(42)
    sample_count = max(20, min(samples, 400))
    weighted_scores: List[float] = []
    for _ in range(sample_count):
        picked = [sentences[int(i)] for i in rng.integers(0, len(sentences), size=len(sentences))]
        sampled_text = " ".join(picked)
        sampled_metrics = _compute_metrics(sampled_text, sampled_text.split(), doc_type)
        weighted_scores.append(sum(sampled_metrics[k] * weights[k] for k in ("RC", "ICC", "DCC", "BI", "SC")))

    arr = np.array(weighted_scores, dtype=np.float32)
    ci_low = float(np.percentile(arr, 2.5))
    ci_high = float(np.percentile(arr, 97.5))
    return {
        "samples": sample_count,
        "weighted_overall_ci95": [round(ci_low, 4), round(ci_high, 4)],
        "weighted_overall_std": round(float(np.std(arr)), 4),
    }


def _jaccard_consecutive(items: List[str]) -> float:
    overlaps: List[float] = []
    for i in range(len(items) - 1):
        a = set(re.findall(r"\b\w+\b", items[i].lower()))
        b = set(re.findall(r"\b\w+\b", items[i + 1].lower()))
        union = a | b
        if union:
            overlaps.append(len(a & b) / len(union))
    return float(np.mean(overlaps)) if overlaps else 0.5


def _split_sentences(text: str) -> List[str]:
    raw = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in raw if s.strip()]


# ---------------------------------------------------------------------------
# Hyperparameter suggestion
# ---------------------------------------------------------------------------

def _suggest_hyperparams(
    token_count: int,
    doc_type: str,
    metrics: Dict[str, float],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    n_min = int(config.get("n_min", 100))
    n_max = int(config.get("n_max", 500))

    # Adjust for document length
    if token_count < 1000:
        n_min = min(n_min, 50)
        n_max = min(n_max, 200)
    elif token_count > 10000:
        n_min = max(n_min, 150)
        n_max = max(n_max, 600)

    # Code docs benefit from tighter chunks
    if doc_type == "code":
        n_min = max(30, n_min - 30)
        n_max = min(300, n_max)

    # Adjust JSD thresholds based on measured coherence
    tau_low = float(config.get("tau_jsd_low", 0.15))
    tau_high = float(config.get("tau_jsd_high", 0.45))
    icc = metrics.get("ICC", 0.5)
    if icc > 0.6:
        # High cohesion doc → be stricter about merging
        tau_low = max(0.08, tau_low - 0.05)
    elif icc < 0.3:
        # Low cohesion → merge less aggressively
        tau_low = min(0.25, tau_low + 0.05)

    return {
        "n_min": n_min,
        "n_max": n_max,
        "tau_jsd_low": round(tau_low, 3),
        "tau_jsd_high": round(tau_high, 3),
        "tau_sem": float(config.get("tau_sem", 0.75)),
    }
