"""
S1 — Document Profiler
Expanded domain detection, adaptive metric weighting, and uncertainty estimates.

This module analyzes document characteristics to guide downstream chunking decisions:
- Classifies document type (code, table, mixed, prose)
- Identifies domain (legal, medical, academic, etc.)
- Computes five quality metrics: RC, ICC, DCC, BI, SC
- Provides domain-specific metric weights
- Estimates confidence via bootstrap sampling
- Suggests optimal hyperparameters based on document properties
"""

import hashlib  # For stable hashing of tokens in embeddings
import re  # Regular expressions for text pattern matching
from typing import Dict, Any, List, Tuple  # Type hints

import numpy as np  # Numerical computations and statistics

# English and French stopwords used to filter content tokens
# These are common words that don't carry semantic meaning
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall", "can",
    "to", "of", "in", "for", "on", "with", "at", "by", "from", "as", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "out", "off", "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how", "all", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "just", "this",
    "that", "these", "those", "it", "its", "he", "she", "they", "we",
    "you", "i", "my", "your", "his", "her", "our", "their", "what",
    "which", "who", "whom", "about", "le", "la", "les", "de", "des",
    "du", "et", "en", "un", "une", "dans", "pour", "que", "est", "sur",
    "par", "avec", "au", "aux",
}

STRUCTURAL_BOUNDARY_RE = re.compile(
    r"^\s*(#{1,6}\s+\S+|(?:Article|Art\.?|ARTICLE)\s+\w+|(?:Section|Chapter|Part|TITRE|CHAPITRE)\s+\w+|\d+(?:\.\d+)*\s+\S+|[-*+]\s+\S+|\|.+\|)",
    re.MULTILINE | re.IGNORECASE,
)


DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "legal": ["statute", "clause", "agreement", "liability", "jurisdiction", "indemnity", "contract", "hereby","statut", "clause", "accord", "responsabilité", "juridiction", "contrat", "par le présent"],
    "medical": ["patient", "diagnosis", "treatment", "clinical", "therapy", "symptom", "hospital", "dosage", "patient", "diagnostic", "traitement", "clinique", "thérapie", "symptôme", "hôpital"],
    "academic": ["abstract", "methodology", "citation", "hypothesis", "literature", "peer review", "thesis", "dataset", "résumé", "méthodologie", "citation", "hypothèse", "littérature", "thèse"],
    "financial": ["revenue", "profit", "loss", "equity", "fiscal", "balance sheet", "cash flow", "valuation", "revenu", "profit", "perte", "capitaux propres", "fiscal", "bilan", "flux de trésorerie", "évaluation"],
    "technical": ["algorithm", "api", "repository", "deployment", "protocol", "database", "framework", "runtime", "algorithme", "dépôt", "déploiement", "protocole", "base de données", "cadre", "runtime"],
    "narrative": ["chapter", "character", "plot", "scene", "dialogue", "protagonist", "story", "novel", "chapitre", "personnage", "intrigue", "scène", "dialogue", "protagoniste", "histoire", "roman"],
    "scientific": ["experiment", "variable", "control group", "statistical", "finding", "evidence", "sample", "observation", "expérience", "variable", "groupe de contrôle", "statistique", "résultat", "preuve", "échantillon", "observation"],
    "regulatory": ["compliance", "regulation", "audit", "governance", "policy", "risk", "control", "obligation", "conformité", "réglementation", "audit", "gouvernance", "politique", "risque", "contrôle", "obligation"],
    "marketing": ["campaign", "conversion", "audience", "brand", "segmentation", "retention", "funnel", "roi", "campagne", "conversion", "audience", "marque", "segmentation", "rétention", "entonnoir", "roi"],
    "education": ["curriculum", "assessment", "learning", "student", "teacher", "pedagogy", "course", "instruction", "curriculum", "évaluation", "apprentissage", "étudiant", "enseignant", "pédagogie", "cours", "instruction"],
    "cybersecurity": ["vulnerability", "threat", "malware", "encryption", "incident", "firewall", "authentication", "exploit", "vulnérabilité", "menace", "malware", "chiffrement", "incident", "pare-feu", "authentification", "exploit"],
    "product": ["roadmap", "feature", "release", "user story", "backlog", "ux", "adoption", "prioritization", "feuille de route", "fonctionnalité", "version", "histoire utilisateur", "backlog", "ux", "adoption", "priorisation"],
    "operations": ["workflow", "throughput", "sla", "capacity", "scheduling", "logistics", "inventory", "downtime", "flux de travail", "débit", "sla", "capacité", "planification", "logistique", "inventaire", "temps d'arrêt"],
    "policy": ["guideline", "directive", "standards", "framework", "mandate", "protocol", "code of conduct", "principle", "ligne directrice", "directive", "normes", "cadre", "mandat", "protocole", "code de conduite", "principe"],
    "research": ["benchmark", "model", "inference", "evaluation", "baseline", "ablation", "metric", "corpus", "benchmark", "modèle", "inférence", "évaluation", "baseline", "ablation", "métrique", "corpus"],
}


DOMAIN_METRIC_WEIGHTS: Dict[str, Dict[str, float]] = {
    # ── Highly structured domains: RC dominates because structure IS the content ──
    # Legal: articles, clauses, and numbered provisions are the primary unit.
    # High RC weight because structural anchors (Article N, §, CHAPITRE) are
    # perfectly reliable split points.  Low ICC because legal writing deliberately
    # uses sparse cross-sentence vocabulary (definitions referenced by number).
    "legal":        {"RC": 0.35, "ICC": 0.10, "DCC": 0.20, "BI": 0.25, "SC": 0.10},

    # Regulatory: same logic as legal.  Compliance documents have explicit article
    # numbering.  BI is weighted to detect extraction noise (scanned regulations).
    "regulatory":   {"RC": 0.35, "ICC": 0.10, "DCC": 0.20, "BI": 0.25, "SC": 0.10},

    # Policy: directive/guideline documents have headings and numbered items.
    # Slightly more ICC weight than legal because policy prose is more narrative.
    "policy":       {"RC": 0.30, "ICC": 0.15, "DCC": 0.20, "BI": 0.20, "SC": 0.15},

    # ── Moderate structure domains ────────────────────────────────────────────
    # Academic: abstract/methods/results/discussion structure.  High RC for
    # section headers; high DCC because adjacent sections are thematically linked.
    "academic":     {"RC": 0.30, "ICC": 0.15, "DCC": 0.25, "BI": 0.15, "SC": 0.15},

    # Research: same as academic but heavier ICC (method descriptions are dense
    # prose with strong sentence continuity within a section).
    "research":     {"RC": 0.25, "ICC": 0.20, "DCC": 0.25, "BI": 0.15, "SC": 0.15},

    # Medical: clinical reports have structured sections (Diagnosis, Treatment).
    # Higher BI weight because OCR-scanned medical PDFs often have extraction noise.
    "medical":      {"RC": 0.20, "ICC": 0.25, "DCC": 0.25, "BI": 0.20, "SC": 0.10},

    # Financial: tables, figures, and section headings coexist with dense prose.
    # SC is higher because irregular unit sizes hurt financial chunk usability.
    "financial":    {"RC": 0.20, "ICC": 0.20, "DCC": 0.25, "BI": 0.20, "SC": 0.15},

    # ── Technical / product domains ───────────────────────────────────────────
    # Technical: API docs, architecture specs.  High ICC (procedure steps are
    # tightly coupled).  High DCC (topics chain: concept → implementation → test).
    "technical":    {"RC": 0.15, "ICC": 0.25, "DCC": 0.25, "BI": 0.20, "SC": 0.15},

    # Cybersecurity: incident reports and threat briefs are dense, structured prose.
    "cybersecurity":{"RC": 0.20, "ICC": 0.25, "DCC": 0.25, "BI": 0.15, "SC": 0.15},

    # Product: roadmaps, backlogs — short items, high variance in unit size.
    # High SC weight because very short items (user stories) need careful packing.
    "product":      {"RC": 0.20, "ICC": 0.20, "DCC": 0.20, "BI": 0.15, "SC": 0.25},

    # Operations: SLA documents, workflow guides — mixed table/prose.
    "operations":   {"RC": 0.20, "ICC": 0.20, "DCC": 0.20, "BI": 0.20, "SC": 0.20},

    # ── Unstructured / narrative domains ─────────────────────────────────────
    # Narrative: fiction/reports without formal structure.  ICC and DCC dominate
    # because story flow is the primary coherence signal.  RC minimal.
    "narrative":    {"RC": 0.05, "ICC": 0.30, "DCC": 0.30, "BI": 0.20, "SC": 0.15},

    # Marketing: campaign briefs — short, punchy, varied.  ICC moderate.
    "marketing":    {"RC": 0.15, "ICC": 0.25, "DCC": 0.25, "BI": 0.15, "SC": 0.20},

    # Education: course materials — structured headings + explanatory prose.
    "education":    {"RC": 0.25, "ICC": 0.25, "DCC": 0.20, "BI": 0.15, "SC": 0.15},

    # Scientific: lab reports, papers — hypothesis/evidence structure.
    "scientific":   {"RC": 0.25, "ICC": 0.20, "DCC": 0.25, "BI": 0.15, "SC": 0.15},
}

# Fallback weights for any domain not explicitly listed above.
# Equal weighting is the most neutral choice when we have no prior knowledge.
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
    metric_details = _compute_metric_details(text, tokens, doc_type, metrics)
    suggested = _suggest_hyperparams(token_count, doc_type, metrics, config)

    return {
        "type": doc_type,
        "domain": domain,
        "domain_scores": domain_scores,
        "length_bucket": length_bucket,
        "token_count": token_count,
        "metrics": metrics,
        "metric_details": metric_details,
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
    """
    Classify the document domain by keyword frequency.

    TIE-BREAKING RULE
    ─────────────────
    When multiple domains score within 20% of the top score, we use a
    priority ordering to pick the most specific one.  This matters for
    documents like tax conventions, which score high on both "financial"
    (revenue, profit, equity...) and "regulatory"/"legal" (statute, clause,
    jurisdiction...).  A tax convention IS a legal instrument — "regulatory"
    or "legal" is more precise than "financial" for chunking purposes because
    it triggers the correct metric weights (RC-dominant) and chunking strategy
    (legal_articles / hybrid_legal_semantic).

    Priority order (most specific → least specific):
      legal > regulatory > policy > medical > academic > scientific
      > cybersecurity > technical > financial > operations
      > product > education > marketing > narrative > general
    """
    text_lower = text.lower()
    custom  = config.get("domain_keywords", {})
    domains = {**DOMAIN_KEYWORDS, **(custom if isinstance(custom, dict) else {})}

    scores: Dict[str, int] = {}
    for domain, kws in domains.items():
        count = 0
        for kw in kws:
            pattern = r"\b" + re.escape(str(kw).lower()) + r"\b"
            count += len(re.findall(pattern, text_lower))
        scores[domain] = count

    if not scores or max(scores.values()) == 0:
        return "general", scores

    top_score = max(scores.values())

    # Collect all domains within 20% of the top score
    close_threshold = top_score * 0.80
    candidates = [d for d, s in scores.items() if s >= close_threshold]

    # Priority ordering — earlier = more specific/preferred for chunking
    _PRIORITY = [
        "legal", "regulatory", "policy",
        "medical", "academic", "scientific",
        "cybersecurity", "technical", "research",
        "financial", "operations", "product",
        "education", "marketing", "narrative",
    ]

    # Among tied candidates, pick the highest-priority one
    for preferred in _PRIORITY:
        if preferred in candidates:
            return preferred, scores

    # Fallback: return the raw maximum
    return max(scores, key=scores.get), scores


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def _compute_metrics(text: str, tokens: List[str], doc_type: str) -> Dict[str, float]:
    sentences = _split_sentences(text)
    paragraphs = _split_blocks(text)
    content_tokens = _content_tokens(text)
    quality_gate = _document_quality_gate(text, content_tokens)

    # RC - Retrieval cues: headings, articles, lists, references, code symbols,
    # table rows, and named anchors that help a chunker find durable boundaries.
    rc = min(_retrieval_cue_score(text, paragraphs, doc_type), max(0.25, quality_gate))

    # ICC - Local semantic continuity. Adjacent sentences should be related, but
    # not identical. We combine lexical overlap and lightweight hashed vectors.
    icc = min(_local_cohesion(sentences, doc_type), quality_gate)

    # DCC - Block-level document continuity. Adjacent paragraphs/sections should
    # carry a manageable amount of shared vocabulary or topic signal.
    dcc = min(_block_coherence(paragraphs, doc_type), quality_gate)

    # BI - Block integrity: detects parseable structural units, clean endings,
    # balanced delimiters, and low extraction noise.
    bi = min(_block_integrity(text, sentences, paragraphs, doc_type), 0.35 + 0.65 * quality_gate)

    # SC - Size regularity for chunking. This is no longer sentence-length only;
    # it estimates whether natural units can be packed into useful chunk sizes.
    sc = min(_size_compliance(sentences, paragraphs, content_tokens, doc_type), 0.30 + 0.70 * quality_gate)

    overall = float(np.mean([rc, icc, dcc, bi, sc]))
    return {
        "RC": round(rc, 4),
        "ICC": round(icc, 4),
        "DCC": round(dcc, 4),
        "BI": round(bi, 4),
        "SC": round(sc, 4),
        "overall": round(overall, 4),
    }


def _compute_metric_details(
    text: str,
    tokens: List[str],
    doc_type: str,
    metrics: Dict[str, float],
) -> Dict[str, Dict[str, Any]]:
    sentences = _split_sentences(text)
    blocks = _split_blocks(text)
    content_tokens = _content_tokens(text)
    quality_gate = _document_quality_gate(text, content_tokens)
    cue_counts = _retrieval_cue_counts(text)
    block_lengths = [len(_content_tokens(b)) for b in blocks if b.strip()]
    sentence_lengths = [len(_content_tokens(s)) for s in sentences if s.strip()]
    delimiter_score = _balanced_delimiter_score(text)
    noisy_lines = sum(1 for ln in text.splitlines() if 0 < len(ln.strip()) <= 2)

    return {
        "RC": {
            "label": "Structure Signals",
            "score": metrics.get("RC", 0.0),
            "reason": _metric_reason("RC", metrics.get("RC", 0.0)),
            "subscores": {
                "headings": cue_counts["headings"],
                "legal_or_sections": cue_counts["legal_or_sections"],
                "lists": cue_counts["lists"],
                "tables": cue_counts["tables"],
                "references": cue_counts["references"],
                "code_anchors": cue_counts["code_anchors"],
                "block_count": len(blocks),
            },
        },
        "ICC": {
            "label": "Sentence Flow",
            "score": metrics.get("ICC", 0.0),
            "reason": _metric_reason("ICC", metrics.get("ICC", 0.0)),
            "subscores": {
                "sentence_count": len(sentences),
                "avg_sentence_tokens": round(float(np.mean(sentence_lengths)), 2) if sentence_lengths else 0.0,
                "quality_gate": round(quality_gate, 4),
            },
        },
        "DCC": {
            "label": "Section Flow",
            "score": metrics.get("DCC", 0.0),
            "reason": _metric_reason("DCC", metrics.get("DCC", 0.0)),
            "subscores": {
                "block_count": len(blocks),
                "avg_block_tokens": round(float(np.mean(block_lengths)), 2) if block_lengths else 0.0,
                "quality_gate": round(quality_gate, 4),
            },
        },
        "BI": {
            "label": "Text Cleanliness",
            "score": metrics.get("BI", 0.0),
            "reason": _metric_reason("BI", metrics.get("BI", 0.0)),
            "subscores": {
                "balanced_delimiters": round(delimiter_score, 4),
                "short_noise_lines": noisy_lines,
                "replacement_chars": text.count("\x00") + text.count("�"),
                "structural_boundaries": len(STRUCTURAL_BOUNDARY_RE.findall(text)),
            },
        },
        "SC": {
            "label": "Chunkability",
            "score": metrics.get("SC", 0.0),
            "reason": _metric_reason("SC", metrics.get("SC", 0.0)),
            "subscores": {
                "natural_units": len(block_lengths),
                "avg_unit_tokens": round(float(np.mean(block_lengths)), 2) if block_lengths else 0.0,
                "unit_token_std": round(float(np.std(block_lengths)), 2) if block_lengths else 0.0,
                "document_tokens": len(tokens),
            },
        },
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
    raw = re.split(r"(?<=[.!?])\s+|\n+(?=\s*(?:#{1,6}\s+|Article\s+\w+|[-*+]\s+|\d+[.)]\s+))", text)
    return [s.strip() for s in raw if s.strip()]


def _split_blocks(text: str) -> List[str]:
    blocks = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(blocks) <= 1:
        blocks = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return blocks or ([text.strip()] if text.strip() else [])


def _content_tokens(text: str) -> List[str]:
    toks = [
        t for t in re.findall(r"\b[\wÀ-ÿ]{2,}\b", text.lower())
        if t not in STOPWORDS
    ]
    return toks or re.findall(r"\b\w+\b", text.lower())


def _document_quality_gate(text: str, content_tokens: List[str]) -> float:
    if not text.strip():
        return 0.0
    chars = len(text)
    alnum = sum(1 for ch in text if ch.isalnum())
    alpha_ratio = alnum / max(chars, 1)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    short_line_ratio = sum(1 for ln in lines if len(ln) <= 2) / max(len(lines), 1)
    replacement_ratio = (text.count("\x00") + text.count("�")) / max(chars, 1)
    punctuation_spam = len(re.findall(r"[^\w\sÀ-ÿ]{4,}", text)) / max(len(lines), 1)

    unique_ratio = len(set(content_tokens)) / max(len(content_tokens), 1)
    if len(content_tokens) < 20:
        diversity = 0.65
    else:
        diversity = min(1.0, unique_ratio * 4.0)

    alpha_score = float(np.clip((alpha_ratio - 0.20) / 0.45, 0.0, 1.0))
    line_score = 1.0 - min(0.75, short_line_ratio)
    noise_score = 1.0 - min(0.95, replacement_ratio * 80.0 + punctuation_spam * 0.18)
    gate = 0.34 * alpha_score + 0.26 * diversity + 0.22 * line_score + 0.18 * noise_score
    return float(np.clip(gate, 0.0, 1.0))


def _retrieval_cue_score(text: str, blocks: List[str], doc_type: str) -> float:
    total_blocks = max(1, len(blocks))
    counts = _retrieval_cue_counts(text)

    cue_density = (
        1.35 * counts["headings"]
        + 1.35 * counts["legal_or_sections"]
        + 0.45 * counts["lists"]
        + 0.55 * counts["tables"]
        + 0.75 * counts["references"]
        + 1.10 * counts["code_anchors"]
    ) / total_blocks
    density_score = 1.0 - float(np.exp(-cue_density))

    if doc_type in {"code", "table", "mixed"}:
        baseline = 0.35
    else:
        baseline = 0.25
    return float(np.clip(max(density_score, baseline if total_blocks >= 2 else 0.15), 0.0, 1.0))


def _retrieval_cue_counts(text: str) -> Dict[str, int]:
    return {
        "headings": len(re.findall(r"(?m)^\s*#{1,6}\s+\S+", text)),
        "legal_or_sections": len(re.findall(r"(?im)^\s*(?:Article|Art\.?|ARTICLE|Section|Chapter|TITRE|CHAPITRE|الفصل|الباب|القسم)\s+\w+", text)),
        "lists": len(re.findall(r"(?m)^\s*(?:[-*+]|\d+[.)])\s+\S+", text)),
        "tables": len(re.findall(r"(?m)^\s*\|.+\|\s*$", text)),
        "references": len(re.findall(r"\[\d+\]|\b(?:section|article|chapter|figure|table)\s+\d+|\([\w\s\-]+,\s*\d{4}\)", text, re.I)),
        "code_anchors": len(re.findall(r"(?m)^\s*(?:def |class |function |import |from \w+ import |#include|const |let )", text)),
    }


def _local_cohesion(sentences: List[str], doc_type: str) -> float:
    if len(sentences) < 2:
        return 0.6
    vals: List[float] = []
    for i in range(len(sentences) - 1):
        a_tokens = set(_content_tokens(sentences[i]))
        b_tokens = set(_content_tokens(sentences[i + 1]))
        union = a_tokens | b_tokens
        jaccard = len(a_tokens & b_tokens) / len(union) if union else 0.0
        cosine = _hashed_cosine(a_tokens, b_tokens)
        pair_score = 0.35 * min(jaccard * 3.0, 1.0) + 0.65 * cosine
        vals.append(pair_score)
    raw = float(np.mean(vals)) if vals else 0.6
    if doc_type in {"code", "table"}:
        raw = 0.75 * raw + 0.15
    return float(np.clip(raw, 0.0, 1.0))


def _block_coherence(blocks: List[str], doc_type: str) -> float:
    if len(blocks) < 2:
        return 0.6
    vals: List[float] = []
    for i in range(len(blocks) - 1):
        a = set(_content_tokens(blocks[i]))
        b = set(_content_tokens(blocks[i + 1]))
        vals.append(0.40 * min((len(a & b) / len(a | b) if (a | b) else 0.0) * 3.0, 1.0) + 0.60 * _hashed_cosine(a, b))
    raw = float(np.mean(vals)) if vals else 0.6
    if doc_type in {"table", "mixed"}:
        raw = 0.80 * raw + 0.10
    return float(np.clip(raw, 0.0, 1.0))


def _block_integrity(text: str, sentences: List[str], blocks: List[str], doc_type: str) -> float:
    if not text.strip():
        return 0.0
    endings = 0
    for unit in sentences or blocks:
        stripped = unit.rstrip()
        if stripped.endswith((".", "!", "?", ":", ";", "}", "]", "`")) or _is_structural_line(stripped):
            endings += 1
    ending_score = endings / max(1, len(sentences or blocks))

    delimiter_score = _balanced_delimiter_score(text)
    noise_chars = text.count("\x00") + text.count("�")
    very_short_lines = sum(1 for ln in text.splitlines() if 0 < len(ln.strip()) <= 2)
    line_count = max(1, len([ln for ln in text.splitlines() if ln.strip()]))
    extraction_score = 1.0 - min(0.45, noise_chars / max(len(text), 1) * 40.0) - min(0.25, very_short_lines / line_count)
    structural_score = min(1.0, len(STRUCTURAL_BOUNDARY_RE.findall(text)) / max(2.0, len(blocks) * 0.25))

    if doc_type == "code":
        return float(np.clip(0.30 * ending_score + 0.45 * delimiter_score + 0.25 * extraction_score, 0.0, 1.0))
    return float(np.clip(0.42 * ending_score + 0.28 * delimiter_score + 0.20 * extraction_score + 0.10 * structural_score, 0.0, 1.0))


def _size_compliance(sentences: List[str], blocks: List[str], tokens: List[str], doc_type: str) -> float:
    total = len(tokens)
    if total == 0:
        return 0.0
    if total < 80:
        return 0.75

    unit_lengths = [len(_content_tokens(b)) for b in blocks if b.strip()]
    if not unit_lengths:
        unit_lengths = [len(_content_tokens(s)) for s in sentences if s.strip()]
    if not unit_lengths:
        return 0.5

    ideal_min, ideal_max = (40, 180)
    if doc_type == "code":
        ideal_min, ideal_max = (20, 140)
    elif doc_type == "table":
        ideal_min, ideal_max = (15, 120)

    packable = sum(1 for n in unit_lengths if 4 <= n <= ideal_max)
    packable_score = packable / len(unit_lengths)
    avg = float(np.mean(unit_lengths))
    target = (ideal_min + ideal_max) / 2.0
    avg_score = 1.0 - min(1.0, abs(avg - target) / target)
    variance_score = 1.0 - min(1.0, float(np.std(unit_lengths)) / max(avg, 1.0))
    enough_units = min(1.0, len(unit_lengths) / max(3.0, total / 450.0))

    return float(np.clip(0.34 * packable_score + 0.26 * avg_score + 0.20 * variance_score + 0.20 * enough_units, 0.0, 1.0))


def _hashed_cosine(a_tokens: set, b_tokens: set, dim: int = 128) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    a = np.zeros(dim, dtype=np.float32)
    b = np.zeros(dim, dtype=np.float32)
    for tok in a_tokens:
        a[_stable_hash(tok, dim)] += 1.0
    for tok in b_tokens:
        b[_stable_hash(tok, dim)] += 1.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _stable_hash(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8", errors="ignore"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def _balanced_delimiter_score(text: str) -> float:
    pairs = [("(", ")"), ("[", "]"), ("{", "}")]
    penalties = []
    for left, right in pairs:
        total = text.count(left) + text.count(right)
        if total == 0:
            penalties.append(0.0)
        else:
            penalties.append(abs(text.count(left) - text.count(right)) / total)
    return float(np.clip(1.0 - np.mean(penalties), 0.0, 1.0))


def _is_structural_line(line: str) -> bool:
    return bool(STRUCTURAL_BOUNDARY_RE.match(line.strip()))


def _metric_reason(key: str, score: float) -> str:
    level = "strong" if score >= 0.70 else "usable" if score >= 0.40 else "weak"
    reasons = {
        "RC": {
            "strong": "The document has clear anchors such as headings, sections, lists, tables, references, or code symbols.",
            "usable": "The document has some structural anchors, but not enough to rely on structure alone.",
            "weak": "Few reliable anchors were found, so semantic or paragraph-based chunking may be safer.",
        },
        "ICC": {
            "strong": "Nearby sentences stay on related topics.",
            "usable": "Sentence-to-sentence flow is mixed but still usable.",
            "weak": "Adjacent sentences share little topic signal, so aggressive merging should be avoided.",
        },
        "DCC": {
            "strong": "Adjacent paragraphs or sections connect consistently.",
            "usable": "Paragraph flow is partially coherent.",
            "weak": "Paragraphs or sections look weakly connected, so boundary detection should be conservative.",
        },
        "BI": {
            "strong": "The extracted text appears clean and structurally complete.",
            "usable": "The text is mostly usable but has some structural or extraction issues.",
            "weak": "The text may contain broken lines, malformed blocks, delimiter imbalance, or extraction noise.",
        },
        "SC": {
            "strong": "Natural document units are well sized for chunking.",
            "usable": "Natural units can be chunked, but size variance may need adjustment.",
            "weak": "Natural units are poorly sized for chunking, so Stage 2 should repack or split carefully.",
        },
    }
    return reasons.get(key, {}).get(level, "Metric computed from document structure and content.")


# ---------------------------------------------------------------------------
# Hyperparameter suggestion
# ---------------------------------------------------------------------------

def _suggest_hyperparams(
    token_count: int,
    doc_type: str,
    metrics: Dict[str, float],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Suggest starting hyperparameters for S2–S3 based on document characteristics.

    These values are returned as suggested_config AND are written back into the
    live config dict so that downstream stages (S2, S3, S4) actually use them
    as defaults when the caller has not explicitly overridden a key.

    Previously, the suggestion was purely informational — it was computed but
    never propagated.  This caused S7 to start from the API default n_max=500
    instead of the document-derived suggestion, wasting early BO trials.

    Propagation rule: write into config[key] only if the key is not already
    set by the user (same rule as warm-start in S7).  This preserves explicit
    user overrides.
    """
    # Start from user-supplied values or sensible defaults
    n_min = int(config.get("n_min", 100))
    n_max = int(config.get("n_max", 500))

    # ── Adjust chunk size for document length ─────────────────────────────────
    if token_count < 1000:
        # Short document: smaller chunks to avoid single-chunk output
        n_min = min(n_min, 50)
        n_max = min(n_max, 250)
    elif token_count > 10_000:
        # Long document: larger chunks are more practical for retrieval
        n_min = max(n_min, 150)
        n_max = max(n_max, 600)

    # ── Doc-type adjustments ──────────────────────────────────────────────────
    if doc_type == "code":
        # Code functions/classes are typically short; tight chunks preserve them
        n_min = max(30, n_min - 30)
        n_max = min(300, n_max)
    elif doc_type == "table":
        # Table rows should stay together; allow small minimum
        n_min = max(15, n_min // 2)

    # ── Adjust JSD thresholds based on measured coherence (ICC) ──────────────
    tau_low  = float(config.get("tau_jsd_low",  0.15))
    tau_high = float(config.get("tau_jsd_high", 0.45))
    rc  = metrics.get("RC",  0.5)
    icc = metrics.get("ICC", 0.5)

    if icc > 0.60:
        # High sentence-level cohesion → raise the merge bar (be stricter)
        tau_low = max(0.08, tau_low - 0.05)
    elif icc < 0.30:
        # Low cohesion → lower the merge bar (merge less aggressively)
        tau_low = min(0.25, tau_low + 0.05)

    if rc > 0.80:
        # Strong structure signal → tighten the hard-split threshold so that
        # structural markers are captured more readily
        tau_high = max(0.30, tau_high - 0.05)

    # Build the suggestion dict
    suggested = {
        "n_min":        n_min,
        "n_max":        n_max,
        "tau_jsd_low":  round(tau_low,  3),
        "tau_jsd_high": round(tau_high, 3),
        "tau_sem":      float(config.get("tau_sem", 0.75)),
    }

    # ── Propagate into live config (only for keys the user did not set) ───────
    # This ensures S2 and S3 use these values as starting defaults rather than
    # ignoring the profiler's analysis entirely.
    for key, val in suggested.items():
        if key not in config:
            config[key] = val

    return suggested