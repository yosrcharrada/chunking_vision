"""
AutoChunker Platform — FastAPI Backend
Provides 5 REST endpoints that orchestrate the 7-stage pipeline.
All heavy work is executed in a background thread so the HTTP server
stays responsive; progress is polled via GET /status/{job_id}.
"""

# Import standard libraries for file I/O, JSON manipulation, regular expressions, threading, logging, etc.
import os # For OS operations like file path handling
os.environ["HF_HUB_OFFLINE"] = "1"   # use local cache only, no network calls
import io
import csv
import json
import re
import uuid  # For generating unique job IDs
import threading  # For running pipeline in background threads
import traceback  # For detailed error reporting
import logging  # For logging events and debugging

import time  # For timing operations
import chardet

from typing import Any, Dict, List, Optional  # Type hints for function parameters and return values

# FastAPI framework and utilities for HTTP server
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware  # Enable cross-origin requests
from fastapi.responses import JSONResponse, Response  # Response types

# Constants
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # Maximum file size limit: 50 MB

# Configure application logger with JSON format for structured logging
logger = logging.getLogger("autochunker")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
    )

# ── Import Pipeline Stages ───────────────────────────────────────────────────────
# S1: Profile document to understand its structure, domain, and quality metrics
from pipeline.s1_profiler import profile_document
from pipeline.s2_chunkers import run_all_chunkers, select_best_strategy
from pipeline.s3_entropy import refine_boundaries, get_jsd_series
from pipeline.s4_boundary import filter_boundaries
from pipeline.s5_graph import enrich_graph, build_entity_graph_data
from pipeline.s6_embedding import embed_chunks
from pipeline.s7_rl import run_rl_loop

# Post-merge: S3 is qentropy-driven, so the old GPT-2 perplexity validator and
# its model preload are gone.  Evaluation (engine/metrics.py) and QA generation
# (engine/qagen.py) use the shared EmbeddingService + optional OpenAI key.
from pipeline.evaluation import build_context, score_run, rank_runs
from engine import openai_client  # loads .env + corporate CA at import; selects OpenAI vs Azure/EY
from engine import qagen
from engine.embeddings import get_embedder

import warnings
# Quieten the harmless third-party noise that otherwise floods the server log
# (HuggingFace `resume_download` FutureWarning on every model load, and the
# requests/urllib3/chardet version-mismatch notice in this environment).
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
warnings.filterwarnings("ignore", message=".*position_ids.*")
warnings.filterwarnings("ignore", message=".*masked_bias.*")
warnings.filterwarnings("ignore", message=".*resume_download.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
try:
    from requests.exceptions import RequestsDependencyWarning
    warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
except Exception:
    pass


# ── In-memory stores ──────────────────────────────────────────────────────
doc_store: Dict[str, Dict] = {}   # document_id → {filename, content, …}
job_store: Dict[str, Dict] = {}   # job_id      → {status, stage, progress, …}

# ── Default pipeline configuration ───────────────────────────────────────
DEFAULT_CONFIG: Dict[str, Any] = {
    # ── S2 sizing ────────────────────────────────────────────────────────
    "n_min": 80,
    "n_max": 500,
    # ── S3 qentropy (the only entropy engine now) ────────────────────────
    "q_entropy_param": 1.0,        # Tsallis q ∈ [-1, 1]; 1.0 == Shannon baseline
    "K": 4,                        # tree branching factor (paper's single knob)
    "min_chunk_tokens": 20,        # qentropy feasibility floor
    "max_chunk_tokens": 320,
    "window": 1,                   # neighbour-averaging window for the shift signal
    "overlap": 0,
    # ── S4 ───────────────────────────────────────────────────────────────
    "tau_sem": 0.75,               # similarity-merge threshold
    "s4_similarity": "cosine",     # "cosine" (classical) | "qcosine" (Fitouhi–Bouzeffour q-cosine, base=q²)
    # ── S6 / metrics embedding backend (engine.embeddings) ───────────────
    "embedding_backend": None,     # None → openai if key else multilingual
    "embedding_model": "all-MiniLM-L6-v2",  # legacy alias kept for s6_embedding
    # ── Evaluation (engine.metrics, Table I) ─────────────────────────────
    "judge_answerability": False,  # opt-in LLM judge (needs OPENAI_API_KEY)
    "qa_count": 12,                # auto-generated eval questions (engine.qagen)
    "rel_threshold": "auto",
    # ── S7 genetic algorithm ─────────────────────────────────────────────
    "ga_population": 6,
    "ga_generations": 3,
    "ga_workers": 4,
    "max_iterations": 30,
}

# ─────────────────────────────────────────────────────────────────────────
app = FastAPI(title="AutoChunker Platform", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _load_config_yaml() -> Dict[str, Any]:
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "config.yaml"))
    if not os.path.exists(path):
        return {}
    try:
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


_yaml_cfg = _load_config_yaml()
DEFAULT_CONFIG = {**DEFAULT_CONFIG, **(_yaml_cfg.get("pipeline", {}) if isinstance(_yaml_cfg, dict) else {})}


# ═══════════════════════════════════════════════════════════════════════════
# Helper: parse uploaded file to plain text
# ═══════════════════════════════════════════════════════════════════════════

def _parse_file(filename: str, content: bytes) -> str:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"

    # Detect PDFs by content too, so files without a .pdf extension still parse.
    if ext == "pdf" or content[:5] == b"%PDF-":
        return _parse_pdf(content)
    # Office/OpenDocument files are zip archives — decoding them as text yields
    # garbage, so fail clearly instead.
    if content[:4] == b"PK\x03\x04":
        raise ValueError(
            "Office documents (.docx/.xlsx/.pptx/.odt) are not supported; "
            "please provide a PDF or a plain-text/code file."
        )
    # TXT, MD, code files — decode as UTF-8 with fallback
    try:
        return _sanitize_text(content.decode("utf-8"))
    except UnicodeDecodeError:
        return _sanitize_text(content.decode("latin-1", errors="replace"))


def _parse_pdf(content: bytes) -> str:
    """
    Extract plain text from PDF bytes.

    Strategy (in order of reliability):
      1. PyMuPDF (fitz)   — best encoding support, handles custom ToUnicode maps
      2. pdfplumber        — good layout preservation
      3. pdfminer.six      — robust for malformed PDFs
      4. PyPDF2            — fast fallback
      5. Raw decode        — last resort

    PyMuPDF is first because it correctly applies PDF ToUnicode CMap tables,
    which fixes the garbled character encoding seen with custom-font PDFs.
    """

    pages: List[str] = []

    # ── Method 1: PyMuPDF / fitz (preferred — best encoding support) ─────
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(stream=io.BytesIO(content), filetype="pdf")
        for page in doc:
            text = page.get_text(
                "text",
                flags=fitz.TEXT_PRESERVE_WHITESPACE | fitz.TEXT_MEDIABOX_CLIP,
            )
            if text:
                pages.append(text)
        doc.close()
    except Exception:
        pass

    # ── Method 2: pdfplumber ─────────────────────────────────────────────
    if sum(len(p.strip()) for p in pages) < 200:
        pages = []
        try:
            import pdfplumber
            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for page in pdf.pages:
                    extracted = page.extract_text(x_tolerance=2, y_tolerance=2)
                    if extracted:
                        pages.append(extracted)
        except Exception:
            pass

    # ── Method 3: pdfminer.six ───────────────────────────────────────────
    if sum(len(p.strip()) for p in pages) < 200:
        try:
            from pdfminer.high_level import extract_text
            text = extract_text(io.BytesIO(content))
            if text and len(text.strip()) > 50:
                pages = [text]
        except Exception:
            pass

    # ── Method 4: PyPDF2 fallback ─────────────────────────────────────────
    if sum(len(p.strip()) for p in pages) < 200:
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            pages = [
                reader.pages[i].extract_text() or ""
                for i in range(len(reader.pages))
            ]
        except Exception:
            pass

    # ── Method 5: raw decode fallback ────────────────────────────────────
    if not pages or sum(len(p.strip()) for p in pages) < 50:
        detected = chardet.detect(content[:10_000])
        enc = detected.get("encoding") or "latin-1"
        try:
            return _sanitize_text(content.decode(enc, errors="replace"))
        except Exception:
            return _sanitize_text(content.decode("utf-8", errors="replace"))

    # ── Clean each page ───────────────────────────────────────────────────
    cleaned = [_clean_pdf_page(p) for p in pages if p.strip()]

    # ── Strip repeated header/footer lines ───────────────────────────────
    if len(cleaned) >= 3:
        cleaned = _strip_repeated_lines(cleaned)

    # ── Join pages and fix encoding artifacts ─────────────────────────────
    raw_text = "\n\n".join(cleaned)
    fixed_text = _fix_pdf_encoding(raw_text)
    return _sanitize_text(fixed_text)

def _fix_pdf_encoding(text: str) -> str:
    """
    Correct encoding mojibake in PDF text extraction.
    
    PDFs with complex encodings often get misinterpreted by text extractors.
    This function tries multiple encoding pairs to detect and fix the issue.
    """
 
    if not text:
        return text
 
    # ── Fix PDF CID character references ────────────────────────────────
    text = re.sub(r"\s*\(cid:\s*\d+\s*\)\s*", " ", text)
    text = re.sub(r" +", " ", text)
 
    # ── Try multiple encoding pairs and pick the best result ────────────
    # This is the key: try different re-encoding combinations
    encoding_pairs = [
        ("latin-1", "windows-1252"),
        ("iso-8859-1", "cp1252"),
        ("utf-8", "latin-1"),
        ("cp1252", "latin-1"),
        ("iso-8859-5", "utf-8"),
    ]
    
    best_text = text
    best_score = _score_text_quality(text)
    
    for source_enc, target_enc in encoding_pairs:
        try:
            # Try to re-interpret the text with different encoding
            fixed = text.encode(source_enc, errors="ignore").decode(target_enc, errors="ignore")
            score = _score_text_quality(fixed)
            
            # Keep the version with the highest quality score
            if score > best_score and fixed.strip():
                best_text = fixed
                best_score = score
        except Exception:
            continue
    
    text = best_text
 
    # ── Fix common PDF ligature and typographic artifacts ─────────────────
    replacements = {
        "\ufb01": "fi",    # ﬁ  fi-ligature
        "\ufb02": "fl",    # ﬂ  fl-ligature
        "\ufb03": "ffi",   # ﬃ  ffi-ligature
        "\ufb04": "ffl",   # ﬄ  ffl-ligature
        "\u2019": "'",     # '  right single quotation mark
        "\u2018": "'",     # '  left single quotation mark
        "\u201c": '"',     # "  left double quotation mark
        "\u201d": '"',     # "  right double quotation mark
        "\u2013": "-",     # –  en dash
        "\u2014": "--",    # —  em dash
        "\u00a0": " ",     # non-breaking space
        "\u00ad": "",      # soft hyphen
        "\u000c": "\n",    # form feed
    }
    for bad_char, replacement in replacements.items():
        text = text.replace(bad_char, replacement)
 
    return text


def _score_text_quality(text: str) -> float:
    """
    Score text quality based on readable characters and patterns.
    Higher score = better quality (fewer mojibake artifacts).
    
    Heuristic: count ratio of readable characters vs. control/unusual chars.
    """
    if not text:
        return 0.0
    
    # French accented characters (common in documents)
    accented = "éèêëàâùûôîïçœæÉÈÊËÀÂÙÛÔÎÏÇŒÆüÜöÖäÄ"
    
    # Count various character types
    ascii_letters = sum(1 for c in text if c.isalpha() and ord(c) < 128)
    accented_count = sum(1 for c in text if c in accented)
    digits = sum(1 for c in text if c.isdigit())
    spaces = sum(1 for c in text if c.isspace())
    
    # Count suspicious characters (mojibake indicators)
    # These are rarely in clean text but common in encoding errors
    suspicious = sum(1 for c in text if ord(c) in range(0x80, 0xA0) or (ord(c) > 127 and c not in accented))
    
    # Calculate quality score
    total_chars = len(text)
    readable_chars = ascii_letters + accented_count + digits + spaces
    
    # Base score: ratio of readable to total characters
    base_score = readable_chars / total_chars if total_chars > 0 else 0.0
    
    # Penalty for suspicious characters
    suspicious_penalty = (suspicious / total_chars) * 0.5 if total_chars > 0 else 0.0
    
    # Bonus for accented characters (indicator of proper encoding for French/European text)
    accented_bonus = (accented_count / total_chars) * 0.3 if total_chars > 0 else 0.0
    
    score = base_score - suspicious_penalty + accented_bonus
    return max(0.0, score)

def _sanitize_text(text: str) -> str:
    clean = text.replace("\x00", " ")
    clean = clean.replace("\r\n", "\n")
    return clean.strip()


# Matches a line that contains only a page number: a bare integer, optionally
# surrounded by dashes or hyphens, e.g. "3", "- 3 -", "3 of 12".
_PAGE_NUM_RE = re.compile(r"^\s*[-–—]?\s*\d+\s*(?:[-–—]|of\s+\d+)?\s*$")


def _clean_pdf_page(page_text: str) -> str:
    """Remove standalone page-number lines from a single page's text."""
    lines = page_text.split("\n")
    kept = [ln for ln in lines if not _PAGE_NUM_RE.match(ln)]
    return "\n".join(kept)


def _strip_repeated_lines(pages: List[str]) -> List[str]:
    """Remove header/footer lines that appear verbatim on most pages."""
    from collections import Counter

    # Count how often each short line (≤ 12 words) appears across pages
    freq: Counter = Counter()
    for page in pages:
        # Only look at the first 3 and last 3 lines (where headers/footers live)
        lines = [ln.strip() for ln in page.split("\n") if ln.strip()]
        candidates = (lines[:3] + lines[-3:]) if len(lines) > 6 else lines
        for ln in set(candidates):
            if 1 <= len(ln.split()) <= 12:
                freq[ln] += 1

    threshold = max(2, len(pages) * 0.60)
    noise = {ln for ln, cnt in freq.items() if cnt >= threshold}
    if not noise:
        return pages

    cleaned: List[str] = []
    for page in pages:
        filtered = "\n".join(
            ln for ln in page.split("\n") if ln.strip() not in noise
        )
        cleaned.append(filtered)
    return cleaned


def _validate_user_config(user_config: Dict[str, Any]) -> Dict[str, Any]:
    from pipeline.s2_chunkers import VALID_STRATEGIES  # import here to avoid circularity at module level

    cfg = dict(user_config or {})
    numeric_ranges = {
        "n_min": (20, 400),
        "n_max": (80, 1200),
        "tau_sem": (0.2, 0.99),
        "max_iterations": (1, 30),
        "q_entropy_param": (-1.0, 1.0),   # Tsallis q (spec range) — tuned by S7 GA
        "K": (2, 8),
        "min_chunk_tokens": (4, 200),
        "max_chunk_tokens": (64, 2000),
        "window": (0, 5),
        "overlap": (0, 4),
        "qa_count": (2, 40),
    }
    int_keys = {"n_min", "n_max", "max_iterations", "K", "min_chunk_tokens",
                "max_chunk_tokens", "window", "overlap", "qa_count"}
    for key, (low, high) in numeric_ranges.items():
        if key in cfg:
            try:
                value = float(cfg[key])
                value = max(low, min(high, value))
                cfg[key] = int(value) if key in int_keys else value
            except Exception:
                cfg.pop(key, None)
    # Embedding backend (engine.embeddings) — validate against known ids.
    if "embedding_backend" in cfg:
        eb = cfg.get("embedding_backend")
        if eb not in {None, "openai", "openai-large", "multilingual", "english", "tfidf"}:
            cfg.pop("embedding_backend", None)
    if "judge_answerability" in cfg:
        cfg["judge_answerability"] = bool(cfg["judge_answerability"])
    # S4 similarity mode (classical cosine vs Fitouhi–Bouzeffour q-cosine)
    if "s4_similarity" in cfg:
        sm = str(cfg.get("s4_similarity", "cosine")).lower()
        cfg["s4_similarity"] = sm if sm in {"cosine", "qcosine"} else "cosine"
    # Validate chunking strategy
    strategy = str(cfg.get("chunking_strategy", "auto")).lower()
    cfg["chunking_strategy"] = strategy if strategy in VALID_STRATEGIES else "auto"
    return cfg


# ═══════════════════════════════════════════════════════════════════════════
# Helper: update job progress
# ═══════════════════════════════════════════════════════════════════════════

def _update_job(job_id: str, **kwargs) -> None:
    if job_id in job_store:
        job_store[job_id].update(kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Pipeline runner (executed in a background thread)
# ═══════════════════════════════════════════════════════════════════════════

def _run_pipeline(job_id: str, doc_id: str, user_config: Dict[str, Any]) -> None:
    try:
        logger.info(f'{{"job_id":"{job_id}","event":"pipeline_start"}}')
        doc = doc_store.get(doc_id)
        if not doc:
            _update_job(job_id, status="error", error="Document not found.")
            return

        text: str = doc.get("content") or ""
        if not text.strip() and doc.get("raw_content") is not None:
            _update_job(
                job_id,
                stage="S1",
                progress=2,
                message="Extracting text from uploaded file…",
            )
            parse_start = time.perf_counter()
            text = _parse_file(doc.get("filename") or "upload.txt", doc["raw_content"])
            doc["content"] = text
            doc["token_count"] = len(text.split())
            logger.info(
                json.dumps(
                    {
                        "job_id": job_id,
                        "event": "document_parsed",
                        "filename": doc.get("filename"),
                        "seconds": round(time.perf_counter() - parse_start, 3),
                        "tokens": doc["token_count"],
                    }
                )
            )
        if not text.strip():
            _update_job(
                job_id,
                status="error",
                message="Could not extract text from the uploaded file.",
                error="Could not extract text from the uploaded file.",
            )
            return
        config = {**DEFAULT_CONFIG, **_validate_user_config(user_config), "job_id": job_id}

        # ── S1: Profile ───────────────────────────────────────────────────
        _update_job(
            job_id,
            stage="S1",
            progress=5,
            message="Profiling document…",
        )
        doc_profile = profile_document(text, config)
        # Blend suggested params into config (user overrides take precedence)
        suggested = doc_profile.get("suggested_config", {})
        for k, v in suggested.items():
            if k not in user_config:
                config[k] = v

        _update_job(job_id, stage_details={"s1": doc_profile})

        # ── QA generation: auto-build the Table-I evaluation set (engine.qagen) ─
        # The qentropy retrieval metrics need (query, ground-truth) pairs.  We
        # synthesise them once from the document; reused by the S7 GA fitness AND
        # the final S8 evaluation.  Degrades gracefully with no OPENAI_API_KEY.
        _update_job(job_id, stage="QA", progress=10,
                    message="Generating evaluation questions…")
        qa_pairs: List[Dict[str, str]] = []
        qa_source = "none"
        _qa_n = int(config.get("qa_count", 12))
        if qagen.openai_configured():
            try:
                qa_pairs = qagen.generate_qa(text, n=_qa_n)
                qa_source = "openai"
            except Exception as exc:
                # Key missing/invalid or provider error → fall back to the
                # offline, document-derived pseudo-queries so the Table-I
                # retrieval metrics are still computed (not label-free).
                logger.warning("LLM QA generation failed (%s); using offline "
                               "document-derived evaluation set.", exc)
        if not qa_pairs:
            try:
                qa_pairs = qagen.generate_qa_offline(text, n=_qa_n)
                if qa_pairs:
                    qa_source = "offline"
            except Exception as exc:
                logger.warning("Offline QA generation failed (%s); evaluation "
                               "will be label-free.", exc)
        config["qa_pairs"] = qa_pairs
        eval_ctx = build_context(
            text, qa_pairs=qa_pairs,
            backend=config.get("embedding_backend"),
            rel_threshold=config.get("rel_threshold", "auto"),
            judge=bool(config.get("judge_answerability", False)),
        )
        _update_job(job_id, stage_details={
            **job_store[job_id].get("stage_details", {}),
            "qa": {"n_pairs": len(qa_pairs), "source": qa_source,
                   "judge": eval_ctx.judge, "rel_threshold": eval_ctx.rel_threshold,
                   "pairs": qa_pairs[:50]},
        })

        # ── S2: Parallel chunkers ─────────────────────────────────────────
        _update_job(
            job_id,
            stage="S2",
            progress=15,
            message="Running parallel chunkers…",
        )
        try:
            all_chunks_map = run_all_chunkers(text, doc_profile["type"], config)
        except Exception:
            logger.exception("S2 failed; falling back to single chunk")
            all_chunks_map = {"fallback": [{"text": text, "start": 0, "end": len(text), "method": "fallback"}]}

        evaluating = {name: chunks for name, chunks in all_chunks_map.items() if chunks}
        if not evaluating:
            evaluating = {"fallback": [{"text": text, "start": 0, "end": len(text), "method": "fallback"}]}

        stage2_table, stage2_scores, stage2_ranked = _evaluate_stage2_outputs(evaluating, text, config)
        forced_strategy = str(config.get("chunking_strategy", "auto")).lower()
        if forced_strategy != "auto" and forced_strategy in evaluating:
            selected_strategy = forced_strategy
        else:
            selected_strategy = stage2_ranked[0][0] if stage2_ranked else next(iter(evaluating.keys()))
        for row in stage2_table:
            row["winner"] = row.get("strategy") == selected_strategy
        initial_chunks = evaluating.get(selected_strategy) or select_best_strategy(evaluating, doc_profile["type"], config) or next(iter(evaluating.values()))

        _update_job(
            job_id,
            stage_details={
                **job_store[job_id].get("stage_details", {}),
                "s2": {
                    "strategies": {
                        k: len(v) for k, v in all_chunks_map.items()
                    },
                    "selected_count": len(initial_chunks),
                    "selected_strategy": selected_strategy,
                    "forced_strategy": config.get("chunking_strategy", "auto"),
                    "evaluating": list(evaluating.keys()),
                    "scores": stage2_scores,
                    "ranked": stage2_ranked,
                    "table": stage2_table,
                },
            },
        )

        # ── S3-S6: run the full scoring pipeline for every chunking method ─
        _update_job(
            job_id,
            stage="S3-S6",
            progress=28,
            message="Running S3-S6 for every chunking method…",
        )

        strategy_outputs: Dict[str, Dict[str, Any]] = {}
        total_methods = max(len(evaluating), 1)
        for idx, (name, chunks) in enumerate(evaluating.items(), start=1):
            _update_job(
                job_id,
                stage="S3-S6",
                progress=28 + int(36 * idx / total_methods),
                message=f"Processing {name.replace('_', ' ')} through S3-S6…",
            )
            strategy_outputs[name] = _run_s3_s6_for_strategy(
                name,
                chunks,
                text,
                doc_profile,
                config,
            )

        # ── S8: Strategy evaluation ───────────────────────────────────────
        _update_job(
            job_id,
            stage="S8",
            progress=68,
            message="Evaluating all chunking methods…",
        )

        # Score EVERY chunking method with the authoritative Table-I metrics
        # (engine.metrics, real auto-generated QA) — the same engine that decides
        # the final winner.  The judge is skipped here (pre-GA) to save cost; it
        # runs once on the final tuned set below.
        strat_metrics = {
            name: score_run(out.get("chunks", []), eval_ctx, judge=False)
            for name, out in strategy_outputs.items()
        }
        import numpy as _np8
        def _cos_rank(m):
            return float(_np8.mean([float(m.get(k, 0.0)) for k in ("mrr", "ndcg", "srgt", "qcs")]))
        forced_strategy = str(config.get("chunking_strategy", "auto")).lower()
        if forced_strategy != "auto" and forced_strategy in strategy_outputs:
            winner = forced_strategy
        elif strat_metrics:
            winner = max(strat_metrics, key=lambda k: _cos_rank(strat_metrics[k]))
        else:
            winner = next(iter(strategy_outputs.keys()))
        evaluation_table, ranked, evaluation_scores = _p2_table(
            strat_metrics, strategy_outputs, pin_winner=winner)
        winning_output = strategy_outputs[winner]
        embedded = winning_output["chunks"]
        embeddings = winning_output["embeddings"]

        s3_s6_details = {
            name: output["details"]
            for name, output in strategy_outputs.items()
        }
        entity_graphs = {
            name: output["details"].get("entity_graph", {})
            for name, output in strategy_outputs.items()
        }
        stage_details = {
            **job_store[job_id].get("stage_details", {}),
            "s3": {
                "jsd_series": winning_output["details"].get("jsd_series", []),
                "chunk_count": winning_output["details"].get("s3_chunk_count", 0),
                "metric": config.get("entropy_metric", "jsd"),
                "thresholds": winning_output["details"].get("thresholds", {}),
                "stats": winning_output["details"].get("s3_stats", {}),
            },
            "s4": {
                "chunk_count": winning_output["details"].get("s4_chunk_count", 0),
                "weighted_decision": True,
                "mean_boundary_score": winning_output["details"].get("mean_boundary_score", 0.0),
            },
            "s5": {
                "entity_graph": winning_output["details"].get("entity_graph", {}),
                "entity_graphs": entity_graphs,
            },
            "s6": {
                "embedding_dim": len(embeddings[0]) if embeddings else 0,
                "model": get_embedder().resolve(config.get("embedding_backend")),
                "strategy": winner,
            },
            "s3_s6": s3_s6_details,
            "s8": {
                "winner": winner,
                "ranked": ranked,
                "scores": evaluation_scores,
                "table": evaluation_table,
            },
        }
        _update_job(
            job_id,
            stage_details=stage_details,
        )

        # ── S7: GA hyperparameter calibration (called ONCE) ──────────────
        _update_job(
            job_id,
            stage="S7",
            progress=82,
            message="Running GA hyperparameter calibration…",
        )

        # Call run_rl_loop ONCE — it internally runs GA per strategy.
        # The old approach called it once per strategy (8× too much work).
        best_chunks, reward_history, final_config = run_rl_loop(
            text,
            doc_profile,
            winning_output.get("chunks", []),
            config,
        )

        # Build s7_outputs from per_strategy_results for the frontend
        per_strat = final_config.get("per_strategy_results", {})
        s7_outputs: Dict[str, Dict[str, Any]] = {}
        for name, res in per_strat.items():
            strat_chunks = res.get("best_chunks") or winning_output.get("chunks", [])
            # Extract embeddings from S8 strategy_outputs if available (for RAG metrics)
            s8_output = strategy_outputs.get(name, {})
            s8_embeddings = s8_output.get("embeddings", [])
            
            s7_outputs[name] = {
                "chunks":         strat_chunks,
                "embeddings":     s8_embeddings,  # ← Add embeddings for RAG computation
                # Use full fitness_history for real convergence curve, not just 2 flat points
                "reward_history": res.get("fitness_history") or [res.get("s2_baseline", 0), res.get("best_score", 0)],
                # Merge best_params into config so tuned values are visible
                "final_config":   {**config, **(res.get("best_params") or {})},
                "reward_breakdown": {
                    "s2_baseline": res.get("s2_baseline", 0),
                    "best_score":  res.get("best_score", 0),
                    "improvement": res.get("improvement", 0),
                },
                "iterations":     res.get("n_evals", 0),
                "final_reward":   res.get("best_score", 0),
            }

        # Fallback: if GA returned no per_strategy_results use S8 outputs
        if not s7_outputs:
            for name, output in strategy_outputs.items():
                s7_outputs[name] = {
                    "chunks":         output.get("chunks", []),
                    "embeddings":     output.get("embeddings", []),  # ← Include embeddings
                    "reward_history": reward_history,
                    "final_config":   final_config,
                    "reward_breakdown": {},
                    "iterations":     len(reward_history),
                    "final_reward":   reward_history[-1] if reward_history else 0.0,
                }

        # ── P2 evaluation (Table-I) — the AUTHORITATIVE metrics, applied AFTER
        # the GA to every tuned strategy, and the basis for the final winner. ──
        p2_runs = []
        for name, out in s7_outputs.items():
            met = score_run(out.get("chunks", []), eval_ctx)
            p2_runs.append({"id": name, "label": name.replace("_", " "), "metrics": met})
        p2_summary = rank_runs(p2_runs, judged=eval_ctx.judge)
        p2_metrics = {r["id"]: r["metrics"] for r in p2_runs}

        from engine.metrics import METRIC_INFO as _METRIC_INFO

        # Decide the winner FIRST (it may win on the token-cost tie-break, not raw
        # score), then build the table with that winner pinned to the top row so
        # the 🏆 is always rank #1 — consistent with the qEntropy panel.
        forced_strategy = str(config.get("chunking_strategy", "auto")).lower()
        if forced_strategy != "auto" and forced_strategy in s7_outputs:
            final_winner = forced_strategy
        elif p2_summary.get("overall"):
            final_winner = p2_summary["overall"]["winner_id"]
        else:
            final_winner = (
                final_config.get("overall_winner_strategy")
                or next(iter(s7_outputs.keys()))
            )
        # The "S7 RL Evaluation" table is the SAME authoritative metric set.
        s7_table, s7_ranked, _ = _p2_table(p2_metrics, s7_outputs, pin_winner=final_winner)

        final_s7 = s7_outputs.get(final_winner) or next(iter(s7_outputs.values()))
        best_chunks    = final_s7["chunks"]
        reward_history = final_s7["reward_history"]

        # ── Safety: detect GA workers returning raw S2 chunks ────────────────
        # When S3 crashes inside a subprocess worker, the worker silently
        # returns S2 output (text, start, end, method only — no S3/S4 fields).
        # This makes chunk_score = 0.35 flat (all defaults).
        # Detect by checking for S3's signature fields.  If absent, use the
        # S8 winner chunks from the main S3→S6 pipeline (always annotated).
        _s3_signature = {"boundary_type", "jsd_score", "metric_score",
                         "boundary_signal", "boundary_features", "icc"}
        if best_chunks and not any(f in best_chunks[0] for f in _s3_signature):
            logger.warning(
                "S7 best_chunks missing S3 annotations — GA workers likely "
                "returned raw S2 output. Using S8 winner chunks instead."
            )
            best_chunks = winning_output.get("chunks", best_chunks)
        # Keep the real GA final_config (has optimizer, overall_winner, baseline_reward etc.)
        # Only merge the winner's best_params into it, don't replace it entirely
        final_config.update(final_s7.get("final_config", {}))

        _update_job(
            job_id,
            stage_details={
                **job_store[job_id].get("stage_details", {}),
                "s7": {
                    "reward_history": reward_history,
                    "iterations":     len(reward_history),
                    "final_config":   final_config,
                    "reward_breakdown": final_config.get("reward_breakdown", {}),
                    "strategy":       final_winner,
                    "winner":         final_winner,
                    "ranked":         s7_ranked,
                    "table":          s7_table,
                    "per_strategy_results": {
                        k: {kk: vv for kk, vv in v.items() if kk not in ("best_chunks", "fitness_history")}
                        for k, v in per_strat.items()
                    },
                    "strategies": {
                        name: {
                            "iterations":     out["iterations"],
                            "final_reward":   out["final_reward"],
                            "reward_history": out["reward_history"],
                            "reward_breakdown": out["reward_breakdown"],
                            "final_config":   out["final_config"],
                            "chunk_count":    len(out["chunks"]),
                            "mean_tokens":    round(
                                sum(len(c.get("text", "").split()) for c in out["chunks"])
                                / max(len(out["chunks"]), 1), 2
                            ),
                        }
                        for name, out in s7_outputs.items()
                    },
                },
            },
        )

        # ── Final scoring ─────────────────────────────────────────────────
        _update_job(
            job_id,
            stage="DONE",
            progress=95,
            message="Finalising results…",
        )
        final_chunks = _finalise_chunks(best_chunks)

        mean_score = (
            sum(c.get("chunk_score", 0.0) for c in final_chunks) / len(final_chunks)
            if final_chunks
            else 0.0
        )

        results = {
            "document_id": doc_id,
            "job_id": job_id,
            "doc_profile": doc_profile,
            "chunks": final_chunks,
            "summary": {
                "doc_type": doc_profile.get("type"),
                "domain": doc_profile.get("domain"),
                "length_bucket": doc_profile.get("length_bucket"),
                "token_count": doc_profile.get("token_count"),
                "chunk_count": len(final_chunks),
                "mean_chunk_score": round(mean_score, 4),
                "rl_iterations": len(reward_history),
                "final_reward": reward_history[-1] if reward_history else 0.0,
                "winning_strategy": final_winner,
                "s8_winning_strategy": winner,
                "stage2_winning_strategy": selected_strategy,
                "evaluated_strategies": len(evaluation_table),
            },
            "stage_details": job_store[job_id].get("stage_details", {}),
            "reward_history": reward_history,
            "reward_histories": {
                name: out["reward_history"]
                for name, out in s7_outputs.items()
            },
            "strategy_evaluation": evaluation_table,
            "stage2_evaluation": stage2_table,
            "s7_evaluation": s7_table,
            # ── Authoritative qentropy / Table-I evaluation (engine.metrics) ──
            "p2_evaluation": {
                "metric_info": _METRIC_INFO,
                "metrics": p2_metrics,            # {strategy: {precision, recall, mrr, ndcg, ...}}
                "summary": p2_summary,            # {best_per_metric, overall winner + ties}
                "winner": final_winner,
                "judged": eval_ctx.judge,
                "n_queries": len(eval_ctx.queries),
                "rel_threshold": eval_ctx.rel_threshold,
                "qentropy": {
                    name: {
                        "q": (out["chunks"][0].get("q") if out.get("chunks") else None),
                        "diversity_number": (out["chunks"][0].get("diversity_number") if out.get("chunks") else None),
                        "shannon_bits": (out["chunks"][0].get("shannon_bits") if out.get("chunks") else None),
                        "tsallis_bits": (out["chunks"][0].get("tsallis_bits") if out.get("chunks") else None),
                        "entropy_mode": (out["chunks"][0].get("entropy_mode") if out.get("chunks") else None),
                        "qlog_bits": (out["chunks"][0].get("qlog_bits") if out.get("chunks") else None),
                        "entropy_rate": (out["chunks"][0].get("entropy_rate") if out.get("chunks") else None),
                    }
                    for name, out in s7_outputs.items()
                },
            },
        }

        _update_job(
            job_id,
            status="complete",
            stage="DONE",
            progress=100,
            message="Pipeline complete.",
            results=results,
        )
        logger.info(f'{{"job_id":"{job_id}","event":"pipeline_complete","chunks":{len(final_chunks)}}}')

    except Exception as exc:
        logger.exception("Pipeline failed")
        _update_job(
            job_id,
            status="error",
            message=str(exc),
            error=traceback.format_exc(),
        )


def _run_s3_s6_for_strategy(
    strategy_name: str,
    chunks: List[Dict],
    text: str,
    doc_profile: Dict[str, Any],
    config: Dict[str, Any],
) -> Dict[str, Any]:
    """Run refinement, filtering, graph enrichment, and embeddings for one strategy."""
    work_config = dict(config)
    work_config["chunking_strategy"] = strategy_name
    refined = refine_boundaries(chunks, work_config)
    jsd_series = get_jsd_series(refined)
    s3_stats = refined[-1].get("s3_stats", {}) if refined else {}
    s3_chunks = _summarize_s3_chunks(refined)
    filtered = filter_boundaries(refined, doc_profile["type"], [], work_config)
    enriched = enrich_graph(filtered, [], work_config)
    graph_data = build_entity_graph_data(enriched)
    embedded, embeddings = embed_chunks(
        enriched,
        text,
        doc_profile,
        work_config["embedding_model"],
        work_config,
    )

    token_counts = [len(c.get("text", "").split()) for c in embedded]
    boundary_scores = [float(c.get("boundary_score", 0.0)) for c in embedded]
    icc_scores = [float(c.get("icc", 0.5)) for c in embedded]
    details = {
        "strategy": strategy_name,
        "initial_chunk_count": len(chunks),
        "s3_chunk_count": len(refined),
        "s4_chunk_count": len(filtered),
        "chunk_count": len(embedded),
        "mean_tokens": round(sum(token_counts) / len(token_counts), 2) if token_counts else 0,
        "jsd_series": jsd_series,
        "s3_chunks": s3_chunks,
        "s3_stats": s3_stats,
        "metric": work_config.get("entropy_metric", "jsd"),
        "thresholds": refined[-1].get("thresholds", {}) if refined else {},
        "s3_merge_count": s3_stats.get("merged_count", 0),
        "s3_hard_count": s3_stats.get("hard_count", 0),
        "s3_soft_count": s3_stats.get("soft_count", 0),
        "s3_protected_count": s3_stats.get("protected_count", 0),
        "s3_mean_signal": s3_stats.get("mean_signal", 0.0),
        "mean_boundary_score": round(sum(boundary_scores) / len(boundary_scores), 4) if boundary_scores else 0.0,
        "mean_icc": round(sum(icc_scores) / len(icc_scores), 4) if icc_scores else 0.0,
        "embedding_dim": len(embeddings[0]) if embeddings else 0,
        "entity_graph": graph_data,
        "entity_count": sum(len(c.get("entities", [])) for c in embedded),
    }
    return {"chunks": embedded, "embeddings": embeddings, "details": details}


def _evaluate_stage2_outputs(
    outputs: Dict[str, List[Dict]],
    full_text: str,
    config: Dict[str, Any],
) -> tuple:
    rows: List[Dict[str, Any]] = []
    scores: Dict[str, Dict[str, Any]] = {}
    total_words = max(1, len(full_text.split()))
    for name, chunks in outputs.items():
        row = _score_stage2_strategy(name, chunks, full_text, total_words, config)
        rows.append(row)
        scores[name] = {
            "score": row["score"],
            "chunk_count": row["chunk_count"],
            "mean_tokens": row["mean_tokens"],
            "size_fit": row["size_fit"],
            "coverage": row["coverage"],
            "count_fit": row["count_fit"],
            "boundary_quality": row["boundary_quality"],
            "structure_integrity": row["structure_integrity"],
            "cohesion": row["cohesion"],
            "distinctness": row["distinctness"],
            "chunking_time": row["chunking_time"],
            "time_efficiency": row["time_efficiency"],
        }

    rows.sort(key=lambda r: r["score"], reverse=True)
    ranked = [(r["strategy"], r["score"]) for r in rows]
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
        row["winner"] = idx == 1
    return rows, scores, ranked


def _summarize_s3_chunks(chunks: List[Dict], limit: int = 240) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "") or ""
        features = chunk.get("boundary_features", {}) or {}
        out.append(
            {
                "index": idx,
                "token_count": len(text.split()),
                "start": int(chunk.get("start", 0) or 0),
                "end": int(chunk.get("end", 0) or 0),
                "boundary_type": chunk.get("boundary_type", "unknown"),
                "merge_reason": chunk.get("merge_reason", ""),
                "boundary_signal": float(chunk.get("boundary_signal", chunk.get("hidden_state", 0.0)) or 0.0),
                "metric_score": float(chunk.get("metric_score", 0.0) or 0.0),
                "jsd_score": float(chunk.get("jsd_score", 0.0) or 0.0),
                "features": {
                    "selected_metric": float(features.get("selected_metric", 0.0) or 0.0),
                    "jsd": float(features.get("jsd", 0.0) or 0.0),
                    "hellinger": float(features.get("hellinger", 0.0) or 0.0),
                    "overlap": float(features.get("overlap", 0.0) or 0.0),
                    "entropy_delta": float(features.get("entropy_delta", 0.0) or 0.0),
                },
                "preview": re.sub(r"\s+", " ", text).strip()[:limit],
            }
        )
    return out


def _score_stage2_strategy(
    strategy_name: str,
    chunks: List[Dict],
    full_text: str,
    total_words: int,
    config: Dict[str, Any],
) -> Dict[str, Any]:
    n_min = int(config.get("n_min", 100))
    n_max = int(config.get("n_max", 500))
    empty = {
        "strategy": strategy_name,
        "rank": 0,
        "winner": False,
        "score": 0.0,
        "chunk_count": 0,
        "mean_tokens": 0,
        "mean_icc": 0.0,
        "inter_separation": 0.0,
        "entropy_strength": 0.0,
        "boundary_distinctiveness": 0.0,
        "size_fit": 0.0,
        "coverage": 0.0,
        "count_fit": 0.0,
        "boundary_quality": 0.0,
        "structure_integrity": 0.0,
        "cohesion": 0.0,
        "non_redundancy": 0.0,
        "distinctness": 0.0,
        "chunking_time": 0.0,
        "time_efficiency": 0.0,
    }
    if not chunks:
        return empty

    sizes = [max(1, len(c.get("text", "").split())) for c in chunks]
    mean_tokens = sum(sizes) / len(sizes)
    target = max(float(n_min), min(float(n_max), float(n_max) * 0.72))
    expected_count = max(1.0, total_words / max(target, 1.0))

    in_range = sum(1 for s in sizes if n_min <= s <= n_max) / len(sizes)
    avg_fit = 1.0 - min(1.0, abs(mean_tokens - target) / max(target, 1.0))
    oversize_penalty = sum(max(0.0, (s - n_max) / max(n_max, 1)) for s in sizes) / len(sizes)
    tiny_penalty = sum(max(0.0, (n_min - s) / max(n_min, 1)) for s in sizes) / len(sizes)
    size_fit = max(0.0, 0.50 * in_range + 0.35 * avg_fit - 0.10 * oversize_penalty - 0.05 * tiny_penalty)

    count_fit = 1.0 - min(1.0, abs(len(chunks) - expected_count) / max(expected_count, 1.0))
    coverage = _span_coverage(chunks, len(full_text))
    span_non_redundancy = _span_non_redundancy(chunks, len(full_text))
    token_distinctness = _stage2_token_distinctness(chunks)
    distinctness = 0.45 * span_non_redundancy + 0.55 * token_distinctness
    boundary_quality = _stage2_boundary_quality(chunks)
    structure_integrity = _stage2_structure_integrity(chunks, full_text)
    cohesion = _stage2_cohesion(chunks)
    chunking_time = float((config.get("_stage2_timings") or {}).get(strategy_name, 0.0) or 0.0)
    time_efficiency = 1.0 / (1.0 + chunking_time)

    score = (
        0.23 * size_fit
        + 0.18 * count_fit
        + 0.17 * boundary_quality
        + 0.14 * structure_integrity
        + 0.12 * cohesion
        + 0.09 * distinctness
        + 0.04 * coverage
        + 0.03 * time_efficiency
    )
    return {
        "strategy": strategy_name,
        "rank": 0,
        "winner": False,
        "score": round(float(max(0.0, min(1.0, score))), 4),
        "chunk_count": len(chunks),
        "mean_tokens": round(mean_tokens, 1),
        "mean_icc": round(float(cohesion), 4),
        "inter_separation": round(float(distinctness), 4),
        "entropy_strength": round(float(count_fit), 4),
        "boundary_distinctiveness": round(float(boundary_quality), 4),
        "size_fit": round(float(max(0.0, min(1.0, size_fit))), 4),
        "coverage": round(float(coverage), 4),
        "count_fit": round(float(count_fit), 4),
        "boundary_quality": round(float(boundary_quality), 4),
        "structure_integrity": round(float(structure_integrity), 4),
        "cohesion": round(float(cohesion), 4),
        "non_redundancy": round(float(span_non_redundancy), 4),
        "distinctness": round(float(distinctness), 4),
        "chunking_time": round(float(chunking_time), 4),
        "time_efficiency": round(float(time_efficiency), 4),
    }

def _auto_generate_queries(text: str, n: int = 10) -> List[str]:
    """
    Generate probe queries from 30-word sliding windows of the document.
    Works for French legal text — no uppercase requirement.
    Windows are guaranteed to be in the same distribution as chunk text,
    so cosine similarities to chunks are meaningful.
    """
    words = text.split()
    if len(words) < 20:
        return [text.strip()] if text.strip() else []
    window = 30
    queries = []
    step = max(1, len(words) // n)
    for i in range(0, len(words), step):
        segment = " ".join(words[i : i + window]).strip()
        if len(segment.split()) >= 10:
            queries.append(segment)
        if len(queries) >= n:
            break
    return queries


def _compute_rag_metrics(
    chunks: List[Dict],
    chunk_embeddings: List,
    full_doc_embedding,
    query_embeddings: List,
    chunking_time: float,
) -> Dict:
    """
    Compute TABLE I metrics from the paper:
    Precision, Recall, MRR, NDCG@5, ss2fd, QCS, Retrieval Token Cost, Chunking Time.

    Key improvements over the original fixed-threshold version:
    - Adaptive threshold  tau = mean(sims) + std(sims)  instead of hard 0.7.
      Adapts to the embedding scale of the 256-dim JL-projected space where
      distances are compressed and 0.7 is almost never reached.
    - Fallback: if no chunk exceeds tau, the top-1 chunk is the ground truth.
    - Continuous graded NDCG: rel_i = cos(retrieved_i, gt_centroid) in [0,1]
      instead of binary, matching the Entropy-branch implementation.
    - Recall bug fixed: empty gt now returns 0.0 (not 1.0).
    """
    import numpy as _np

    TOP_K = 5

    if not chunks or not chunk_embeddings or len(chunk_embeddings) != len(chunks):
        return {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0,
                "ss2fd": 0.0, "qcs": 0.0, "retrieval_token_cost": 0.0,
                "chunking_time": round(chunking_time, 4)}

    # Stack and normalise chunk embeddings
    emb = _np.array([_np.array(e) for e in chunk_embeddings], dtype=_np.float32)
    norms = _np.linalg.norm(emb, axis=1, keepdims=True)
    norms = _np.where(norms == 0, 1.0, norms)
    emb_n = emb / norms

    # ss2fd — average cosine of each chunk to the full-document embedding
    fd = _np.array(full_doc_embedding, dtype=_np.float32)
    fd_n = fd / max(float(_np.linalg.norm(fd)), 1e-8)
    ss2fd = float(_np.mean(emb_n @ fd_n))

    if not query_embeddings:
        return {"precision": 0.0, "recall": 0.0, "mrr": 0.0, "ndcg": 0.0,
                "ss2fd": round(ss2fd, 4), "qcs": 0.0,
                "retrieval_token_cost": float(
                    sum(len(c.get("text", "").split()) for c in chunks[:TOP_K])
                ),
                "chunking_time": round(chunking_time, 4)}

    precisions, recalls, mrrs, ndcgs, qcss, token_costs = [], [], [], [], [], []

    for q_emb in query_embeddings:
        q = _np.array(q_emb, dtype=_np.float32)
        q_n = q / max(float(_np.linalg.norm(q)), 1e-8)

        sims = emb_n @ q_n                        # cosine similarity to all chunks
        ranked = list(_np.argsort(sims)[::-1])    # descending

        # ── Adaptive threshold (from Entropy branch) ─────────────────────
        # tau_q = mean_c cos(chunk_c, query) + std_c cos(chunk_c, query)
        # This is 1 standard deviation above the mean — self-calibrates to
        # the embedding scale so it works in any projected dimension.
        tau = float(sims.mean()) + float(sims.std())
        gt = set(i for i, s in enumerate(sims) if float(s) >= tau)
        if not gt:
            gt = {int(_np.argmax(sims))}   # fallback: best chunk is GT

        top5 = ranked[:TOP_K]
        tp = len(set(top5) & gt)

        precisions.append(tp / max(len(top5), 1))
        recalls.append(tp / len(gt))        # gt always non-empty due to fallback

        mrr = 0.0
        for rank, idx in enumerate(ranked, 1):
            if idx in gt:
                mrr = 1.0 / rank
                break
        mrrs.append(mrr)

        # ── Continuous graded NDCG ────────────────────────────────────────
        # rel_i = cosine(retrieved_i, gt_centroid), clipped to [0, 1]
        # This gives partial credit for near-relevant chunks (no binary cliff).
        gt_list = list(gt)
        gt_centroid = _np.mean(emb_n[gt_list], axis=0)
        gt_norm = float(_np.linalg.norm(gt_centroid))
        gt_centroid_n = gt_centroid / max(gt_norm, 1e-8)
        gt_sims = _np.clip(emb_n @ gt_centroid_n, 0.0, 1.0)

        k = min(TOP_K, len(ranked))
        dcg = sum(
            float(gt_sims[ranked[i]]) / _np.log2(i + 2)
            for i in range(k)
        )
        # Proper IDCG: pair the LARGEST relevances with the LARGEST (earliest)
        # discounts — the true maximum achievable DCG.  The previous version
        # multiplied the relevance sum by an *averaged* discount, which under-
        # estimated the ideal and let NDCG exceed 1.0 (the "101%" bug).
        ideal_rels = _np.sort(gt_sims)[::-1][:k]
        idcg = sum(float(ideal_rels[i]) / _np.log2(i + 2) for i in range(k))
        ndcgs.append(float(_np.clip(dcg / idcg if idcg > 1e-12 else 0.0, 0.0, 1.0)))

        qcss.append(float(_np.clip(_np.mean([sims[i] for i in top5]), 0.0, 1.0)) if top5 else 0.0)
        token_costs.append(sum(len(chunks[i].get("text", "").split()) for i in top5))

    def _c(x):  # clamp every reported metric to [0, 1]
        return round(float(_np.clip(x, 0.0, 1.0)), 4)

    return {
        "precision":            _c(_np.mean(precisions)),
        "recall":               _c(_np.mean(recalls)),
        "mrr":                  _c(_np.mean(mrrs)),
        "ndcg":                 _c(_np.mean(ndcgs)),
        "ss2fd":                _c(ss2fd),
        "qcs":                  _c(_np.mean(qcss)),
        "retrieval_token_cost": round(float(_np.mean(token_costs)), 1),
        "chunking_time":        round(chunking_time, 4),
    }

def _p2_table(metrics_by_strategy: Dict[str, Dict], outputs: Dict[str, Dict],
              pin_winner: Optional[str] = None) -> tuple:
    """Build a frontend evaluation table from the AUTHORITATIVE Table-I metrics
    (engine.metrics, scored against the real auto-generated QA).

    Replaces the old self-referential RAG proxy whose ground truth was derived
    from the same cosine similarities it ranked by — which made MRR and recall
    saturate to 1.0 for every strategy.  `score` here is the cosine-rank mean
    (mrr/ndcg/srgt/qcs), the same primary signal used to pick the winner.
    """
    import numpy as _np
    rows = []
    for name, m in metrics_by_strategy.items():
        chunks = (outputs.get(name) or {}).get("chunks", []) or []
        toks = [len(str(c.get("text", "")).split()) for c in chunks]
        score = float(_np.mean([float(m.get(k, 0.0)) for k in ("mrr", "ndcg", "srgt", "qcs")]))
        rows.append({
            "strategy": name,
            "score": round(score, 4),
            "mrr": m.get("mrr", 0.0), "ndcg": m.get("ndcg", 0.0),
            "precision": m.get("precision", 0.0), "recall": m.get("recall", 0.0),
            "ss2fd": m.get("ss2fd", 0.0), "qcs": m.get("qcs", 0.0), "srgt": m.get("srgt", 0.0),
            "retrieval_token_cost": m.get("retrieval_token_cost", 0.0),
            "chunking_time": round(float(m.get("chunking_time_ms", 0.0)) / 1000.0, 4),
            "chunk_count": len(chunks),
            "mean_tokens": round(sum(toks) / max(len(toks), 1), 1),
            "winner": False,
        })
    # Sort by score; if a decided winner is given (it may win on the token-cost
    # tie-break rather than raw score), pin it to the top so the 🏆 row is always
    # rank #1 and the highlight is consistent across every table.
    if pin_winner:
        rows.sort(key=lambda r: (r["strategy"] != pin_winner, -r["score"]))
    else:
        rows.sort(key=lambda r: -r["score"])
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        r["winner"] = (pin_winner is not None and r["strategy"] == pin_winner)
    ranked = [(r["strategy"], r["score"]) for r in rows]
    scores = {r["strategy"]: {"score": r["score"]} for r in rows}
    return rows, ranked, scores


def _evaluate_strategy_outputs(
    outputs: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    queries: List[str] = None,
    query_embeddings: List = None,
    full_doc_embedding = None,
) -> tuple:
    rows, scores = [], {}
    for name, output in outputs.items():
        chunks = output.get("chunks", [])
        chunk_embeddings = output.get("embeddings", [])
        chunking_time = float((config.get("_stage2_timings") or {}).get(name, 0.0))

        row = _score_strategy(name, chunks, config)  # internal quality score (0-1)

        # Add RAG metrics from the paper's TABLE I and blend into final score
        if full_doc_embedding is not None:
            rag = _compute_rag_metrics(
                chunks, chunk_embeddings, full_doc_embedding,
                query_embeddings or [], chunking_time,
            )
            row.update(rag)

            # Blended score: 50% internal quality + 50% RAG retrieval performance
            # Internal score already in row["score"]; RAG score computed below.
            rag_score = (
                0.25 * rag.get("mrr",       0.0)
              + 0.20 * rag.get("ndcg",      0.0)
              + 0.20 * rag.get("precision",  0.0)
              + 0.15 * rag.get("ss2fd",     0.0)
              + 0.10 * rag.get("qcs",       0.0)
              + 0.10 * (1.0 - min(rag.get("retrieval_token_cost", 500) / 2000.0, 1.0))
            )
            internal_score = row["score"]
            row["internal_score"] = round(internal_score, 4)
            row["rag_score"]      = round(rag_score, 4)
            row["score"]          = round(0.50 * internal_score + 0.50 * rag_score, 4)

        rows.append(row)
        scores[name] = {"score": row["score"], "chunk_count": row["chunk_count"],
                        "mean_tokens": row["mean_tokens"]}

    rows.sort(key=lambda r: r["score"], reverse=True)
    ranked = [(r["strategy"], r["score"]) for r in rows]
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
        row["winner"] = idx == 1
    return rows, scores, ranked


def _evaluate_s7_outputs(
    outputs: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    queries: List[str] = None,
    query_embeddings: List = None,
    full_doc_embedding = None,
) -> tuple:
    rows: List[Dict[str, Any]] = []
    for name, output in outputs.items():
        chunks = output.get("chunks", [])
        chunk_embeddings = output.get("embeddings", []) if full_doc_embedding is not None else []
        chunking_time = float((config.get("_stage2_timings") or {}).get(name, 0.0))
        
        # Get structural metrics (size_fit, boundary_distinctiveness, etc.)
        row = _score_strategy(name, chunks, config)
        structural_score = row["score"]  # Keep original structural score
        
        # Add GA final reward
        final_reward = float(output.get("final_reward", 0.0) or 0.0)
        row["rl_final_reward"] = round(final_reward, 4)
        row["iterations"] = int(output.get("iterations", 0) or 0)
        breakdown = output.get("reward_breakdown", {}) or {}
        row["reward_quality"] = float(breakdown.get("quality", 0.0) or 0.0)
        row["reward_coverage"] = float(breakdown.get("coverage", 0.0) or 0.0)
        row["reward_consistency"] = float(breakdown.get("consistency", 0.0) or 0.0)
        row["reward_efficiency"] = float(breakdown.get("efficiency", 0.0) or 0.0)
        
        # Add RAG metrics (same as S8) if embeddings available
        if full_doc_embedding is not None and chunk_embeddings:
            rag = _compute_rag_metrics(
                chunks, chunk_embeddings, full_doc_embedding,
                query_embeddings or [], chunking_time,
            )
            row.update(rag)
            row["rag_score"] = round(
                0.25 * rag.get("mrr", 0.0)
                + 0.20 * rag.get("ndcg", 0.0)
                + 0.20 * rag.get("precision", 0.0)
                + 0.15 * rag.get("ss2fd", 0.0)
                + 0.10 * rag.get("qcs", 0.0)
                + 0.10 * (1.0 - min(rag.get("retrieval_token_cost", 500) / 2000.0, 1.0)),
                4
            )
        else:
            row["rag_score"] = 0.0
        
        # Blended score: 40% GA fitness + 40% structural + 20% RAG
        ga_fitness = round(max(0.0, min(1.0, final_reward)), 4)
        row["ga_fitness_score"] = ga_fitness
        row["structural_score"] = round(structural_score, 4)
        
        # Final score: GA is primary, structural adds confidence, RAG is bonus
        final_score = (
            0.50 * ga_fitness           # GA fitness (primary — direct optimization)
            + 0.30 * structural_score   # Structural quality (should align with GA)
            + 0.20 * row.get("rag_score", 0.0)  # RAG performance (retrieval)
        )
        row["score"] = round(max(0.0, min(1.0, final_score)), 4)
        
        rows.append(row)

    rows.sort(key=lambda r: r["score"], reverse=True)
    ranked = [(r["strategy"], r["score"]) for r in rows]
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
        row["winner"] = idx == 1
    return rows, ranked


def _score_strategy(strategy_name: str, chunks: List[Dict], config: Dict[str, Any]) -> Dict[str, Any]:
    n_min = int(config.get("n_min", 100))
    n_max = int(config.get("n_max", 500))
    if not chunks:
        return {
            "strategy": strategy_name,
            "rank": 0,
            "winner": False,
            "score": 0.0,
            "chunk_count": 0,
            "mean_tokens": 0,
            "mean_icc": 0.0,
            "inter_separation": 0.0,
            "entropy_strength": 0.0,
            "boundary_distinctiveness": 0.0,
            "size_fit": 0.0,
        }

    sizes = [max(1, len(c.get("text", "").split())) for c in chunks]
    mean_tokens = sum(sizes) / len(sizes)
    mean_icc = sum(float(c.get("icc", 0.5)) for c in chunks) / len(chunks)
    entropy_strength = sum(min(float(c.get("jsd_score", c.get("metric_score", 0.0))) * 2.0, 1.0) for c in chunks) / len(chunks)
    boundary_distinctiveness = sum(1.0 - float(c.get("boundary_score", 0.5)) for c in chunks) / len(chunks)
    inter_separation = _average_chunk_separation(chunks)

    target = max(float(n_min), float(n_max) * 0.70)
    avg_fit = 1.0 - min(1.0, abs(mean_tokens - target) / max(target, 1.0))
    in_range = sum(1 for s in sizes if n_min <= s <= n_max) / len(sizes)
    size_fit = 0.65 * avg_fit + 0.35 * in_range

    score = (
        0.25 * mean_icc
        + 0.25 * boundary_distinctiveness
        + 0.20 * size_fit
        + 0.20 * inter_separation
        + 0.10 * entropy_strength
    )
    return {
        "strategy": strategy_name,
        "rank": 0,
        "winner": False,
        "score": round(float(max(0.0, min(1.0, score))), 4),
        "chunk_count": len(chunks),
        "mean_tokens": round(mean_tokens, 1),
        "mean_icc": round(float(mean_icc), 4),
        "inter_separation": round(float(inter_separation), 4),
        "entropy_strength": round(float(entropy_strength), 4),
        "boundary_distinctiveness": round(float(boundary_distinctiveness), 4),
        "size_fit": round(float(size_fit), 4),
    }


def _average_chunk_separation(chunks: List[Dict]) -> float:
    if len(chunks) < 2:
        return 0.5
    vals: List[float] = []
    for idx in range(len(chunks) - 1):
        a = set(re.findall(r"\b\w+\b", chunks[idx].get("text", "").lower()))
        b = set(re.findall(r"\b\w+\b", chunks[idx + 1].get("text", "").lower()))
        union = a | b
        vals.append(1.0 - (len(a & b) / len(union) if union else 0.0))
    return sum(vals) / len(vals) if vals else 0.5


def _span_coverage(chunks: List[Dict], text_len: int) -> float:
    if text_len <= 0:
        return 0.0
    spans = []
    for chunk in chunks:
        start = int(chunk.get("start", 0) or 0)
        end = int(chunk.get("end", 0) or 0)
        if end > start:
            spans.append((max(0, start), min(text_len, end)))
    if not spans:
        return 0.0
    spans.sort()
    merged = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    covered = sum(end - start for start, end in merged)
    return float(max(0.0, min(1.0, covered / text_len)))


def _span_non_redundancy(chunks: List[Dict], text_len: int) -> float:
    if text_len <= 0:
        return 0.0
    spans = []
    raw_span_chars = 0
    for chunk in chunks:
        start = int(chunk.get("start", 0) or 0)
        end = int(chunk.get("end", 0) or 0)
        if end > start:
            start = max(0, start)
            end = min(text_len, end)
            spans.append((start, end))
            raw_span_chars += max(0, end - start)
    if not spans or raw_span_chars <= 0:
        return 0.0
    spans.sort()
    merged = []
    for start, end in spans:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    unique_chars = sum(end - start for start, end in merged)
    return float(max(0.0, min(1.0, unique_chars / raw_span_chars)))


def _stage2_boundary_quality(chunks: List[Dict]) -> float:
    if not chunks:
        return 0.0
    good = 0.0
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        first = text.splitlines()[0].strip()
        last = text.rstrip()[-1:]
        start_ok = bool(re.match(r"^(#{1,6}\s+|Article\s+\w+|Art\.?\s+\w+|Section\s+\w+|Chapter\s+\w+|\d+(?:\.\d+)*\s+|[A-Z0-9])", first, re.I))
        end_ok = last in {".", "!", "?", ":", ";", "}", "]", "`"} or len(text.split()) < 30
        good += 0.55 * float(start_ok) + 0.45 * float(end_ok)
    return good / len(chunks)


def _stage2_structure_integrity(chunks: List[Dict], full_text: str) -> float:
    anchors = list(
        re.finditer(
            r"(?im)^\s*(?:"
            r"(?:article|art\.?)\s+\d+(?:\s*(?:er|e|ème|bis|ter|quater))?"
            r"|(?:titre|chapitre|section|sous-section|paragraphe)\s+(?:[ivxlcdm]+|\d+|premier|première)"
            r"|#{1,6}\s+\S+"
            r"|\d+(?:\.\d+){1,3}\s+\S+"
            r")",
            full_text,
        )
    )
    if not anchors:
        return _stage2_boundary_quality(chunks)

    chunk_starts = sorted(max(0, int(c.get("start", 0) or 0)) for c in chunks)
    if not chunk_starts:
        return 0.0

    aligned = 0
    tolerance = 80
    for anchor in anchors:
        pos = anchor.start()
        if any(abs(start - pos) <= tolerance for start in chunk_starts):
            aligned += 1
    anchor_alignment = aligned / len(anchors)

    split_penalties = []
    for idx, anchor in enumerate(anchors):
        start = anchor.start()
        end = anchors[idx + 1].start() if idx + 1 < len(anchors) else len(full_text)
        if end <= start:
            continue
        boundaries_inside = sum(1 for cs in chunk_starts if start + tolerance < cs < end - tolerance)
        unit_words = max(1, len(full_text[start:end].split()))
        expected_splits = max(1, round(unit_words / 420))
        split_penalties.append(1.0 - min(1.0, max(0, boundaries_inside - expected_splits) / max(expected_splits, 1)))

    unit_integrity = sum(split_penalties) / len(split_penalties) if split_penalties else 1.0
    return float(max(0.0, min(1.0, 0.62 * anchor_alignment + 0.38 * unit_integrity)))


def _stage2_token_distinctness(chunks: List[Dict]) -> float:
    if len(chunks) < 2:
        return 1.0
    vals: List[float] = []
    for idx in range(len(chunks) - 1):
        a = _content_token_set(chunks[idx].get("text", ""))
        b = _content_token_set(chunks[idx + 1].get("text", ""))
        if not a or not b:
            vals.append(0.7)
            continue
        overlap = len(a & b) / max(1, min(len(a), len(b)))
        vals.append(1.0 - min(1.0, overlap))
    return float(max(0.0, min(1.0, sum(vals) / len(vals)))) if vals else 1.0


def _content_token_set(text: str) -> set:
    stop = {
        "the", "and", "for", "that", "with", "from", "this", "dans", "pour",
        "avec", "des", "les", "une", "sur", "par", "aux", "que", "est",
        "article", "titre", "chapitre", "section",
    }
    return {
        tok
        for tok in re.findall(r"\b[\wÀ-ÿ]{3,}\b", text.lower())
        if tok not in stop and not tok.isdigit()
    }


def _stage2_separation(chunks: List[Dict]) -> float:
    if len(chunks) < 2:
        return 0.0
    return _average_chunk_separation(chunks)


def _stage2_cohesion(chunks: List[Dict]) -> float:
    if not chunks:
        return 0.0
    vals = []
    for chunk in chunks:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", chunk.get("text", "")) if s.strip()]
        if len(sentences) < 2:
            vals.append(0.55)
            continue
        overlaps = []
        for idx in range(len(sentences) - 1):
            a = set(re.findall(r"\b\w+\b", sentences[idx].lower()))
            b = set(re.findall(r"\b\w+\b", sentences[idx + 1].lower()))
            union = a | b
            if union:
                overlaps.append(len(a & b) / len(union))
        vals.append(sum(overlaps) / len(overlaps) if overlaps else 0.55)
    return float(sum(vals) / len(vals)) if vals else 0.0


def _finalise_chunks(chunks: list) -> list:
    """
    Finalise the winning chunk set for delivery.

    Safety-net: try multiple field names for jsd, boundary, and icc so that
    any naming variation from older pipeline versions never silently defaults
    to the flat 35% score (= jsd=0, boundary=0.5, icc=0.5 defaults).
    """
    out = []
    for i, c in enumerate(chunks):
        chunk = dict(c)

        # jsd_score — try all known field names S3 has used across versions
        jsd = float(
            chunk.get("jsd_score")
            or chunk.get("metric_score")
            or chunk.get("boundary_signal")
            or 0.0
        )

        # boundary_score from S4 — high = similar = bad boundary
        boundary = float(chunk.get("boundary_score") or 0.5)

        # icc — intra-chunk coherence from S4
        icc = float(chunk.get("icc") or 0.5)

        chunk["chunk_index"] = i
        chunk["chunk_score"] = round(
            0.4 * (1.0 - boundary) + 0.3 * icc + 0.3 * min(jsd * 2.0, 1.0), 4
        )
        chunk["token_count"] = len(chunk.get("text", "").split())
        chunk.pop("embedding", None)
        chunk.pop("graph_vector", None)
        if isinstance(chunk.get("lstm_cell"), list):
            chunk["lstm_cell"] = [round(float(v), 4) for v in chunk["lstm_cell"]]
        out.append(chunk)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Endpoints
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> JSONResponse:
    """Accept a file upload. PDFs defer text extraction until pipeline start."""
    upload_start = time.perf_counter()
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(413, "File too large (max 50 MB).")

    filename = file.filename or "upload.txt"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "txt"
    text = ""
    token_count = None
    if ext != "pdf":
        text = _parse_file(filename, content)
        if not text.strip():
            raise HTTPException(400, "Could not extract text from the uploaded file.")
        token_count = len(text.split())

    doc_id = str(uuid.uuid4())
    doc_store[doc_id] = {
        "filename": filename,
        "content": text,
        "raw_content": content if ext == "pdf" else None,
        "size": len(content),
        "content_type": file.content_type or "application/octet-stream",
        "token_count": token_count,
    }

    return JSONResponse(
        {
            "document_id": doc_id,
            "filename": filename,
            "char_count": len(text) if text else None,
            "token_count": token_count,
            "parse_deferred": ext == "pdf",
            "upload_seconds": round(time.perf_counter() - upload_start, 3),
        }
    )


@app.post("/run/{document_id}")
async def run_pipeline(
    document_id: str,
    config: str = Form(default="{}"),
) -> JSONResponse:
    """Trigger the full pipeline for a previously uploaded document."""
    if document_id not in doc_store:
        raise HTTPException(404, "Document not found.")

    try:
        user_config = _validate_user_config(json.loads(config))
    except json.JSONDecodeError:
        user_config = {}

    job_id = str(uuid.uuid4())
    job_store[job_id] = {
        "status": "running",
        "stage": "QUEUED",
        "progress": 0,
        "message": "Job queued…",
        "doc_id": document_id,
        "results": None,
        "error": None,
        "stage_details": {},
        "reward_history": [],
    }

    thread = threading.Thread(
        target=_run_pipeline,
        args=(job_id, document_id, user_config),
        daemon=True,
    )
    thread.start()

    return JSONResponse({"job_id": job_id, "status": "running"})


@app.get("/status/{job_id}")
async def get_status(job_id: str) -> JSONResponse:
    """Return the current pipeline stage and progress percentage."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")

    return JSONResponse(
        {
            "job_id": job_id,
            "status": job["status"],
            "stage": job["stage"],
            "progress": job["progress"],
            "message": job.get("message", ""),
        }
    )


@app.get("/results/{job_id}")
async def get_results(job_id: str) -> JSONResponse:
    """Return the full chunk results once the pipeline is complete."""
    job = job_store.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")

    if job["status"] == "error":
        raise HTTPException(500, f"Pipeline error: {job.get('message', 'Unknown error')}")

    if job["status"] != "complete":
        raise HTTPException(202, "Pipeline still running.")

    return JSONResponse(job["results"])


@app.get("/export/{job_id}/{fmt}")
async def export_results(job_id: str, fmt: str) -> Response:
    """Download results as json / csv / markdown."""
    job = job_store.get(job_id)
    if not job or job["status"] != "complete":
        raise HTTPException(404, "Results not available.")

    results = job["results"]
    chunks = results.get("chunks", [])

    if fmt == "json":
        payload = json.dumps(results, indent=2, ensure_ascii=False)
        return Response(
            content=payload,
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="chunks_{job_id[:8]}.json"'},
        )

    if fmt == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(
            ["index", "token_count", "chunk_score", "jsd_score",
             "boundary_score", "boundary_type", "entities", "text_preview"]
        )
        for c in chunks:
            ents = ", ".join(e.get("text", "") for e in c.get("entities", []))
            preview = c.get("text", "")[:120].replace("\n", " ")
            writer.writerow(
                [
                    c.get("chunk_index", ""),
                    c.get("token_count", ""),
                    c.get("chunk_score", ""),
                    c.get("jsd_score", ""),
                    c.get("boundary_score", ""),
                    c.get("boundary_type", ""),
                    ents,
                    preview,
                ]
            )
        return Response(
            content=buf.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="chunks_{job_id[:8]}.csv"'},
        )

    if fmt == "markdown":
        lines = []
        summary = results.get("summary", {})
        lines.append(f"# AutoChunker Results\n")
        lines.append(f"- Document type: {summary.get('doc_type')}")
        lines.append(f"- Domain: {summary.get('domain')}")
        lines.append(f"- Chunks: {summary.get('chunk_count')}")
        lines.append(f"- Mean score: {summary.get('mean_chunk_score')}\n")
        for c in chunks:
            ents = ", ".join(e.get("text", "") for e in c.get("entities", []))
            lines.append("---")
            lines.append(
                f"<!-- chunk_index: {c.get('chunk_index')} | "
                f"tokens: {c.get('token_count')} | "
                f"score: {c.get('chunk_score')} | "
                f"entities: {ents} -->"
            )
            lines.append(c.get("text", ""))
            lines.append("")
        return Response(
            content="\n".join(lines),
            media_type="text/markdown",
            headers={"Content-Disposition": f'attachment; filename="chunks_{job_id[:8]}.md"'},
        )

    raise HTTPException(400, f"Unsupported format '{fmt}'. Use json, csv, or markdown.")


@app.get("/health")
async def health() -> JSONResponse:
    return JSONResponse({"status": "ok"})


@app.get("/backends")
async def backends() -> JSONResponse:
    """Embedding backends available to the qentropy engine + judge/QA capability."""
    try:
        embedder = get_embedder()
        return JSONResponse({
            "backends": embedder.available_backends(),
            "default": embedder.default_backend(),
            "openai_configured": qagen.openai_configured(),
            "judge_available": qagen.openai_configured(),
            "provider": openai_client.provider(),   # "openai" | "azure" | "none"
        })
    except Exception as exc:
        return JSONResponse({"backends": [], "default": None, "error": str(exc)}, status_code=200)
