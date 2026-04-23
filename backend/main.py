"""
AutoChunker Platform — FastAPI Backend
Provides 5 REST endpoints that orchestrate the 7-stage pipeline.
All heavy work is executed in a background thread so the HTTP server
stays responsive; progress is polled via GET /status/{job_id}.
"""

# Import standard libraries for file I/O, JSON manipulation, regular expressions, threading, logging, etc.
import io
import csv
import json
import re
import uuid  # For generating unique job IDs
import threading  # For running pipeline in background threads
import traceback  # For detailed error reporting
import logging  # For logging events and debugging
import os  # For OS operations like file path handling
import time  # For timing operations
import chardet

from typing import Any, Dict, List  # Type hints for function parameters and return values

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

# ── In-memory stores ──────────────────────────────────────────────────────
doc_store: Dict[str, Dict] = {}   # document_id → {filename, content, …}
job_store: Dict[str, Dict] = {}   # job_id      → {status, stage, progress, …}

# ── Default pipeline configuration ───────────────────────────────────────
DEFAULT_CONFIG: Dict[str, Any] = {
    "n_min": 50,
    "n_max": 1000,
    "tau_jsd_low": 0.15,
    "tau_jsd_high": 0.45,
    "tau_sem": 0.75,
    "max_iterations": 10,
    "alpha": 0.4,
    "beta": 0.4,
    "lambda": 0.2,
    "embedding_model": "all-MiniLM-L6-v2",
    "entropy_metric": "hybrid",
    "hybrid_lambda": 0.6,
    "threshold_mode": "percentile",
    "tau_percentile_low": 25,
    "tau_percentile_high": 75,
    "ensemble_models": [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "jina-embeddings-v2-base-en",
    ],
    "reward_objectives": {
        "quality": 0.35,
        "coverage": 0.30,
        "consistency": 0.20,
        "efficiency": 0.15,
    },
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

    if ext == "pdf":
        return _parse_pdf(content)
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
        "tau_jsd_low": (0.01, 0.6),
        "tau_jsd_high": (0.05, 0.95),
        "tau_sem": (0.2, 0.99),
        "max_iterations": (1, 30),
        "alpha": (0.0, 1.0),
        "beta": (0.0, 1.0),
        "lambda": (0.0, 1.0),
        "hybrid_lambda": (0.0, 1.0),
    }
    for key, (low, high) in numeric_ranges.items():
        if key in cfg:
            try:
                value = float(cfg[key])
                value = max(low, min(high, value))
                cfg[key] = int(value) if key in {"n_min", "n_max", "max_iterations"} else value
            except Exception:
                cfg.pop(key, None)
    if cfg.get("tau_jsd_low", DEFAULT_CONFIG["tau_jsd_low"]) >= cfg.get("tau_jsd_high", DEFAULT_CONFIG["tau_jsd_high"]):
        cfg["tau_jsd_high"] = float(cfg.get("tau_jsd_low", 0.15)) + 0.1
    metric = str(cfg.get("entropy_metric", DEFAULT_CONFIG["entropy_metric"])).lower()
    cfg["entropy_metric"] = metric if metric in {"jsd", "hellinger", "hybrid", "pmi", "depth", "drift"} else DEFAULT_CONFIG["entropy_metric"]
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
        evaluation_table, evaluation_scores, ranked = _evaluate_strategy_outputs(strategy_outputs, config)
        forced_strategy = str(config.get("chunking_strategy", "auto")).lower()
        if forced_strategy != "auto" and forced_strategy in strategy_outputs:
            winner = forced_strategy
        else:
            winner = ranked[0][0] if ranked else next(iter(strategy_outputs.keys()))
        for row in evaluation_table:
            row["winner"] = row.get("strategy") == winner
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
                "model": config["embedding_model"],
                "ensemble_models": config.get("ensemble_models", []),
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

        # ── S7: RL reward calibration per strategy ────────────────────────
        _update_job(
            job_id,
            stage="S7",
            progress=82,
            message="Running RL calibration for every chunking method…",
        )
        s7_outputs: Dict[str, Dict[str, Any]] = {}
        total_methods = max(len(strategy_outputs), 1)
        for idx, (name, output) in enumerate(strategy_outputs.items(), start=1):
            _update_job(
                job_id,
                stage="S7",
                progress=82 + int(10 * idx / total_methods),
                message=f"Running RL calibration on {name.replace('_', ' ')}…",
            )
            work_config = dict(config)
            work_config["chunking_strategy"] = name
            work_config["rl_history_key"] = f"{doc_profile.get('domain', 'general')}::{name}"
            best_chunks, reward_history, final_config = run_rl_loop(
                text,
                doc_profile,
                output.get("chunks", []),
                work_config,
            )
            s7_outputs[name] = {
                "chunks": best_chunks,
                "reward_history": reward_history,
                "final_config": final_config,
                "reward_breakdown": final_config.get("reward_breakdown", {}),
                "iterations": len(reward_history),
                "final_reward": reward_history[-1] if reward_history else 0.0,
            }

        s7_table, s7_ranked = _evaluate_s7_outputs(s7_outputs, config)
        forced_strategy = str(config.get("chunking_strategy", "auto")).lower()
        if forced_strategy != "auto" and forced_strategy in s7_outputs:
            final_winner = forced_strategy
        else:
            final_winner = s7_ranked[0][0] if s7_ranked else winner
        for row in s7_table:
            row["winner"] = row.get("strategy") == final_winner

        final_s7 = s7_outputs.get(final_winner) or next(iter(s7_outputs.values()))
        best_chunks = final_s7["chunks"]
        reward_history = final_s7["reward_history"]
        final_config = final_s7["final_config"]

        _update_job(
            job_id,
            stage_details={
                **job_store[job_id].get("stage_details", {}),
                "s7": {
                    "reward_history": reward_history,
                    "iterations": len(reward_history),
                    "final_config": final_config,
                    "reward_breakdown": final_config.get("reward_breakdown", {}),
                    "strategy": final_winner,
                    "winner": final_winner,
                    "ranked": s7_ranked,
                    "table": s7_table,
                    "strategies": {
                        name: {
                            "iterations": out["iterations"],
                            "final_reward": out["final_reward"],
                            "reward_history": out["reward_history"],
                            "reward_breakdown": out["reward_breakdown"],
                            "final_config": out["final_config"],
                            "chunk_count": len(out["chunks"]),
                            "mean_tokens": round(
                                sum(len(c.get("text", "").split()) for c in out["chunks"]) / len(out["chunks"]),
                                2,
                            ) if out["chunks"] else 0,
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


def _evaluate_strategy_outputs(
    outputs: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
) -> tuple:
    rows: List[Dict[str, Any]] = []
    scores: Dict[str, Dict[str, Any]] = {}
    for name, output in outputs.items():
        chunks = output.get("chunks", [])
        row = _score_strategy(name, chunks, config)
        rows.append(row)
        scores[name] = {
            "score": row["score"],
            "chunk_count": row["chunk_count"],
            "mean_tokens": row["mean_tokens"],
        }

    rows.sort(key=lambda r: r["score"], reverse=True)
    ranked = [(r["strategy"], r["score"]) for r in rows]
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
        row["winner"] = idx == 1
    return rows, scores, ranked


def _evaluate_s7_outputs(
    outputs: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
) -> tuple:
    rows: List[Dict[str, Any]] = []
    for name, output in outputs.items():
        chunks = output.get("chunks", [])
        row = _score_strategy(name, chunks, config)
        final_reward = float(output.get("final_reward", 0.0) or 0.0)
        row["score"] = round(max(0.0, min(1.0, final_reward)), 4)
        row["rl_final_reward"] = round(final_reward, 4)
        row["iterations"] = int(output.get("iterations", 0) or 0)
        breakdown = output.get("reward_breakdown", {}) or {}
        row["reward_quality"] = float(breakdown.get("quality", 0.0) or 0.0)
        row["reward_coverage"] = float(breakdown.get("coverage", 0.0) or 0.0)
        row["reward_consistency"] = float(breakdown.get("consistency", 0.0) or 0.0)
        row["reward_efficiency"] = float(breakdown.get("efficiency", 0.0) or 0.0)
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
    """Assign a composite chunk_score and clean up non-serialisable fields."""
    out = []
    for i, c in enumerate(chunks):
        chunk = dict(c)

        jsd = float(chunk.get("jsd_score", 0.0))
        boundary = float(chunk.get("boundary_score", 0.5))
        icc = float(chunk.get("icc", 0.5))

        # Composite chunk quality score (higher = better)
        # Low JSD = similar to neighbours (less coherent boundary) → penalise
        # High boundary_score = similar to neighbours → penalise (should differ)
        chunk_score = float(
            0.4 * (1.0 - boundary)        # boundary distinctiveness
            + 0.3 * icc                    # internal cohesion
            + 0.3 * min(jsd * 2.0, 1.0)   # JSD strength (rescaled)
        )
        chunk["chunk_index"] = i
        chunk["chunk_score"] = round(chunk_score, 4)
        chunk["token_count"] = len(chunk.get("text", "").split())

        # Drop raw embedding vectors from results (too large)
        chunk.pop("embedding", None)
        chunk.pop("graph_vector", None)

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
