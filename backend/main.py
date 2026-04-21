"""
AutoChunker Platform — FastAPI Backend
Provides 5 REST endpoints that orchestrate the 7-stage pipeline.
All heavy work is executed in a background thread so the HTTP server
stays responsive; progress is polled via GET /status/{job_id}.
"""

import io
import csv
import json
import re
import uuid
import threading
import traceback
import logging
import os
from typing import Any, Dict, List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB
logger = logging.getLogger("autochunker")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}',
    )

# ── Pipeline stages ───────────────────────────────────────────────────────
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
    "n_min": 100,
    "n_max": 500,
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
    # Try pdfplumber first, then PyPDF2
    pages: List[str] = []
    try:
        import pdfplumber  # noqa: E402
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text)
    except Exception:
        pass

    if not pages:
        try:
            import PyPDF2  # noqa: E402
            reader = PyPDF2.PdfReader(io.BytesIO(content))
            pages = [
                reader.pages[i].extract_text() or ""
                for i in range(len(reader.pages))
            ]
        except Exception:
            pass

    if not pages:
        return _sanitize_text(content.decode("utf-8", errors="replace"))

    # ── Clean each page before joining ────────────────────────────────────
    cleaned = [_clean_pdf_page(p) for p in pages if p.strip()]

    # Detect & strip repeated header/footer lines that appear on most pages.
    # A line that appears verbatim (after stripping) in ≥ 60 % of pages and
    # is ≤ 12 words is almost certainly a running header or footer.
    if len(cleaned) >= 3:
        cleaned = _strip_repeated_lines(cleaned)

    return _sanitize_text("\n\n".join(cleaned))


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
    cfg["entropy_metric"] = metric if metric in {"jsd", "hellinger", "hybrid"} else DEFAULT_CONFIG["entropy_metric"]
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

        text: str = doc["content"]
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

        # ── S7: RL reward calibration ─────────────────────────────────────
        _update_job(
            job_id,
            stage="S7",
            progress=82,
            message=f"Running RL calibration on {winner.replace('_', ' ')}…",
        )
        best_chunks, reward_history, final_config = run_rl_loop(
            text, doc_profile, embedded, config
        )

        _update_job(
            job_id,
            stage_details={
                **job_store[job_id].get("stage_details", {}),
                "s7": {
                    "reward_history": reward_history,
                    "iterations": len(reward_history),
                    "final_config": final_config,
                    "reward_breakdown": final_config.get("reward_breakdown", {}),
                    "strategy": winner,
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
                "winning_strategy": winner,
                "stage2_winning_strategy": selected_strategy,
                "evaluated_strategies": len(evaluation_table),
            },
            "stage_details": job_store[job_id].get("stage_details", {}),
            "reward_history": reward_history,
            "strategy_evaluation": evaluation_table,
            "stage2_evaluation": stage2_table,
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
        "metric": work_config.get("entropy_metric", "jsd"),
        "thresholds": refined[-1].get("thresholds", {}) if refined else {},
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
        }

    rows.sort(key=lambda r: r["score"], reverse=True)
    ranked = [(r["strategy"], r["score"]) for r in rows]
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
        row["winner"] = idx == 1
    return rows, scores, ranked


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
    boundary_quality = _stage2_boundary_quality(chunks)
    separation = _stage2_separation(chunks)
    cohesion = _stage2_cohesion(chunks)
    boundary_signal = separation * min(1.0, len(chunks) / max(expected_count * 0.45, 1.0))

    score = (
        0.32 * size_fit
        + 0.20 * count_fit
        + 0.16 * coverage
        + 0.14 * boundary_quality
        + 0.10 * separation
        + 0.08 * cohesion
    )
    return {
        "strategy": strategy_name,
        "rank": 0,
        "winner": False,
        "score": round(float(max(0.0, min(1.0, score))), 4),
        "chunk_count": len(chunks),
        "mean_tokens": round(mean_tokens, 1),
        "mean_icc": round(float(cohesion), 4),
        "inter_separation": round(float(separation), 4),
        "entropy_strength": round(float(boundary_signal), 4),
        "boundary_distinctiveness": round(float(boundary_quality), 4),
        "size_fit": round(float(max(0.0, min(1.0, size_fit))), 4),
        "coverage": round(float(coverage), 4),
        "count_fit": round(float(count_fit), 4),
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
    """Accept a file upload and store its parsed text content."""
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(413, "File too large (max 50 MB).")

    text = _parse_file(file.filename or "upload.txt", content)
    if not text.strip():
        raise HTTPException(400, "Could not extract text from the uploaded file.")

    doc_id = str(uuid.uuid4())
    doc_store[doc_id] = {
        "filename": file.filename,
        "content": text,
        "size": len(content),
        "content_type": file.content_type or "application/octet-stream",
    }

    return JSONResponse(
        {
            "document_id": doc_id,
            "filename": file.filename,
            "char_count": len(text),
            "token_count": len(text.split()),
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
