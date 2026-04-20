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
            initial_chunks = select_best_strategy(all_chunks_map, doc_profile["type"], config)
        except Exception:
            logger.exception("S2 failed; falling back to single chunk")
            all_chunks_map = {"fallback": [{"text": text, "start": 0, "end": len(text), "method": "fallback"}]}
            initial_chunks = all_chunks_map["fallback"]

        if not initial_chunks:
            # Ultra-short document: treat the whole text as one chunk
            initial_chunks = [
                {"text": text, "start": 0, "end": len(text), "method": "fallback"}
            ]

        # Identify which strategy was actually chosen
        selected_strategy = initial_chunks[0].get("method", "unknown") if initial_chunks else "unknown"

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
                },
            },
        )

        # ── S3: Entropy refinement ────────────────────────────────────────
        _update_job(
            job_id,
            stage="S3",
            progress=28,
            message="Computing entropy boundaries…",
        )
        refined = refine_boundaries(initial_chunks, config)
        jsd_series = get_jsd_series(refined)

        _update_job(
            job_id,
            stage_details={
                **job_store[job_id].get("stage_details", {}),
                "s3": {
                    "jsd_series": jsd_series,
                    "chunk_count": len(refined),
                    "metric": config.get("entropy_metric", "jsd"),
                    "thresholds": refined[-1].get("thresholds", {}) if refined else {},
                },
            },
        )

        # ── S4: Boundary quality filter ───────────────────────────────────
        _update_job(
            job_id,
            stage="S4",
            progress=42,
            message="Filtering boundaries…",
        )
        filtered = filter_boundaries(refined, doc_profile["type"], [], config)

        _update_job(
            job_id,
            stage_details={
                **job_store[job_id].get("stage_details", {}),
                "s4": {
                    "chunk_count": len(filtered),
                    "weighted_decision": True,
                },
            },
        )

        # ── S5: Graph enrichment ──────────────────────────────────────────
        _update_job(
            job_id,
            stage="S5",
            progress=55,
            message="Building entity graph…",
        )
        enriched = enrich_graph(filtered, [], config)
        graph_data = build_entity_graph_data(enriched)

        _update_job(
            job_id,
            stage_details={
                **job_store[job_id].get("stage_details", {}),
                "s5": {"entity_graph": graph_data},
            },
        )

        # ── S6: Contextual embedding ──────────────────────────────────────
        _update_job(
            job_id,
            stage="S6",
            progress=68,
            message="Generating embeddings…",
        )
        embedded, embeddings = embed_chunks(
            enriched,
            text,
            doc_profile,
            config["embedding_model"],
            config,
        )

        _update_job(
            job_id,
            stage_details={
                **job_store[job_id].get("stage_details", {}),
                "s6": {
                    "embedding_dim": len(embeddings[0]) if embeddings else 0,
                    "model": config["embedding_model"],
                    "ensemble_models": config.get("ensemble_models", []),
                },
            },
        )

        # ── S7: RL reward calibration ─────────────────────────────────────
        _update_job(
            job_id,
            stage="S7",
            progress=78,
            message="Running RL calibration loop…",
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
            },
            "stage_details": job_store[job_id].get("stage_details", {}),
            "reward_history": reward_history,
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
