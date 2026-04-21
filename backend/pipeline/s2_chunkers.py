"""
S2 - Expert Chunk Candidate Generation

This stage deliberately produces several strong candidate chunk sets. Later
stages score/refine them, so S2 should preserve structure and avoid obvious
bad candidates: tiny fragments, oversized chunks, broken articles/sections,
or arbitrary cuts through sentences/code blocks.
"""

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "of", "to", "in", "for", "on", "with", "as", "by",
    "from", "at", "that", "this", "these", "those", "it", "its", "we",
    "you", "they", "he", "she", "not", "no", "if", "then", "than",
    "le", "la", "les", "de", "des", "du", "et", "en", "un", "une",
    "dans", "pour", "que", "est", "sur", "par", "avec", "au", "aux",
}

LEGAL_BOUNDARY_RE = re.compile(
    r"^[ \t]*(?:"
    r"(?:TITRE|CHAPITRE|SECTION|SOUS-SECTION|PARAGRAPHE|BOOK|PART|CHAPTER)"
    r"\s+(?:[IVXLCDM]+|\d+|PREMIER|PREMIERE|PREMIÈRE|premier|premiere|première)"
    r"|(?:Article|Art\.?|ARTICLE)\s+(?:\d+(?:\s*(?:er|eme|ème|e|bis|ter|quater))?|premier|1er|[IVX]+)"
    r"|§\s*\d+"
    r")(?:[ \t].*)?$",
    re.MULTILINE | re.IGNORECASE,
)

SECTION_HEADER_RE = re.compile(
    r"^\s*("
    r"#{1,6}\s+\S+"
    r"|<h[1-6][^>]*>.*?</h[1-6]>"
    r"|\\(?:chapter|section|subsection|subsubsection)\{[^}]+\}"
    r"|\d+(?:\.\d+){0,3}\s+\S.+"
    r"|[A-Z][A-Z0-9\s\-_/,]{4,}"
    r"|[A-Z][^:\n]{2,80}:"
    r")\s*$",
    re.IGNORECASE,
)

CODE_BOUNDARY_RE = re.compile(
    r"^\s*(?:"
    r"(?:async\s+)?def\s+\w+\s*\("
    r"|class\s+\w+"
    r"|function\s+\w+"
    r"|(?:public|private|protected|static)\s+"
    r"|const\s+\w+\s*=|let\s+\w+\s*=|var\s+\w+\s*="
    r"|#include\s+[<\"]|#define\s+"
    r")",
    re.MULTILINE,
)

TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$|^[^,\n]+(?:,[^,\n]+){2,}$", re.MULTILINE)

VALID_STRATEGIES = {
    "auto",
    "recursive",
    "sliding_window",
    "structure",
    "semantic_boundaries",
    "sentence_clustering",
    "paragraph_pack",
    "legal_articles",
}


def run_all_chunkers(text: str, doc_type: str, config: Dict[str, Any]) -> Dict[str, List[Dict]]:
    n_min = int(config.get("n_min", 100))
    n_max = int(config.get("n_max", 500))
    structure_type = _detect_structure_type(text, doc_type)
    eff_min, eff_max = _adaptive_window(n_min, n_max, text, doc_type, structure_type)
    # Later stages may merge adjacent chunks. Defaulting to overlap here can
    # duplicate text after those merges, so overlap is explicit opt-in.
    overlap = int(config.get("overlap_tokens", 0))

    tasks = {
        "recursive": lambda: recursive_character_split(text, eff_min, eff_max, doc_type, structure_type),
        "sliding_window": lambda: sliding_window_split(text, eff_max, overlap),
        "structure": lambda: structure_based_split(text, doc_type, eff_min, eff_max, structure_type),
        "semantic_boundaries": lambda: semantic_boundary_split(text, eff_min, eff_max, config, structure_type),
        "sentence_clustering": lambda: sentence_cluster_split(text, eff_min, eff_max, config),
        "paragraph_pack": lambda: paragraph_pack_split(text, eff_min, eff_max),
    }
    if structure_type == "legal_article":
        tasks["legal_articles"] = lambda: legal_article_split(text, eff_min, eff_max)

    out: Dict[str, List[Dict]] = {}
    with ThreadPoolExecutor(max_workers=min(len(tasks), 8)) as ex:
        futures = {name: ex.submit(fn) for name, fn in tasks.items()}
        for name, fut in futures.items():
            try:
                out[name] = _quality_pass(fut.result(), text, eff_min, eff_max, name)
            except Exception:
                out[name] = []
    return out


def select_best_strategy(all_chunks: Dict[str, List[Dict]], doc_type: str, config: Dict[str, Any]) -> List[Dict]:
    n_min = int(config.get("n_min", 100))
    n_max = int(config.get("n_max", 500))

    forced = str(config.get("chunking_strategy", "auto")).lower()
    if forced and forced != "auto" and forced in VALID_STRATEGIES:
        chunks = all_chunks.get(forced)
        if chunks:
            return chunks

    preferred = {
        "code": ["structure", "recursive", "semantic_boundaries", "sentence_clustering", "sliding_window", "paragraph_pack"],
        "table": ["structure", "paragraph_pack", "sliding_window", "recursive", "semantic_boundaries", "sentence_clustering"],
        "mixed": ["structure", "semantic_boundaries", "paragraph_pack", "recursive", "sentence_clustering", "sliding_window"],
        "prose": ["semantic_boundaries", "paragraph_pack", "sentence_clustering", "recursive", "structure", "sliding_window"],
    }.get(doc_type, ["semantic_boundaries", "paragraph_pack", "recursive", "structure", "sentence_clustering", "sliding_window"])

    best_name = None
    best_score = -1.0
    for name, chunks in all_chunks.items():
        if not chunks:
            continue
        strategy_bias = 0.025 * max(0, (len(preferred) - preferred.index(name))) if name in preferred else 0.0
        if name == "legal_articles":
            strategy_bias += 0.06
        score = _strategy_quality_score(chunks, n_min, n_max) + strategy_bias
        if score > best_score:
            best_score = score
            best_name = name
    return all_chunks.get(best_name, []) if best_name else []


def _strategy_quality_score(chunks: List[Dict], n_min: int, n_max: int) -> float:
    if not chunks:
        return 0.0
    sizes = [max(1, len(c.get("text", "").split())) for c in chunks]
    avg = float(np.mean(sizes))
    std = float(np.std(sizes))
    target = max(float(n_min), float(n_max) * 0.72)
    fit_score = 1.0 - min(1.0, abs(avg - target) / max(target, 1.0))
    stability = 1.0 - min(1.0, std / max(avg, 1.0))
    in_range = sum(1 for s in sizes if n_min <= s <= n_max) / len(sizes)
    boundary_div = _avg_boundary_divergence(chunks)
    integrity = _boundary_integrity(chunks)
    complete = _sentence_completion_score(chunks)

    return float(
        0.24 * fit_score
        + 0.18 * stability
        + 0.18 * in_range
        + 0.17 * boundary_div
        + 0.13 * integrity
        + 0.10 * complete
    )


def recursive_character_split(
    text: str,
    n_min: int,
    n_max: int,
    doc_type: str = "prose",
    structure_type: str = "plain",
) -> List[Dict]:
    separators = _separators_for(doc_type, structure_type)
    raw_chunks = _recursive_split(text, separators, n_min, n_max)
    return _chunks_from_texts(text, raw_chunks, "recursive")


def sliding_window_split(text: str, window_size: int, overlap: int) -> List[Dict]:
    tokens = text.split()
    if not tokens:
        return []
    step = max(1, window_size - overlap)
    offsets = _build_token_offsets(text, tokens)
    chunks: List[Dict] = []
    i = 0
    while i < len(tokens):
        end_idx = min(i + window_size, len(tokens))
        chunk_text = " ".join(tokens[i:end_idx])
        chunks.append({
            "text": chunk_text,
            "start": offsets[i],
            "end": offsets[end_idx - 1] + len(tokens[end_idx - 1]),
            "method": "sliding_window",
        })
        if end_idx == len(tokens):
            break
        i += step
    return chunks


def structure_based_split(
    text: str,
    doc_type: str,
    n_min: int = 100,
    n_max: int = 500,
    structure_type: str = "plain",
) -> List[Dict]:
    if structure_type == "legal_article":
        return legal_article_split(text, n_min, n_max)
    if doc_type == "code":
        return _split_code(text, n_min, n_max)
    if doc_type == "table" or structure_type == "table":
        return _split_table(text, n_min, n_max)
    sections = _split_sections(text)
    chunks = _pack_units_with_offsets(sections, n_min, n_max, "structure", "\n\n")
    return chunks if len(chunks) > 1 else paragraph_pack_split(text, n_min, n_max)


def paragraph_pack_split(text: str, n_min: int, n_max: int) -> List[Dict]:
    units = _paragraph_units(text)
    if len(units) <= 1:
        units = _sentence_units(text)
    return _pack_units_with_offsets(units, n_min, n_max, "paragraph_pack", "\n\n")


def legal_article_split(text: str, n_min: int, n_max: int) -> List[Dict]:
    spans = _boundary_spans(text, LEGAL_BOUNDARY_RE)
    if len(spans) <= 1:
        return structure_based_split(text, "prose", n_min, n_max, "sectioned")
    units = [(text[start:end].strip(), start, end) for start, end in spans if text[start:end].strip()]
    return _pack_units_with_offsets(units, n_min, n_max, "legal_articles", "\n\n")


def semantic_boundary_split(
    text: str,
    n_min: int,
    n_max: int,
    config: Dict[str, Any],
    structure_type: str = "plain",
) -> List[Dict]:
    sentences = _sentence_units(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return _chunks_from_texts(text, [sentences[0][0]], "semantic_boundaries")

    vectors = np.array([_hash_embedding(s[0], 128) for s in sentences], dtype=np.float32)
    distances = [1.0 - float(_cosine(vectors[i], vectors[i + 1])) for i in range(len(vectors) - 1)]
    if distances:
        dynamic = float(np.percentile(distances, 72))
    else:
        dynamic = 0.45
    threshold = float(config.get("semantic_shift_threshold", dynamic))
    threshold = max(0.25, min(0.78, min(threshold, dynamic + 0.08)))

    chunks: List[Dict] = []
    buf: List[Tuple[str, int, int]] = [sentences[0]]
    for i in range(1, len(sentences)):
        sent = sentences[i]
        buf_wc = _unit_word_count(buf)
        sent_wc = len(sent[0].split())
        hard_boundary = _looks_like_heading(sent[0]) or (
            structure_type == "legal_article" and LEGAL_BOUNDARY_RE.match(sent[0])
        )
        semantic_shift = distances[i - 1] >= threshold if i - 1 < len(distances) else False
        exceeds_max = (buf_wc + sent_wc) > n_max
        should_split = (hard_boundary and buf_wc >= max(20, int(n_min * 0.45))) or (semantic_shift and buf_wc >= n_min) or exceeds_max
        if should_split:
            chunks.append(_build_chunk_from_units(buf, "semantic_boundaries"))
            buf = [sent]
        else:
            buf.append(sent)
    if buf:
        chunks.append(_build_chunk_from_units(buf, "semantic_boundaries"))
    return chunks


def sentence_cluster_split(text: str, n_min: int, n_max: int, config: Dict[str, Any]) -> List[Dict]:
    sentences = _sentence_units(text)
    if not sentences:
        return []

    threshold = float(config.get("sentence_cluster_similarity", 0.62))
    chunks: List[Dict] = []
    buf: List[Tuple[str, int, int]] = []
    centroid: Optional[np.ndarray] = None

    for sent in sentences:
        vec = _hash_embedding(sent[0], 128)
        buf_wc = _unit_word_count(buf)
        sent_wc = len(sent[0].split())
        sim = float(_cosine(vec, centroid)) if centroid is not None else 1.0
        should_split = bool(buf) and (
            (sim < threshold and buf_wc >= n_min)
            or (buf_wc + sent_wc > n_max)
            or (_looks_like_heading(sent[0]) and buf_wc >= max(20, int(n_min * 0.45)))
        )
        if should_split:
            chunks.append(_build_chunk_from_units(buf, "sentence_clustering"))
            buf = [sent]
            centroid = vec
        else:
            buf.append(sent)
            if centroid is None:
                centroid = vec
            else:
                centroid = (centroid * (len(buf) - 1) + vec) / len(buf)

    if buf:
        chunks.append(_build_chunk_from_units(buf, "sentence_clustering"))
    return chunks


def _detect_structure_type(text: str, doc_type: str) -> str:
    if doc_type == "code":
        return "code"
    legal_count = len(list(LEGAL_BOUNDARY_RE.finditer(text)))
    if legal_count >= 2:
        return "legal_article"
    if doc_type == "table" or len(TABLE_ROW_RE.findall(text)) >= 3:
        return "table"
    headings = sum(1 for line in text.splitlines() if _looks_like_heading(line))
    if headings >= 2:
        return "sectioned"
    paragraphs = [p for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(paragraphs) >= 3:
        return "paragraph"
    return "plain"


def _adaptive_window(n_min: int, n_max: int, text: str, doc_type: str, structure_type: str) -> Tuple[int, int]:
    total = max(1, len(text.split()))
    eff_min, eff_max = n_min, n_max
    if total < n_max * 2:
        eff_min = max(20, min(n_min, int(total * 0.18)))
        eff_max = max(eff_min + 40, min(n_max, int(total * 0.55)))
    elif structure_type == "legal_article":
        eff_min = max(40, int(n_min * 0.75))
        eff_max = max(eff_min + 80, int(n_max * 0.90))
    elif doc_type == "code":
        eff_min = max(30, int(n_min * 0.70))
        eff_max = max(eff_min + 80, int(n_max * 0.85))
    elif structure_type in {"sectioned", "paragraph"}:
        eff_max = max(eff_min + 80, int(n_max * 1.05))
    return eff_min, max(eff_min + 20, eff_max)


def _separators_for(doc_type: str, structure_type: str) -> List[str]:
    if doc_type == "code":
        return ["\nclass ", "\ndef ", "\nasync def ", "\nfunction ", "\n\n", "\n", "; ", " "]
    if structure_type == "legal_article":
        return ["\n\nArticle ", "\n\nARTICLE ", "\n\nSection ", "\n\nSECTION ", "\n\n", "\n", ". ", "; ", " "]
    if structure_type == "table":
        return ["\n\n", "\n", "|", ",", " "]
    return ["\n\n\n", "\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " "]


def _recursive_split(text: str, separators: List[str], n_min: int, n_max: int) -> List[str]:
    if not text.strip():
        return []
    if len(text.split()) <= n_max:
        return [text.strip()]
    sep = separators[0] if separators else " "
    rest = separators[1:] if len(separators) > 1 else []
    parts = text.split(sep) if sep else list(text)
    if len(parts) <= 1 and rest:
        return _recursive_split(text, rest, n_min, n_max)

    chunks: List[str] = []
    current = ""
    for part in parts:
        piece = part.strip()
        if not piece:
            continue
        candidate = (current + sep + piece).strip() if current else piece
        if len(candidate.split()) <= n_max:
            current = candidate
            continue
        if current:
            chunks.extend(_recursive_split(current, rest, n_min, n_max) if len(current.split()) > n_max and rest else [current])
        current = piece
    if current:
        chunks.extend(_recursive_split(current, rest, n_min, n_max) if len(current.split()) > n_max and rest else [current])
    return _merge_text_units(chunks, n_min, n_max)


def _split_code(text: str, n_min: int, n_max: int) -> List[Dict]:
    spans = _boundary_spans(text, CODE_BOUNDARY_RE)
    if len(spans) <= 1:
        return recursive_character_split(text, n_min, n_max, "code", "code")
    units = [(text[start:end].strip(), start, end) for start, end in spans if text[start:end].strip()]
    return _pack_units_with_offsets(units, n_min, n_max, "structure", "\n\n")


def _split_table(text: str, n_min: int, n_max: int) -> List[Dict]:
    lines = [(line, start, start + len(line)) for line, start in _iter_lines_with_offsets(text) if line.strip()]
    return _pack_units_with_offsets(lines, max(10, int(n_min * 0.5)), n_max, "structure", "\n")


def _split_sections(text: str) -> List[Tuple[str, int, int]]:
    starts = []
    pos = 0
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped and _looks_like_heading(stripped):
            starts.append(pos)
        pos += len(line)
    if len(starts) <= 1:
        return _paragraph_units(text)
    starts = sorted(set(starts))
    spans = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        chunk = text[start:end].strip()
        if chunk:
            spans.append((chunk, start, end))
    prefix = text[:starts[0]].strip()
    if prefix:
        spans.insert(0, (prefix, 0, starts[0]))
    return spans


def _boundary_spans(text: str, pattern: re.Pattern) -> List[Tuple[int, int]]:
    matches = list(pattern.finditer(text))
    if not matches:
        return [(0, len(text))]
    starts = [m.start() for m in matches]
    if starts[0] > 0 and text[:starts[0]].strip():
        starts.insert(0, 0)
    spans = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(text)
        if text[start:end].strip():
            spans.append((start, end))
    return spans


def _paragraph_units(text: str) -> List[Tuple[str, int, int]]:
    units = []
    for match in re.finditer(r"\S(?:.*?)(?=\n\s*\n|\Z)", text, re.DOTALL):
        raw = match.group(0).strip()
        if raw:
            units.append((raw, match.start(), match.end()))
    return units or [(text.strip(), 0, len(text))]


def _sentence_units(text: str) -> List[Tuple[str, int, int]]:
    pattern = r"(?m)^\s*#{1,6}\s+.+$|[^.!?\n]+(?:[.!?]+|$)"
    units = []
    for match in re.finditer(pattern, text):
        sent = match.group(0).strip()
        if sent:
            units.append((sent, match.start(), match.end()))
    return units or [(text.strip(), 0, len(text))]


def _pack_units_with_offsets(
    units: List[Tuple[str, int, int]],
    n_min: int,
    n_max: int,
    method: str,
    joiner: str,
) -> List[Dict]:
    if not units:
        return []
    chunks: List[Dict] = []
    buf: List[Tuple[str, int, int]] = []
    for unit in units:
        unit_wc = len(unit[0].split())
        buf_wc = _unit_word_count(buf)
        if buf and buf_wc + unit_wc > n_max:
            chunks.append(_build_chunk_from_units(buf, method, joiner))
            buf = [unit]
        else:
            buf.append(unit)
        if _unit_word_count(buf) >= n_max:
            chunks.append(_build_chunk_from_units(buf, method, joiner))
            buf = []
    if buf:
        chunks.append(_build_chunk_from_units(buf, method, joiner))
    return _merge_small_chunks(chunks, n_min, n_max)


def _build_chunk_from_units(
    units: List[Tuple[str, int, int]],
    method: str,
    joiner: str = " ",
) -> Dict[str, Any]:
    text = joiner.join(u[0].strip() for u in units if u[0].strip()).strip()
    return {"text": text, "start": units[0][1], "end": units[-1][2], "method": method}


def _chunks_from_texts(text: str, chunk_texts: List[str], method: str) -> List[Dict]:
    chunks: List[Dict] = []
    cursor = 0
    for chunk in chunk_texts:
        cleaned = chunk.strip()
        if not cleaned:
            continue
        start = text.find(cleaned, cursor)
        if start == -1:
            start = text.find(cleaned)
        if start == -1:
            start = cursor
        end = min(len(text), start + len(cleaned))
        chunks.append({"text": cleaned, "start": start, "end": end, "method": method})
        cursor = max(cursor, end)
    return chunks


def _quality_pass(chunks: List[Dict], full_text: str, n_min: int, n_max: int, method: str) -> List[Dict]:
    cleaned = []
    for chunk in sorted((dict(c) for c in chunks if c.get("text", "").strip()), key=lambda c: c.get("start", 0)):
        chunk["text"] = chunk["text"].strip()
        chunk["method"] = method
        if len(chunk["text"].split()) > int(n_max * 1.35):
            pieces = _sentence_units(chunk["text"])
            if len(pieces) > 1:
                base = int(chunk.get("start", 0))
                rel_units = [(u[0], base + u[1], base + u[2]) for u in pieces]
                cleaned.extend(_pack_units_with_offsets(rel_units, n_min, n_max, method, " "))
                continue
            cleaned.extend(recursive_character_split(chunk["text"], n_min, n_max, "prose", "plain"))
        else:
            cleaned.append(chunk)
    merged = _merge_small_chunks(cleaned, n_min, n_max)
    return _reindex(merged, full_text, method)


def _merge_small_chunks(chunks: List[Dict], n_min: int, n_max: int) -> List[Dict]:
    if not chunks:
        return []
    merged: List[Dict] = []
    current = dict(chunks[0])
    for nxt in chunks[1:]:
        current_wc = len(current["text"].split())
        nxt_wc = len(nxt["text"].split())
        combined_wc = len((current["text"] + " " + nxt["text"]).split())
        if (current_wc < n_min or nxt_wc < max(20, int(n_min * 0.45))) and combined_wc <= int(n_max * 1.18):
            current["text"] = (current["text"] + "\n\n" + nxt["text"]).strip()
            current["end"] = nxt.get("end", current.get("end", 0))
        else:
            merged.append(current)
            current = dict(nxt)
    merged.append(current)

    if len(merged) >= 2 and len(merged[-1]["text"].split()) < n_min:
        prev = merged[-2]
        tail = merged[-1]
        if len((prev["text"] + " " + tail["text"]).split()) <= int(n_max * 1.25):
            prev["text"] = (prev["text"] + "\n\n" + tail["text"]).strip()
            prev["end"] = tail.get("end", prev.get("end", 0))
            merged.pop()
    return merged


def _merge_text_units(units: List[str], n_min: int, n_max: int) -> List[str]:
    chunks = [{"text": u, "start": 0, "end": 0, "method": "recursive"} for u in units if u.strip()]
    return [c["text"] for c in _merge_small_chunks(chunks, n_min, n_max)]


def _reindex(chunks: List[Dict], full_text: str, method: str) -> List[Dict]:
    out = []
    cursor = 0
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue
        start = int(chunk.get("start", -1))
        end = int(chunk.get("end", -1))
        if start < 0 or end <= start or full_text[start:end].strip()[:20] != text[:20]:
            found = full_text.find(text, cursor)
            if found == -1:
                found = full_text.find(text)
            if found != -1:
                start, end = found, found + len(text)
        out.append({
            "text": text,
            "start": max(0, start),
            "end": max(max(0, start), min(len(full_text), end if end > start else start + len(text))),
            "method": method,
        })
        cursor = out[-1]["end"]
    return out


def _build_token_offsets(text: str, tokens: List[str]) -> List[int]:
    offsets: List[int] = []
    pos = 0
    for token in tokens:
        idx = text.find(token, pos)
        idx = pos if idx == -1 else idx
        offsets.append(idx)
        pos = idx + len(token)
    return offsets


def _iter_lines_with_offsets(text: str):
    pos = 0
    for line in text.splitlines(keepends=True):
        yield line.rstrip("\n"), pos
        pos += len(line)


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 120:
        return False
    return bool(SECTION_HEADER_RE.match(stripped) or LEGAL_BOUNDARY_RE.match(stripped))


def _unit_word_count(units: List[Tuple[str, int, int]]) -> int:
    return sum(len(u[0].split()) for u in units)


def _avg_boundary_divergence(chunks: List[Dict]) -> float:
    if len(chunks) < 2:
        return 0.5
    vals = []
    for i in range(len(chunks) - 1):
        a = set(_content_tokens(chunks[i].get("text", "")))
        b = set(_content_tokens(chunks[i + 1].get("text", "")))
        union = a | b
        vals.append(1.0 - (len(a & b) / len(union) if union else 0.0))
    return float(np.mean(vals)) if vals else 0.5


def _boundary_integrity(chunks: List[Dict]) -> float:
    if not chunks:
        return 0.0
    good = 0
    for chunk in chunks:
        text = chunk.get("text", "").strip()
        first = text.splitlines()[0].strip() if text else ""
        if _looks_like_heading(first) or first[:1].isupper() or first[:1].isdigit():
            good += 1
    return good / len(chunks)


def _sentence_completion_score(chunks: List[Dict]) -> float:
    if not chunks:
        return 0.0
    complete = 0
    for chunk in chunks:
        tail = chunk.get("text", "").rstrip()
        if not tail or tail.endswith((".", "!", "?", ":", ";", "}", "]", "```")):
            complete += 1
    return complete / len(chunks)


def _hash_embedding(text: str, dim: int = 96) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in _content_tokens(text):
        vec[hash(tok) % dim] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _content_tokens(text: str) -> List[str]:
    return [
        tok for tok in re.findall(r"\b\w+\b", text.lower())
        if len(tok) > 1 and tok not in STOPWORDS
    ] or re.findall(r"\b\w+\b", text.lower())


def _cosine(a: np.ndarray, b: Optional[np.ndarray]) -> float:
    if b is None:
        return 0.0
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
