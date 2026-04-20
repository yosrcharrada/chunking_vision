"""
S2 — Parallel Chunkers
Runs 5 chunking strategies:
  1) Recursive character split
  2) Sliding window
  3) Structure-based
  4) Semantic boundaries
  5) Sentence clustering
"""

import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Tuple

import numpy as np


def run_all_chunkers(text: str, doc_type: str, config: Dict[str, Any]) -> Dict[str, List[Dict]]:
    n_min = int(config.get("n_min", 100))
    n_max = int(config.get("n_max", 500))
    overlap = max(10, int(n_max * 0.2))

    tasks = {
        "recursive": lambda: recursive_character_split(text, n_min, n_max),
        "sliding_window": lambda: sliding_window_split(text, n_max, overlap),
        "structure": lambda: structure_based_split(text, doc_type),
        "semantic_boundaries": lambda: semantic_boundary_split(text, n_min, n_max, config),
        "sentence_clustering": lambda: sentence_cluster_split(text, n_min, n_max, config),
    }
    out: Dict[str, List[Dict]] = {}
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {name: ex.submit(fn) for name, fn in tasks.items()}
        for name, fut in futures.items():
            try:
                out[name] = fut.result()
            except Exception:
                out[name] = []
    return out


VALID_STRATEGIES = {"auto", "recursive", "sliding_window", "structure", "semantic_boundaries", "sentence_clustering"}


def select_best_strategy(all_chunks: Dict[str, List[Dict]], doc_type: str, config: Dict[str, Any]) -> List[Dict]:
    n_max = int(config.get("n_max", 500))

    # Allow the user to force a specific strategy
    forced = str(config.get("chunking_strategy", "auto")).lower()
    if forced and forced != "auto" and forced in VALID_STRATEGIES:
        chunks = all_chunks.get(forced)
        if chunks:
            return chunks
        # Forced strategy produced no chunks (e.g. structure on plain prose) – fall through to auto

    preferred = {
        "code": ["structure", "semantic_boundaries", "recursive", "sliding_window", "sentence_clustering"],
        "table": ["structure", "sliding_window", "recursive", "semantic_boundaries", "sentence_clustering"],
        "mixed": ["semantic_boundaries", "structure", "recursive", "sentence_clustering", "sliding_window"],
        "prose": ["semantic_boundaries", "sentence_clustering", "recursive", "sliding_window", "structure"],
    }.get(doc_type, ["semantic_boundaries", "recursive", "sliding_window", "structure", "sentence_clustering"])

    best_name = None
    best_score = -1.0
    for name, chunks in all_chunks.items():
        if len(chunks) < 1:
            continue
        strategy_bias = 0.03 * max(0, (len(preferred) - preferred.index(name))) if name in preferred else 0.0
        score = _strategy_quality_score(chunks, n_max) + strategy_bias
        if score > best_score:
            best_score = score
            best_name = name
    if best_name:
        return all_chunks.get(best_name, [])
    return []


def _strategy_quality_score(chunks: List[Dict], n_max: int) -> float:
    if not chunks:
        return 0.0
    sizes = [max(1, len(c.get("text", "").split())) for c in chunks]
    avg = float(np.mean(sizes))
    std = float(np.std(sizes))
    target = n_max * 0.7
    fit_score = 1.0 - min(1.0, abs(avg - target) / max(target, 1.0))
    stability = 1.0 - min(1.0, std / max(avg, 1.0))
    boundary_div = _avg_boundary_divergence(chunks)
    return 0.45 * fit_score + 0.30 * stability + 0.25 * boundary_div


def _avg_boundary_divergence(chunks: List[Dict]) -> float:
    if len(chunks) < 2:
        return 0.5
    vals = []
    for i in range(len(chunks) - 1):
        a = set(re.findall(r"\b\w+\b", chunks[i]["text"].lower()))
        b = set(re.findall(r"\b\w+\b", chunks[i + 1]["text"].lower()))
        union = a | b
        vals.append(1.0 - (len(a & b) / len(union) if union else 0.0))
    return float(np.mean(vals)) if vals else 0.5


def recursive_character_split(text: str, n_min: int, n_max: int) -> List[Dict]:
    separators = ["\n\n", "\n", " ", ""]
    raw_chunks = _recursive_split(text, separators, n_min, n_max)
    result: List[Dict] = []
    search_start = 0
    for chunk in raw_chunks:
        if not chunk.strip():
            continue
        idx = text.find(chunk, search_start)
        idx = search_start if idx == -1 else idx
        end = idx + len(chunk)
        result.append({"text": chunk, "start": idx, "end": end, "method": "recursive"})
        search_start = end
    return result


def _recursive_split(text: str, separators: List[str], n_min: int, n_max: int) -> List[str]:
    if not text.strip():
        return []
    sep = separators[0] if separators else ""
    remaining = separators[1:] if len(separators) > 1 else []
    if sep == "":
        words = text.split()
        return [" ".join(words[i: i + n_max]) for i in range(0, len(words), n_max)]

    parts = text.split(sep)
    chunks: List[str] = []
    current_parts: List[str] = []
    for part in parts:
        candidate = " ".join(current_parts + [part]).split()
        if len(candidate) <= n_max:
            current_parts.append(part)
            continue
        if current_parts:
            ctext = sep.join(current_parts).strip()
            wc = len(ctext.split())
            if wc > n_max and remaining:
                chunks.extend(_recursive_split(ctext, remaining, n_min, n_max))
            elif ctext:
                chunks.append(ctext)
        if len(part.split()) > n_max and remaining:
            chunks.extend(_recursive_split(part, remaining, n_min, n_max))
            current_parts = []
        else:
            current_parts = [part]

    if current_parts:
        tail = sep.join(current_parts).strip()
        wc = len(tail.split())
        if chunks and wc < n_min and len((chunks[-1] + " " + tail).split()) <= int(n_max * 1.3):
            chunks[-1] = chunks[-1] + (sep if sep else " ") + tail
        elif wc > n_max and remaining:
            chunks.extend(_recursive_split(tail, remaining, n_min, n_max))
        elif tail:
            chunks.append(tail)
    return [c for c in chunks if c.strip()]


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


def _build_token_offsets(text: str, tokens: List[str]) -> List[int]:
    offsets: List[int] = []
    pos = 0
    for token in tokens:
        idx = text.find(token, pos)
        idx = pos if idx == -1 else idx
        offsets.append(idx)
        pos = idx + len(token)
    return offsets


def structure_based_split(text: str, doc_type: str) -> List[Dict]:
    chunks = _split_code(text) if doc_type == "code" else _split_markup(text)
    return _split_paragraphs(text) if len(chunks) <= 1 else chunks


def _split_markup(text: str) -> List[Dict]:
    # Matches: Markdown headings, HTML headings, LaTeX sectioning, AND
    # legal/formal article/section headings such as:
    #   "Article 1:", "ARTICLE 2 :", "Section 3.", "SECTION IV -", "Chapter 2 —"
    header_re = re.compile(
        r"^("
        r"#{1,6}\s+.+"                                             # Markdown
        r"|<h[1-6][^>]*>.+?</h[1-6]>"                             # HTML
        r"|\\(?:chapter|section|subsection|subsubsection)\{[^}]+\}"  # LaTeX
        r"|(?:ARTICLE|Article|SECTION|Section|CHAPTER|Chapter|TITRE|Titre|PART|Part)"
        r"\s+[\w\d]+\s*[\:\.\-—–].*"                              # Legal (EN/FR)
        r")",
        re.MULTILINE | re.IGNORECASE,
    )
    lines = text.split("\n")
    chunks: List[Dict] = []
    current: List[str] = []
    current_start = 0
    pos = 0
    for line in lines:
        if header_re.match(line) and current:
            chunk_text = "\n".join(current).strip()
            if chunk_text:
                chunks.append({"text": chunk_text, "start": current_start, "end": current_start + len(chunk_text), "method": "structure"})
            current = [line]
            current_start = pos
        else:
            current.append(line)
        pos += len(line) + 1
    if current:
        chunk_text = "\n".join(current).strip()
        if chunk_text:
            chunks.append({"text": chunk_text, "start": current_start, "end": current_start + len(chunk_text), "method": "structure"})
    return chunks


def _split_code(text: str) -> List[Dict]:
    boundary_re = re.compile(
        r"^(async\s+)?(?:def |class |public |private |protected |static |function |const\s+\w+\s*=|let\s+\w+\s*=)",
        re.MULTILINE,
    )
    lines = text.split("\n")
    chunks: List[Dict] = []
    current: List[str] = []
    start = 0
    pos = 0
    for line in lines:
        if boundary_re.match(line) and len("\n".join(current).split()) > 5:
            ctext = "\n".join(current).strip()
            if ctext:
                chunks.append({"text": ctext, "start": start, "end": start + len(ctext), "method": "structure"})
            current = [line]
            start = pos
        else:
            current.append(line)
        pos += len(line) + 1
    if current:
        ctext = "\n".join(current).strip()
        if ctext:
            chunks.append({"text": ctext, "start": start, "end": start + len(ctext), "method": "structure"})
    return chunks


def _split_paragraphs(text: str) -> List[Dict]:
    chunks: List[Dict] = []
    search_pos = 0
    for para in re.split(r"\n{2,}", text):
        p = para.strip()
        if not p:
            continue
        idx = text.find(p, search_pos)
        idx = search_pos if idx == -1 else idx
        end = idx + len(p)
        chunks.append({"text": p, "start": idx, "end": end, "method": "structure"})
        search_pos = end
    return chunks


def semantic_boundary_split(text: str, n_min: int, n_max: int, config: Dict[str, Any]) -> List[Dict]:
    sentences = _split_sentences_with_offsets(text)
    if not sentences:
        return []
    embeddings = np.array([_hash_embedding(s, 96) for s, _, _ in sentences], dtype=np.float32)
    sims = [float(_cosine(embeddings[i], embeddings[i + 1])) for i in range(len(embeddings) - 1)]
    shift_threshold = float(config.get("semantic_shift_threshold", 0.55))
    chunks: List[Dict] = []
    buf: List[Tuple[str, int, int]] = [sentences[0]]
    for i in range(1, len(sentences)):
        candidate_len = len(" ".join(x[0] for x in buf).split())
        sim = sims[i - 1] if i - 1 < len(sims) else 1.0
        should_split = sim < shift_threshold and candidate_len >= n_min
        if should_split or candidate_len >= n_max:
            chunks.append(_build_chunk_from_sentences(buf, "semantic_boundaries"))
            buf = [sentences[i]]
        else:
            buf.append(sentences[i])
    if buf:
        chunks.append(_build_chunk_from_sentences(buf, "semantic_boundaries"))
    return _merge_small_chunks(chunks, n_min, n_max)


def sentence_cluster_split(text: str, n_min: int, n_max: int, config: Dict[str, Any]) -> List[Dict]:
    sentences = _split_sentences_with_offsets(text)
    if not sentences:
        return []
    threshold = float(config.get("sentence_cluster_similarity", 0.68))
    max_cluster = int(config.get("sentence_cluster_max", 8))
    clusters: List[List[Tuple[str, int, int]]] = []
    centroids: List[np.ndarray] = []
    for sent in sentences:
        vec = _hash_embedding(sent[0], 96)
        assigned = False
        for i, centroid in enumerate(centroids):
            if _cosine(vec, centroid) >= threshold and len(clusters[i]) < max_cluster:
                clusters[i].append(sent)
                centroids[i] = (centroid * (len(clusters[i]) - 1) + vec) / len(clusters[i])
                assigned = True
                break
        if not assigned:
            clusters.append([sent])
            centroids.append(vec)

    ordered = sorted(clusters, key=lambda cl: cl[0][1])
    chunks = [_build_chunk_from_sentences(cluster, "sentence_clustering") for cluster in ordered]
    return _merge_small_chunks(chunks, n_min, n_max)


def _split_sentences_with_offsets(text: str) -> List[Tuple[str, int, int]]:
    matches = list(re.finditer(r"[^.!?\n]+(?:[.!?]+|$)", text))
    out: List[Tuple[str, int, int]] = []
    for m in matches:
        sent = m.group(0).strip()
        if sent:
            out.append((sent, m.start(), m.end()))
    return out or [(text.strip(), 0, len(text))]


def _build_chunk_from_sentences(sentences: List[Tuple[str, int, int]], method: str) -> Dict[str, Any]:
    text = " ".join(s[0] for s in sentences).strip()
    return {"text": text, "start": sentences[0][1], "end": sentences[-1][2], "method": method}


def _merge_small_chunks(chunks: List[Dict], n_min: int, n_max: int) -> List[Dict]:
    if not chunks:
        return chunks
    merged = [dict(chunks[0])]
    for curr in chunks[1:]:
        prev = merged[-1]
        curr_wc = len(curr["text"].split())
        if curr_wc < n_min and len((prev["text"] + " " + curr["text"]).split()) <= int(n_max * 1.4):
            prev["text"] = prev["text"] + "\n\n" + curr["text"]
            prev["end"] = curr["end"]
        else:
            merged.append(dict(curr))
    return merged


def _hash_embedding(text: str, dim: int = 96) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"\b\w+\b", text.lower()):
        idx = hash(tok) % dim
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
