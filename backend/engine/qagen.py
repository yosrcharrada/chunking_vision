"""Auto-generate evaluation Q&A pairs from a document using OpenAI.

The platform needs (query, ground-truth) pairs to compute the retrieval metrics
in Table I.  Instead of asking the user to write them, we ask an LLM to read the
document and produce N diverse questions, each with a concise answer grounded in
the text.  These become the evaluation set for the q-sweep benchmark.

The API key is read from the environment (loaded from .env at startup) and never
logged.  If no key is configured, callers should fall back gracefully.
"""

from __future__ import annotations

import json
import os
import re
from typing import List, Optional

import numpy as np

_SYSTEM = (
    "You are an expert evaluation-set author for retrieval systems. "
    "You read a document and write questions that a real user might ask whose "
    "answer is fully contained in the document. Questions must be diverse and "
    "cover different parts/sections of the document. Each answer must be concise "
    "(1-3 sentences), factual, and grounded strictly in the document. "
    "CRITICAL: make the questions HARD for a retriever — real users do not quote "
    "the document. PARAPHRASE: phrase each question in your own words, using "
    "synonyms and different sentence structure than the source passage (never "
    "copy its wording). Include several questions about specific details buried "
    "in the middle of sections (figures, dates, conditions, exceptions), and at "
    "least a quarter multi-hop questions whose answer combines facts from two "
    "different parts of the document. Mix difficulty: some direct, some hard. "
    "Write BOTH the questions and the answers in the SAME language as the "
    "document itself (e.g. French questions for a French document); never "
    "translate to English. Answers should be faithful but also paraphrased, not "
    "verbatim quotes."
)


def openai_configured() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _truncate(text: str, max_chars: int = 48000) -> str:
    if len(text) <= max_chars:
        return text
    head = text[: int(max_chars * 0.7)]
    tail = text[-int(max_chars * 0.3):]
    return head + "\n\n[... document truncated for question generation ...]\n\n" + tail


def generate_qa(text: str, n: int = 20, model: Optional[str] = None) -> List[dict]:
    """Return a list of {"query","ground_truth"} dicts. Raises on hard failure."""
    if not openai_configured():
        raise RuntimeError("OPENAI_API_KEY not configured")

    from .openai_client import get_client, chat_model

    client = get_client()                  # OpenAI or Azure/EY, per environment
    model = model or chat_model("gpt-4o-mini")

    user = (
        f"Document:\n\"\"\"\n{_truncate(text)}\n\"\"\"\n\n"
        f"Write exactly {n} question-answer pairs evaluating comprehension of this "
        f"document. Return ONLY JSON of the form:\n"
        f'{{"pairs": [{{"query": "...", "ground_truth": "..."}}]}}'
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": user},
        ],
        temperature=0.0,  # deterministic eval set (part of the determinism contract)
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content or "{}"
    pairs = _parse_pairs(content, n)
    if not pairs:
        raise RuntimeError("LLM returned no usable Q&A pairs")
    return pairs


def _parse_pairs(content: str, n: int) -> List[dict]:
    data = None
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", content, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None
    if not isinstance(data, dict):
        return []
    raw = data.get("pairs") or data.get("qa") or data.get("questions") or []
    out = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        q = (item.get("query") or item.get("question") or "").strip()
        a = (item.get("ground_truth") or item.get("answer") or "").strip()
        if q:
            out.append({"query": q, "ground_truth": a})
    return out[:n]


# ─────────────────────────────────────────────────────────────────────────────
# OFFLINE, KEY-FREE evaluation-set generation (document-derived pseudo-queries)
# ─────────────────────────────────────────────────────────────────────────────
# When there is no working OpenAI key we still need (query, ground_truth) pairs
# so the Table-I retrieval metrics (Precision/Recall/MRR/NDCG/QCS) are non-zero
# and COMPARABLE across strategies.  We derive them from the document itself —
# never a hardcoded list, so it works for ANY document in ANY language:
#
#   1. Split the document into sentences.
#   2. TF-IDF over those sentences → each sentence's salient content terms.
#   3. Pick the most salient sentence in each of N positional bins (coverage of
#      the whole document, not just the top).
#   4. query        = that sentence's top content terms, in original order,
#                     with the boilerplate/function words (low TF-IDF) dropped —
#                     so the query is lexically DIFFERENT from the passage and
#                     the retriever must actually locate it (not string-match).
#      ground_truth = the full source sentence.  The scorer marks a chunk
#                     relevant when it is similar to this ground_truth, i.e. the
#                     chunk the sentence came from — exactly the passage a good
#                     chunking should keep findable.
#
# This is the classic self-retrieval / Inverse-Cloze evaluation.  Absolute
# scores read a little high vs LLM-written multi-hop questions, but the RELATIVE
# ranking between methodologies (q-log vs Tsallis, q-cosine vs cosine) — which
# is what "best methodology" needs — is preserved.

_SENT_SPLIT = re.compile(r"(?<=[\.\!\?؟。])\s+|\n{2,}")


def _split_sentences(text: str, min_chars: int = 40, max_chars: int = 400) -> List[str]:
    out: List[str] = []
    for raw in _SENT_SPLIT.split(text or ""):
        s = " ".join(raw.split())            # collapse whitespace/newlines
        if min_chars <= len(s) <= max_chars:
            out.append(s)
    return out


def generate_qa_offline(text: str, n: int = 12) -> List[dict]:
    """Key-free (query, ground_truth) pairs derived from the document itself.

    Returns [] only for documents too small to yield distinct sentences (the
    caller then falls back to the label-free quality fitness)."""
    sents = _split_sentences(text)
    if len(sents) < 2:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception:
        return []

    vec = TfidfVectorizer(lowercase=True, token_pattern=r"(?u)\b\w[\w\-]+\b",
                          max_df=0.6, min_df=1)
    try:
        X = vec.fit_transform(sents)         # (n_sents, n_terms), L2-normalised rows
    except ValueError:
        return []
    terms = vec.get_feature_names_out()
    salience = X.sum(axis=1).A1              # per-sentence total TF-IDF mass

    # Positional bins → coverage of the WHOLE document, picking the single most
    # salient sentence in each bin (dedup by index).
    n = max(1, min(int(n), len(sents)))
    bins = np.array_split(np.arange(len(sents)), n)
    chosen: List[int] = []
    for b in bins:
        if b.size:
            chosen.append(int(b[np.argmax(salience[b])]))
    chosen = sorted(dict.fromkeys(chosen))   # unique, in document order

    pairs: List[dict] = []
    seen_q = set()
    for i in chosen:
        row = X.getrow(i)
        if row.nnz == 0:
            continue
        # top content terms of this sentence by TF-IDF weight
        order = np.argsort(row.data)[::-1]
        top_terms = [terms[row.indices[j]] for j in order[:8]]
        # keep them in the order they appear in the sentence for a natural query
        low = sents[i].lower()
        kept = sorted(set(top_terms), key=lambda t: (low.find(t) if t in low else 1e9))
        query = " ".join([t for t in kept if (low.find(t) >= 0)][:6]).strip()
        if len(query) < 3 or query in seen_q:
            continue
        seen_q.add(query)
        pairs.append({"query": query, "ground_truth": sents[i]})
    return pairs
