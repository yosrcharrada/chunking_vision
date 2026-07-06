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
