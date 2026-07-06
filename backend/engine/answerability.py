"""LLM-judged answerability — the honest evaluation.

The cosine metrics in metrics.py can be fooled: a chunk may look word-similar to
a question without actually containing the answer.  The only trustworthy test of
"did we retrieve a good chunk?" is to give the retrieved text to an LLM and ask:
*can you answer this question from it, and is the answer correct?*

This module does exactly that, judging the top-retrieved chunks for each query.
It is OPT-IN (it costs LLM calls) and degrades gracefully with no API key.
Results are cached per (question, context) so re-runs are cheap and identical
chunks across methods aren't re-judged.
"""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

_SYSTEM = (
    "You are a strict retrieval evaluator. You are given a QUESTION, the GROUND "
    "TRUTH answer, and a CONTEXT passage retrieved by a system. Decide whether the "
    "CONTEXT alone is sufficient to answer the question, and whether the answer it "
    "supports matches the ground truth. Be strict: if the key fact is missing from "
    "the context, it is not answerable. Reply ONLY as JSON: "
    '{"answerable": true|false, "correct": true|false}.'
)

_CACHE: dict = {}


def available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


def _key(question: str, gt: str, context: str, model: str) -> str:
    h = hashlib.sha256(f"{model} {question} {gt} {context}".encode()).hexdigest()
    return h


def _judge_one(client, model: str, question: str, gt: str, context: str) -> dict:
    ck = _key(question, gt, context, model)
    if ck in _CACHE:
        return _CACHE[ck]
    user = (
        f"QUESTION:\n{question}\n\nGROUND TRUTH:\n{gt or '(none provided)'}\n\n"
        f"CONTEXT:\n{context[:6000]}\n\nReturn the JSON verdict."
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": user}],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    try:
        data = json.loads(resp.choices[0].message.content or "{}")
        out = {"answerable": bool(data.get("answerable")),
               "correct": bool(data.get("correct"))}
    except Exception:
        out = {"answerable": False, "correct": False}
    _CACHE[ck] = out
    return out


def judge_many(
    tasks: List[Tuple[str, str, str]],
    model: Optional[str] = None,
    max_workers: int = 8,
) -> List[dict]:
    """Judge many (question, ground_truth, context) triples IN PARALLEL.

    Identical triples are judged only once (dedup by cache key), and the unique
    calls run concurrently — so judging a whole benchmark (dozens of calls) takes
    a few seconds instead of ~a minute, which keeps it under the proxy timeout.
    """
    if not tasks or not available():
        return [{"answerable": False, "correct": False} for _ in tasks]
    from .openai_client import get_client, chat_model

    client = get_client()                  # OpenAI or Azure/EY, per environment
    model = model or chat_model("gpt-4o-mini")

    keys = [_key(q, gt, ctx, model) for (q, gt, ctx) in tasks]
    uniq = {}
    for k, t in zip(keys, tasks):
        uniq.setdefault(k, t)

    def work(item):
        k, (q, gt, ctx) = item
        try:
            return k, _judge_one(client, model, q, gt, ctx)
        except Exception:
            return k, {"answerable": False, "correct": False}

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for k, v in ex.map(work, list(uniq.items())):
            results[k] = v
    return [results[k] for k in keys]


def judge_run(
    queries: List[str],
    ground_truths: List[str],
    contexts: List[str],
    model: Optional[str] = None,
) -> dict:
    """Judge one chunking run.  `contexts[i]` is the concatenated text of the
    chunks retrieved for query i.  Returns {answerable, correct, n} fractions."""
    if not queries or not available():
        return {"answerable": 0.0, "correct": 0.0, "n": 0}
    tasks = [
        (q, ground_truths[i] if i < len(ground_truths) else "",
         contexts[i] if i < len(contexts) else "")
        for i, q in enumerate(queries) if q.strip()
    ]
    verdicts = judge_many(tasks, model)
    if not verdicts:
        return {"answerable": 0.0, "correct": 0.0, "n": 0}
    ans = sum(1 for v in verdicts if v["answerable"])
    corr = sum(1 for v in verdicts if v["correct"])
    n = len(verdicts)
    return {"answerable": round(ans / n, 4), "correct": round(corr / n, 4), "n": n}
