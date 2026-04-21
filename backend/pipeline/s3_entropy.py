"""
S3 — Deterministic Entropy Boundary Refinement

This stage evaluates adjacent raw chunks and merges only weak boundaries.
Signals are deterministic: lexical distribution distance, token overlap,
entropy shift, and protected document boundaries such as articles/sections.
"""

import re
from typing import Any, Dict, List, Tuple

import numpy as np


class _LSTMCell:
    def __init__(self, input_dim: int, hidden_dim: int, rng: np.random.RandomState):
        scale = np.sqrt(2.0 / (input_dim + hidden_dim))
        self.Wi = rng.randn(hidden_dim, input_dim) * scale
        self.Ui = rng.randn(hidden_dim, hidden_dim) * scale
        self.bi = np.zeros(hidden_dim)
        self.Wf = rng.randn(hidden_dim, input_dim) * scale
        self.Uf = rng.randn(hidden_dim, hidden_dim) * scale
        self.bf = np.ones(hidden_dim)
        self.Wg = rng.randn(hidden_dim, input_dim) * scale
        self.Ug = rng.randn(hidden_dim, hidden_dim) * scale
        self.bg = np.zeros(hidden_dim)
        self.Wo = rng.randn(hidden_dim, input_dim) * scale
        self.Uo = rng.randn(hidden_dim, hidden_dim) * scale
        self.bo = np.zeros(hidden_dim)
        self.hidden_dim = hidden_dim
        self.h = np.zeros(hidden_dim)
        self.c = np.zeros(hidden_dim)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def reset(self):
        self.h = np.zeros(self.hidden_dim)
        self.c = np.zeros(self.hidden_dim)

    def step(self, x: np.ndarray) -> np.ndarray:
        i = self._sigmoid(self.Wi @ x + self.Ui @ self.h + self.bi)
        f = self._sigmoid(self.Wf @ x + self.Uf @ self.h + self.bf)
        g = np.tanh(self.Wg @ x + self.Ug @ self.h + self.bg)
        o = self._sigmoid(self.Wo @ x + self.Uo @ self.h + self.bo)
        self.c = f * self.c + i * g
        self.h = o * np.tanh(self.c)
        return self.h.copy()


class BiMultiLayerEntropyMemory:
    def __init__(self, input_dim: int = 3, hidden_dim: int = 8, layers: int = 2, seed: int = 42):
        self.layers = max(1, layers)
        self.hidden_dim = hidden_dim
        self.fwd_cells = []
        self.bwd_cells = []
        rng = np.random.RandomState(seed)
        in_dim = input_dim
        for _ in range(self.layers):
            self.fwd_cells.append(_LSTMCell(in_dim, hidden_dim, rng))
            self.bwd_cells.append(_LSTMCell(in_dim, hidden_dim, rng))
            in_dim = hidden_dim
        scale = np.sqrt(2.0 / (2 * hidden_dim + 1))
        self.Wp = rng.randn(1, hidden_dim * 2) * scale

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))

    def score_sequence(self, sequence: List[np.ndarray]) -> List[float]:
        if not sequence:
            return []
        for c in self.fwd_cells + self.bwd_cells:
            c.reset()

        fwd_states: List[np.ndarray] = []
        for x in sequence:
            y = x
            for cell in self.fwd_cells:
                y = cell.step(y)
            fwd_states.append(y)

        bwd_states = [np.zeros(self.hidden_dim) for _ in sequence]
        for idx in range(len(sequence) - 1, -1, -1):
            y = sequence[idx]
            for cell in self.bwd_cells:
                y = cell.step(y)
            bwd_states[idx] = y

        scores: List[float] = []
        for i in range(len(sequence)):
            merged = np.concatenate([fwd_states[i], bwd_states[i]])
            scores.append(float(self._sigmoid(self.Wp @ merged)[0]))
        return scores


def refine_boundaries(chunks: List[Dict], config: Dict[str, Any]) -> List[Dict]:
    if not chunks:
        return chunks
    if len(chunks) == 1:
        one = dict(chunks[0])
        one.update({
            "metric_score": 0.0,
            "jsd_score": 0.0,
            "boundary_signal": 0.0,
            "hidden_state": 0.0,
            "boundary_type": "single",
            "s3_stats": {
                "initial_count": 1,
                "final_count": 1,
                "merged_count": 0,
                "hard_count": 0,
                "soft_count": 0,
                "protected_count": 0,
                "mean_signal": 0.0,
            },
        })
        return [one]

    metric = str(config.get("entropy_metric", "jsd")).lower()
    hybrid_lambda = float(config.get("hybrid_lambda", 0.6))
    mode = str(config.get("threshold_mode", "fixed")).lower()
    tau_low = float(config.get("tau_jsd_low", 0.15))
    tau_high = float(config.get("tau_jsd_high", 0.45))
    n_max = int(config.get("n_max", 500))

    initial_signals: List[float] = []
    for i in range(len(chunks) - 1):
        stats = _boundary_stats(chunks[i]["text"], chunks[i + 1]["text"], metric, hybrid_lambda)
        initial_signals.append(stats["signal"])

    if mode == "percentile" and initial_signals:
        low_p = float(config.get("tau_percentile_low", 25))
        high_p = float(config.get("tau_percentile_high", 75))
        tau_low = float(np.percentile(initial_signals, low_p))
        tau_high = float(np.percentile(initial_signals, high_p))
    if tau_low >= tau_high:
        tau_high = min(1.0, tau_low + 0.08)

    work = [dict(c) for c in chunks]
    out: List[Dict] = []
    signal_history: List[float] = []
    merged_count = 0
    hard_count = 0
    soft_count = 0
    protected_count = 0
    i = 0
    while i < len(work):
        curr = dict(work[i])
        if i < len(work) - 1:
            nxt = work[i + 1]
            stats = _boundary_stats(curr["text"], nxt["text"], metric, hybrid_lambda)
            signal = _local_smoothed_signal(work, i, metric, hybrid_lambda)
            protected = _is_protected_boundary(nxt.get("text", ""))
            curr_words = len(curr.get("text", "").split())
            next_words = len(nxt.get("text", "").split())
            m = stats["metric"]
            j = stats["jsd"]
            curr["metric_score"] = round(m, 4)
            curr["jsd_score"] = round(j, 4)
            curr["entropy_metric"] = metric
            curr["hidden_state"] = round(signal, 4)
            curr["boundary_signal"] = round(signal, 4)
            curr["boundary_features"] = {
                "selected_metric": round(stats["metric"], 4),
                "jsd": round(stats["jsd"], 4),
                "hellinger": round(stats["hellinger"], 4),
                "overlap": round(stats["overlap"], 4),
                "entropy_delta": round(stats["entropy_delta"], 4),
            }
            signal_history.append(signal)
            can_merge = (
                signal < tau_low
                and not protected
                and (curr_words + next_words) <= max(n_max * 1.35, n_max + 80)
            )
            if can_merge:
                curr["text"] = curr["text"] + "\n\n" + nxt["text"]
                curr["end"] = nxt.get("end", curr.get("end", 0))
                curr["boundary_type"] = "merged"
                curr["merge_reason"] = "low_entropy_boundary"
                work = work[: i + 1] + work[i + 2:]
                work[i] = curr
                merged_count += 1
                continue
            if protected:
                curr["boundary_type"] = "hard"
                curr["merge_reason"] = "protected_structure_boundary"
                protected_count += 1
                hard_count += 1
            elif signal > tau_high:
                curr["boundary_type"] = "hard"
                curr["merge_reason"] = "high_entropy_shift"
                hard_count += 1
            else:
                curr["boundary_type"] = "soft"
                curr["merge_reason"] = "moderate_entropy_shift"
                soft_count += 1
        else:
            curr["metric_score"] = 0.0
            curr["jsd_score"] = 0.0
            curr["entropy_metric"] = metric
            curr["hidden_state"] = 0.0
            curr["boundary_signal"] = 0.0
            curr["boundary_type"] = "end"
        out.append(curr)
        i += 1

    if out:
        out[-1]["thresholds"] = {"low": round(tau_low, 4), "high": round(tau_high, 4), "mode": mode}
        out[-1]["s3_stats"] = {
            "initial_count": len(chunks),
            "final_count": len(out),
            "merged_count": merged_count,
            "hard_count": hard_count,
            "soft_count": soft_count,
            "protected_count": protected_count,
            "mean_signal": round(float(np.mean(signal_history)), 4) if signal_history else 0.0,
            "merge_ratio": round(merged_count / max(1, len(chunks) - 1), 4),
        }
    return out


def get_jsd_series(chunks: List[Dict]) -> List[float]:
    return [c.get("metric_score", c.get("jsd_score", 0.0)) for c in chunks]


def _boundary_stats(text_a: str, text_b: str, metric: str, hybrid_lambda: float) -> Dict[str, float]:
    jsd = _compute_jsd(text_a, text_b)
    hell = _compute_hellinger(text_a, text_b)
    selected = _select_metric(metric, jsd, hell, hybrid_lambda)
    overlap = _compute_token_overlap(text_a, text_b)
    entropy_delta = abs(_compute_shannon_entropy(text_a) - _compute_shannon_entropy(text_b))
    signal = float(np.clip(0.70 * selected + 0.20 * (1.0 - overlap) + 0.10 * entropy_delta, 0.0, 1.0))
    return {
        "metric": selected,
        "jsd": jsd,
        "hellinger": hell,
        "overlap": overlap,
        "entropy_delta": entropy_delta,
        "signal": signal,
    }


def _local_smoothed_signal(work: List[Dict], idx: int, metric: str, hybrid_lambda: float) -> float:
    vals: List[float] = []
    weights: List[float] = []
    for j, weight in ((idx - 1, 0.25), (idx, 0.50), (idx + 1, 0.25)):
        if 0 <= j < len(work) - 1:
            vals.append(_boundary_stats(work[j]["text"], work[j + 1]["text"], metric, hybrid_lambda)["signal"])
            weights.append(weight)
    if not vals:
        return 0.0
    total = sum(weights) or 1.0
    return float(np.clip(sum(v * w for v, w in zip(vals, weights)) / total, 0.0, 1.0))


def _is_protected_boundary(text: str) -> bool:
    first = ""
    for line in text.splitlines():
        if line.strip():
            first = line.strip()
            break
    if not first:
        return False
    return bool(
        re.match(
            r"^(?:"
            r"#{1,6}\s+"
            r"|(?:article|art\.?)\s+\d+(?:\s*(?:er|e|ème|bis|ter|quater))?\b"
            r"|(?:titre|chapitre|section|sous-section|paragraphe)\s+(?:[ivxlcdm]+|\d+|premier|première)\b"
            r"|\d+(?:\.\d+){1,3}\s+\S+"
            r")",
            first,
            re.IGNORECASE,
        )
    )


def _select_metric(metric: str, jsd: float, hell: float, lam: float) -> float:
    if metric == "hellinger":
        return hell
    if metric == "hybrid":
        return float(np.clip(lam * hell + (1.0 - lam) * jsd, 0.0, 1.0))
    return jsd


def _compute_shannon_entropy(text: str) -> float:
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    count: Dict[str, int] = {}
    for t in tokens:
        count[t] = count.get(t, 0) + 1
    probs = np.array([v / len(tokens) for v in count.values()], dtype=np.float32)
    H = -float(np.sum(probs * np.log2(np.clip(probs, 1e-12, 1.0))))
    return float(H / (np.log2(max(2, len(tokens)))))


def _compute_token_overlap(text_a: str, text_b: str) -> float:
    va = set(_tokenize(text_a))
    vb = set(_tokenize(text_b))
    union = va | vb
    return float(len(va & vb) / len(union)) if union else 0.0


def _compute_jsd(text1: str, text2: str) -> float:
    p, q = _distribution_pair(text1, text2)
    if p is None or q is None:
        return 0.5
    m = (p + q) / 2.0
    return float(np.clip(0.5 * _kl(p, m) + 0.5 * _kl(q, m), 0.0, 1.0))


def _compute_hellinger(text1: str, text2: str) -> float:
    p, q = _distribution_pair(text1, text2)
    if p is None or q is None:
        return 0.5
    return float(np.clip(np.linalg.norm(np.sqrt(p) - np.sqrt(q)) / np.sqrt(2), 0.0, 1.0))


def _distribution_pair(text1: str, text2: str) -> Tuple[np.ndarray, np.ndarray]:
    tokens1 = _tokenize(text1)
    tokens2 = _tokenize(text2)
    if not tokens1 or not tokens2:
        return None, None  # type: ignore[return-value]
    vocab = list(set(tokens1) | set(tokens2))
    c1: Dict[str, int] = {}
    c2: Dict[str, int] = {}
    for t in tokens1:
        c1[t] = c1.get(t, 0) + 1
    for t in tokens2:
        c2[t] = c2.get(t, 0) + 1
    p = np.array([c1.get(w, 0) / len(tokens1) for w in vocab], dtype=np.float64)
    q = np.array([c2.get(w, 0) / len(tokens2) for w in vocab], dtype=np.float64)
    return p, q


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / np.clip(q[mask], 1e-12, 1.0))))


def _tokenize(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())
