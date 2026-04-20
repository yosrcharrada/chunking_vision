"""
S3 — Entropy Boundary Refinement
Supports selectable metrics (JSD, Hellinger, Hybrid), bidirectional multi-layer
LSTM smoothing, and percentile thresholding.
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
        one.update({"metric_score": 0.0, "jsd_score": 0.0, "hidden_state": 0.0, "boundary_type": "single"})
        return [one]

    metric = str(config.get("entropy_metric", "jsd")).lower()
    hybrid_lambda = float(config.get("hybrid_lambda", 0.6))
    mode = str(config.get("threshold_mode", "fixed")).lower()
    tau_low = float(config.get("tau_jsd_low", 0.15))
    tau_high = float(config.get("tau_jsd_high", 0.45))

    features: List[np.ndarray] = []
    metric_scores: List[float] = []
    jsd_scores: List[float] = []
    for i in range(len(chunks) - 1):
        a, b = chunks[i]["text"], chunks[i + 1]["text"]
        jsd = _compute_jsd(a, b)
        hell = _compute_hellinger(a, b)
        score = _select_metric(metric, jsd, hell, hybrid_lambda)
        overlap = _compute_token_overlap(a, b)
        entropy = _compute_shannon_entropy(a)
        features.append(np.array([score, entropy, overlap], dtype=np.float32))
        metric_scores.append(score)
        jsd_scores.append(jsd)

    if mode == "percentile" and metric_scores:
        low_p = float(config.get("tau_percentile_low", 25))
        high_p = float(config.get("tau_percentile_high", 75))
        tau_low = float(np.percentile(metric_scores, low_p))
        tau_high = float(np.percentile(metric_scores, high_p))

    memory = BiMultiLayerEntropyMemory(
        input_dim=3,
        hidden_dim=int(config.get("entropy_hidden_dim", 8)),
        layers=int(config.get("entropy_layers", 2)),
        seed=42,
    )
    lstm_scores = memory.score_sequence(features)

    work = [dict(c) for c in chunks]
    out: List[Dict] = []
    i = 0
    while i < len(work):
        curr = dict(work[i])
        if i < len(work) - 1:
            m = metric_scores[i] if i < len(metric_scores) else 0.0
            j = jsd_scores[i] if i < len(jsd_scores) else m
            l = lstm_scores[i] if i < len(lstm_scores) else 0.0
            signal = 0.55 * m + 0.45 * l
            curr["metric_score"] = round(m, 4)
            curr["jsd_score"] = round(j, 4)
            curr["entropy_metric"] = metric
            curr["hidden_state"] = round(l, 4)
            if signal < tau_low:
                nxt = work[i + 1]
                curr["text"] = curr["text"] + "\n\n" + nxt["text"]
                curr["end"] = nxt.get("end", curr.get("end", 0))
                curr["boundary_type"] = "merged"
                work = work[: i + 1] + work[i + 2:]
                work[i] = curr
                continue
            curr["boundary_type"] = "hard" if signal > tau_high else "soft"
        else:
            curr["metric_score"] = 0.0
            curr["jsd_score"] = 0.0
            curr["entropy_metric"] = metric
            curr["hidden_state"] = 0.0
            curr["boundary_type"] = "end"
        out.append(curr)
        i += 1

    if out:
        out[-1]["thresholds"] = {"low": round(tau_low, 4), "high": round(tau_high, 4), "mode": mode}
    return out


def get_jsd_series(chunks: List[Dict]) -> List[float]:
    return [c.get("metric_score", c.get("jsd_score", 0.0)) for c in chunks]


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
