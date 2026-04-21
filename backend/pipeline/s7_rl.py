"""
S7 — Advanced RL Calibration
Implements DQN-style updates with replay memory, hybrid action magnitudes,
multi-objective rewards, and warm-start meta-learning across documents.
"""

import copy
import json
import os
import re
from collections import deque
from typing import Any, Dict, List, Tuple

import numpy as np

from .s2_chunkers import run_all_chunkers, select_best_strategy
from .s3_entropy import refine_boundaries
from .s4_boundary import filter_boundaries
from .s5_graph import enrich_graph
from .s6_embedding import embed_chunks

_TARGET_WORDS_PER_CHUNK = 300
_RL_HISTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "rl_history.json")


class DQNAgent:
    ACTIONS = ["tau_jsd_low", "tau_jsd_high", "n_max", "tau_sem", "hybrid_lambda", "merge_weight"]
    MAGNITUDES = [0.5, 1.0, 1.5]

    def __init__(self, state_dim: int = 8, hidden_dim: int = 24, lr: float = 0.02, gamma: float = 0.9, epsilon: float = 0.25):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.RandomState(42)
        self.action_size = len(self.ACTIONS) * len(self.MAGNITUDES)
        self.W1 = self.rng.randn(hidden_dim, state_dim).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)
        self.W2 = self.rng.randn(self.action_size, hidden_dim).astype(np.float32) * 0.1
        self.b2 = np.zeros(self.action_size, dtype=np.float32)
        self.replay = deque(maxlen=500)

    def _forward(self, state: np.ndarray) -> np.ndarray:
        h = np.tanh(self.W1 @ state + self.b1)
        return self.W2 @ h + self.b2

    def select_action(self, state: np.ndarray) -> int:
        if self.rng.rand() < self.epsilon:
            return int(self.rng.randint(0, self.action_size))
        q = self._forward(state)
        return int(np.argmax(q))

    def remember(self, transition: Tuple[np.ndarray, int, float, np.ndarray]) -> None:
        self.replay.append(transition)

    def learn(self, batch_size: int = 16) -> None:
        if len(self.replay) < 8:
            return
        idxs = self.rng.choice(len(self.replay), size=min(batch_size, len(self.replay)), replace=False)
        for i in idxs:
            state, action_idx, reward, next_state = self.replay[i]
            q = self._forward(state)
            target = q.copy()
            next_q = self._forward(next_state)
            target[action_idx] = reward + self.gamma * float(np.max(next_q))

            h = np.tanh(self.W1 @ state + self.b1)
            pred = self.W2 @ h + self.b2
            err = pred - target
            grad_W2 = np.outer(err, h)
            grad_b2 = err
            dh = (1 - h ** 2) * (self.W2.T @ err)
            grad_W1 = np.outer(dh, state)
            grad_b1 = dh

            self.W2 -= self.lr * grad_W2
            self.b2 -= self.lr * grad_b2
            self.W1 -= self.lr * grad_W1
            self.b1 -= self.lr * grad_b1

    def apply_action(self, action_idx: int, config: Dict[str, Any]) -> Dict[str, Any]:
        cfg = copy.deepcopy(config)
        action_id = action_idx // len(self.MAGNITUDES)
        mag_id = action_idx % len(self.MAGNITUDES)
        key = self.ACTIONS[action_id]
        scale = self.MAGNITUDES[mag_id]
        deltas = {
            "tau_jsd_low": 0.02 * scale,
            "tau_jsd_high": 0.03 * scale,
            "n_max": 20 * scale,
            "tau_sem": 0.02 * scale,
            "hybrid_lambda": 0.05 * scale,
            "merge_weight": 0.08 * scale,
        }
        signs = [-1, 1]
        sign = signs[action_idx % 2]
        cfg[key] = (cfg.get(key) or 0.0) + sign * deltas[key]

        cfg["tau_jsd_low"] = float(np.clip(cfg.get("tau_jsd_low") or 0.15, 0.05, 0.45))
        cfg["tau_jsd_high"] = float(np.clip(cfg.get("tau_jsd_high") or 0.45, 0.2, 0.8))
        if cfg["tau_jsd_low"] >= cfg["tau_jsd_high"]:
            cfg["tau_jsd_high"] = cfg["tau_jsd_low"] + 0.1
        cfg["n_max"] = int(np.clip(cfg.get("n_max") or 500, 200, 900))
        cfg["tau_sem"] = float(np.clip(cfg.get("tau_sem") or 0.75, 0.4, 0.95))
        cfg["hybrid_lambda"] = float(np.clip(cfg.get("hybrid_lambda") or 0.6, 0.1, 0.9))
        cfg["boundary_merge_weight"] = float(np.clip(cfg.get("merge_weight") or cfg.get("boundary_merge_weight") or 1.0, 0.5, 1.5))
        cfg["entropy_metric"] = cfg.get("entropy_metric") or "hybrid"
        return cfg


def run_rl_loop(
    text: str,
    doc_profile: Dict[str, Any],
    initial_chunks: List[Dict],
    config: Dict[str, Any],
) -> Tuple[List[Dict], List[float], Dict[str, Any]]:
    max_iters = int(config.get("max_iterations", 10))
    model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
    doc_type = doc_profile.get("type", "prose")
    domain = doc_profile.get("domain", "general")
    objective_weights = _objective_weights(config)
    history = _load_history()

    warm_cfg = _warm_start_config(config, history, domain)
    probe_queries = _generate_probes(text, n=6)
    agent = DQNAgent(
        state_dim=8,
        hidden_dim=int(warm_cfg.get("dqn_hidden_dim", 24)),
        lr=float(warm_cfg.get("dqn_lr", 0.02)),
        gamma=float(warm_cfg.get("dqn_gamma", 0.9)),
        epsilon=float(warm_cfg.get("dqn_epsilon", 0.25)),
    )

    def _state_vec(chs: List[Dict], reward_components: Dict[str, float]) -> np.ndarray:
        return np.array([
            float(np.mean([c.get("metric_score", c.get("jsd_score", 0.4)) for c in chs])) if chs else 0.4,
            float(np.mean([c.get("boundary_score", 0.5) for c in chs])) if chs else 0.5,
            float(np.mean([c.get("icc", 0.5) for c in chs])) if chs else 0.5,
            reward_components.get("quality", 0.0),
            reward_components.get("coverage", 0.0),
            reward_components.get("consistency", 0.0),
            reward_components.get("efficiency", 0.0),
            float(np.clip(len(chs) / max(_target_count(chs), 1.0), 0.0, 2.0)) if chs else 1.0,
        ], dtype=np.float32)

    current_cfg = copy.deepcopy(warm_cfg)
    best_chunks = initial_chunks
    best_components = _compute_reward_components(initial_chunks, probe_queries, objective_weights)
    best_reward = best_components["total"]
    reward_history = [round(best_reward, 4)]
    reward_breakdown = [best_components]
    current_components = best_components
    current_chunks = initial_chunks

    for _ in range(max_iters):
        state_vec = _state_vec(current_chunks, current_components)
        action = agent.select_action(state_vec)
        trial_cfg = agent.apply_action(action, current_cfg)
        try:
            all_chunks = run_all_chunkers(text, doc_type, trial_cfg)
            trial_chunks = select_best_strategy(all_chunks, doc_type, trial_cfg)
            if not trial_chunks:
                reward_history.append(round(best_reward, 4))
                reward_breakdown.append(best_components)
                continue
            trial_chunks = refine_boundaries(trial_chunks, trial_cfg)
            trial_chunks = filter_boundaries(trial_chunks, doc_type, [], trial_cfg)
            trial_chunks = enrich_graph(trial_chunks, [], trial_cfg)
            trial_chunks, _ = embed_chunks(trial_chunks, text, doc_profile, model_name, trial_cfg)
        except Exception:
            reward_history.append(round(best_reward, 4))
            reward_breakdown.append(best_components)
            continue

        trial_components = _compute_reward_components(trial_chunks, probe_queries, objective_weights)
        reward = trial_components["total"]
        next_state = _state_vec(trial_chunks, trial_components)
        agent.remember((state_vec, action, reward, next_state))
        agent.learn()

        if reward > best_reward:
            best_reward = reward
            best_chunks = trial_chunks
            best_components = trial_components
            current_cfg = trial_cfg
            current_chunks = trial_chunks
            current_components = trial_components
        reward_history.append(round(reward, 4))
        reward_breakdown.append(trial_components)

    final_cfg = copy.deepcopy(current_cfg)
    final_cfg["reward_breakdown"] = best_components
    final_cfg["reward_history_breakdown"] = reward_breakdown
    final_cfg["dqn_action_space"] = {"discrete": len(agent.ACTIONS), "continuous_magnitudes": agent.MAGNITUDES}
    final_cfg["replay_buffer_size"] = len(agent.replay)
    _save_history(domain, final_cfg, best_components)
    return best_chunks, reward_history, final_cfg


def _compute_reward_components(chunks: List[Dict], probes: List[str], weights: Dict[str, float]) -> Dict[str, float]:
    if not chunks:
        return {"quality": 0.0, "coverage": 0.0, "consistency": 0.0, "efficiency": 0.0, "total": -1.0}
    quality = float(np.mean([1.0 - c.get("boundary_score", 0.5) for c in chunks]))
    coverage = _recall_proxy(chunks, probes)
    size = np.array([len(c.get("text", "").split()) for c in chunks], dtype=np.float32)
    consistency = float(1.0 - min(1.0, np.std(size) / max(np.mean(size), 1.0)))
    target = _target_count(chunks)
    efficiency = float(1.0 - min(1.0, abs(len(chunks) - target) / max(target, 1.0)))
    total = (
        weights["quality"] * quality
        + weights["coverage"] * coverage
        + weights["consistency"] * consistency
        + weights["efficiency"] * efficiency
    )
    return {
        "quality": round(quality, 4),
        "coverage": round(coverage, 4),
        "consistency": round(consistency, 4),
        "efficiency": round(efficiency, 4),
        "total": round(float(total), 4),
    }


def _objective_weights(config: Dict[str, Any]) -> Dict[str, float]:
    defaults = {"quality": 0.35, "coverage": 0.30, "consistency": 0.20, "efficiency": 0.15}
    incoming = config.get("reward_objectives", {})
    if not isinstance(incoming, dict):
        incoming = {}
    raw = {k: float(incoming.get(k, v)) for k, v in defaults.items()}
    s = sum(raw.values()) or 1.0
    return {k: v / s for k, v in raw.items()}


def _recall_proxy(chunks: List[Dict], probes: List[str]) -> float:
    if not probes:
        return 0.5
    hits = 0
    for q in probes:
        terms = set(re.findall(r"\b\w{4,}\b", q.lower()))
        if not terms:
            hits += 1
            continue
        found = any(bool(terms & set(re.findall(r"\b\w+\b", c.get("text", "").lower()))) for c in chunks)
        if found:
            hits += 1
    return hits / len(probes)

def _target_count(chunks: List[Dict]) -> float:
    total_words = sum(len(c.get("text", "").split()) for c in chunks)
    return max(3.0, total_words / _TARGET_WORDS_PER_CHUNK)

def _generate_probes(text: str, n: int = 5) -> List[str]:
    probes: List[str] = []
    for m in re.finditer(r"^#{1,3}\s+(.+)$", text, re.MULTILINE):
        probes.append(m.group(1).strip())
        if len(probes) >= n:
            return probes
    for para in re.split(r"\n{2,}", text):
        p = para.strip()
        if not p:
            continue
        sent = re.split(r"(?<=[.!?])\s+", p)
        if sent and len(sent[0].split()) >= 5:
            probes.append(sent[0].strip())
        if len(probes) >= n:
            break
    return probes[:n]

def _load_history() -> Dict[str, Any]:
    if not os.path.exists(_RL_HISTORY_PATH):
        return {}
    try:
        with open(_RL_HISTORY_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}

def _warm_start_config(config: Dict[str, Any], history: Dict[str, Any], domain: str) -> Dict[str, Any]:
    out = copy.deepcopy(config)
    record = history.get(domain, {})
    for k in ("tau_jsd_low", "tau_jsd_high", "n_max", "tau_sem", "hybrid_lambda", "merge_weight"):
        v = record.get(k)
        if v is not None and k not in out:
            out[k] = v
    return out


def _save_history(domain: str, config: Dict[str, Any], reward_components: Dict[str, float]) -> None:
    history = _load_history()
    record: Dict[str, Any] = {"last_reward_components": reward_components}
    for k in ("tau_jsd_low", "tau_jsd_high", "n_max", "tau_sem", "hybrid_lambda", "merge_weight"):
        v = config.get(k)
        if v is not None:
            record[k] = v
    history[domain] = record
    try:
        with open(_RL_HISTORY_PATH, "w", encoding="utf-8") as fh:
            json.dump(history, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass
