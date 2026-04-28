#!/usr/bin/env python3
"""Rewrite s7_rl.py with stable RL implementation."""

content = '''"""
S7 — Advanced RL Calibration (v3 — STABLE with Monotonic Improvement Guarantee)

CRITICAL FIXES:
✓ Best-State Tracking: Maintains best_state, best_reward, best_chunks
✓ Safe Rollback: Enforces rollback when reward degrades
✓ Replay Buffer Purification: Only stores improving transitions
✓ Exploration Decay: Epsilon decays + immediate reduction on degradation
✓ Early Stopping: Stops after K steps with no improvement
✓ Episodic Loop: 1 document = 1 episode with full reset

CORE GUARANTEE:
  "Chunk quality NEVER degrades within an episode.
   Any action worsening performance is IMMEDIATELY reverted."
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
_EARLY_STOPPING_PATIENCE = 3


class DQNAgent:
    """Deep Q-Network agent with stable learning and improving transitions only."""

    ACTIONS = [
        "tau_jsd_low",
        "tau_jsd_high",
        "n_max",
        "tau_sem",
        "tau_percentile_low",
        "tau_percentile_high",
    ]

    MAGNITUDES = [0.5, 1.0, 1.5]

    def __init__(
        self,
        state_dim: int = 11,
        hidden_dim: int = 24,
        lr: float = 0.02,
        gamma: float = 0.9,
        epsilon: float = 0.25,
    ):
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = 0.05

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

    def remember_improving(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray,
        is_improving: bool
    ) -> None:
        """Store transition ONLY if improving (purified replay buffer)."""
        if is_improving:
            self.replay.append((state, action, reward, next_state))

    def decay_exploration(self, reward_improved: bool) -> None:
        """Decay epsilon based on reward improvement."""
        if not reward_improved:
            self.epsilon *= 0.85
        else:
            self.epsilon *= 0.98
        self.epsilon = max(self.epsilon, self.epsilon_min)

    def learn(self, batch_size: int = 16) -> None:
        if len(self.replay) < 8:
            return
        idxs = self.rng.choice(
            len(self.replay),
            size=min(batch_size, len(self.replay)),
            replace=False,
        )
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
            "tau_percentile_low": 2.0 * scale,
            "tau_percentile_high": 2.0 * scale,
        }
        sign = -1 if (action_idx % 2 == 0) else 1
        cfg[key] = (cfg.get(key) or 0.0) + sign * deltas[key]
        cfg["tau_jsd_low"] = float(np.clip(cfg.get("tau_jsd_low", 0.15), 0.05, 0.45))
        cfg["tau_jsd_high"] = float(np.clip(cfg.get("tau_jsd_high", 0.45), 0.20, 0.80))
        if cfg["tau_jsd_low"] >= cfg["tau_jsd_high"]:
            cfg["tau_jsd_high"] = cfg["tau_jsd_low"] + 0.10
        cfg["n_max"] = int(np.clip(cfg.get("n_max", 500), 200, 900))
        cfg["tau_sem"] = float(np.clip(cfg.get("tau_sem", 0.75), 0.40, 0.95))
        cfg["tau_percentile_low"] = float(np.clip(cfg.get("tau_percentile_low", 25), 5, 45))
        cfg["tau_percentile_high"] = float(np.clip(cfg.get("tau_percentile_high", 75), 55, 95))
        cfg["entropy_metric"] = cfg.get("entropy_metric") or "hybrid"
        return cfg


def run_rl_loop(
    text: str,
    doc_profile: Dict[str, Any],
    initial_chunks: List[Dict],
    config: Dict[str, Any],
) -> Tuple[List[Dict], List[float], Dict[str, Any]]:
    """
    EPISODIC RL LOOP WITH MONOTONIC IMPROVEMENT GUARANTEE.
    Each document = 1 episode.
    Guarantee: Chunk quality never degrades within episode.
    """
    max_iters = int(config.get("max_iterations", 10))
    model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
    doc_type = doc_profile.get("type", "prose")
    domain = doc_profile.get("domain", "general")
    history_key = str(config.get("rl_history_key") or domain)
    objective_weights = _objective_weights(config)
    history = _load_history()
    warm_cfg = _warm_start_config(config, history, history_key)
    probe_queries = _generate_probes(text, n=6)

    agent = DQNAgent(
        state_dim=11,
        hidden_dim=int(warm_cfg.get("dqn_hidden_dim", 24)),
        lr=float(warm_cfg.get("dqn_lr", 0.02)),
        gamma=float(warm_cfg.get("dqn_gamma", 0.90)),
        epsilon=float(warm_cfg.get("dqn_epsilon", 0.25)),
    )

    def _state_vec(
        chs: List[Dict], reward_components: Dict[str, float]
    ) -> np.ndarray:
        if not chs:
            return np.zeros(11, dtype=np.float32)

        def _mean_field(key: str, default: float) -> float:
            vals = []
            for c in chs:
                bf = c.get("boundary_features", {})
                if key in bf:
                    vals.append(float(bf[key]))
                elif key in c:
                    vals.append(float(c[key]))
                else:
                    vals.append(default)
            return float(np.mean(vals)) if vals else default

        return np.array(
            [
                _mean_field("jsd", 0.4),
                _mean_field("hellinger", 0.4),
                _mean_field("pmi_drop", 0.5),
                _mean_field("depth_change", 0.2),
                _mean_field("drift", 0.4),
                float(np.mean([c.get("boundary_score", 0.5) for c in chs])),
                float(np.mean([c.get("icc", 0.5) for c in chs])),
                reward_components.get("quality", 0.0),
                reward_components.get("coverage", 0.0),
                reward_components.get("consistency", 0.0),
                float(np.clip(len(chs) / max(_target_count(chs), 1.0), 0.0, 2.0)),
            ],
            dtype=np.float32,
        )

    # === INITIALIZE EPISODE STATE ===
    best_chunks = initial_chunks
    best_components = _compute_reward_components(
        initial_chunks, probe_queries, objective_weights
    )
    best_reward = best_components["total"]
    best_config = copy.deepcopy(warm_cfg)

    reward_history = [round(best_reward, 4)]
    reward_breakdown = [best_components]
    consecutive_non_improving = 0

    current_chunks = initial_chunks
    current_config = copy.deepcopy(best_config)
    current_components = best_components

    # === MAIN ITERATION LOOP ===
    for step_idx in range(max_iters):
        state_vec = _state_vec(current_chunks, current_components)
        action = agent.select_action(state_vec)
        trial_config = agent.apply_action(action, current_config)

        # Try pipeline with trial config
        try:
            all_chunks = run_all_chunkers(text, doc_type, trial_config)
            trial_chunks = select_best_strategy(all_chunks, doc_type, trial_config)

            if not trial_chunks:
                consecutive_non_improving += 1
                agent.decay_exploration(reward_improved=False)
                reward_history.append(round(best_reward, 4))
                reward_breakdown.append(best_components)
                continue

            trial_chunks = refine_boundaries(trial_chunks, trial_config)
            trial_chunks = filter_boundaries(trial_chunks, doc_type, [], trial_config)
            trial_chunks = enrich_graph(trial_chunks, [], trial_config)
            trial_chunks, _ = embed_chunks(
                trial_chunks, text, doc_profile, model_name, trial_config
            )

        except Exception:
            consecutive_non_improving += 1
            agent.decay_exploration(reward_improved=False)
            reward_history.append(round(best_reward, 4))
            reward_breakdown.append(best_components)
            continue

        # Compute reward
        trial_components = _compute_reward_components(
            trial_chunks, probe_queries, objective_weights
        )
        trial_reward = trial_components["total"]

        # === SAFE STEP: Enforce Monotonic Improvement ===
        is_improving = trial_reward > best_reward
        next_state = _state_vec(trial_chunks, trial_components)

        if is_improving:
            # IMPROVEMENT: Accept and update
            best_reward = trial_reward
            best_chunks = trial_chunks
            best_components = trial_components
            best_config = copy.deepcopy(trial_config)

            current_chunks = best_chunks
            current_config = best_config
            current_components = best_components

            # Store in replay buffer (only improving)
            agent.remember_improving(state_vec, action, trial_reward, next_state, True)
            agent.decay_exploration(reward_improved=True)
            consecutive_non_improving = 0

        else:
            # DEGRADATION: ROLLBACK
            # Current state remains unchanged (implicit rollback)
            # Do NOT store in replay buffer
            agent.remember_improving(state_vec, action, trial_reward, next_state, False)
            agent.decay_exploration(reward_improved=False)
            consecutive_non_improving += 1

        agent.learn()
        reward_history.append(round(best_reward, 4))
        reward_breakdown.append(best_components)

        # Early stopping
        if consecutive_non_improving >= _EARLY_STOPPING_PATIENCE:
            break

    # === END OF EPISODE ===
    final_config = copy.deepcopy(best_config)
    final_config["reward_breakdown"] = best_components
    final_config["reward_history_breakdown"] = reward_breakdown
    final_config["dqn_action_space"] = {
        "discrete": len(agent.ACTIONS),
        "continuous_magnitudes": agent.MAGNITUDES,
    }
    final_config["replay_buffer_size"] = len(agent.replay)
    final_config["rl_history_key"] = history_key
    final_config["dqn_state_dim"] = agent.state_dim
    final_config["early_stopped"] = consecutive_non_improving >= _EARLY_STOPPING_PATIENCE

    _save_history(history_key, final_config, best_components)
    return best_chunks, reward_history, final_config


def _compute_reward_components(
    chunks: List[Dict],
    probes: List[str],
    weights: Dict[str, float],
) -> Dict[str, float]:
    if not chunks:
        return {
            "quality": 0.0,
            "coverage": 0.0,
            "consistency": 0.0,
            "efficiency": 0.0,
            "structural": 0.0,
            "total": -1.0,
        }

    quality = float(
        np.mean([1.0 - c.get("boundary_score", 0.5) for c in chunks])
    )
    coverage = _recall_proxy(chunks, probes)
    size = np.array(
        [len(c.get("text", "").split()) for c in chunks], dtype=np.float32
    )
    consistency = float(
        1.0 - min(1.0, np.std(size) / max(np.mean(size), 1.0))
    )
    target = _target_count(chunks)
    efficiency = float(
        1.0 - min(1.0, abs(len(chunks) - target) / max(target, 1.0))
    )
    hard_ratio = sum(
        1
        for c in chunks
        if c.get("boundary_type") in {"hard", "protected_structure_boundary"}
    ) / max(len(chunks), 1)
    mean_pmi = float(
        np.mean(
            [
                c.get("boundary_features", {}).get("pmi_drop", c.get("pmi_drop", 0.5))
                for c in chunks
            ]
        )
    )
    structural = float(np.clip(0.5 * hard_ratio + 0.5 * mean_pmi, 0.0, 1.0))
    total = (
        weights["quality"] * quality
        + weights["coverage"] * coverage
        + weights["consistency"] * consistency
        + weights["efficiency"] * efficiency
        + 0.15 * structural
    )
    return {
        "quality": round(quality, 4),
        "coverage": round(coverage, 4),
        "consistency": round(consistency, 4),
        "efficiency": round(efficiency, 4),
        "structural": round(structural, 4),
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
        found = any(
            bool(terms & set(re.findall(r"\b\w+\b", c.get("text", "").lower())))
            for c in chunks
        )
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


def _warm_start_config(
    config: Dict[str, Any],
    history: Dict[str, Any],
    domain: str,
) -> Dict[str, Any]:
    out = copy.deepcopy(config)
    record = history.get(domain, {})
    tunable_keys = (
        "tau_jsd_low",
        "tau_jsd_high",
        "n_max",
        "tau_sem",
        "tau_percentile_low",
        "tau_percentile_high",
        "hybrid_lambda",
        "merge_weight",
    )
    for k in tunable_keys:
        v = record.get(k)
        if v is not None and k not in out:
            out[k] = v
    return out


def _save_history(
    domain: str,
    config: Dict[str, Any],
    reward_components: Dict[str, float],
) -> None:
    history = _load_history()
    record: Dict[str, Any] = {"last_reward_components": reward_components}
    tunable_keys = (
        "tau_jsd_low",
        "tau_jsd_high",
        "n_max",
        "tau_sem",
        "tau_percentile_low",
        "tau_percentile_high",
        "hybrid_lambda",
        "merge_weight",
    )
    for k in tunable_keys:
        v = config.get(k)
        if v is not None:
            record[k] = v
    history[domain] = record
    try:
        with open(_RL_HISTORY_PATH, "w", encoding="utf-8") as fh:
            json.dump(history, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass
'''

# Write the file
file_path = r'c:\Users\yosrc\OneDrive\Desktop\chunking final2\chunker\backend\pipeline\s7_rl.py'
with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ s7_rl.py rewritten successfully!")
print("✅ Best-state tracking with explicit rollback installed")
print("✅ Replay buffer purified (only improving transitions)")
print("✅ Exploration decay with 0.85x on degradation")
print("✅ Early stopping: 3 consecutive non-improving steps")
print("✅ Monotonic improvement guarantee: Quality never degrades")
print("✅ 1 document = 1 episode (episodic reset)")
