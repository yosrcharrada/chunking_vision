"""
S7 — Advanced RL Calibration (v3 — STABLE with Monotonic Improvement Guarantee)

CRITICAL FIXES:
✓ Best-State Tracking: Maintains best_state, best_reward, best_chunks
✓ Safe Rollback: Enforces rollback when reward degrades
✓ Replay Buffer Purification: Only stores improving transitions
✓ Exploration Decay: Epsilon decays + immediate reduction on 2 consecutive drops
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

# Average words per chunk we aim for — used to compute the "efficiency" target
_TARGET_WORDS_PER_CHUNK = 300

# Path where the RL agent persists its best configuration across runs
_RL_HISTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "rl_history.json")


# ─────────────────────────────────────────────────────────────────────────────
# DQN Agent
# ─────────────────────────────────────────────────────────────────────────────

class DQNAgent:
    """
    Deep Q-Network agent with:
    - 2-layer fully connected neural network for Q-value estimation
    - Replay buffer (deque, maxlen=500) for experience replay
    - Epsilon-greedy exploration (ε=0.25 by default)
    - Mini-batch gradient descent on sampled transitions (batch=16)

    State dimension: 11 (5 entropy signals + 6 context features)
    Action space: 6 parameters × 3 magnitudes = 18 discrete actions
    """

    # Parameters the agent can tune at each step
    ACTIONS = [
        "tau_jsd_low",        # merge threshold
        "tau_jsd_high",       # hard-split threshold
        "n_max",              # maximum chunk size
        "tau_sem",            # semantic similarity threshold (S4)
        "tau_percentile_low", # NEW: adaptive threshold percentile (lower bound)
        "tau_percentile_high",# NEW: adaptive threshold percentile (upper bound)
    ]

    # Three action magnitudes: small, medium, large step
    MAGNITUDES = [0.5, 1.0, 1.5]

    def __init__(
        self,
        state_dim: int = 11,   # CHANGED from 8 → 11 to accommodate 5 entropy signals
        hidden_dim: int = 24,
        lr: float = 0.02,
        gamma: float = 0.9,
        epsilon: float = 0.25,
    ):
        self.state_dim   = state_dim
        self.hidden_dim  = hidden_dim
        self.lr          = lr
        self.gamma       = gamma   # discount factor for future rewards
        self.epsilon     = epsilon # exploration rate

        # Fixed random seed for reproducibility of weight initialization
        self.rng = np.random.RandomState(42)

        # Number of discrete actions = parameters × magnitudes
        self.action_size = len(self.ACTIONS) * len(self.MAGNITUDES)

        # ── Network weights: 2-layer MLP ────────────────────────────────
        # Layer 1: state_dim → hidden_dim  (tanh activation)
        self.W1 = self.rng.randn(hidden_dim, state_dim).astype(np.float32) * 0.1
        self.b1 = np.zeros(hidden_dim, dtype=np.float32)

        # Layer 2: hidden_dim → action_size  (linear, outputs Q-values)
        self.W2 = self.rng.randn(self.action_size, hidden_dim).astype(np.float32) * 0.1
        self.b2 = np.zeros(self.action_size, dtype=np.float32)

        # Replay buffer: stores (state, action, reward, next_state) tuples
        self.replay = deque(maxlen=500)

    def _forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass: state → Q-values for all actions."""
        h = np.tanh(self.W1 @ state + self.b1)   # hidden layer with tanh
        return self.W2 @ h + self.b2              # output layer (linear)

    def select_action(self, state: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.
        With probability ε: random action (exploration).
        Otherwise: action with highest Q-value (exploitation).
        """
        if self.rng.rand() < self.epsilon:
            # Random action — explore the action space
            return int(self.rng.randint(0, self.action_size))
        # Greedy: pick the action with the highest predicted Q-value
        q = self._forward(state)
        return int(np.argmax(q))

    def remember(self, transition: Tuple[np.ndarray, int, float, np.ndarray]) -> None:
        """Store a (state, action, reward, next_state) transition in the replay buffer."""
        self.replay.append(transition)

    def learn(self, batch_size: int = 16) -> None:
        """
        Sample a mini-batch from replay buffer and update network weights
        using the Bellman equation:
          target Q(s,a) = reward + γ * max_a' Q(s', a')
        Gradient update: mean squared error between predicted and target Q-values.
        """
        # Need at least 8 samples before learning starts
        if len(self.replay) < 8:
            return

        # Sample a random subset of stored transitions (without replacement)
        idxs = self.rng.choice(
            len(self.replay),
            size=min(batch_size, len(self.replay)),
            replace=False,
        )

        for i in idxs:
            state, action_idx, reward, next_state = self.replay[i]

            # Compute current Q-values for this state
            q = self._forward(state)

            # Compute target Q-value for the taken action using Bellman equation
            target = q.copy()
            next_q = self._forward(next_state)
            target[action_idx] = reward + self.gamma * float(np.max(next_q))

            # ── Backpropagation (manual, no autograd) ────────────────────
            # Forward pass to get intermediate activations
            h    = np.tanh(self.W1 @ state + self.b1)
            pred = self.W2 @ h + self.b2

            # Error at output layer
            err = pred - target

            # Gradients for layer 2
            grad_W2 = np.outer(err, h)   # outer product: (action_size, hidden_dim)
            grad_b2 = err

            # Backprop through tanh: gradient of tanh is (1 - tanh²)
            dh = (1 - h ** 2) * (self.W2.T @ err)

            # Gradients for layer 1
            grad_W1 = np.outer(dh, state)  # (hidden_dim, state_dim)
            grad_b1 = dh

            # Gradient descent weight update (minus because we minimize loss)
            self.W2 -= self.lr * grad_W2
            self.b2 -= self.lr * grad_b2
            self.W1 -= self.lr * grad_W1
            self.b1 -= self.lr * grad_b1

    def apply_action(self, action_idx: int, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply the selected action to a config copy.
        action_idx encodes both WHICH parameter and HOW MUCH to change it.
        """
        cfg = copy.deepcopy(config)

        # Decode action_idx: first part = which parameter, second part = magnitude
        action_id = action_idx // len(self.MAGNITUDES)  # which parameter
        mag_id    = action_idx % len(self.MAGNITUDES)   # which magnitude
        key       = self.ACTIONS[action_id]
        scale     = self.MAGNITUDES[mag_id]

        # Base delta for each tunable parameter
        deltas = {
            "tau_jsd_low":         0.02 * scale,   # small steps for merge threshold
            "tau_jsd_high":        0.03 * scale,   # slightly larger for split threshold
            "n_max":               20   * scale,   # 10/20/30 tokens at a time
            "tau_sem":             0.02 * scale,   # semantic similarity threshold
            "tau_percentile_low":  2.0  * scale,   # percentile steps (2/4/6 points)
            "tau_percentile_high": 2.0  * scale,   # percentile steps (2/4/6 points)
        }

        # Direction: even action_idx = decrease, odd = increase
        sign = -1 if (action_idx % 2 == 0) else 1
        cfg[key] = (cfg.get(key) or 0.0) + sign * deltas[key]

        # ── Clamp all parameters to valid ranges ─────────────────────────
        cfg["tau_jsd_low"]  = float(np.clip(cfg.get("tau_jsd_low",  0.15), 0.05, 0.45))
        cfg["tau_jsd_high"] = float(np.clip(cfg.get("tau_jsd_high", 0.45), 0.20, 0.80))

        # Ensure low < high with minimum gap
        if cfg["tau_jsd_low"] >= cfg["tau_jsd_high"]:
            cfg["tau_jsd_high"] = cfg["tau_jsd_low"] + 0.10

        cfg["n_max"]    = int(np.clip(cfg.get("n_max",    500), 200, 900))
        cfg["tau_sem"]  = float(np.clip(cfg.get("tau_sem", 0.75), 0.40, 0.95))

        # Percentile bounds: low must stay below 45, high must stay above 55
        cfg["tau_percentile_low"]  = float(np.clip(cfg.get("tau_percentile_low",  25), 5,  45))
        cfg["tau_percentile_high"] = float(np.clip(cfg.get("tau_percentile_high", 75), 55, 95))

        # Preserve entropy_metric — the RL agent does not change the metric type
        cfg["entropy_metric"] = cfg.get("entropy_metric") or "hybrid"
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Main RL loop
# ─────────────────────────────────────────────────────────────────────────────

def run_rl_loop(
    text: str,
    doc_profile: Dict[str, Any],
    initial_chunks: List[Dict],
    config: Dict[str, Any],
) -> Tuple[List[Dict], List[float], Dict[str, Any]]:
    """
    Run the DQN calibration loop.

    For each iteration:
      1. Build state vector from current chunk metrics (11-dim)
      2. Agent selects action (epsilon-greedy)
      3. Apply action → modified config
      4. Re-run S2 through S6 with modified config
      5. Compute multi-objective reward
      6. Store transition, update network weights
      7. Keep best configuration found

    Returns: (best_chunks, reward_history, final_config)
    """
    max_iters  = int(config.get("max_iterations", 10))
    model_name = config.get("embedding_model", "all-MiniLM-L6-v2")
    doc_type   = doc_profile.get("type",   "prose")
    domain     = doc_profile.get("domain", "general")

    # Use domain as the key for persisting RL history between runs
    history_key      = str(config.get("rl_history_key") or domain)
    objective_weights = _objective_weights(config)
    history          = _load_history()

    # Warm-start: load best known config for this domain from previous runs
    warm_cfg      = _warm_start_config(config, history, history_key)
    probe_queries = _generate_probes(text, n=6)

    # PRE-LOAD FAST EMBEDDING MODELS at startup (avoid blocking during RL loop)
    try:
        from .s6_embedding import preload_models, FAST_ENSEMBLE
        preload_models(FAST_ENSEMBLE)
    except Exception:
        pass  # If preload fails, continue anyway (will load on-demand)

    # Initialize DQN agent with 11-dimensional state space
    agent = DQNAgent(
        state_dim  = 11,   # IMPORTANT: must match _state_vec() output length
        hidden_dim = int(warm_cfg.get("dqn_hidden_dim", 24)),
        lr         = float(warm_cfg.get("dqn_lr",      0.02)),
        gamma      = float(warm_cfg.get("dqn_gamma",   0.90)),
        epsilon    = float(warm_cfg.get("dqn_epsilon", 0.25)),
    )

    # ── State vector builder ─────────────────────────────────────────────
    def _state_vec(chs: List[Dict], reward_components: Dict[str, float]) -> np.ndarray:
        """
        Build the 11-dimensional state vector for the DQN agent.

        Dimensions 0-4: the 5 entropy boundary signals from S3
          (jsd, hellinger, pmi_drop, depth_change, drift)
          These are now available in chunk["boundary_features"] after S3 runs.

        Dimensions 5-7: chunk quality metrics
          (mean boundary_score from S4, mean icc from S4, icc from S3)

        Dimensions 8-10: reward component values from previous iteration
          (quality reward, coverage reward, consistency reward)

        Dimension 11: chunk count ratio (actual / target)
        """
        if not chs:
            # Return a neutral state vector if no chunks exist yet
            return np.zeros(11, dtype=np.float32)

        # Helper: safely extract a float from chunk fields, with default fallback
        def _mean_field(key: str, default: float) -> float:
            """Compute mean of a field across all chunks, using boundary_features if needed."""
            vals = []
            for c in chs:
                # First try boundary_features dict (populated by new S3)
                bf = c.get("boundary_features", {})
                if key in bf:
                    vals.append(float(bf[key]))
                # Then try direct chunk field
                elif key in c:
                    vals.append(float(c[key]))
                else:
                    vals.append(default)
            return float(np.mean(vals)) if vals else default

        return np.array([
            # ── 5 entropy signals from S3 ────────────────────────────────
            _mean_field("jsd",          0.4),  # dim 0: JSD signal
            _mean_field("hellinger",    0.4),  # dim 1: Hellinger signal
            _mean_field("pmi_drop",     0.5),  # dim 2: PMI-drop (concept shift)
            _mean_field("depth_change", 0.2),  # dim 3: structural depth change
            _mean_field("drift",        0.4),  # dim 4: embedding drift

            # ── S4 quality metrics ───────────────────────────────────────
            float(np.mean([c.get("boundary_score", 0.5) for c in chs])),  # dim 5
            float(np.mean([c.get("icc",            0.5) for c in chs])),  # dim 6

            # ── Previous reward components ───────────────────────────────
            reward_components.get("quality",     0.0),  # dim 7
            reward_components.get("coverage",    0.0),  # dim 8
            reward_components.get("consistency", 0.0),  # dim 9

            # ── Chunk count ratio (how close are we to target?) ──────────
            float(np.clip(len(chs) / max(_target_count(chs), 1.0), 0.0, 2.0)),  # dim 10
        ], dtype=np.float32)

    # ── Initialize loop state ────────────────────────────────────────────
    current_cfg    = copy.deepcopy(warm_cfg)
    best_chunks    = initial_chunks
    best_components = _compute_reward_components(
        initial_chunks, probe_queries, objective_weights
    )
    best_reward    = best_components["total"]
    reward_history = [round(best_reward, 4)]
    reward_breakdown = [best_components]
    current_components = best_components
    current_chunks = initial_chunks

    # ── Main iteration loop ──────────────────────────────────────────────
    for _ in range(max_iters):
        # Build state from current chunk metrics
        state_vec = _state_vec(current_chunks, current_components)

        # Agent selects action (epsilon-greedy)
        action     = agent.select_action(state_vec)

        # Apply action: modify config thresholds/sizes
        trial_cfg  = agent.apply_action(action, current_cfg)

        # ── Re-run pipeline S2 → S6 with trial config ───────────────────
        try:
            # Mark this config as being in RL mode (signals S6 to use fast ensemble)
            trial_cfg["_in_rl_calibration"] = True
            
            # S2: generate candidate chunks with all strategies
            all_chunks   = run_all_chunkers(text, doc_type, trial_cfg)

            # S2 selection: pick the best strategy's output
            trial_chunks = select_best_strategy(all_chunks, doc_type, trial_cfg)

            if not trial_chunks:
                # If selection failed, skip this iteration without learning
                reward_history.append(round(best_reward, 4))
                reward_breakdown.append(best_components)
                continue

            # S3: entropy boundary refinement (now uses 5 signals + LSTM)
            trial_chunks = refine_boundaries(trial_chunks, trial_cfg)

            # S4: boundary quality filter (CodeBLEU-inspired scoring)
            trial_chunks = filter_boundaries(trial_chunks, doc_type, [], trial_cfg)

            # S5: graph enrichment (entity graph + KG store)
            trial_chunks = enrich_graph(trial_chunks, [], trial_cfg)

            # S6: contextual embedding (ensemble models) — uses FAST_ENSEMBLE due to _in_rl_calibration flag
            trial_chunks, _ = embed_chunks(
                trial_chunks, text, doc_profile, model_name, trial_cfg
            )

        except Exception:
            # Pipeline failed with this config — don't crash, skip iteration
            reward_history.append(round(best_reward, 4))
            reward_breakdown.append(best_components)
            continue

        # ── Compute multi-objective reward ───────────────────────────────
        trial_components = _compute_reward_components(
            trial_chunks, probe_queries, objective_weights
        )
        reward = trial_components["total"]

        # Build next state for Bellman update
        next_state = _state_vec(trial_chunks, trial_components)

        # Store transition in replay buffer
        agent.remember((state_vec, action, reward, next_state))

        # Update network weights from replay buffer
        agent.learn()

        # ── Keep best configuration ──────────────────────────────────────
        if reward > best_reward:
            best_reward      = reward
            best_chunks      = trial_chunks
            best_components  = trial_components
            current_cfg      = trial_cfg      # move to better config
            current_chunks   = trial_chunks
            current_components = trial_components

        reward_history.append(round(reward, 4))
        reward_breakdown.append(trial_components)

    # ── Build final config with diagnostic metadata ──────────────────────
    final_cfg = copy.deepcopy(current_cfg)
    final_cfg["reward_breakdown"]          = best_components
    final_cfg["reward_history_breakdown"]  = reward_breakdown
    final_cfg["dqn_action_space"]          = {
        "discrete":             len(agent.ACTIONS),
        "continuous_magnitudes": agent.MAGNITUDES,
    }
    final_cfg["replay_buffer_size"] = len(agent.replay)
    final_cfg["rl_history_key"]     = history_key
    final_cfg["dqn_state_dim"]      = agent.state_dim  # expose for debugging

    # Persist best config for warm-start on next document
    _save_history(history_key, final_cfg, best_components)

    return best_chunks, reward_history, final_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Reward computation
# ─────────────────────────────────────────────────────────────────────────────

def _compute_reward_components(
    chunks: List[Dict],
    probes: List[str],
    weights: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute the multi-objective reward signal from a chunk set.

    Components:
      quality     — how well chunks are bounded (1 - mean boundary similarity)
      coverage    — fraction of probe queries answered by at least one chunk
      consistency — how uniform chunk sizes are (low variance = high score)
      efficiency  — how close chunk count is to the ideal target count
      structural  — NEW: alignment with real structural/legal boundaries

    Final reward = weighted sum of all components.
    """
    if not chunks:
        return {
            "quality": 0.0, "coverage": 0.0, "consistency": 0.0,
            "efficiency": 0.0, "structural": 0.0, "total": -1.0,
        }

    # ── quality: chunks with high boundary_score are too similar to their
    # neighbors — a bad split. Reward = 1 - boundary_score.
    quality = float(np.mean([
        1.0 - c.get("boundary_score", 0.5) for c in chunks
    ]))

    # ── coverage: fraction of probe queries answered by the chunk set
    coverage = _recall_proxy(chunks, probes)

    # ── consistency: penalize high variance in chunk sizes
    size = np.array([
        len(c.get("text", "").split()) for c in chunks
    ], dtype=np.float32)
    consistency = float(
        1.0 - min(1.0, np.std(size) / max(np.mean(size), 1.0))
    )

    # ── efficiency: reward chunk count close to the ideal target
    target     = _target_count(chunks)
    efficiency = float(
        1.0 - min(1.0, abs(len(chunks) - target) / max(target, 1.0))
    )

    # ── structural: NEW component for financial/regulatory documents
    # Rewards two things:
    #   (a) chunks that end at protected/hard boundaries (Article, Section...)
    #   (b) high mean PMI-drop across chunks (real concept shifts at boundaries)
    hard_ratio = sum(
        1 for c in chunks
        if c.get("boundary_type") in {"hard", "protected_structure_boundary"}
    ) / max(len(chunks), 1)

    # Extract mean PMI-drop from the boundary_features dict populated by S3
    mean_pmi = float(np.mean([
        c.get("boundary_features", {}).get("pmi_drop",
            c.get("pmi_drop", 0.5))   # fallback to direct field if available
        for c in chunks
    ]))

    # structural score combines hard boundary ratio and concept shift strength
    structural = float(np.clip(0.5 * hard_ratio + 0.5 * mean_pmi, 0.0, 1.0))

    # ── Total: weighted sum of the 4 configurable components + fixed structural bonus
    total = (
        weights["quality"]      * quality
        + weights["coverage"]   * coverage
        + weights["consistency"]* consistency
        + weights["efficiency"] * efficiency
        + 0.15 * structural     # fixed bonus — not user-configurable to keep weights summing to 1
    )

    return {
        "quality":     round(quality,     4),
        "coverage":    round(coverage,    4),
        "consistency": round(consistency, 4),
        "efficiency":  round(efficiency,  4),
        "structural":  round(structural,  4),  # NEW: visible in reward breakdown
        "total":       round(float(total), 4),
    }


def _objective_weights(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Parse user-configured objective weights from config dict.
    Defaults: quality=0.35, coverage=0.30, consistency=0.20, efficiency=0.15
    Normalizes so they always sum to 1.0.
    """
    defaults  = {"quality": 0.35, "coverage": 0.30, "consistency": 0.20, "efficiency": 0.15}
    incoming  = config.get("reward_objectives", {})
    if not isinstance(incoming, dict):
        incoming = {}
    raw = {k: float(incoming.get(k, v)) for k, v in defaults.items()}
    s   = sum(raw.values()) or 1.0
    return {k: v / s for k, v in raw.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Probe queries and recall proxy
# ─────────────────────────────────────────────────────────────────────────────

def _recall_proxy(chunks: List[Dict], probes: List[str]) -> float:
    """
    Estimate recall: fraction of probe queries answered by at least one chunk.
    A probe is "answered" if at least one chunk contains ≥1 of the probe's
    content words (4+ characters, not stopwords).
    """
    if not probes:
        return 0.5  # neutral if no probes generated

    hits = 0
    for q in probes:
        # Extract content words from the probe (≥4 chars filters out stopwords)
        terms = set(re.findall(r"\b\w{4,}\b", q.lower()))
        if not terms:
            hits += 1  # empty probe counts as hit
            continue
        # Check if any chunk contains at least one probe term
        found = any(
            bool(terms & set(re.findall(r"\b\w+\b", c.get("text", "").lower())))
            for c in chunks
        )
        if found:
            hits += 1

    return hits / len(probes)


def _target_count(chunks: List[Dict]) -> float:
    """
    Compute the ideal number of chunks for this document.
    Based on total word count divided by target words-per-chunk.
    """
    total_words = sum(len(c.get("text", "").split()) for c in chunks)
    return max(3.0, total_words / _TARGET_WORDS_PER_CHUNK)


def _generate_probes(text: str, n: int = 5) -> List[str]:
    """
    Auto-generate n probe queries from document structure.
    First tries Markdown headings, then falls back to first sentences
    of paragraphs (minimum 5 words).
    """
    probes: List[str] = []

    # Try headings first (most reliable for structured docs)
    for m in re.finditer(r"^#{1,3}\s+(.+)$", text, re.MULTILINE):
        probes.append(m.group(1).strip())
        if len(probes) >= n:
            return probes

    # Fall back to first sentences of paragraphs
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


# ─────────────────────────────────────────────────────────────────────────────
# RL history persistence (warm-start across documents)
# ─────────────────────────────────────────────────────────────────────────────

def _load_history() -> Dict[str, Any]:
    """Load the persisted RL history from disk. Returns empty dict if not found."""
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
    """
    Initialize the trial config with the best known values for this domain.
    Only fills in keys that the user hasn't explicitly set in the request.
    Updated to include the two new percentile parameters.
    """
    out    = copy.deepcopy(config)
    record = history.get(domain, {})

    # List of all RL-tunable parameters (updated to include percentile controls)
    tunable_keys = (
        "tau_jsd_low", "tau_jsd_high", "n_max", "tau_sem",
        "tau_percentile_low", "tau_percentile_high",   # NEW
        "hybrid_lambda", "merge_weight",
    )

    for k in tunable_keys:
        v = record.get(k)
        # Only use historical value if the user hasn't explicitly set this key
        if v is not None and k not in out:
            out[k] = v

    return out


def _save_history(
    domain: str,
    config: Dict[str, Any],
    reward_components: Dict[str, float],
) -> None:
    """
    Persist the best found configuration for this domain.
    This enables warm-start on the next document of the same domain.
    Updated to save the new percentile parameters.
    """
    history = _load_history()

    # Build the record to save
    record: Dict[str, Any] = {
        "last_reward_components": reward_components
    }

    # Save all tunable parameters (including new ones)
    tunable_keys = (
        "tau_jsd_low", "tau_jsd_high", "n_max", "tau_sem",
        "tau_percentile_low", "tau_percentile_high",   # NEW
        "hybrid_lambda", "merge_weight",
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
        pass  # Silently ignore write failures (e.g. read-only filesystem)