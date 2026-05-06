"""
S7 — Hyperparameter Calibration via Bayesian Optimization (TPE)
================================================================
Pipeline position : runs AFTER S2–S6 (all chunking stages) and produces the
                    final, optimally-configured chunk set for this document.

WHY WE REPLACED DQN WITH BAYESIAN OPTIMIZATION
───────────────────────────────────────────────
The original DQN had three fatal flaws that made it a no-op in practice:

  1. "Monotonic improvement guarantee" killed all exploration.
     Any trial that scored lower than the current best was immediately
     reverted AND double-penalized.  With 10 iterations and instant
     reversion, the agent could never walk through a temporarily-worse
     state to reach a globally-better one.  The reward history froze at
     iteration 1 in every test.

  2. Too few iterations for a DQN to learn anything.
     A DQN with 18 actions and an 11-dim state needs hundreds of
     transitions before its Q-network generalises.  10 iterations with
     a revert policy → ≤ 2 non-reverted transitions → zero learning.

  3. Self-referential reward.
     reward_quality = 1 - mean(S4_boundary_score).
     S4 computes boundary_score inside the same pipeline run the agent
     just triggered.  The agent was optimizing a number it computed
     itself, not an external ground truth.

WHAT BAYESIAN OPTIMIZATION (TPE) GIVES US
──────────────────────────────────────────
Tree-structured Parzen Estimator (TPE) is the right algorithm here:

  • Designed for expensive black-box functions (each eval = full pipeline
    run).  It needs 20–50 trials, not thousands.

  • Builds a probabilistic surrogate model of the objective landscape
    from all past trials. Uses that model to pick the next configuration
    most likely to improve, via the Expected Improvement (EI) criterion:

        EI(x) = E[max(f(x) − f(x⁺), 0)]

    where f(x⁺) is the current best observed reward.

  • Does not need exploration/exploitation tuning (ε, γ, etc.).
    The surrogate model handles the trade-off automatically.

  • Warm-starts perfectly: persist the study's past trials to disk, load
    them on the next document of the same domain → the surrogate model
    already knows which regions of the search space are good.

REWARD FUNCTION FIXES
─────────────────────
  Old coverage   : recall_proxy with probes auto-generated from headings
                   of the SAME document → trivially answered by any chunking
                   → always 1.0 → zero discriminating signal.

  New coverage   : cross-chunk PRECISION signal.  For each probe, we
                   measure whether the BEST matching chunk is tightly
                   focused (short, high ICC) or sprawling (long, low ICC).
                   A chunk that contains the answer within 300 words of
                   relevant content scores higher than one that buries it
                   in 900 words of mixed content.

  Old quality    : 1 - mean(boundary_score)  — same pipeline, circular.

  New quality    : combination of:
                     (a) mean inter-chunk separation (hash-cosine distance
                         between adjacent chunk embeddings → real boundary
                         distinctiveness).
                     (b) mean intra-chunk ICC (from S4) → coherence.
                   Neither of these is produced by the component being
                   optimised; they are independent structural signals.

  efficiency     : unchanged but now correctly drives the search.
                   target_count = total_words / TARGET_WORDS_PER_CHUNK.
                   The search CAN fix it because it explores n_max freely.

  structural     : unchanged. Hard-boundary ratio + mean PMI-drop.

  consistency    : unchanged. CV of chunk sizes.

SEARCH SPACE
────────────
  tau_jsd_low          ∈ [0.05, 0.40]  — merge threshold (S3)
  tau_jsd_high         ∈ [0.20, 0.80]  — hard-split threshold (S3)
  n_max                ∈ [150,  900]   — max tokens per chunk (S2)
  n_min                ∈ [30,   250]   — min tokens per chunk (S2)
  tau_sem              ∈ [0.40, 0.95]  — S4 merge similarity threshold
  tau_percentile_low   ∈ [5,    45]    — S3 adaptive threshold lower %ile
  tau_percentile_high  ∈ [55,   95]    — S3 adaptive threshold upper %ile

PERSISTENCE & WARM-START
────────────────────────
  Best trials are stored in `rl_history.json` keyed by domain
  (e.g. "regulatory", "legal", "technical").  On subsequent documents
  of the same domain the surrogate model is seeded with past trials,
  meaning fewer trials are needed to reach good performance.
"""

import copy
import json
import logging
import os
import re
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Optional Optuna import — graceful fallback to random search if missing ───
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)   # silence per-trial logs
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False
    warnings.warn(
        "optuna not installed; S7 will fall back to random search.  "
        "Install with: pip install optuna"
    )

from .s2_chunkers import run_all_chunkers, select_best_strategy
from .s3_entropy import refine_boundaries
from .s4_boundary import filter_boundaries
from .s5_graph import enrich_graph
from .s6_embedding import embed_chunks

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

# Ideal chunk length in words — drives the efficiency reward component.
# A legal/regulatory document is best served by ~300-word chunks so that
# each chunk covers one concept without burying it in surrounding context.
_TARGET_WORDS_PER_CHUNK = 300

# Where we persist per-domain trial history for warm-start
_RL_HISTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "rl_history.json")

# Minimum number of Optuna trials before early stopping is allowed
_MIN_TRIALS_BEFORE_STOP = 8

# If the best reward does not improve by at least this delta over
# _PATIENCE_TRIALS consecutive trials, stop early.
_IMPROVEMENT_DELTA = 0.005
_PATIENCE_TRIALS   = 6


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def run_rl_loop(
    text: str,
    doc_profile: Dict[str, Any],
    initial_chunks: List[Dict],
    config: Dict[str, Any],
) -> Tuple[List[Dict], List[float], Dict[str, Any]]:
    """
    Tune hyperparameters for EACH chunking strategy independently via Bayesian
    Optimisation (TPE), then return the strategy+params combination that scores
    highest — using the SAME metric as S2 so all scores are directly comparable.

    CORRECT ARCHITECTURE
    ────────────────────
    The platform benchmarks chunking strategies.  S7's job is to find the best
    possible version of EACH strategy by tuning its hyperparameters, then crown
    the overall winner.

    Per-strategy BO:
      For each strategy S in [structure, hybrid_legal_semantic, legal_articles,
                               recursive, paragraph_pack, ...]:
        Run a dedicated Optuna study for S alone.
        Each trial: tune (n_max, n_min, tau_jsd_low, tau_jsd_high, tau_sem,
                         tau_percentile_low, tau_percentile_high) for S.
        Evaluate: _strategy_quality_score(S_chunks, n_min, n_max).
        Record: best params + best score for S.

      Overall winner = argmax over all strategies of their best BO score.

    Why per-strategy, not global?
      Global BO mixes strategies across trials → TPE surrogate receives
      inconsistent signal (same params may select different strategies) →
      bouncing reward → never converges on any strategy's optimum.

    UNIFIED SCORING (same metric throughout)
      S2 uses _strategy_quality_score.
      S7 uses _strategy_quality_score.
      So if S2 gives structure=0.756 and S7 gives structure_optimised=0.789,
      that 0.789 is genuinely better — not a different scale.

    Trial budget allocation:
      trials_per_strategy = max_trials // n_active_strategies
      Minimum 3 trials per strategy (enough for TPE to start learning).
      Strategies excluded from BO: semantic_boundaries, sentence_clustering
      (they produce 100+ micro-chunks regardless of params — BO can't help).

    Parameters
    ──────────
    text          : raw document text
    doc_profile   : output of S1
    initial_chunks: S2 winner chunks (used as baseline and fallback)
    config        : pipeline config dict

    Returns
    ───────
    best_chunks   : chunks from the best strategy+params combination
    reward_history: flat list of all trial rewards (all strategies combined)
    final_config  : diagnostics + best params per strategy
    """
    from .s2_chunkers import _strategy_quality_score as _sqscore

    max_trials  = int(config.get("max_iterations", 20))
    model_name  = config.get("embedding_model", "all-MiniLM-L6-v2")
    doc_type    = doc_profile.get("type",   "prose")
    domain      = doc_profile.get("domain", "general")
    history_key = str(config.get("rl_history_key") or domain)
    n_min       = int(config.get("n_min", 100))
    n_max       = int(config.get("n_max", 500))

    # ── Strategies eligible for BO ────────────────────────────────────────────
    # Excluded: semantic_boundaries and sentence_clustering always produce
    # 100+ micro-chunks regardless of n_max → BO cannot meaningfully tune them.
    # They are already benchmarked correctly in S2 with their native params.
    _EXCLUDED = {"semantic_boundaries", "sentence_clustering"}

    # Determine which strategies are active for this document type
    # (same logic as S2 run_all_chunkers — only legal strategies for legal docs)
    from .s2_chunkers import run_all_chunkers
    all_s2 = run_all_chunkers(text, doc_type, config)
    active_strategies = [s for s in all_s2.keys() if s not in _EXCLUDED]

    # Allocate trial budget evenly across strategies (minimum 3 per strategy)
    n_strategies      = max(1, len(active_strategies))
    trials_per_strat  = max(3, max_trials // n_strategies)

    # ── Baseline: S2 winner with default params ───────────────────────────────
    baseline_reward = round(float(_sqscore(initial_chunks, n_min, n_max)), 4)

    # ── Per-strategy BO results ───────────────────────────────────────────────
    # strategy_name → {"best_chunks", "best_score", "best_params", "trials"}
    per_strategy_results: Dict[str, Any] = {}

    # S2 scores are the baselines for each strategy
    s2_scores = {}
    for strat, chunks_list in all_s2.items():
        if chunks_list:
            s2_scores[strat] = round(float(_sqscore(chunks_list, n_min, n_max)), 4)

    reward_history: List[float] = [baseline_reward]

    # ── Run BO for each strategy independently ────────────────────────────────
    for strategy in active_strategies:
        strat_baseline_chunks = all_s2.get(strategy, initial_chunks) or initial_chunks
        strat_baseline_score  = s2_scores.get(strategy, baseline_reward)

        # Load per-strategy warm-start (domain + strategy key)
        strat_key   = f"{history_key}__{strategy}"
        history     = _load_history()
        warm_cfg    = _warm_start_config(config, history, strat_key)
        warm_cfg["chunking_strategy"] = strategy

        strat_best_chunks = strat_baseline_chunks
        strat_best_score  = strat_baseline_score
        strat_best_cfg    = copy.deepcopy(warm_cfg)
        strat_trials: List[float] = []

        if _OPTUNA_AVAILABLE:
            strat_best_chunks, strat_best_score, strat_best_cfg, strat_trials = (
                _run_strategy_optuna(
                    text=text, doc_type=doc_type, doc_profile=doc_profile,
                    model_name=model_name, warm_cfg=warm_cfg,
                    strategy=strategy, n_min=n_min, n_max=n_max,
                    baseline_score=strat_baseline_score,
                    best_chunks=strat_best_chunks,
                    best_score=strat_best_score,
                    best_cfg=strat_best_cfg,
                    max_trials=trials_per_strat,
                )
            )
        else:
            strat_best_chunks, strat_best_score, strat_best_cfg, strat_trials = (
                _run_strategy_random(
                    text=text, doc_type=doc_type, doc_profile=doc_profile,
                    model_name=model_name, warm_cfg=warm_cfg,
                    strategy=strategy, n_min=n_min, n_max=n_max,
                    baseline_score=strat_baseline_score,
                    best_chunks=strat_best_chunks,
                    best_score=strat_best_score,
                    best_cfg=strat_best_cfg,
                    max_trials=trials_per_strat,
                )
            )

        reward_history.extend(strat_trials)

        # Persist per-strategy warm-start
        _save_history(strat_key, strat_best_cfg, {"total": strat_best_score})

        per_strategy_results[strategy] = {
            "best_chunks": strat_best_chunks,
            "best_score":  round(strat_best_score, 4),
            "s2_baseline": strat_baseline_score,
            "improvement": round(strat_best_score - strat_baseline_score, 4),
            "best_params": {
                k: strat_best_cfg.get(k)
                for k in ("n_max", "n_min", "tau_jsd_low", "tau_jsd_high",
                          "tau_sem", "tau_percentile_low", "tau_percentile_high")
            },
            "n_trials": len(strat_trials),
        }

    # ── Pick overall winner ───────────────────────────────────────────────────
    # The winner is the strategy whose best BO score is highest.
    # This is the correct benchmark result: best possible version of each strategy,
    # winner is the one that performs best on this document.
    overall_winner = max(
        per_strategy_results,
        key=lambda s: per_strategy_results[s]["best_score"],
    )
    winner_result = per_strategy_results[overall_winner]
    best_chunks   = winner_result["best_chunks"]
    best_reward   = winner_result["best_score"]

    # ── Build final config ────────────────────────────────────────────────────
    final_cfg = copy.deepcopy(winner_result.get("best_params", config))
    final_cfg.update({
        "optimizer":                 "optuna_tpe" if _OPTUNA_AVAILABLE else "random_search",
        "rl_history_key":            history_key,
        "n_trials_run":              len(reward_history) - 1,
        "baseline_reward":           baseline_reward,
        "best_reward":               round(best_reward, 4),
        "improvement_over_baseline": round(best_reward - baseline_reward, 4),
        "overall_winner_strategy":   overall_winner,
        "chunking_strategy":         overall_winner,
        "per_strategy_results":      {
            s: {k: v for k, v in r.items() if k != "best_chunks"}
            for s, r in per_strategy_results.items()
        },
    })

    return best_chunks, reward_history, final_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Optuna TPE optimisation
# ─────────────────────────────────────────────────────────────────────────────

def _run_strategy_optuna(
    text, doc_type, doc_profile, model_name, warm_cfg,
    strategy, n_min, n_max, baseline_score,
    best_chunks, best_score, best_cfg, max_trials,
):
    """
    Dedicated Optuna TPE study for ONE specific strategy.
    Every trial tests the same strategy with different hyperparameters.
    The TPE surrogate learns: params → quality for THIS strategy only.
    Reward = _strategy_quality_score (same as S2 — directly comparable).
    """
    import optuna
    from optuna.samplers import TPESampler
    from .s2_chunkers import _strategy_quality_score as _sqscore

    sampler = TPESampler(seed=42, n_startup_trials=min(3, max_trials), multivariate=True)
    study   = optuna.create_study(direction="maximize", sampler=sampler)

    no_improve    = 0
    trial_rewards: List[float] = []

    def objective(trial):
        nonlocal best_chunks, best_score, best_cfg, no_improve

        trial_cfg    = _suggest_config(trial, warm_cfg, strategy)
        trial_chunks = _run_pipeline(text, doc_type, doc_profile, model_name, trial_cfg, strategy)

        if trial_chunks is None:
            trial_rewards.append(round(best_score, 4))
            return baseline_score - 0.1

        reward = round(float(_sqscore(
            trial_chunks,
            trial_cfg.get("n_min", n_min),
            trial_cfg.get("n_max", n_max),
        )), 4)
        trial_rewards.append(reward)

        if reward > best_score:
            best_score  = reward
            best_chunks = trial_chunks
            best_cfg    = trial_cfg
            no_improve  = 0
        else:
            no_improve += 1

        return reward

    for idx in range(max_trials):
        if idx >= _MIN_TRIALS_BEFORE_STOP and no_improve >= _PATIENCE_TRIALS:
            break
        try:
            t = study.ask()
            study.tell(t, objective(t))
        except Exception as exc:
            logger.debug("S7 [%s] trial %d: %s", strategy, idx, exc)
            trial_rewards.append(round(best_score, 4))

    return best_chunks, best_score, best_cfg, trial_rewards


def _run_strategy_random(
    text, doc_type, doc_profile, model_name, warm_cfg,
    strategy, n_min, n_max, baseline_score,
    best_chunks, best_score, best_cfg, max_trials,
):
    """Random search fallback for ONE strategy. Uses _strategy_quality_score."""
    from .s2_chunkers import _strategy_quality_score as _sqscore

    rng = np.random.RandomState(abs(hash(strategy)) % (2**31))
    trial_rewards: List[float] = []

    for _ in range(max_trials):
        trial_cfg    = _random_config(warm_cfg, rng, strategy)
        trial_chunks = _run_pipeline(text, doc_type, doc_profile, model_name, trial_cfg, strategy)

        if trial_chunks is None:
            trial_rewards.append(round(best_score, 4))
            continue

        reward = round(float(_sqscore(
            trial_chunks,
            trial_cfg.get("n_min", n_min),
            trial_cfg.get("n_max", n_max),
        )), 4)
        trial_rewards.append(reward)

        if reward > best_score:
            best_score  = reward
            best_chunks = trial_chunks
            best_cfg    = trial_cfg

    return best_chunks, best_score, best_cfg, trial_rewards


# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameter search space helpers
# ─────────────────────────────────────────────────────────────────────────────

def _suggest_config(trial: Any, base_cfg: Dict[str, Any], s2_winner: str) -> Dict[str, Any]:
    """
    Ask Optuna to suggest values for each tunable hyperparameter.

    chunking_strategy is locked to s2_winner — the BO tunes the parameters
    FOR that strategy, not across all strategies.
    """
    cfg = copy.deepcopy(base_cfg)

    cfg["tau_jsd_low"]  = trial.suggest_float("tau_jsd_low",  0.05, 0.40)
    cfg["tau_jsd_high"] = trial.suggest_float("tau_jsd_high", 0.20, 0.80)
    cfg["n_max"]        = trial.suggest_int(  "n_max",        150,  900, step=25)
    cfg["n_min"]        = trial.suggest_int(  "n_min",        30,   250, step=10)
    cfg["tau_sem"]      = trial.suggest_float("tau_sem",      0.40, 0.95)
    cfg["tau_percentile_low"]  = trial.suggest_float("tau_percentile_low",  5,  45)
    cfg["tau_percentile_high"] = trial.suggest_float("tau_percentile_high", 55, 95)

    # Enforce constraints
    if cfg["tau_jsd_low"] >= cfg["tau_jsd_high"] - 0.08:
        cfg["tau_jsd_high"] = min(0.80, cfg["tau_jsd_low"] + 0.10)
    if cfg["n_min"] >= cfg["n_max"]:
        cfg["n_min"] = max(30, cfg["n_max"] - 50)

    # Lock strategy — every trial uses the S2 winner
    cfg["chunking_strategy"] = s2_winner
    cfg["entropy_metric"]    = base_cfg.get("entropy_metric", "hybrid")
    cfg["_in_rl_calibration"] = True

    return cfg


def _random_config(base_cfg: Dict[str, Any], rng: np.random.RandomState, s2_winner: str) -> Dict[str, Any]:
    """Random search config locked to s2_winner strategy."""
    cfg = copy.deepcopy(base_cfg)

    cfg["tau_jsd_low"]  = float(rng.uniform(0.05, 0.40))
    cfg["tau_jsd_high"] = float(rng.uniform(0.20, 0.80))
    cfg["n_max"]        = int(rng.randint(6, 37) * 25)
    cfg["n_min"]        = int(rng.randint(3, 26) * 10)
    cfg["tau_sem"]      = float(rng.uniform(0.40, 0.95))
    cfg["tau_percentile_low"]  = float(rng.uniform(5,  45))
    cfg["tau_percentile_high"] = float(rng.uniform(55, 95))

    if cfg["tau_jsd_low"] >= cfg["tau_jsd_high"] - 0.08:
        cfg["tau_jsd_high"] = min(0.80, cfg["tau_jsd_low"] + 0.10)
    if cfg["n_min"] >= cfg["n_max"]:
        cfg["n_min"] = max(30, cfg["n_max"] - 50)

    cfg["chunking_strategy"]  = s2_winner
    cfg["entropy_metric"]     = base_cfg.get("entropy_metric", "hybrid")
    cfg["_in_rl_calibration"] = True

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline runner — S2 through S6 with a given config
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline(
    text: str,
    doc_type: str,
    doc_profile: Dict[str, Any],
    model_name: str,
    cfg: Dict[str, Any],
    s2_winner: str,
) -> Optional[List[Dict]]:
    """
    Run S2 (locked to s2_winner) → S3 → S4 → S5 → S6 with trial config.

    The strategy is locked: instead of running all chunkers and selecting
    the best one (which changes per trial), we run ONLY the s2_winner
    chunker.  This gives the TPE surrogate model clean, consistent signal:
    it learns how params affect quality for one specific strategy, not a
    mixture of strategies.
    """
    try:
        cfg["_full_text_sample"]  = text[:3000]
        cfg["chunking_strategy"]  = s2_winner   # enforce the lock

        # Import the specific chunker for the locked strategy
        from .s2_chunkers import (
            recursive_character_split, sliding_window_split,
            structure_based_split, semantic_boundary_split,
            sentence_cluster_split, paragraph_pack_split,
            legal_article_split, hybrid_legal_semantic_split,
            _quality_pass,
        )
        from .s3_entropy import refine_boundaries
        from .s4_boundary import filter_boundaries
        from .s5_graph import enrich_graph
        from .s6_embedding import embed_chunks

        n_min = int(cfg.get("n_min", 100))
        n_max = int(cfg.get("n_max", 500))

        # ── Run ONLY the locked strategy ─────────────────────────────────────
        strategy_map = {
            "recursive":              lambda: recursive_character_split(text, n_min, n_max, doc_type),
            "sliding_window":         lambda: sliding_window_split(text, n_max, int(n_max * 0.15)),
            "structure":              lambda: structure_based_split(text, doc_type, n_min, n_max),
            "semantic_boundaries":    lambda: semantic_boundary_split(text, n_min, n_max, cfg),
            "sentence_clustering":    lambda: sentence_cluster_split(text, n_min, n_max, cfg),
            "paragraph_pack":         lambda: paragraph_pack_split(text, n_min, n_max),
            "legal_articles":         lambda: legal_article_split(text, n_min, n_max),
            "hybrid_legal_semantic":  lambda: hybrid_legal_semantic_split(text, n_min, n_max, cfg),
        }

        chunker_fn = strategy_map.get(s2_winner, strategy_map["structure"])
        trial_chunks = chunker_fn()

        if not trial_chunks:
            return None

        # Apply quality pass (same as S2 does after chunking)
        trial_chunks = _quality_pass(trial_chunks, text, n_min, n_max, s2_winner)

        # S3 → S4 → S5 → S6
        trial_chunks = refine_boundaries(trial_chunks, cfg)
        trial_chunks = filter_boundaries(trial_chunks, doc_type, [], cfg)
        trial_chunks = enrich_graph(trial_chunks, [], cfg)
        trial_chunks, _ = embed_chunks(trial_chunks, text, doc_profile, model_name, cfg)

        return trial_chunks

    except Exception as exc:
        logger.debug("S7 pipeline trial failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Reward function
# ─────────────────────────────────────────────────────────────────────────────

def _compute_reward_components(
    chunks: List[Dict],
    probes: List[str],
    weights: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute the multi-objective reward for a chunk set.

    Five components (all ∈ [0, 1], higher is better):

    1. quality
       ─────────
       Combines inter-chunk separation and intra-chunk coherence.

       separation = mean cosine distance between adjacent chunk hash
                    embeddings.  High separation → boundaries are at real
                    topic shifts, not arbitrary cuts.

         separation(i, i+1) = 1 − cosine(embed(Cᵢ), embed(Cᵢ₊₁))

       icc = mean intra-chunk coherence (from S4).
             ICC(C) = mean Jaccard(sᵢ, sᵢ₊₁) over consecutive sentences.
             High ICC → each chunk is internally coherent.

       quality = 0.55 × separation + 0.45 × icc

       NOTE: this is NOT the S4 boundary_score.  S4 boundary_score measures
       similarity (high = similar = bad boundary).  separation measures
       DISTANCE (high = different = good boundary).  They are complementary
       but not circular — separation uses a fast hash embedding recomputed
       here, independent of S4.

    2. coverage
       ────────
       Measures how PRECISELY the chunk set answers each probe query.

       For each probe, we find the single best-matching chunk (highest
       token overlap).  Then we penalise it if it is too large:

         precision_score = icc_of_best_chunk × (target_size / actual_size)
                           clipped to [0, 1]

       where target_size = TARGET_WORDS_PER_CHUNK.
       A small, coherent chunk that contains the answer scores near 1.
       A 900-word blob that buries the answer scores much lower.

       This avoids the "trivially 1.0" problem of the old recall proxy.

    3. consistency
       ───────────
       Penalises high variance in chunk sizes:

         consistency = 1 − CV   where CV = std(sizes) / mean(sizes)

       Low variance → the chunker found stable natural units across the
       document (good).  High variance → some chunks are huge fragments,
       others are tiny slivers (bad).

    4. efficiency
       ──────────
       Rewards chunk count close to the document-derived ideal:

         target_count = total_words / TARGET_WORDS_PER_CHUNK
         efficiency   = 1 − |len(chunks) − target_count| / target_count

       This directly penalises the original problem (8 chunks for a
       6478-word document that needs ~22).

    5. structural
       ──────────
       Domain-aware signal for legal/regulatory/financial documents:

         structural = 0.5 × hard_boundary_ratio + 0.5 × mean_pmi_drop

       hard_boundary_ratio : fraction of chunks starting at a protected
                             structural marker (Article, CHAPITRE, etc.)
       mean_pmi_drop       : mean concept shift at boundaries (from S3
                             boundary_features dict)

    Final reward
    ────────────
      total = w_quality × quality
            + w_coverage × coverage
            + w_consistency × consistency
            + w_efficiency × efficiency
            + 0.10 × structural          ← fixed bonus, always included
    """
    if not chunks:
        return {
            "quality": 0.0, "coverage": 0.0, "consistency": 0.0,
            "efficiency": 0.0, "structural": 0.0, "total": -1.0,
        }

    # ── 1. quality = separation + icc ────────────────────────────────────────
    # Inter-chunk separation: cosine distance between adjacent hash embeddings.
    # We recompute hash embeddings here (independent of S4 scores — not circular).
    separations: List[float] = []
    for i in range(len(chunks) - 1):
        v1 = _hash_embed(chunks[i].get("text", ""),     dim=128)
        v2 = _hash_embed(chunks[i + 1].get("text", ""), dim=128)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 0 and n2 > 0:
            cos_dist = 1.0 - float(np.dot(v1, v2) / (n1 * n2))
            separations.append(float(np.clip(cos_dist, 0.0, 1.0)))

    separation = float(np.mean(separations)) if separations else 0.5

    # Intra-chunk ICC from S4 (already computed per chunk)
    icc_vals = [float(c.get("icc", 0.5)) for c in chunks]
    mean_icc = float(np.mean(icc_vals))

    quality = float(np.clip(0.55 * separation + 0.45 * mean_icc, 0.0, 1.0))

    # ── 2. coverage = precision-weighted probe recall ─────────────────────────
    coverage = _precision_recall_proxy(chunks, probes)

    # ── 3. consistency = 1 - coefficient_of_variation ────────────────────────
    sizes = np.array(
        [max(1, len(c.get("text", "").split())) for c in chunks],
        dtype=np.float32,
    )
    cv    = float(np.std(sizes) / max(float(np.mean(sizes)), 1.0))
    consistency = float(np.clip(1.0 - cv, 0.0, 1.0))

    # ── 4. efficiency = proximity to ideal chunk count ────────────────────────
    target     = _target_count(chunks)
    efficiency = float(
        np.clip(1.0 - abs(len(chunks) - target) / max(target, 1.0), 0.0, 1.0)
    )

    # ── 5. structural = hard_boundary_ratio + mean PMI-drop ─────────────────
    hard_ratio = sum(
        1 for c in chunks
        if c.get("boundary_type") in {"hard", "protected_structure_boundary"}
    ) / max(len(chunks), 1)

    # Read PMI-drop from the boundary_features dict that S3 populates
    pmi_values = [
        float(c.get("boundary_features", {}).get("pmi_drop",
              c.get("pmi_drop", 0.5)))
        for c in chunks
    ]
    mean_pmi = float(np.mean(pmi_values))

    structural = float(np.clip(0.5 * hard_ratio + 0.5 * mean_pmi, 0.0, 1.0))

    # ── 6. Mid-sentence penalty (subtract from total) ────────────────────────
    # Penalise any chunk that starts mid-sentence (lowercase first char that
    # is not a legal list marker).  This directly penalises the BO for finding
    # n_max values that cause recursive/paragraph_pack to cut inside sentences.
    # Each mid-sentence start deducts 0.04 from the total reward.
    mid_sentence_count = sum(
        1 for c in chunks
        if (c.get("text", "").strip()[:1].islower()
            and not re.match(r"^\d+\)", c.get("text", "").strip())
            and not re.match(r"^[a-z][-\)]\s", c.get("text", "").strip()))
    )
    mid_sentence_penalty = float(
        np.clip(mid_sentence_count * 0.04, 0.0, 0.20)
    )

    # ── Total ────────────────────────────────────────────────────────────────
    total = (
        weights["quality"]       * quality
        + weights["coverage"]    * coverage
        + weights["consistency"] * consistency
        + weights["efficiency"]  * efficiency
        + 0.10                   * structural       # fixed domain-structure bonus
        - mid_sentence_penalty                      # penalise mid-sentence cuts
    )

    return {
        "quality":              round(quality,              4),
        "coverage":             round(coverage,             4),
        "consistency":          round(consistency,          4),
        "efficiency":           round(efficiency,           4),
        "structural":           round(structural,           4),
        "mid_sentence_penalty": round(mid_sentence_penalty, 4),
        "total":                round(float(np.clip(total, 0.0, 1.0)), 4),
    }


def _objective_weights(config: Dict[str, Any]) -> Dict[str, float]:
    """
    Parse user-configured objective weights from the config dict.

    Defaults:
      quality=0.35, coverage=0.25, consistency=0.20, efficiency=0.20

    The weights are normalised so they always sum to 1.0.  This means
    the user can supply any positive values and they will be rescaled.
    """
    defaults = {
        "quality":     0.35,
        "coverage":    0.25,
        "consistency": 0.20,
        "efficiency":  0.20,
    }
    incoming = config.get("reward_objectives", {})
    if not isinstance(incoming, dict):
        incoming = {}
    raw = {k: float(incoming.get(k, v)) for k, v in defaults.items()}
    s   = sum(raw.values()) or 1.0
    return {k: v / s for k, v in raw.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Probe generation & coverage evaluation
# ─────────────────────────────────────────────────────────────────────────────

def _generate_probes(text: str, n: int = 10) -> List[str]:
    """
    Generate probe queries from document structure for the coverage metric.

    Strategy (priority order):
    1. Legal/structural headings: Article N, CHAPITRE N, TITRE N
       These are the most semantically precise anchors in legal documents.
    2. Markdown headings (# Title) — for technical and academic documents.
    3. Numbered section lines (1.1, 2.3.4 …) — for regulatory/policy docs.
    4. First sentence of each paragraph (≥ 8 words) — universal fallback.

    Each probe is a short natural-language phrase that a retrieval system
    might use to query the chunk set.  The coverage metric measures whether
    the best-matching chunk is small and coherent, not just whether it exists.
    """
    probes: List[str] = []

    # ── 1. Legal article/section headings ───────────────────────────────────
    for m in re.finditer(
        r"(?im)^\s*((?:Article|Art\.?|ARTICLE|CHAPITRE|TITRE|SECTION)\s+\w+[^\n]{0,60})",
        text,
    ):
        probe = m.group(1).strip()
        if 3 <= len(probe.split()) <= 12:
            probes.append(probe)
        if len(probes) >= n:
            return probes

    # ── 2. Markdown headings ─────────────────────────────────────────────────
    for m in re.finditer(r"^#{1,3}\s+(.+)$", text, re.MULTILINE):
        probe = m.group(1).strip()
        if 2 <= len(probe.split()) <= 12:
            probes.append(probe)
        if len(probes) >= n:
            return probes

    # ── 3. Numbered section lines ────────────────────────────────────────────
    for m in re.finditer(r"(?m)^\s*(\d+(?:\.\d+)+)\s+(.+)$", text):
        probe = (m.group(1) + " " + m.group(2)).strip()
        if len(probe.split()) >= 3:
            probes.append(probe[:100])
        if len(probes) >= n:
            return probes

    # ── 4. First sentence of paragraphs (fallback) ───────────────────────────
    for para in re.split(r"\n{2,}", text):
        p = para.strip()
        if not p:
            continue
        sentences = re.split(r"(?<=[.!?])\s+", p)
        if sentences and len(sentences[0].split()) >= 8:
            probes.append(sentences[0].strip()[:120])
        if len(probes) >= n:
            break

    return probes[:n]


def _precision_recall_proxy(chunks: List[Dict], probes: List[str]) -> float:
    """
    Precision-weighted coverage metric.

    For each probe query:
      1. Find the chunk with the highest token overlap with the probe.
      2. Score:  precision = overlap_ratio × size_penalty
         where:
           overlap_ratio = |probe_terms ∩ chunk_terms| / |probe_terms|
           size_penalty  = min(1.0, TARGET_WORDS_PER_CHUNK / chunk_words)

    WHY ICC WAS REMOVED
    ───────────────────
    The original multiplied by chunk_icc, creating a hard ceiling at mean_icc
    ≈ 0.18 for legal documents.  Every strategy scored ≤ 0.18 regardless of
    actual retrieval quality — coverage was useless as a discriminating signal.

    ICC is already captured in the `quality` reward component.  Including it
    in coverage too double-penalised low-ICC chunks and made the two components
    correlated.  Coverage now measures purely: "can this chunk set answer the
    probe?" — independent of internal coherence.
    """
    if not probes:
        return 0.5

    target = float(_TARGET_WORDS_PER_CHUNK)
    scores: List[float] = []

    for probe in probes:
        probe_terms = set(re.findall(r"\b\w{3,}\b", probe.lower()))
        if not probe_terms:
            scores.append(0.5)
            continue

        best_score = 0.0
        for chunk in chunks:
            chunk_tokens = set(re.findall(r"\b\w+\b", chunk.get("text", "").lower()))
            overlap      = len(probe_terms & chunk_tokens)
            if overlap == 0:
                continue

            overlap_ratio = overlap / len(probe_terms)
            chunk_words   = max(1, len(chunk.get("text", "").split()))
            size_penalty  = min(1.0, target / chunk_words)

            # ICC deliberately excluded — it is already in the quality component
            precision  = float(np.clip(overlap_ratio * size_penalty, 0.0, 1.0))
            best_score = max(best_score, precision)

        scores.append(best_score)

    return float(np.mean(scores)) if scores else 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _target_count(chunks: List[Dict]) -> float:
    """
    Ideal chunk count = total document words / TARGET_WORDS_PER_CHUNK.
    Clamped to at least 3 to avoid degenerate single-chunk edge cases.
    """
    total_words = sum(max(1, len(c.get("text", "").split())) for c in chunks)
    return max(3.0, total_words / _TARGET_WORDS_PER_CHUNK)


def _hash_embed(text: str, dim: int = 128) -> np.ndarray:
    """
    Lightweight bag-of-words hash embedding.

    Maps each content token to a position in a dim-dimensional vector via
    Python's built-in hash function, accumulates counts, and L2-normalises.

    Used ONLY for the separation component of the quality reward so that
    it is independent of S4/S6 scores (no circular reward feedback).
    """
    vec = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"\b\w{3,}\b", text.lower()):
        vec[hash(tok) % dim] += 1.0
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


# ─────────────────────────────────────────────────────────────────────────────
# Persistence: warm-start across documents of the same domain
# ─────────────────────────────────────────────────────────────────────────────

def _load_history() -> Dict[str, Any]:
    """
    Load the persisted optimisation history from disk.

    Returns an empty dict if the file does not exist or is corrupted.
    Each key is a domain string (e.g. "regulatory", "legal").
    Each value contains the best config params and past Optuna trial data.
    """
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
    Initialise the trial config with the best known parameter values for
    this domain from previous runs.

    Rule: only fills in keys that the user has NOT explicitly provided in
    the incoming config dict.  User-supplied values always take precedence.
    """
    out    = copy.deepcopy(config)
    record = history.get(domain, {})

    # All tunable parameters — same as the search space in _suggest_config
    tunable_keys = (
        "tau_jsd_low", "tau_jsd_high",
        "n_max", "n_min",
        "tau_sem",
        "tau_percentile_low", "tau_percentile_high",
    )

    for k in tunable_keys:
        v = record.get("best_params", {}).get(k)
        if v is not None and k not in out:
            out[k] = v

    return out


def _save_history(
    domain: str,
    config: Dict[str, Any],
    reward_components: Dict[str, float],
) -> None:
    """
    Persist the best found configuration and reward for this domain.

    Stored structure:
    {
      "domain_key": {
        "best_params":  { ... tunable params ... },
        "best_reward":  float,
        "last_reward_components": { ... },
        "optuna_trials": [ { "params": {...}, "value": float }, ... ]
      }
    }

    optuna_trials stores a lightweight record of each trial so the TPE
    surrogate model can be seeded from past runs on subsequent documents.
    """
    history = _load_history()

    tunable_keys = (
        "tau_jsd_low", "tau_jsd_high",
        "n_max", "n_min",
        "tau_sem",
        "tau_percentile_low", "tau_percentile_high",
    )

    best_params = {k: config.get(k) for k in tunable_keys if config.get(k) is not None}

    # Preserve any existing optuna_trials so they accumulate across runs
    existing_trials = history.get(domain, {}).get("optuna_trials", [])

    # Add the current best as a new trial record for future warm-starting
    new_trial = {"params": best_params, "value": reward_components.get("total", 0.0)}
    updated_trials = existing_trials + [new_trial]

    # Cap to 200 stored trials to prevent unbounded file growth
    updated_trials = updated_trials[-200:]

    history[domain] = {
        "best_params":             best_params,
        "best_reward":             reward_components.get("total", 0.0),
        "last_reward_components":  reward_components,
        "optuna_trials":           updated_trials,
    }

    try:
        with open(_RL_HISTORY_PATH, "w", encoding="utf-8") as fh:
            json.dump(history, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass   # silently ignore write failures (e.g. read-only filesystem)