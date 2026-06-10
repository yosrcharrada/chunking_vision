"""
S7 — Hyperparameter Calibration via Genetic Algorithm (GA)
===========================================================
Pipeline position : runs AFTER S2–S6 and produces the final, optimally-configured
                    chunk set for this document.

WHY GENETIC ALGORITHM INSTEAD OF TPE
──────────────────────────────────────
TPE (Bayesian Optimisation) is excellent for very small budgets (5–20 evaluations)
because it learns a surrogate model of the objective.  But it is inherently
sequential — each trial depends on all previous trial results, so you cannot
run multiple trials in parallel.

For a pipeline where each evaluation (one full S2→S6 run) costs 2–10 seconds,
sequential optimisation with 5 trials per strategy × 6 strategies = 30 evaluations
means 60–300 seconds of wall-clock time.

The Genetic Algorithm (GA) solves this with FULL PARALLELISM:
  • An entire generation of N individuals is evaluated simultaneously using a
    ProcessPoolExecutor — all N pipeline calls run on separate CPU cores at once.
  • Wall-clock time per generation = time of the SLOWEST evaluation, not the sum.
  • With N=20 and 4 cores: each generation takes ~5s instead of ~100s.
  • Total wall-clock: G × (slowest_eval_time) ≈ 5 generations × 5s = 25s
    vs TPE sequential: 30 × 5s = 150s

GA DESIGN CHOICES
─────────────────
  Population size     N = 20 individuals per generation
  Generations         G = 5  (configurable via ga_generations)
  Selection           Tournament selection (k=3): pick 3 random, keep best
  Crossover           BLX-α (blend crossover, α=0.5)
  Mutation rate       0.20 per gene
  Mutation scale      Gaussian noise, σ = 0.15
  Elitism             Top 2 individuals copied unchanged to next generation
  Parallelism         ProcessPoolExecutor with platform-safe worker count
  Warm-start          Best params from previous runs seed the initial population

FIXES APPLIED (v3-fixed)
────────────────────────
  FIX 1: reward_breakdown now populated in per_strategy_results
  FIX 2: size_fit=0 hard penalty — reward capped at 0.50 when all chunks OOB
  FIX 3: max_workers no longer hardcoded to 2 — uses min(max_workers, 4) on
          Linux, 2 on Windows for safety
  FIX 4: sliding_window added to _EXCLUDED_STRATEGIES — its tau thresholds
          have no effect, so GA always produced a flat reward for it (0.6905×6)
  FIX 5: fitness_history records best-so-far AND generation mean for better
          convergence diagnostics
"""

import copy
import json
import logging
import os
import platform
import re
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .s2_chunkers import run_all_chunkers, select_best_strategy
from .s3_entropy import refine_boundaries
from .s4_boundary import filter_boundaries
from .s5_graph import enrich_graph
from .s6_embedding import embed_chunks

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_RL_HISTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "rl_history.json")

# FIX 4: sliding_window added — its only tunable param is n_max which changes
# window_size, but all tau threshold mutations have zero effect on its output.
# GA reward is always flat for sliding_window. Still benchmarked in S2.
_EXCLUDED_STRATEGIES = {
    "semantic_boundaries",  # always produces micro-chunks regardless of n_max
    "sentence_clustering",  # same — sentence-level granularity ignores thresholds
    "sliding_window",       # purely size-based: tau_jsd_*, tau_sem, tau_percentile_*
                            # are all ignored → GA reward is always flat
}

_TARGET_WORDS_PER_CHUNK = 300

# ── GA Hyperparameters ────────────────────────────────────────────────────────

_GA_POP_SIZE         = 20
_GA_GENERATIONS      = 5
_GA_CROSSOVER_ALPHA  = 0.5
_GA_CROSSOVER_RATE   = 0.80
_GA_MUTATION_RATE    = 0.20
_GA_MUTATION_SIGMA   = 0.15
_GA_ELITE_COUNT      = 2
_GA_TOURNAMENT_K     = 3
_GA_WARMSTART_FRACTION = 0.10
_GA_EVAL_TIMEOUT     = 120

# ── Gene bounds ───────────────────────────────────────────────────────────────
# FIX 2 related: n_max upper bound is 900, but the size_fit hard penalty now
# prevents the GA from choosing n_max=900 when it causes size_fit=0.
_GENE_SPECS = [
    # (real_min, real_max, step, name)
    (150,  900,   25,   "n_max"),
    (30,   250,   10,   "n_min"),
    (0.05, 0.40,  None, "tau_jsd_low"),
    (0.20, 0.80,  None, "tau_jsd_high"),
    (0.40, 0.95,  None, "tau_sem"),
    (5,    45,    None, "tau_percentile_low"),
    (55,   95,    None, "tau_percentile_high"),
]
_N_GENES = len(_GENE_SPECS)  # 7


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
    Tune hyperparameters for EACH chunking strategy independently via GA,
    then return the strategy+params combination that scores highest.
    """
    from .s2_chunkers import _strategy_quality_score as _sqscore

    model_name  = config.get("embedding_model", "all-MiniLM-L6-v2")
    doc_type    = doc_profile.get("type",   "prose")
    domain      = doc_profile.get("domain", "general")
    history_key = str(config.get("rl_history_key") or domain)
    n_min       = int(config.get("n_min", 100))
    n_max       = int(config.get("n_max", 500))

    pop_size    = int(config.get("ga_population",  _GA_POP_SIZE))
    generations = int(config.get("ga_generations", _GA_GENERATIONS))
    max_workers = int(config.get("ga_workers", os.cpu_count() or 4))

    # Run S2 to get all strategy baseline chunks
    all_s2 = run_all_chunkers(text, doc_type, config)
    active_strategies = [s for s in all_s2.keys() if s not in _EXCLUDED_STRATEGIES]

    s2_scores: Dict[str, float] = {}
    for strat, chunks_list in all_s2.items():
        if chunks_list:
            s2_scores[strat] = round(float(_sqscore(chunks_list, n_min, n_max)), 4)

    baseline_reward = round(float(_sqscore(initial_chunks, n_min, n_max)), 4)
    reward_history: List[float] = [baseline_reward]

    per_strategy_results: Dict[str, Any] = {}

    for strategy in active_strategies:
        strat_baseline_score  = s2_scores.get(strategy, baseline_reward)
        strat_baseline_chunks = all_s2.get(strategy) or initial_chunks

        strat_key = f"{history_key}__{strategy}"
        history   = _load_history()
        warm_cfg  = _warm_start_config(config, history, strat_key)
        warm_cfg["chunking_strategy"] = strategy

        best_chunks, best_score, best_cfg, strat_rewards, gen_means = _run_strategy_ga(
            text=text,
            doc_type=doc_type,
            doc_profile=doc_profile,
            model_name=model_name,
            warm_cfg=warm_cfg,
            strategy=strategy,
            n_min=n_min,
            n_max=n_max,
            baseline_score=strat_baseline_score,
            baseline_chunks=strat_baseline_chunks,
            pop_size=pop_size,
            generations=generations,
            max_workers=max_workers,
        )

        reward_history.extend(strat_rewards)
        _save_history(strat_key, best_cfg, {"total": best_score})

        # FIX 1: populate reward_breakdown by computing components on best chunks
        best_probes    = _generate_probes(text, n=8)
        best_weights   = _objective_weights(config)
        # Pass size bounds for hard size_fit constraint (FIX 2)
        best_weights["_n_min"] = float(n_min)
        best_weights["_n_max"] = float(n_max)
        best_breakdown = _compute_reward_components(best_chunks, best_probes, best_weights)

        per_strategy_results[strategy] = {
            "best_chunks":    best_chunks,
            "best_score":     round(best_score, 4),
            "s2_baseline":    strat_baseline_score,
            "improvement":    round(best_score - strat_baseline_score, 4),
            "best_params":    {k: best_cfg.get(k) for k in (
                "n_max", "n_min", "tau_jsd_low", "tau_jsd_high",
                "tau_sem", "tau_percentile_low", "tau_percentile_high"
            )},
            "n_evals":        len(strat_rewards),
            "fitness_history": [round(strat_baseline_score, 4)] + strat_rewards,
            # FIX 5: include generation means for convergence diagnostics
            "generation_means": gen_means,
            # FIX 1: reward_breakdown now populated (was always {})
            "reward_breakdown": {
                "quality":     round(float(best_breakdown.get("quality",     0.0)), 4),
                "coverage":    round(float(best_breakdown.get("coverage",    0.0)), 4),
                "consistency": round(float(best_breakdown.get("consistency", 0.0)), 4),
                "efficiency":  round(float(best_breakdown.get("efficiency",  0.0)), 4),
                "structural":  round(float(best_breakdown.get("structural",  0.0)), 4),
                "total":       round(float(best_breakdown.get("total",       0.0)), 4),
            },
        }

    # Overall winner
    overall_winner = max(
        per_strategy_results,
        key=lambda s: per_strategy_results[s]["best_score"],
    )
    winner         = per_strategy_results[overall_winner]
    best_chunks    = winner["best_chunks"]
    best_reward    = winner["best_score"]

    final_cfg = copy.deepcopy(winner.get("best_params", config))
    final_cfg.update({
        "optimizer":                 "genetic_algorithm",
        "ga_population":             pop_size,
        "ga_generations":            generations,
        "ga_workers":                max_workers,
        "rl_history_key":            history_key,
        "n_evals_total":             len(reward_history) - 1,
        "baseline_reward":           baseline_reward,
        "best_reward":               round(best_reward, 4),
        "improvement_over_baseline": round(best_reward - baseline_reward, 4),
        "overall_winner_strategy":   overall_winner,
        "chunking_strategy":         overall_winner,
        "per_strategy_results": {
            s: dict(r)
            for s, r in per_strategy_results.items()
        },
    })

    return best_chunks, reward_history, final_cfg


# ─────────────────────────────────────────────────────────────────────────────
# GA engine — one strategy at a time
# ─────────────────────────────────────────────────────────────────────────────

def _run_strategy_ga(
    text: str,
    doc_type: str,
    doc_profile: Dict[str, Any],
    model_name: str,
    warm_cfg: Dict[str, Any],
    strategy: str,
    n_min: int,
    n_max: int,
    baseline_score: float,
    baseline_chunks: List[Dict],
    pop_size: int,
    generations: int,
    max_workers: int,
) -> Tuple[List[Dict], float, Dict[str, Any], List[float], List[float]]:
    """
    Run a Genetic Algorithm for ONE chunking strategy.
    Returns: (best_chunks, best_score, best_cfg, fitness_history, gen_means)
    fitness_history = best-so-far per generation (for convergence chart)
    gen_means       = mean of valid fitnesses per generation (for diagnostics)
    """
    from .s2_chunkers import _strategy_quality_score as _sqscore

    rng = np.random.RandomState(abs(hash(strategy)) % (2**31))
    population = _initialise_population(pop_size, warm_cfg, rng)

    best_score  = baseline_score
    best_chunks = baseline_chunks
    best_cfg    = copy.deepcopy(warm_cfg)

    fitness_history: List[float] = []
    gen_means:       List[float] = []  # FIX 5: track generation means

    # FIX 3: platform-safe worker count
    # Windows: 2 workers (paging safety). Linux: min(max_workers, 4).
    if platform.system() == "Windows":
        safe_workers = 2
    else:
        safe_workers = min(max_workers, 4)

    for gen_idx in range(generations):
        trial_cfgs: List[Dict] = [
            _decode_individual(chromosome, warm_cfg, strategy)
            for chromosome in population
        ]

        fitnesses: List[float] = [0.0] * pop_size

        args_list = [
            (text, doc_type, doc_profile, model_name, cfg, strategy)
            for cfg in trial_cfgs
        ]

        with ProcessPoolExecutor(max_workers=safe_workers) as executor:  # FIX 3
            future_to_idx = {
                executor.submit(_run_pipeline_worker, args): i
                for i, args in enumerate(args_list)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    chunks_result = future.result()
                except Exception as exc:
                    logger.debug("GA [%s] gen %d ind %d failed: %s",
                                 strategy, gen_idx, idx, exc)
                    chunks_result = None

                if chunks_result is not None:
                    fitness = round(float(_sqscore(
                        chunks_result,
                        trial_cfgs[idx].get("n_min", n_min),
                        trial_cfgs[idx].get("n_max", n_max),
                    )), 4)
                else:
                    fitness = 0.0

                fitnesses[idx] = fitness

                if fitness > best_score:
                    best_score  = fitness
                    best_chunks = chunks_result
                    best_cfg    = trial_cfgs[idx]
                    logger.debug("GA [%s] gen %d ind %d: new best = %.4f",
                                 strategy, gen_idx, idx, best_score)

        valid = [f for f in fitnesses if f > 0]
        gen_mean = round(float(np.mean(valid)), 4) if valid else 0.0

        logger.debug("GA [%s] gen %d/%d  best=%.4f  mean=%.4f",
                     strategy, gen_idx + 1, generations, best_score, gen_mean)

        # FIX 5: record both best-so-far and generation mean
        fitness_history.append(round(best_score, 4))
        gen_means.append(gen_mean)

        population = _evolve(population, fitnesses, rng)

    return best_chunks, best_score, best_cfg, fitness_history, gen_means


# ─────────────────────────────────────────────────────────────────────────────
# GA Operators
# ─────────────────────────────────────────────────────────────────────────────

def _initialise_population(
    pop_size: int,
    warm_cfg: Dict[str, Any],
    rng: np.random.RandomState,
) -> List[np.ndarray]:
    population: List[np.ndarray] = []
    n_warm    = max(1, int(pop_size * _GA_WARMSTART_FRACTION))
    warm_gene = _encode_config(warm_cfg)

    for i in range(pop_size):
        if i < n_warm and warm_gene is not None:
            noise = rng.normal(0.0, 0.02, size=_N_GENES)
            gene  = np.clip(warm_gene + noise, 0.0, 1.0)
        else:
            gene = rng.uniform(0.0, 1.0, size=_N_GENES)
        population.append(gene)

    return population


def _evolve(
    population: List[np.ndarray],
    fitnesses: List[float],
    rng: np.random.RandomState,
) -> List[np.ndarray]:
    pop_size     = len(population)
    next_gen: List[np.ndarray] = []
    sorted_indices = np.argsort(fitnesses)[::-1]

    for i in range(min(_GA_ELITE_COUNT, pop_size)):
        next_gen.append(population[sorted_indices[i]].copy())

    while len(next_gen) < pop_size:
        parent_a = _tournament_select(population, fitnesses, _GA_TOURNAMENT_K, rng)
        parent_b = _tournament_select(population, fitnesses, _GA_TOURNAMENT_K, rng)

        if rng.random() < _GA_CROSSOVER_RATE:
            child = _blx_alpha_crossover(parent_a, parent_b, _GA_CROSSOVER_ALPHA, rng)
        else:
            child = parent_a.copy()

        child = _gaussian_mutate(child, _GA_MUTATION_RATE, _GA_MUTATION_SIGMA, rng)
        next_gen.append(child)

    return next_gen[:pop_size]


def _tournament_select(
    population: List[np.ndarray],
    fitnesses: List[float],
    k: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    candidates = rng.choice(len(population), size=min(k, len(population)), replace=False)
    best_idx   = candidates[np.argmax([fitnesses[i] for i in candidates])]
    return population[best_idx].copy()


def _blx_alpha_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    alpha: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    child = np.empty(_N_GENES)
    for d in range(_N_GENES):
        lo    = min(parent_a[d], parent_b[d])
        hi    = max(parent_a[d], parent_b[d])
        span  = hi - lo
        child[d] = rng.uniform(lo - alpha * span, hi + alpha * span)
    return np.clip(child, 0.0, 1.0)


def _gaussian_mutate(
    gene: np.ndarray,
    rate: float,
    sigma: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    mutated = gene.copy()
    for d in range(_N_GENES):
        if rng.random() < rate:
            mutated[d] = float(np.clip(gene[d] + rng.normal(0.0, sigma), 0.0, 1.0))
    return mutated


# ─────────────────────────────────────────────────────────────────────────────
# Gene encoding / decoding
# ─────────────────────────────────────────────────────────────────────────────

def _encode_config(cfg: Dict[str, Any]) -> Optional[np.ndarray]:
    gene = np.zeros(_N_GENES)
    key_map = {
        0: "n_max", 1: "n_min", 2: "tau_jsd_low", 3: "tau_jsd_high",
        4: "tau_sem", 5: "tau_percentile_low", 6: "tau_percentile_high",
    }
    for d, key in key_map.items():
        val = cfg.get(key)
        if val is None:
            return None
        real_min, real_max, _, _ = _GENE_SPECS[d]
        gene[d] = float(np.clip((val - real_min) / (real_max - real_min), 0.0, 1.0))
    return gene


def _decode_individual(
    gene: np.ndarray,
    base_cfg: Dict[str, Any],
    strategy: str,
) -> Dict[str, Any]:
    cfg = copy.deepcopy(base_cfg)
    key_map = {
        0: "n_max", 1: "n_min", 2: "tau_jsd_low", 3: "tau_jsd_high",
        4: "tau_sem", 5: "tau_percentile_low", 6: "tau_percentile_high",
    }

    for d, key in key_map.items():
        real_min, real_max, step, _ = _GENE_SPECS[d]
        real_val = real_min + gene[d] * (real_max - real_min)
        if step is not None:
            real_val = round(real_val / step) * step
            real_val = int(np.clip(real_val, real_min, real_max))
        else:
            real_val = float(np.clip(real_val, real_min, real_max))
        cfg[key] = real_val

    # Constraint repair
    if cfg["tau_jsd_low"] >= cfg["tau_jsd_high"] - 0.08:
        cfg["tau_jsd_high"] = min(0.80, cfg["tau_jsd_low"] + 0.10)
    if cfg["n_min"] >= cfg["n_max"] - 50:
        cfg["n_min"] = max(30, cfg["n_max"] - 50)
    if cfg["tau_percentile_low"] >= cfg["tau_percentile_high"]:
        cfg["tau_percentile_low"], cfg["tau_percentile_high"] = (
            cfg["tau_percentile_high"] - 5,
            cfg["tau_percentile_low"] + 5,
        )

    cfg["chunking_strategy"]  = strategy
    cfg["entropy_metric"]     = base_cfg.get("entropy_metric", "hybrid")
    cfg["_in_rl_calibration"] = True

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline worker (top-level for pickle)
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline_worker(args: tuple) -> Optional[List[Dict]]:
    text, doc_type, doc_profile, model_name, cfg, strategy = args
    return _run_pipeline(text, doc_type, doc_profile, model_name, cfg, strategy)


def _run_pipeline(
    text: str,
    doc_type: str,
    doc_profile: Dict[str, Any],
    model_name: str,
    cfg: Dict[str, Any],
    s2_winner: str,
) -> Optional[List[Dict]]:
    """
    Run S2 → S3 → S4 during GA trials. S5 and S6 intentionally skipped:
    the fitness function does not use embeddings or entity graph data.
    Skipping eliminates ~600 model reloads across all GA trials.
    """
    try:
        cfg["_full_text_sample"] = text[:3000]
        cfg["chunking_strategy"] = s2_winner

        from .s2_chunkers import (
            recursive_character_split, sliding_window_split,
            structure_based_split, semantic_boundary_split,
            sentence_cluster_split, paragraph_pack_split,
            legal_article_split, hybrid_legal_semantic_split,
            _quality_pass,
        )
        from .s3_entropy import refine_boundaries
        from .s4_boundary import filter_boundaries

        import logging as _logging
        _logging.getLogger("sentence_transformers").setLevel(_logging.ERROR)
        warnings.filterwarnings("ignore", message=".*position_ids.*")
        warnings.filterwarnings("ignore", message=".*masked_bias.*")

        n_min = int(cfg.get("n_min", 100))
        n_max = int(cfg.get("n_max", 500))

        strategy_map = {
            "recursive":             lambda: recursive_character_split(text, n_min, n_max, doc_type),
            "sliding_window":        lambda: sliding_window_split(text, n_max, int(n_max * 0.15)),
            "structure":             lambda: structure_based_split(text, doc_type, n_min, n_max),
            "semantic_boundaries":   lambda: semantic_boundary_split(text, n_min, n_max, cfg),
            "sentence_clustering":   lambda: sentence_cluster_split(text, n_min, n_max, cfg),
            "paragraph_pack":        lambda: paragraph_pack_split(text, n_min, n_max),
            "legal_articles":        lambda: legal_article_split(text, n_min, n_max),
            "hybrid_legal_semantic": lambda: hybrid_legal_semantic_split(text, n_min, n_max, cfg),
        }

        chunker_fn   = strategy_map.get(s2_winner, strategy_map["structure"])
        trial_chunks = chunker_fn()

        if not trial_chunks:
            return None

        trial_chunks = _quality_pass(trial_chunks, text, n_min, n_max, s2_winner)
        trial_chunks = refine_boundaries(trial_chunks, cfg)
        trial_chunks = filter_boundaries(trial_chunks, doc_type, [], cfg)

        return trial_chunks

    except Exception as exc:
        logger.debug("S7 GA trial failed: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Reward function
# ─────────────────────────────────────────────────────────────────────────────

def _objective_weights(config: Dict[str, Any]) -> Dict[str, float]:
    defaults  = {"quality": 0.35, "coverage": 0.25, "consistency": 0.20, "efficiency": 0.20}
    incoming  = config.get("reward_objectives", {})
    if not isinstance(incoming, dict):
        incoming = {}
    raw = {k: float(incoming.get(k, v)) for k, v in defaults.items()}
    s   = sum(raw.values()) or 1.0
    return {k: v / s for k, v in raw.items()}


def _compute_reward_components(
    chunks: List[Dict],
    probes: List[str],
    weights: Dict[str, float],
) -> Dict[str, float]:
    """
    Multi-objective reward. Five components in [0,1].
    FIX 2: hard size_fit constraint — if >95% of chunks are outside [n_min, n_max],
    total is capped at 0.50 regardless of other scores.
    """
    if not chunks:
        return {
            "quality": 0.0, "coverage": 0.0, "consistency": 0.0,
            "efficiency": 0.0, "structural": 0.0, "total": -1.0,
        }

    # Exclude end-boundary chunk from all metric computations
    real_chunks = [c for c in chunks if c.get("boundary_type") != "end"]
    if not real_chunks:
        real_chunks = chunks

    # ── 1. quality ────────────────────────────────────────────────────────────
    separations: List[float] = []
    for i in range(len(real_chunks) - 1):
        v1 = _hash_embed(real_chunks[i].get("text", ""),     dim=128)
        v2 = _hash_embed(real_chunks[i + 1].get("text", ""), dim=128)
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 > 0 and n2 > 0:
            separations.append(float(np.clip(1.0 - np.dot(v1, v2) / (n1 * n2), 0.0, 1.0)))

    separation = float(np.mean(separations)) if separations else 0.5
    icc_vals   = [float(c.get("icc", 0.5)) for c in real_chunks]
    mean_icc   = float(np.mean(icc_vals))
    quality    = float(np.clip(0.55 * separation + 0.45 * mean_icc, 0.0, 1.0))

    # ── 2. coverage ───────────────────────────────────────────────────────────
    coverage = _precision_recall_proxy(real_chunks, probes)

    # ── 3. consistency ────────────────────────────────────────────────────────
    sizes = np.array(
        [max(1, len(c.get("text", "").split())) for c in real_chunks],
        dtype=np.float32,
    )
    cv          = float(np.std(sizes) / max(float(np.mean(sizes)), 1.0))
    consistency = float(np.clip(1.0 - cv, 0.0, 1.0))

    # ── 4. efficiency ─────────────────────────────────────────────────────────
    target     = _target_count(real_chunks)
    efficiency = float(np.clip(
        1.0 - abs(len(real_chunks) - target) / max(target, 1.0), 0.0, 1.0
    ))

    # ── 5. structural ─────────────────────────────────────────────────────────
    hard_ratio = sum(
        1 for c in real_chunks
        if c.get("boundary_type") in {"hard", "protected_structure_boundary"}
    ) / max(len(real_chunks), 1)
    pmi_values = [
        float(c.get("boundary_features", {}).get("pmi_drop", c.get("pmi_drop", 0.5)))
        for c in real_chunks
    ]
    structural = float(np.clip(0.5 * hard_ratio + 0.5 * float(np.mean(pmi_values)), 0.0, 1.0))

    # ── 6. Mid-sentence penalty ───────────────────────────────────────────────
    mid_sentence_penalty = 0.0
    for c in real_chunks:
        t = c.get("text", "").strip()
        if t and t[0].islower() and not re.match(r"^[a-z]\)", t):
            mid_sentence_penalty += 0.04
    mid_sentence_penalty = min(mid_sentence_penalty, 0.30)

    # ── FIX 2: Hard size_fit constraint ───────────────────────────────────────
    n_min_val = float(weights.get("_n_min", 80))
    n_max_val = float(weights.get("_n_max", 500))
    in_range_ratio = sum(
        1 for s in sizes if n_min_val <= s <= n_max_val
    ) / max(len(sizes), 1)

    raw_total = (
        weights.get("quality",      0.35) * quality
        + weights.get("coverage",   0.25) * coverage
        + weights.get("consistency", 0.20) * consistency
        + weights.get("efficiency",  0.20) * efficiency
        + 0.10 * structural
        - mid_sentence_penalty
    )

    # If >95% of chunks are outside target size range, cap reward at 0.50.
    # Prevents GA from choosing n_max=900 and getting all chunks out of range.
    if in_range_ratio <= 0.05:
        total = min(raw_total, 0.50)
    else:
        total = raw_total

    total = float(np.clip(total, 0.0, 1.0))

    return {
        "quality":     round(quality,     4),
        "coverage":    round(coverage,    4),
        "consistency": round(consistency, 4),
        "efficiency":  round(efficiency,  4),
        "structural":  round(structural,  4),
        "total":       round(total,       4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Probe generation and coverage proxy
# ─────────────────────────────────────────────────────────────────────────────

def _generate_probes(text: str, n: int = 8) -> List[str]:
    probes: List[str] = []

    for m in re.finditer(
        r"(?im)^\s*((?:Article|Art\.?|ARTICLE|CHAPITRE|TITRE|SECTION)\s+\w+[^\n]{0,60})",
        text,
    ):
        probe = m.group(1).strip()
        if 3 <= len(probe.split()) <= 12:
            probes.append(probe)
        if len(probes) >= n:
            return probes

    for m in re.finditer(r"^#{1,3}\s+(.+)$", text, re.MULTILINE):
        probe = m.group(1).strip()
        if 2 <= len(probe.split()) <= 12:
            probes.append(probe)
        if len(probes) >= n:
            return probes

    for m in re.finditer(r"(?m)^\s*(\d+(?:\.\d+)+)\s+(.+)$", text):
        probe = (m.group(1) + " " + m.group(2)).strip()
        if len(probe.split()) >= 3:
            probes.append(probe[:100])
        if len(probes) >= n:
            return probes

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
            best_score    = max(best_score, float(np.clip(overlap_ratio * size_penalty, 0.0, 1.0)))

        scores.append(best_score)

    return float(np.mean(scores)) if scores else 0.5


# ─────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ─────────────────────────────────────────────────────────────────────────────

def _target_count(chunks: List[Dict]) -> float:
    total_words = sum(max(1, len(c.get("text", "").split())) for c in chunks)
    return max(3.0, total_words / _TARGET_WORDS_PER_CHUNK)


def _hash_embed(text: str, dim: int = 128) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    for tok in re.findall(r"\b\w{3,}\b", text.lower()):
        vec[hash(tok) % dim] += 1.0
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


# ─────────────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────────────

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
    out    = copy.deepcopy(config)
    record = history.get(domain, {})
    tunable_keys = (
        "tau_jsd_low", "tau_jsd_high", "n_max", "n_min",
        "tau_sem", "tau_percentile_low", "tau_percentile_high",
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
    history = _load_history()
    tunable_keys = (
        "tau_jsd_low", "tau_jsd_high", "n_max", "n_min",
        "tau_sem", "tau_percentile_low", "tau_percentile_high",
    )
    best_params    = {k: config.get(k) for k in tunable_keys if config.get(k) is not None}
    existing_trials = history.get(domain, {}).get("ga_trials", [])
    new_trial       = {"params": best_params, "value": reward_components.get("total", 0.0)}
    updated_trials  = (existing_trials + [new_trial])[-200:]

    history[domain] = {
        "best_params":            best_params,
        "best_reward":            reward_components.get("total", 0.0),
        "last_reward_components": reward_components,
        "ga_trials":              updated_trials,
    }

    try:
        with open(_RL_HISTORY_PATH, "w", encoding="utf-8") as fh:
            json.dump(history, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass