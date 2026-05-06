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

GA is also better at escaping local optima than TPE because:
  • Crossover combines GOOD parts from two different configurations — it can
    discover that (n_max=375 from parent A) + (τ_low=0.12 from parent B) works
    better than either parent alone.
  • Mutation perturbs every gene independently — it explores the search space
    breadth-first across the full population simultaneously.
  • Elitism guarantees the best solution found is never lost.

GA DESIGN CHOICES
─────────────────
  Population size     N = 20 individuals per generation
  Generations         G = 5  (configurable via ga_generations)
  Selection           Tournament selection (k=3): pick 3 random, keep best
  Crossover           BLX-α (blend crossover, α=0.5): child genes can fall
                      slightly outside the parent range → better exploration
  Mutation rate       0.20 per gene — higher than typical (0.1) because we
                      have few generations; need fast diversity generation
  Mutation scale      Gaussian noise, σ = 0.15 × (gene_max - gene_min)
  Elitism             Top 2 individuals copied unchanged to next generation
  Parallelism         ProcessPoolExecutor with max_workers = os.cpu_count()
  Warm-start          Best params from previous runs seed the initial population
                      (10% of population = warm-started, 90% = random)

SEARCH SPACE (7 parameters, same as TPE version)
──────────────────────────────────────────────────
  n_max                ∈ [150, 900]   — max words per chunk (S2)
  n_min                ∈ [30,  250]   — min words per chunk (S2)
  tau_jsd_low          ∈ [0.05, 0.40] — S3 merge threshold
  tau_jsd_high         ∈ [0.20, 0.80] — S3 hard-split threshold
  tau_sem              ∈ [0.40, 0.95] — S4 similarity merge threshold
  tau_percentile_low   ∈ [5,   45]    — S3 adaptive percentile (low)
  tau_percentile_high  ∈ [55,  95]    — S3 adaptive percentile (high)

GENE ENCODING
─────────────
  Each individual is a numpy array of 7 floats in [0, 1] (normalised).
  Decoding maps back to the real parameter range:
    gene[0] → n_max      = 150 + gene[0] × 750    (rounded to nearest 25)
    gene[1] → n_min      = 30  + gene[1] × 220    (rounded to nearest 10)
    gene[2] → τ_low      = 0.05 + gene[2] × 0.35
    gene[3] → τ_high     = 0.20 + gene[3] × 0.60
    gene[4] → τ_sem      = 0.40 + gene[4] × 0.55
    gene[5] → p_low      = 5   + gene[5] × 40
    gene[6] → p_high     = 55  + gene[6] × 40

  Normalised encoding makes crossover and mutation scale-invariant — a mutation
  of σ=0.15 in [0,1] space is proportionally the same for every gene regardless
  of its real-world scale (n_max ranges 750 words, τ_low ranges only 0.35).

UNIFIED SCORING
───────────────
  Every fitness evaluation uses _strategy_quality_score — the same function S2
  uses for its benchmark.  A score of 0.921 in GA genuinely means the same as
  0.899 in S2: the GA found better hyperparameters, not a different metric.

PERSISTENCE & WARM-START
────────────────────────
  After each run, best params are saved to rl_history.json under
  "{domain}__{strategy}".  The next document of the same domain seeds
  10% of the initial population with these known-good params.
"""

import copy
import json
import logging
import os
import re
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FuturesTimeout
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .s2_chunkers import run_all_chunkers, select_best_strategy
from .s3_entropy import refine_boundaries
from .s4_boundary import filter_boundaries
from .s5_graph import enrich_graph
from .s6_embedding import embed_chunks

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Where we persist per-strategy best params for warm-start
_RL_HISTORY_PATH = os.path.join(os.path.dirname(__file__), "..", "rl_history.json")

# Strategies that always produce micro-chunks regardless of n_max —
# GA cannot improve them; they are benchmarked in S2 at native params only.
_EXCLUDED_STRATEGIES = {"semantic_boundaries", "sentence_clustering"}

# ── GA Hyperparameters ────────────────────────────────────────────────────────

# Number of individuals in the population.
# Larger = more diversity, slower per generation. 20 is a good balance.
_GA_POP_SIZE = 20

# Number of generations to evolve.
# With parallelism, each generation costs ~slowest_eval seconds, not N × eval.
_GA_GENERATIONS = 5

# BLX-α crossover parameter.
# α=0.5 means children can extend 50% beyond the parents' range → good exploration.
_GA_CROSSOVER_ALPHA = 0.5

# Crossover probability: fraction of the population that undergoes crossover.
# The rest are reproduced unchanged (after selection).
_GA_CROSSOVER_RATE = 0.80

# Per-gene mutation probability: each gene mutates independently.
# 0.20 is higher than classical GA (0.05–0.10) because we have few generations.
_GA_MUTATION_RATE = 0.20

# Gaussian mutation noise scale (fraction of gene's normalised range [0,1]).
_GA_MUTATION_SIGMA = 0.15

# Number of elites copied unchanged to next generation.
_GA_ELITE_COUNT = 2

# Tournament size for selection.
# k=3: pick 3 random individuals, the fittest wins. Higher k = more selection pressure.
_GA_TOURNAMENT_K = 3

# Fraction of the initial population seeded from warm-start params.
# 0.10 = 2 individuals out of 20 come from previous best-known params.
_GA_WARMSTART_FRACTION = 0.10

# Timeout per individual pipeline evaluation in seconds.
# If a trial hangs (rare), it is skipped and scored as 0.0.
_GA_EVAL_TIMEOUT = 120

# ── Gene bounds (all in normalised [0,1] space) ──────────────────────────────
# Each gene maps linearly from [0,1] to its real parameter range.
# Stored as (real_min, real_max, snap_to_step) tuples.
_GENE_SPECS = [
    # (real_min, real_max,  step,  name)
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
    Tune hyperparameters for EACH chunking strategy independently via a
    Genetic Algorithm with full parallel fitness evaluation, then return
    the strategy+params combination that scores highest.

    Architecture
    ────────────
    For each active strategy S (excluding micro-chunk strategies):
      1. Build an initial population of N=20 parameter vectors, seeding
         warm-start individuals from rl_history.json.
      2. Evaluate the entire population in PARALLEL using ProcessPoolExecutor.
      3. Evolve for G=5 generations using tournament selection, BLX-α
         crossover, Gaussian mutation, and elitism.
      4. Record the best (score, params, chunks) for strategy S.

    Overall winner = argmax over all strategies of their best GA score.

    All fitness evaluations use _strategy_quality_score — the same function
    S2 uses — so scores are directly comparable throughout the pipeline.

    Parameters
    ──────────
    text          : raw document text
    doc_profile   : output of S1 (domain, type, metrics)
    initial_chunks: S2 winner chunks — used as the global baseline
    config        : pipeline config dict

    Returns
    ───────
    best_chunks    : chunk set from the winning strategy+params combination
    reward_history : flat list of all fitness evaluations (all strategies)
    final_config   : diagnostics including per_strategy_results
    """
    from .s2_chunkers import _strategy_quality_score as _sqscore

    model_name  = config.get("embedding_model", "all-MiniLM-L6-v2")
    doc_type    = doc_profile.get("type",   "prose")
    domain      = doc_profile.get("domain", "general")
    history_key = str(config.get("rl_history_key") or domain)
    n_min       = int(config.get("n_min", 100))
    n_max       = int(config.get("n_max", 500))

    # Read GA settings from config (allow caller overrides)
    pop_size    = int(config.get("ga_population", _GA_POP_SIZE))
    generations = int(config.get("ga_generations", _GA_GENERATIONS))
    max_workers = int(config.get("ga_workers", min(4, os.cpu_count() or 2)))

    # ── Run S2 to get all strategy baseline chunks ────────────────────────────
    all_s2 = run_all_chunkers(text, doc_type, config)
    active_strategies = [s for s in all_s2.keys() if s not in _EXCLUDED_STRATEGIES]

    # S2 baseline scores (default params) — used as the floor for each strategy
    s2_scores: Dict[str, float] = {}
    for strat, chunks_list in all_s2.items():
        if chunks_list:
            s2_scores[strat] = round(float(_sqscore(chunks_list, n_min, n_max)), 4)

    baseline_reward = round(float(_sqscore(initial_chunks, n_min, n_max)), 4)
    reward_history: List[float] = [baseline_reward]

    # ── Per-strategy GA ───────────────────────────────────────────────────────
    per_strategy_results: Dict[str, Any] = {}

    for strategy in active_strategies:
        strat_baseline_score  = s2_scores.get(strategy, baseline_reward)
        strat_baseline_chunks = all_s2.get(strategy) or initial_chunks

        strat_key = f"{history_key}__{strategy}"
        history   = _load_history()
        warm_cfg  = _warm_start_config(config, history, strat_key)
        warm_cfg["chunking_strategy"] = strategy

        best_chunks, best_score, best_cfg, strat_rewards = _run_strategy_ga(
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

        # Persist best params for this strategy+domain combination
        _save_history(strat_key, best_cfg, {"total": best_score})

        per_strategy_results[strategy] = {
            "best_chunks": best_chunks,
            "best_score":  round(best_score, 4),
            "s2_baseline": strat_baseline_score,
            "improvement": round(best_score - strat_baseline_score, 4),
            "best_params": {k: best_cfg.get(k) for k in (
                "n_max", "n_min", "tau_jsd_low", "tau_jsd_high",
                "tau_sem", "tau_percentile_low", "tau_percentile_high"
            )},
            "n_evals": len(strat_rewards),
        }

    # ── Overall winner ────────────────────────────────────────────────────────
    overall_winner = max(
        per_strategy_results,
        key=lambda s: per_strategy_results[s]["best_score"],
    )
    winner         = per_strategy_results[overall_winner]
    best_chunks    = winner["best_chunks"]
    best_reward    = winner["best_score"]

    # ── Final config ──────────────────────────────────────────────────────────
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
            s: {k: v for k, v in r.items() if k != "best_chunks"}
            for s, r in per_strategy_results.items()
        },
    })

    return best_chunks, reward_history, final_cfg


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  GENETIC ALGORITHM ENGINE
#  ─────────────────────────────────────────────────────────────────────────────
#
#  MULTITHREADING vs MULTIPROCESSING — why we use MULTIPROCESSING here
#  ─────────────────────────────────────────────────────────────────────
#  Multithreading (threading.Thread) and Multiprocessing (ProcessPoolExecutor)
#  are both ways to run code concurrently, but they are NOT equivalent in Python:
#
#  • Python has a Global Interpreter Lock (GIL): only ONE thread can execute
#    Python bytecode at a time. Threads are blocked waiting for the GIL while
#    another thread runs. This makes threading useless for CPU-bound tasks.
#
#  • Threading IS useful for I/O-bound tasks (waiting for disk, network, etc.)
#    because the GIL is released while a thread is waiting for I/O.
#
#  • Multiprocessing spawns separate OS processes, each with its own Python
#    interpreter, its own GIL, and its own memory space. They truly run in
#    parallel on separate CPU cores — no GIL contention.
#
#  Our GA evaluations are CPU-BOUND (running a full S2→S6 pipeline). Using
#  threading would run them sequentially (GIL). Using multiprocessing gives
#  us true parallelism across all available CPU cores.
#
#  GENETIC ALGORITHM KEY CONCEPTS
#  ────────────────────────────────
#  A GA is a population-based search algorithm inspired by biological evolution.
#  It maintains a POPULATION of candidate solutions and evolves it over multiple
#  GENERATIONS using four key operators:
#
#  ┌─────────────────┬────────────────────────────────────────────────────────┐
#  │ CONCEPT         │ MEANING IN THIS PIPELINE                               │
#  ├─────────────────┼────────────────────────────────────────────────────────┤
#  │ Individual      │ One set of 7 hyperparameters                           │
#  │ (chromosome)    │ [n_max, n_min, τ_low, τ_high, τ_sem, p_low, p_high]   │
#  ├─────────────────┼────────────────────────────────────────────────────────┤
#  │ Gene            │ One hyperparameter (e.g. n_max=375)                    │
#  ├─────────────────┼────────────────────────────────────────────────────────┤
#  │ Population      │ N=20 individuals evaluated simultaneously per generation│
#  ├─────────────────┼────────────────────────────────────────────────────────┤
#  │ Fitness         │ _strategy_quality_score(chunks) — same metric as S2    │
#  │ function        │ Higher = better chunk quality for this strategy        │
#  ├─────────────────┼────────────────────────────────────────────────────────┤
#  │ Selection       │ Tournament selection (k=3): pick 3 random, keep best   │
#  │                 │ (see SELECTION METHODS comparison below)               │
#  ├─────────────────┼────────────────────────────────────────────────────────┤
#  │ Crossover       │ BLX-α (Blend Crossover, α=0.5): combine two parents   │
#  │ (recombination) │ to produce a child that can explore beyond them        │
#  ├─────────────────┼────────────────────────────────────────────────────────┤
#  │ Mutation        │ Per-gene Gaussian perturbation (rate=0.20, σ=0.15)     │
#  │                 │ Introduces random diversity to avoid local optima      │
#  ├─────────────────┼────────────────────────────────────────────────────────┤
#  │ Elitism         │ Top 2 individuals copied unchanged to next generation   │
#  │                 │ Guarantees best solution found is NEVER lost            │
#  └─────────────────┴────────────────────────────────────────────────────────┘
#
#  SELECTION METHODS — why TOURNAMENT and not ROULETTE WHEEL
#  ──────────────────────────────────────────────────────────
#  Three classical selection methods exist:
#
#  1. ROULETTE WHEEL (fitness-proportional selection):
#     Each individual is selected with probability proportional to its fitness.
#     P(select i) = fitness(i) / Σ fitness(j)
#     ✗ PROBLEM: if one individual has fitness 0.95 and the rest have 0.50,
#       it dominates selection completely → premature convergence.
#     ✗ PROBLEM: breaks when fitnesses are negative or nearly equal (low spread).
#
#  2. RANK-BASED (selection pressure):
#     Sort by fitness, assign probability by RANK not raw value.
#     P(select i) = (2i) / (n(n+1))  for rank i out of n individuals.
#     ✓ More stable than roulette wheel (never dominated by one individual).
#     ✗ Slower convergence because rank differences ignore magnitude gaps.
#
#  3. TOURNAMENT SELECTION (our choice):
#     Randomly pick k individuals from the population. The fittest wins.
#     ✓ Tunable selection pressure via k (higher k = more pressure).
#     ✓ Works with any fitness scale — no normalisation needed.
#     ✓ Never dominated by a single super-fit individual.
#     ✓ Parallelisable — selection decisions are independent.
#     → We use k=3: moderate pressure, maintains good diversity, converges well
#       in 5 generations.
#
# ══════════════════════════════════════════════════════════════════════════════


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
) -> Tuple[List[Dict], float, Dict[str, Any], List[float]]:
    """
    Run a Genetic Algorithm for ONE chunking strategy to find its best hyperparams.

    Each individual (chromosome) = one set of 7 hyperparameters encoded as a
    normalised [0,1]^7 numpy array. The fitness of an individual is the chunk
    quality score produced by running the full pipeline with those hyperparams.

    Parallelism: every individual in a generation is evaluated SIMULTANEOUSLY
    using ProcessPoolExecutor (multiprocessing, not threading — see module header
    for the GIL explanation). Wall-clock time per generation = slowest eval,
    not the sum of all evals.

    Args:
        strategy        : the chunking strategy being optimised (locked throughout)
        pop_size        : number of individuals (chromosomes) per generation
        generations     : number of evolutionary cycles to run
        max_workers     : number of CPU cores to use for parallel evaluation
        baseline_score  : S2 score at default params — the floor we must beat
        baseline_chunks : S2 chunks — used as fallback if GA finds nothing better

    Returns:
        best_chunks     : chunk set produced by the best individual ever seen
        best_score      : fitness score of that individual
        best_cfg        : decoded config dict that produced it
        fitness_history : list of best-so-far fitness after each generation
                          (used by the frontend to draw the convergence chart)
    """
    from .s2_chunkers import _strategy_quality_score as _sqscore

    # Seed the random number generator deterministically per strategy.
    # Same strategy = same seed = reproducible GA runs.
    rng = np.random.RandomState(abs(hash(strategy)) % (2**31))

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1: INITIALISE POPULATION
    # The population is a list of 'pop_size' chromosomes.
    # Each chromosome = numpy array of 7 floats in [0,1] (normalised gene space).
    # ──────────────────────────────────────────────────────────────────────────
    population = _initialise_population(pop_size, warm_cfg, rng)

    # All-time best tracker — starts at the S2 baseline so we never regress
    best_score  = baseline_score
    best_chunks = baseline_chunks
    best_cfg    = copy.deepcopy(warm_cfg)

    # Records best-so-far fitness at end of each generation (for convergence plot)
    fitness_history: List[float] = []

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2: GENERATIONAL LOOP
    # Repeat for G generations. Each generation:
    #   (a) evaluate all individuals in PARALLEL
    #   (b) apply elitism, selection, crossover, mutation to produce next gen
    # ──────────────────────────────────────────────────────────────────────────
    for gen_idx in range(generations):

        # ── (a) PARALLEL FITNESS EVALUATION ───────────────────────────────────
        # Decode every chromosome in the current population into a real config dict,
        # then evaluate all of them simultaneously using multiprocessing.
        #
        # WHY MULTIPROCESSING NOT MULTITHREADING:
        # The pipeline (S2→S6) is CPU-bound. Python threads share one GIL and
        # cannot run CPU-bound code in parallel. ProcessPoolExecutor spawns
        # separate OS processes — each has its own Python interpreter and GIL,
        # so they truly run concurrently on separate CPU cores.
        #
        # WHY _run_pipeline_worker MUST BE TOP-LEVEL:
        # multiprocessing uses pickle to send tasks to worker processes.
        # pickle cannot serialise lambdas or closures (they reference the
        # enclosing scope which doesn't exist in the worker process).
        # _run_pipeline_worker is a top-level module function → picklable.

        # Decode all chromosomes to real config dicts before submitting
        trial_cfgs: List[Dict] = [
            _decode_individual(chromosome, warm_cfg, strategy)
            for chromosome in population
        ]

        # Build argument tuples (all values must be pickle-serialisable)
        args_list = [
            (text, doc_type, doc_profile, model_name, cfg, strategy)
            for cfg in trial_cfgs
        ]

        # Fitness array: one score per individual (0.0 if pipeline crashed)
        fitnesses: List[float] = [0.0] * pop_size

        # Submit ALL pop_size individuals to the process pool simultaneously.
        # as_completed() yields futures as they finish (fastest first),
        # so we don't wait for all of them before processing any results.
        with ThreadPoolExecutor(max_workers=max_workers) as executor:

            # Map: future → individual index (so we know which result goes where)
            future_to_idx = {
                executor.submit(_run_pipeline_worker, args): i
                for i, args in enumerate(args_list)
            }

            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    # Retrieve pipeline result with a timeout guard
                    chunks_result = future.result(timeout=_GA_EVAL_TIMEOUT)
                except (FuturesTimeout, Exception) as exc:
                    # Pipeline crashed or timed out — score 0.0, GA avoids this region
                    logger.debug("GA [%s] gen %d ind %d failed: %s",
                                 strategy, gen_idx, idx, exc)
                    chunks_result = None

                # ── FITNESS FUNCTION ──────────────────────────────────────────
                # Compute the fitness of this individual.
                # We use _strategy_quality_score — the EXACT same function S2 uses
                # to score strategies at default params. This makes GA scores
                # directly comparable to S2 baseline scores.
                # A GA score of 0.921 means the same thing as S2 score of 0.899:
                # the GA found better hyperparams, not a different metric.
                if chunks_result is not None:
                    fitness = round(float(_sqscore(
                        chunks_result,
                        trial_cfgs[idx].get("n_min", n_min),
                        trial_cfgs[idx].get("n_max", n_max),
                    )), 4)
                else:
                    fitness = 0.0  # failed evaluation → worst possible score

                fitnesses[idx] = fitness

                # Track the all-time best individual across all generations
                if fitness > best_score:
                    best_score  = fitness
                    best_chunks = chunks_result
                    best_cfg    = trial_cfgs[idx]

        # Log generation summary
        mean_fitness = float(np.mean(fitnesses))
        logger.debug(
            "GA [%s] gen %d/%d  best=%.4f  mean=%.4f  pop_best=%.4f",
            strategy, gen_idx + 1, generations, best_score,
            mean_fitness, max(fitnesses),
        )

        # Record the all-time best after this generation (convergence history)
        fitness_history.append(round(best_score, 4))

        # ── (b) PRODUCE NEXT GENERATION ───────────────────────────────────────
        # Apply selection + crossover + mutation to evolve the population.
        # This is done sequentially (it's just numpy ops — very fast).
        population = _evolve(population, fitnesses, rng)

    return best_chunks, best_score, best_cfg, fitness_history


# ─────────────────────────────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════
#  GA OPERATORS
#  Each operator corresponds to a biological concept in natural selection.
# ══════════════════════════════════════════════════════════════════════════════


def _initialise_population(
    pop_size: int,
    warm_cfg: Dict[str, Any],
    rng: np.random.RandomState,
) -> List[np.ndarray]:
    """
    ┌─────────────────────────────────────────────────────────┐
    │  GA CONCEPT: POPULATION INITIALISATION                  │
    │  The starting population is the "generation 0" before   │
    │  any evolution has occurred. Its quality directly        │
    │  affects how fast the GA converges.                      │
    └─────────────────────────────────────────────────────────┘

    We use a HYBRID initialisation strategy:

    • WARM-START individuals (10% of population):
      Decoded from the best hyperparameters known for this
      {domain}__{strategy} combination, stored in rl_history.json
      from previous document runs. These individuals start near
      a known-good region of the search space.
      A small noise σ=0.02 is added so warm-start individuals
      are not all identical — we need diversity even in the seed.

    • RANDOM individuals (90% of population):
      Sampled uniformly from [0,1]^7. These ensure the population
      covers the full search space and doesn't prematurely converge
      on the warm-start solution.

    Gene representation:
      Each individual (chromosome) is a numpy float array of length 7.
      Each gene is normalised to [0,1] — the actual parameter value is
      recovered by _decode_individual. Normalisation makes all GA operators
      scale-invariant (a mutation of σ=0.15 means the same relative
      perturbation for n_max (range=750) and τ_low (range=0.35)).
    """
    population: List[np.ndarray] = []

    # Number of individuals seeded from warm-start history
    n_warm = max(1, int(pop_size * _GA_WARMSTART_FRACTION))

    # Encode the warm-start config into normalised gene space
    # Returns None if the warm-start config is incomplete (cold start)
    warm_gene = _encode_config(warm_cfg)

    for i in range(pop_size):
        if i < n_warm and warm_gene is not None:
            # ── WARM-START individual ─────────────────────────────────────────
            # Add tiny Gaussian noise so warm-start individuals are
            # not all identical. σ=0.02 = 2% perturbation in gene space.
            noise = rng.normal(0.0, 0.02, size=_N_GENES)
            gene  = np.clip(warm_gene + noise, 0.0, 1.0)
        else:
            # ── RANDOM individual ─────────────────────────────────────────────
            # Uniform random in [0,1]^7 — explores the full search space
            gene = rng.uniform(0.0, 1.0, size=_N_GENES)

        population.append(gene)

    return population


def _evolve(
    population: List[np.ndarray],
    fitnesses: List[float],
    rng: np.random.RandomState,
) -> List[np.ndarray]:
    """
    ┌─────────────────────────────────────────────────────────┐
    │  GA CONCEPT: GENERATIONAL EVOLUTION                     │
    │  Produces the next generation from the current one       │
    │  using elitism, selection, crossover, and mutation.      │
    └─────────────────────────────────────────────────────────┘

    The next generation is assembled in this order:
      1. ELITISM: top-K individuals copied unchanged (quality guarantee)
      2. SELECTION + CROSSOVER + MUTATION: fill remaining slots

    This function is the GA's main evolutionary step. It is called once
    per generation after all fitness evaluations complete.
    """
    pop_size = len(population)
    next_gen: List[np.ndarray] = []

    # Sort individuals by fitness (descending) — best first
    sorted_indices = np.argsort(fitnesses)[::-1]

    # ──────────────────────────────────────────────────────────────────────────
    # OPERATOR 1: ELITISM
    # ──────────────────────────────────────────────────────────────────────────
    # The top _GA_ELITE_COUNT individuals are copied UNCHANGED to the next
    # generation. This guarantees that the best solution found so far is
    # never destroyed by crossover or mutation.
    #
    # Without elitism, a lucky good individual could be crossed or mutated
    # into a worse one, losing the best solution. Elitism prevents this.
    #
    # We use 2 elites — enough to preserve the best solution and its close
    # neighbour, without reducing population diversity too much.
    for i in range(min(_GA_ELITE_COUNT, pop_size)):
        elite_gene = population[sorted_indices[i]].copy()
        next_gen.append(elite_gene)
        # Log elites for debugging
        logger.debug("  Elite %d: fitness=%.4f", i, fitnesses[sorted_indices[i]])

    # ──────────────────────────────────────────────────────────────────────────
    # Fill the rest of the next generation via SELECTION + CROSSOVER + MUTATION
    # ──────────────────────────────────────────────────────────────────────────
    while len(next_gen) < pop_size:

        # OPERATOR 2: SELECTION (Tournament)
        # Pick two parents via tournament selection.
        # Each parent is selected independently (with replacement allowed),
        # so both parents can sometimes be the same individual.
        parent_a = _tournament_select(population, fitnesses, _GA_TOURNAMENT_K, rng)
        parent_b = _tournament_select(population, fitnesses, _GA_TOURNAMENT_K, rng)

        # OPERATOR 3: CROSSOVER (BLX-α)
        # With probability _GA_CROSSOVER_RATE, produce a child by combining
        # parent_a and parent_b. Otherwise, clone parent_a (no crossover).
        if rng.random() < _GA_CROSSOVER_RATE:
            child = _blx_alpha_crossover(parent_a, parent_b, _GA_CROSSOVER_ALPHA, rng)
        else:
            # No crossover: child is a copy of the fitter parent
            child = parent_a.copy()

        # OPERATOR 4: MUTATION (Gaussian)
        # Perturb the child's genes randomly.
        # Applied AFTER crossover so we mutate the recombined individual.
        child = _gaussian_mutate(child, _GA_MUTATION_RATE, _GA_MUTATION_SIGMA, rng)

        next_gen.append(child)

    # Return exactly pop_size individuals (trim any excess from elitism rounding)
    return next_gen[:pop_size]


def _tournament_select(
    population: List[np.ndarray],
    fitnesses: List[float],
    k: int,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    ┌─────────────────────────────────────────────────────────┐
    │  GA OPERATOR: TOURNAMENT SELECTION                      │
    │                                                          │
    │  Why TOURNAMENT and not ROULETTE WHEEL?                  │
    │                                                          │
    │  ROULETTE WHEEL: P(select i) ∝ fitness(i) / Σ fitness   │
    │  • Breaks when one individual dominates (fitness 0.95    │
    │    vs all others at 0.50 → 63% chance of picking it).   │
    │  • Requires normalised, positive fitness values.         │
    │  • Leads to premature convergence: diversity collapses   │
    │    in generations 1-2 as the dominant individual takes   │
    │    over the entire population.                           │
    │                                                          │
    │  TOURNAMENT (k=3): pick 3 random, the fittest wins.     │
    │  • No dominance problem — a super-fit individual can     │
    │    only be selected if it appears in the tournament.     │
    │  • Works with any fitness scale (no normalisation).      │
    │  • Tunable pressure via k: k=2 = low pressure (more     │
    │    diversity), k=10 = high pressure (fast convergence).  │
    │  • We use k=3: moderate pressure, balances exploration   │
    │    vs exploitation for our 5-generation budget.          │
    └─────────────────────────────────────────────────────────┘

    Args:
        population : list of chromosomes
        fitnesses  : fitness score for each chromosome (same order)
        k          : tournament size (number of candidates to pick)
        rng        : seeded random generator

    Returns:
        A COPY of the winning chromosome (does not modify original population)
    """
    # Pick k unique random indices from the population (without replacement)
    candidates = rng.choice(len(population), size=min(k, len(population)), replace=False)

    # The winner is whichever candidate has the highest fitness
    best_candidate_idx = candidates[np.argmax([fitnesses[i] for i in candidates])]

    # Return a COPY (not a reference) so the original population is not mutated
    return population[best_candidate_idx].copy()


def _blx_alpha_crossover(
    parent_a: np.ndarray,
    parent_b: np.ndarray,
    alpha: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    ┌─────────────────────────────────────────────────────────┐
    │  GA OPERATOR: BLX-α CROSSOVER (Blend Crossover)        │
    │                                                          │
    │  Crossover (recombination) combines genetic material     │
    │  from two parents to produce a child that inherits the   │
    │  best traits of both.                                    │
    │                                                          │
    │  BLX-α vs uniform crossover:                            │
    │  • Uniform crossover: child[d] ∈ {parent_a[d],          │
    │    parent_b[d]} — can ONLY produce values the parents    │
    │    already had. It explores within the convex hull of    │
    │    the population, never outside it.                     │
    │  • BLX-α: child[d] ~ Uniform(lo - α×range, hi + α×range)│
    │    With α=0.5, the child can go 50% beyond either parent │
    │    in any dimension. This allows the GA to DISCOVER      │
    │    configurations that neither parent ever tested.       │
    │                                                          │
    │  Formula per gene dimension d:                           │
    │    lo    = min(parent_a[d], parent_b[d])                 │
    │    hi    = max(parent_a[d], parent_b[d])                 │
    │    range = hi - lo                                       │
    │    child[d] ~ Uniform(lo - α×range,  hi + α×range)       │
    │    → clipped to [0, 1] to stay in valid gene space      │
    └─────────────────────────────────────────────────────────┘

    Example:
        parent_a[n_max_gene] = 0.30  → real n_max ≈ 375
        parent_b[n_max_gene] = 0.50  → real n_max ≈ 525
        range = 0.20
        alpha = 0.50
        child can sample from [0.30-0.10, 0.50+0.10] = [0.20, 0.60]
        → real n_max can be anywhere from 300 to 600
        (the parents covered 375-525; child covers 300-600)

    Args:
        parent_a : first parent chromosome [0,1]^7
        parent_b : second parent chromosome [0,1]^7
        alpha    : blend extent factor (0.5 = extend 50% beyond parents)
        rng      : seeded random generator

    Returns:
        child chromosome [0,1]^7 (clipped to valid range)
    """
    child = np.empty(_N_GENES)

    for d in range(_N_GENES):
        # Find the interval spanned by the two parents for this gene
        lo    = min(parent_a[d], parent_b[d])
        hi    = max(parent_a[d], parent_b[d])
        span  = hi - lo

        # Extend the interval by alpha × span on each side
        extended_lo = lo - alpha * span   # may go below 0
        extended_hi = hi + alpha * span   # may go above 1

        # Sample uniformly from the extended interval
        child[d] = rng.uniform(extended_lo, extended_hi)

    # Clip to [0,1] — genes must stay in valid normalised space
    return np.clip(child, 0.0, 1.0)


def _gaussian_mutate(
    gene: np.ndarray,
    rate: float,
    sigma: float,
    rng: np.random.RandomState,
) -> np.ndarray:
    """
    ┌─────────────────────────────────────────────────────────┐
    │  GA OPERATOR: GAUSSIAN MUTATION                         │
    │                                                          │
    │  Mutation introduces random changes to prevent the GA   │
    │  from getting stuck in local optima. Without mutation,  │
    │  once all individuals in the population share the same  │
    │  gene value for dimension d, crossover cannot recover   │
    │  diversity in d — only mutation can.                    │
    │                                                          │
    │  Per-gene Gaussian mutation:                             │
    │    For each gene d independently:                        │
    │      with probability `rate` (0.20):                    │
    │        gene[d] += N(0, sigma)                           │
    │    Result clipped to [0, 1]                             │
    │                                                          │
    │  Why σ=0.15?                                            │
    │  In [0,1] gene space, σ=0.15 means a typical mutation   │
    │  moves a gene by 15% of its total normalised range.     │
    │  For n_max (real range 750 words): ~112 words shift.    │
    │  For τ_low (real range 0.35): ~0.05 shift.             │
    │  This is large enough to escape local optima, small     │
    │  enough not to completely randomise the individual.      │
    │                                                          │
    │  Why rate=0.20?                                         │
    │  Classical GAs use rate≈0.05-0.10. We use 0.20 because  │
    │  we have only 5 generations — we need faster diversity   │
    │  injection to cover the search space adequately.        │
    └─────────────────────────────────────────────────────────┘

    Args:
        gene  : chromosome to mutate [0,1]^7  (NOT modified in-place)
        rate  : per-gene mutation probability (0.20 = 20% of genes mutate)
        sigma : Gaussian noise standard deviation in [0,1] space
        rng   : seeded random generator

    Returns:
        mutated chromosome [0,1]^7 (a new array, original unchanged)
    """
    mutated = gene.copy()  # never mutate the original in-place

    # Generate a boolean mask: True where mutation occurs
    # Each gene mutates independently with probability `rate`
    mutation_mask = rng.random(_N_GENES) < rate

    if mutation_mask.any():
        # Gaussian noise applied only to mutating genes
        noise = rng.normal(0.0, sigma, _N_GENES)
        mutated[mutation_mask] += noise[mutation_mask]

        # Clip back to [0,1] — genes must stay in valid normalised space
        mutated = np.clip(mutated, 0.0, 1.0)

    return mutated


# ─────────────────────────────────────────────────────────────────────────────
# Gene encoding / decoding
# ─────────────────────────────────────────────────────────────────────────────

def _encode_config(cfg: Dict[str, Any]) -> Optional[np.ndarray]:
    """
    Encode a config dict as a normalised [0,1]^7 gene vector.

    Returns None if any required key is missing (e.g. cold-start with no
    warm history). The formula is the inverse of _decode_individual:
        gene[d] = (real_value - real_min) / (real_max - real_min)
    Integers (n_max, n_min) are used as-is without snapping.
    """
    gene = np.zeros(_N_GENES)
    key_map = {
        0: "n_max", 1: "n_min", 2: "tau_jsd_low", 3: "tau_jsd_high",
        4: "tau_sem", 5: "tau_percentile_low", 6: "tau_percentile_high",
    }
    for d, key in key_map.items():
        val = cfg.get(key)
        if val is None:
            return None   # missing key — can't encode
        real_min, real_max, _, _ = _GENE_SPECS[d]
        gene[d] = float(np.clip((val - real_min) / (real_max - real_min), 0.0, 1.0))
    return gene


def _decode_individual(
    gene: np.ndarray,
    base_cfg: Dict[str, Any],
    strategy: str,
) -> Dict[str, Any]:
    """
    Decode a normalised [0,1]^7 gene vector into a pipeline config dict.

    Decoding formula:
        real_value = real_min + gene[d] × (real_max - real_min)
        if step is not None: round to nearest step

    After decoding, inter-parameter constraints are enforced:
        τ_jsd_low  ≤ τ_jsd_high - 0.08   (gap between merge/split thresholds)
        n_min      ≤ n_max - 50           (minimum chunk size must be smaller)
        τ_percentile_low < τ_percentile_high (lower percentile must be lower)
    """
    cfg = copy.deepcopy(base_cfg)

    key_map = {
        0: "n_max", 1: "n_min", 2: "tau_jsd_low", 3: "tau_jsd_high",
        4: "tau_sem", 5: "tau_percentile_low", 6: "tau_percentile_high",
    }

    for d, key in key_map.items():
        real_min, real_max, step, _ = _GENE_SPECS[d]
        real_val = real_min + gene[d] * (real_max - real_min)
        if step is not None:
            # Round to nearest step (e.g. n_max to nearest 25)
            real_val = round(real_val / step) * step
            real_val = int(np.clip(real_val, real_min, real_max))
        else:
            real_val = float(np.clip(real_val, real_min, real_max))
        cfg[key] = real_val

    # ── Constraint repair ─────────────────────────────────────────────────────
    # If τ_low ≥ τ_high - 0.08: push τ_high up to τ_low + 0.10
    if cfg["tau_jsd_low"] >= cfg["tau_jsd_high"] - 0.08:
        cfg["tau_jsd_high"] = min(0.80, cfg["tau_jsd_low"] + 0.10)

    # If n_min ≥ n_max - 50: pull n_min down
    if cfg["n_min"] >= cfg["n_max"] - 50:
        cfg["n_min"] = max(30, cfg["n_max"] - 50)

    # If p_low ≥ p_high: swap them
    if cfg["tau_percentile_low"] >= cfg["tau_percentile_high"]:
        cfg["tau_percentile_low"], cfg["tau_percentile_high"] = (
            cfg["tau_percentile_high"] - 5,
            cfg["tau_percentile_low"] + 5,
        )

    # Non-tunable settings preserved from base config
    cfg["chunking_strategy"]  = strategy
    cfg["entropy_metric"]     = base_cfg.get("entropy_metric", "hybrid")
    cfg["_in_rl_calibration"] = True

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Parallel pipeline worker — must be a top-level function for pickle
# ─────────────────────────────────────────────────────────────────────────────

def _run_pipeline_worker(args: tuple) -> Optional[List[Dict]]:
    """
    Top-level wrapper for parallel evaluation in ProcessPoolExecutor.

    Must be a top-level function (not a lambda, not a closure) because
    Python's multiprocessing uses pickle to send functions to worker processes,
    and pickle cannot serialise closures or lambdas.

    Receives a single tuple argument `args` because ProcessPoolExecutor.submit
    passes only one argument to the worker. The tuple contains:
        (text, doc_type, doc_profile, model_name, cfg, strategy)

    Returns the chunk list or None if the pipeline failed.
    """
    text, doc_type, doc_profile, model_name, cfg, strategy = args
    return _run_pipeline(text, doc_type, doc_profile, model_name, cfg, strategy)


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
    Run S2 → S3 → S4 during GA trials (S5 and S6 intentionally skipped).

    WHY S5 AND S6 ARE SKIPPED IN GA TRIALS
    ────────────────────────────────────────
    The fitness function (_strategy_quality_score) measures:
      - chunk size fit and count fit        (from S2 output)
      - chunk size stability/variance       (from S2 output)
      - boundary divergence (JSD)           (computed fresh here)
      - structural integrity                (from S2/S3 output)
      - sentence completion rate            (from S2 output)

    NONE of these components use S6 embeddings or S5 entity graph data.
    S6 loads sentence-transformer models (all-MiniLM-L6-v2, all-mpnet-base-v2)
    which are ~400MB each. In multiprocessing, each worker process has its own
    memory — the main process model cache is NOT shared with workers. So every
    single GA trial would reload 400MB of models from disk.

    With pop_size=20, generations=5, strategies=6: that is 600 model loads,
    each taking 1-3 seconds. Skipping S6 removes ~600-1800 seconds of overhead
    with zero impact on fitness scoring accuracy.

    S5 is skipped for the same reason: entity graph enrichment does not affect
    any component of _strategy_quality_score and adds unnecessary I/O.

    S5 and S6 run in full on the FINAL pipeline pass (after GA completes) when
    the best params are applied to produce the actual output chunks.
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

        # Suppress the noisy "UNEXPECTED key" warnings from transformers
        # that fire even when S6 is not called (some imports trigger them)
        import warnings
        import logging
        logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
        warnings.filterwarnings("ignore", message=".*position_ids.*")
        warnings.filterwarnings("ignore", message=".*masked_bias.*")

        n_min = int(cfg.get("n_min", 100))
        n_max = int(cfg.get("n_max", 500))

        # ── S2: run the locked strategy only ─────────────────────────────────
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

        # S2 post-processing (quality pass)
        trial_chunks = _quality_pass(trial_chunks, text, n_min, n_max, s2_winner)

        # ── S3: entropy boundary refinement ──────────────────────────────────
        trial_chunks = refine_boundaries(trial_chunks, cfg)

        # ── S4: boundary quality filter ───────────────────────────────────────
        trial_chunks = filter_boundaries(trial_chunks, doc_type, [], cfg)

        # ── S5 and S6 intentionally SKIPPED in GA trials ─────────────────────
        # Fitness function does not use embeddings or entity graph data.
        # Skipping these eliminates ~600 model-reload operations across all
        # GA trials, saving several minutes of wall-clock time.

        return trial_chunks

    except Exception as exc:
        logger.debug("S7 GA trial failed: %s", exc)
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
    Each value contains the best config params and past GA trial data.
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
        "ga_trials": [ { "params": {...}, "value": float }, ... ]
      }
    }

    ga_trials stores a lightweight record of each trial so the TPE
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

    # Preserve any existing ga_trials so they accumulate across runs
    existing_trials = history.get(domain, {}).get("ga_trials", [])

    # Add the current best as a new trial record for future warm-starting
    new_trial = {"params": best_params, "value": reward_components.get("total", 0.0)}
    updated_trials = existing_trials + [new_trial]

    # Cap to 200 stored trials to prevent unbounded file growth
    updated_trials = updated_trials[-200:]

    history[domain] = {
        "best_params":             best_params,
        "best_reward":             reward_components.get("total", 0.0),
        "last_reward_components":  reward_components,
        "ga_trials":           updated_trials,
    }

    try:
        with open(_RL_HISTORY_PATH, "w", encoding="utf-8") as fh:
            json.dump(history, fh, ensure_ascii=False, indent=2)
    except Exception:
        pass   # silently ignore write failures (e.g. read-only filesystem)