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

SEARCH SPACE (6 genes — see _GENE_SPECS)
──────────────────────────────────────────────────
  n_max                ∈ [150, 900]   — max words per chunk (S2)
  n_min                ∈ [30,  250]   — min words per chunk (S2)
  tau_sem              ∈ [0.40, 0.95] — S4 similarity merge threshold
  q_entropy_param      ∈ [-1.0, 1.0]  — S3 Tsallis q (drives the D_q boundary count)
  min_chunk_tokens     ∈ [12,  80]    — S3 qentropy feasibility floor
  q_s4_kernel          ∈ [-1.0, 1.0]  — S4 q-cosine kernel base=q^2 (qcosine mode only)

GENE ENCODING
─────────────
  Each individual is a numpy array of 6 floats in [0, 1] (normalised).
  Decoding maps back to the real parameter range:
    gene[0] → n_max            = 150 + gene[0] × 750  (snapped to nearest 25)
    gene[1] → n_min            = 30  + gene[1] × 220  (snapped to nearest 10)
    gene[2] → τ_sem            = 0.40 + gene[2] × 0.55
    gene[3] → q_entropy_param  = -1.0 + gene[3] × 2.0
    gene[4] → min_chunk_tokens = 12  + gene[4] × 68   (snapped to nearest 1)
    gene[5] → q_s4_kernel      = -1.0 + gene[5] × 2.0  (S4 q-cosine only)

  Normalised encoding makes crossover and mutation scale-invariant — a mutation
  of σ=0.15 in [0,1] space is proportionally the same for every gene regardless
  of its real-world scale (n_max ranges 750 words, τ_sem ranges only 0.55).

UNIFIED SCORING
───────────────
  Every fitness evaluation uses evaluation.ga_fitness — the same Table-I
  cosine-rank / answer-correctness signal (or label-free quality when there
  are no queries) used everywhere else in the platform, so a GA score is
  directly comparable to the same document's S2/S8 scores.

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
# After the merge S3 is qentropy-driven: the only S3 knob is the Tsallis q.
# The old JSD/percentile thresholds no longer affect chunking and were removed
# from the genome so the whole evolutionary budget is spent on parameters that
# actually move the metric: S2 sizing, the S4 merge threshold, the Tsallis q
# (q ∈ [-1, 1] per the platform spec) and the qentropy min-chunk-token floor.
#
# q_s4_kernel is DECOUPLED from q_entropy_param: the latter drives S3's D_q
# (boundary count) via Tsallis entropy; the former is the base=q^2 deformation
# used ONLY by S4's q-cosine kernel (s4_similarity="qcosine").  Without this
# split, one gene had to be simultaneously good for S3's boundary count AND
# S4's merge geometry — a strictly harder joint-optimisation problem than
# ga_base's (where S4=cosine doesn't depend on q at all), which was masking
# whatever the q-cosine kernel could independently contribute.  It is a no-op
# for ga_base (cosine ignores it entirely) and for any run that never sets
# s4_similarity="qcosine" — those runs' genome is unaffected in every other
# respect.
_GENE_SPECS = [
    # (real_min, real_max,  step,  name)
    (150,  900,   25,   "n_max"),            # S2 max words/chunk
    (30,   250,   10,   "n_min"),            # S2 min words/chunk
    (0.40, 0.95,  None, "tau_sem"),          # S4 similarity-merge threshold
    (-1.0, 1.0,   None, "q_entropy_param"),  # S3 Tsallis q (drives D_q boundary count)
    (12,   80,    1,    "min_chunk_tokens"),  # S3 qentropy feasibility floor
    (-1.0, 1.0,   None, "q_s4_kernel"),      # S4 q-cosine kernel base=q^2 (qcosine mode only)
]
_N_GENES = len(_GENE_SPECS)  # 6



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

    All fitness evaluations use evaluation.ga_fitness (Table-I cosine-rank /
    answer-correctness, or label-free quality without queries).

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
    from . import evaluation as EV

    model_name  = config.get("embedding_model", "all-MiniLM-L6-v2")
    doc_type    = doc_profile.get("type",   "prose")
    domain      = doc_profile.get("domain", "general")
    history_key = str(config.get("rl_history_key") or domain)
    n_min       = int(config.get("n_min", 100))
    n_max       = int(config.get("n_max", 500))

    # Read GA settings from config (allow caller overrides)
    pop_size    = int(config.get("ga_population", _GA_POP_SIZE))
    generations = int(config.get("ga_generations", _GA_GENERATIONS))
    max_workers = int(config.get("ga_workers", os.cpu_count() or 4))

    # ── P2 evaluation context — the GA fitness is the Table-I retrieval signal ──
    # Built ONCE: query / ground-truth / document embeddings are reused across
    # every individual of every generation (mirrors the qentropy benchmark).
    ctx = EV.build_context(
        text,
        qa_pairs=config.get("qa_pairs"),
        backend=config.get("embedding_backend"),
        rel_threshold=config.get("rel_threshold", "auto"),
        judge=bool(config.get("judge_answerability", False)),
    )

    # ── Run S2 to get all strategy baseline chunks ────────────────────────────
    all_s2 = run_all_chunkers(text, doc_type, config)
    active_strategies = [s for s in all_s2.keys() if s not in _EXCLUDED_STRATEGIES]

    # S2 baseline scores (default params) — used as the floor for each strategy
    s2_scores: Dict[str, float] = {}
    for strat, chunks_list in all_s2.items():
        if chunks_list:
            s2_scores[strat] = round(float(EV.ga_fitness(chunks_list, ctx)), 4)

    baseline_reward = round(float(EV.ga_fitness(initial_chunks, ctx)), 4)
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

        # Optional elite seed (e.g. a previously-run GA's per-strategy winner,
        # such as the cosine-kernel arm's result) — guarantees this run cannot
        # finish worse than it.  See _run_strategy_ga's elite_seed_cfg docstring.
        elite_seed_cfg = (config.get("elite_seeds") or {}).get(strategy)

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
            ctx=ctx,
            elite_seed_cfg=elite_seed_cfg,
        )

        reward_history.extend(strat_rewards)

        # Persist best params for this strategy+domain combination
        _save_history(strat_key, best_cfg, {"total": best_score})

        per_strategy_results[strategy] = {
            "best_chunks":    best_chunks,       # kept — needed by main.py for per-strategy metrics
            "best_score":     round(best_score, 4),
            "s2_baseline":    strat_baseline_score,
            "improvement":    round(best_score - strat_baseline_score, 4),
            "best_params":    {k: best_cfg.get(k) for k in (
                "n_max", "n_min", "tau_sem", "q_entropy_param",
                "min_chunk_tokens", "q_s4_kernel",
            )},
            "n_evals":        len(strat_rewards),
            # Full per-trial fitness scores — gives the real convergence curve
            # instead of just [baseline, best] (2 flat points)
            "fitness_history": [round(strat_baseline_score, 4)] + strat_rewards,
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
        # best_chunks kept per strategy so main.py can compute real per-strategy metrics
        "per_strategy_results": {
            s: dict(r)   # include best_chunks — main.py pops it when building s7_outputs
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
#  │ Individual      │ One set of 6 genes                                     │
#  │ (chromosome)    │ [n_max, n_min, tau_sem, q_entropy_param,               │
#  │                 │  min_chunk_tokens, q_s4_kernel]                        │
#  ├─────────────────┼────────────────────────────────────────────────────────┤
#  │ Gene            │ One hyperparameter (e.g. n_max=375)                    │
#  ├─────────────────┼────────────────────────────────────────────────────────┤
#  │ Population      │ N=20 individuals evaluated simultaneously per generation│
#  ├─────────────────┼────────────────────────────────────────────────────────┤
#  │ Fitness         │ evaluation.ga_fitness(chunks) — same signal as S2/S8   │
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
    ctx: Any = None,
    elite_seed_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Dict], float, Dict[str, Any], List[float]]:
    """
    Run a Genetic Algorithm for ONE chunking strategy to find its best hyperparams.

    Each individual (chromosome) = one set of 6 genes encoded as a
    normalised [0,1]^6 numpy array. The fitness of an individual is the chunk
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
        elite_seed_cfg  : optional gene values (n_max/n_min/tau_sem/q_entropy_param/
                          min_chunk_tokens — q_s4_kernel deliberately excluded,
                          see below) from a PRIOR GA run — e.g. the cosine
                          kernel's winning config — to seed into this run.  It is
                          (1) folded into the warm-start individual so the initial
                          population starts from a known-good point instead of a
                          cold/random one, and (2) evaluated once up front so
                          best_score/best_chunks never regress below it.  This is
                          what guarantees ga_qcos cannot finish worse than
                          ga_base: elitism never discards the tracked best, and
                          the tracked best starts at >= the seed's fitness.
                          q_s4_kernel is excluded from the copy because a
                          cosine-only seed never exercised that gene (cosine
                          ignores q entirely) — its value would be meaningless
                          noise, so this run's q_s4_kernel is left free to be
                          searched from warm-start/random as usual.

    Returns:
        best_chunks     : chunk set produced by the best individual ever seen
        best_score      : fitness score of that individual
        best_cfg        : decoded config dict that produced it
        fitness_history : list of best-so-far fitness after each generation
                          (used by the frontend to draw the convergence chart)
    """
    from . import evaluation as EV

    # Seed the random number generator deterministically per strategy.
    # Same strategy = same seed = reproducible GA runs.
    rng = np.random.RandomState(abs(hash(strategy)) % (2**31))

    # Elite seed: fold a prior GA winner's gene values into warm_cfg so the
    # warm-start individual (see _initialise_population) starts from it instead
    # of from rl_history.json's possibly-unrelated record.  Everything else
    # (s4_similarity, embedding_backend, ...) stays as configured for THIS run.
    # q_s4_kernel is deliberately EXCLUDED: if the seed came from a cosine-only
    # run, that gene was never exercised (cosine ignores q) so its value is
    # meaningless noise — copying it in would bias this run's genuinely NEW
    # search dimension instead of leaving it free to explore.
    if elite_seed_cfg:
        for k in ("n_max", "n_min", "tau_sem", "q_entropy_param", "min_chunk_tokens"):
            if elite_seed_cfg.get(k) is not None:
                warm_cfg[k] = elite_seed_cfg[k]

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1: INITIALISE POPULATION
    # The population is a list of 'pop_size' chromosomes.
    # Each chromosome = numpy array of 6 floats in [0,1] (normalised gene space).
    # ──────────────────────────────────────────────────────────────────────────
    population = _initialise_population(pop_size, warm_cfg, rng)

    # All-time best tracker — starts at the S2 baseline so we never regress
    best_score  = baseline_score
    best_chunks = baseline_chunks
    best_cfg    = copy.deepcopy(warm_cfg)

    # Evaluate the elite seed itself (exact config, not the noised warm-start
    # copy already in the population) so best_score/best_chunks are raised to at
    # least its fitness BEFORE the generational loop begins.  Combined with
    # elitism (which never discards the tracked best), this makes it provable
    # that this GA run cannot finish worse than the seed's own fitness.
    if elite_seed_cfg:
        seed_cfg = copy.deepcopy(warm_cfg)
        seed_chunks = _run_pipeline(text, doc_type, doc_profile, model_name, seed_cfg, strategy)
        if seed_chunks:
            seed_score = float(EV.ga_fitness(seed_chunks, ctx)) if ctx is not None else 0.0
            if seed_score > best_score:
                best_score, best_chunks, best_cfg = seed_score, seed_chunks, seed_cfg

    # Records best-so-far fitness at end of each generation (for convergence plot)
    fitness_history: List[float] = []

    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2: GENERATIONAL LOOP — sequential evaluation, no executor, no deadline
    # Budget is fully controlled by pop_size × generations (e.g. 4×2 = 8 evals).
    # No wall-clock deadline: the sequential loop has zero deadlock risk, and the
    # 60s deadline was causing n_evals=1 (pipeline runs take 15-30s each, so the
    # deadline expired after the very first evaluation → no actual GA learning).
    # ──────────────────────────────────────────────────────────────────────────
    for gen_idx in range(generations):

        # ── (a) SEQUENTIAL FITNESS EVALUATION ────────────────────────────────
        # Why sequential (not ThreadPoolExecutor / ProcessPoolExecutor)?
        # ThreadPoolExecutor: GIL prevents true CPU parallelism, AND when
        #   future.result(timeout) fires the thread keeps running, causing
        #   executor.shutdown(wait=True) to block forever → deadlock on Windows.
        # ProcessPoolExecutor: Windows 'spawn' re-imports full Python env per
        #
        # PARALLELISM: ProcessPoolExecutor with max_workers=2
        # ──────────────────────────────────────────────────
        # Previous crashes were caused by:
        #   1. Too many workers (20) → paging file exhaustion on Windows
        #   2. future.result(timeout=N) → TimeoutError in main thread BUT
        #      worker kept running → executor.shutdown(wait=True) deadlocked
        #
        # Fix:
        #   1. Only 2 workers → safe memory footprint (~400MB total)
        #   2. NO timeout on future.result() → workers always complete,
        #      shutdown is always clean, zero deadlock risk
        #   3. _run_pipeline has try/except so it always returns (never hangs)
        #
        # Speedup: 4 individuals → 2 run at a time → 2× faster than sequential
        # With pop=4, gen=2, 6 strategies: total ~6 min instead of ~12 min.

        # Decode all chromosomes to real config dicts
        trial_cfgs: List[Dict] = [
            _decode_individual(chromosome, warm_cfg, strategy)
            for chromosome in population
        ]

        # Fitness array: one score per individual (0.0 if pipeline crashed)
        fitnesses: List[float] = [0.0] * pop_size

        args_list = [
            (text, doc_type, doc_profile, model_name, cfg, strategy)
            for cfg in trial_cfgs
        ]

        # Evaluate the population IN-PROCESS (sequential).  Why not a process
        # pool?  Every S3 evaluation embeds via the shared EmbeddingService, and a
        # fresh worker process reloads the ~120 MB embedding model from scratch on
        # each new pool — which is what produced the endless HuggingFace download
        # warnings and most of the wall-clock time.  Running in-process keeps the
        # model loaded ONCE and lets the S3 unit-embedding cache hit across
        # individuals, so a warm sequential loop is faster here than spawning
        # workers, and the console stays quiet.
        for idx, args in enumerate(args_list):
            try:
                chunks_result = _run_pipeline_worker(args)
            except Exception as exc:
                logger.debug("GA [%s] gen %d ind %d failed: %s",
                             strategy, gen_idx, idx, exc)
                chunks_result = None

            # ── FITNESS FUNCTION ───────────────────────────────────────────
            # P2 Table-I retrieval signal (cosine-rank mean / answer-correct),
            # or label-free quality offline.
            if chunks_result is not None and ctx is not None:
                fitness = round(float(EV.ga_fitness(chunks_result, ctx)), 4)
            else:
                fitness = 0.0   # failed → worst score → GA avoids this region

            fitnesses[idx] = fitness

            if fitness > best_score:
                best_score  = fitness
                best_chunks = chunks_result
                best_cfg    = trial_cfgs[idx]
                logger.debug("GA [%s] gen %d ind %d: new best = %.4f",
                             strategy, gen_idx, idx, best_score)

        # Log generation summary
        valid = [f for f in fitnesses if f > 0]
        logger.debug("GA [%s] gen %d/%d  best=%.4f  mean=%.4f",
                     strategy, gen_idx + 1, generations,
                     best_score, float(np.mean(valid)) if valid else 0.0)

        # Record all-time best after this generation (for convergence chart)
        fitness_history.append(round(best_score, 4))

        # ── (b) PRODUCE NEXT GENERATION ──────────────────────────────────────
        # Selection + crossover + mutation — pure numpy, runs in milliseconds.
        population = _evolve(population, fitnesses, rng)

    # If no individual beat the S2 baseline, best_chunks is still the RAW S2
    # output (no S3 qentropy annotations).  Re-run S2→S3→S4 once with best_cfg so
    # the returned winner always carries q / D_q / entropy fields for evaluation.
    if best_chunks and not any("diversity_number" in c for c in best_chunks):
        annotated = _run_pipeline(text, doc_type, doc_profile, model_name,
                                  copy.deepcopy(best_cfg), strategy)
        if annotated:
            best_chunks = annotated

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
      Sampled uniformly from [0,1]^6. These ensure the population
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
            # Uniform random in [0,1]^6 — explores the full search space
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
        parent_a : first parent chromosome [0,1]^6
        parent_b : second parent chromosome [0,1]^6
        alpha    : blend extent factor (0.5 = extend 50% beyond parents)
        rng      : seeded random generator

    Returns:
        child chromosome [0,1]^6 (clipped to valid range)
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
        gene  : chromosome to mutate [0,1]^6  (NOT modified in-place)
        rate  : per-gene mutation probability (0.20 = 20% of genes mutate)
        sigma : Gaussian noise standard deviation in [0,1] space
        rng   : seeded random generator

    Returns:
        mutated chromosome [0,1]^6 (a new array, original unchanged)
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
    Encode a config dict as a normalised [0,1]^6 gene vector.

    Returns None if any required key is missing (e.g. cold-start with no
    warm history). The formula is the inverse of _decode_individual:
        gene[d] = (real_value - real_min) / (real_max - real_min)
    Integers (n_max, n_min) are used as-is without snapping.
    """
    gene = np.zeros(_N_GENES)
    key_map = {
        0: "n_max", 1: "n_min", 2: "tau_sem",
        3: "q_entropy_param", 4: "min_chunk_tokens", 5: "q_s4_kernel",
    }
    for d, key in key_map.items():
        val = cfg.get(key)
        if val is None and key in ("q_entropy_param", "q_s4_kernel"):
            val = 1.0   # default q (Shannon / classical-limit kernel) — safe
                        # fallback for warm-start records predating this gene
        if val is None and key == "min_chunk_tokens":
            val = 20
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
    Decode a normalised [0,1]^6 gene vector into a pipeline config dict.

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
        0: "n_max", 1: "n_min", 2: "tau_sem",
        3: "q_entropy_param", 4: "min_chunk_tokens", 5: "q_s4_kernel",
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
    # If n_min ≥ n_max - 50: pull n_min down
    if cfg["n_min"] >= cfg["n_max"] - 50:
        cfg["n_min"] = max(30, cfg["n_max"] - 50)

    # Non-tunable settings preserved from base config
    cfg["chunking_strategy"] = strategy
    # q_entropy_param is a tunable gene — decoded above, clamp to the spec range.
    # It drives ONLY S3's D_q (Tsallis entropy) and spans the FULL [-1, 1] —
    # unrestricted, even when s4_similarity="qcosine", because the S4 kernel now
    # has its OWN independent gene (q_s4_kernel) below.
    cfg["q_entropy_param"] = float(np.clip(cfg.get("q_entropy_param", 1.0), -1.0, 1.0))
    cfg["q_s4_kernel"] = float(np.clip(cfg.get("q_s4_kernel", 1.0), -1.0, 1.0))
    # When this GA run is searching under the q-cosine S4 kernel, keep q_s4_kernel
    # strictly away from -1 (excluded per instructor guidance — see the note in
    # s4_boundary._q_cosine_similarity: q=-1 duplicates the q=+1 classical-limit
    # anchor via base=q^2 and is never considered by the source paper).
    if cfg.get("s4_similarity") == "qcosine" and cfg["q_s4_kernel"] <= -0.999:
        cfg["q_s4_kernel"] = -0.999

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
    The fitness function (evaluation.ga_fitness — the Table-I cosine-rank /
    answer-correctness signal, or label-free quality when there are no queries)
    scores the chunk texts directly via engine.embeddings.  It does not use
    S6's sentence-transformer ensemble or S5's entity graph at all.
    S6 loads sentence-transformer models (all-MiniLM-L6-v2, all-mpnet-base-v2)
    which are ~400MB each. In multiprocessing, each worker process has its own
    memory — the main process model cache is NOT shared with workers. So every
    single GA trial would reload 400MB of models from disk.

    With pop_size=20, generations=5, strategies=6: that is 600 model loads,
    each taking 1-3 seconds. Skipping S6 removes ~600-1800 seconds of overhead
    with zero impact on fitness scoring accuracy.

    S5 is skipped for the same reason: entity graph enrichment does not affect
    ga_fitness at all and adds unnecessary I/O.

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
        try:
            trial_chunks = refine_boundaries(trial_chunks, cfg)
        except Exception as s3_exc:
            # S3 crash in worker — log and return None so caller detects failure.
            # Do NOT silently fall back to S2 chunks (that causes chunk_score=0.35).
            logger.warning("S7 GA worker: S3 refine_boundaries failed: %s", s3_exc)
            return None

        # ── S4: boundary quality filter ───────────────────────────────────────
        try:
            trial_chunks = filter_boundaries(trial_chunks, doc_type, [], cfg)
        except Exception as s4_exc:
            logger.warning("S7 GA worker: S4 filter_boundaries failed: %s", s4_exc)
            return None

        # ── S5 and S6 intentionally SKIPPED in GA trials ─────────────────────
        # Fitness function does not use embeddings or entity graph data.
        # Skipping these eliminates ~600 model-reload operations across all
        # GA trials, saving several minutes of wall-clock time.

        return trial_chunks

    except Exception as exc:
        logger.debug("S7 GA trial failed: %s", exc)
        return None


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
    tunable_keys = (  # the 6 live genes (see _GENE_SPECS)
        "n_max", "n_min", "tau_sem",
        "q_entropy_param", "min_chunk_tokens", "q_s4_kernel",
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

    tunable_keys = (  # the 6 live genes (see _GENE_SPECS)
        "n_max", "n_min", "tau_sem",
        "q_entropy_param", "min_chunk_tokens", "q_s4_kernel",
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
