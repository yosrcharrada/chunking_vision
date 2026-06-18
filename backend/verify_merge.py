"""Offline end-to-end smoke test of the merged qentropy pipeline.
Run: python verify_merge.py   (no OPENAI_API_KEY needed)."""
import warnings
warnings.filterwarnings("ignore")

DOC = (
    "ARTICLE 1. The contracting parties agree to uphold the terms herein. "
    "Payment shall be made within thirty days of invoice. Late payment incurs a penalty. "
    "Photosynthesis converts sunlight into chemical energy in plant cells. "
    "Chlorophyll absorbs light most strongly in the blue and red wavelengths. "
    "The mitochondria is the powerhouse of the cell, producing ATP. "
    "Quantum entanglement links the states of two distant particles. "
    "Measuring one particle instantaneously affects the other regardless of distance. "
    "ARTICLE 2. Disputes shall be resolved through binding arbitration. "
    "The governing law is that of the jurisdiction named in the schedule. "
    "Revenue grew fourteen percent year over year driven by cloud demand. "
    "Operating margin expanded as headcount growth was controlled."
) * 3


def main():
    from pipeline.s2_chunkers import run_all_chunkers
    from pipeline.s3_entropy import refine_boundaries, get_jsd_series
    from pipeline.s4_boundary import filter_boundaries
    from pipeline import evaluation as EV
    from pipeline.s7_rl import run_rl_loop

    cfg = {"q_entropy_param": 0.5, "K": 4, "min_chunk_tokens": 20, "max_chunk_tokens": 320,
           "window": 1, "tau_sem": 0.75, "n_min": 60, "n_max": 300,
           "ga_population": 2, "ga_generations": 1, "ga_workers": 1,
           "embedding_backend": None, "judge_answerability": False, "qa_pairs": []}
    doc_profile = {"type": "prose", "domain": "general",
                   "token_count": len(DOC.split()), "length_bucket": "medium"}

    print("── S2 ──")
    s2 = run_all_chunkers(DOC, "prose", cfg)
    strat = next(k for k, v in s2.items() if v)
    chunks = s2[strat]
    print(f"strategy={strat}  S2 chunks={len(chunks)}")

    print("── S3 (qentropy) ──")
    refined = refine_boundaries(chunks, cfg)
    st = refined[-1]["s3_stats"]
    print(f"S3 chunks={len(refined)}  D_q={st['diversity_number']}  q={refined[0]['q']} "
          f"shannon={st['shannon_bits']}  tsallis={st['tsallis_bits']}  "
          f"h_K={st['tree_entropy']['entropy_rate']:.4f}  merged={st['merged_count']} hard={st['hard_count']}")
    assert all("diversity_number" in c for c in refined), "S3 missing qentropy fields"
    assert len(get_jsd_series(refined)) == len(refined)

    print("── S4 ──")
    filtered = filter_boundaries(refined, "prose", [], cfg)
    print(f"S4 chunks={len(filtered)}")

    print("── P2 evaluation (label-free, offline) ──")
    ctx = EV.build_context(DOC, qa_pairs=[], backend=None)
    met = EV.score_run(filtered, ctx)
    print(f"ss2fd={met['ss2fd']}  n_chunks={met['n_chunks']}  avg_tokens={met['avg_chunk_tokens']}")
    print(f"ga_fitness(label-free)={EV.ga_fitness(filtered, ctx):.4f}")

    print("── S7 GA (tiny, qentropy + P2 fitness) ──")
    best, hist, final = run_rl_loop(DOC, doc_profile, chunks, dict(cfg))
    print(f"GA best chunks={len(best)}  winner={final.get('overall_winner_strategy')}  "
          f"tuned q={final.get('q_entropy_param')}  n_evals={final.get('n_evals_total')}")
    assert best and any("diversity_number" in c for c in best), "GA winner missing qentropy fields"
    print("\nALL OK ✓  qentropy drives S3, GA tunes q, P2 metrics computed.")


if __name__ == "__main__":
    main()
