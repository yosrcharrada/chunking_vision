from engine import quality as Q
from engine.chunking import ChunkParams, recursive_chunk_document
from engine.structure import split_sentences

DOC = (
    "Cats are mammals that purr when content. Dogs are loyal companions that bark. "
    "The stock market fell sharply today as investors fled to safer bonds. "
    "Inflation figures spooked traders across the board. Photosynthesis converts "
    "sunlight into chemical energy. Chlorophyll absorbs light in plants."
)


def test_quality_score_in_range(embedder):
    p = ChunkParams(q=0.0, K=4, min_chunk_tokens=8, max_chunk_tokens=160)
    sent_emb = embedder.embed(split_sentences(DOC), "multilingual")
    res = recursive_chunk_document(DOC, p, embedder, backend="multilingual")
    score, details = Q.chunking_quality(res.chunks, sent_emb)
    assert 0.0 <= score <= 1.0
    assert details["n_chunks"] == len(res.chunks)


def test_choose_best_q_returns_candidate(embedder):
    p = ChunkParams(q=1.0, K=4, min_chunk_tokens=8, max_chunk_tokens=160)
    cands = [-1.0, 0.0, 1.0]
    bq, res, scores = Q.choose_best_q(DOC, p, embedder, candidates=cands, backend="multilingual")
    assert bq in cands
    assert len(scores) == len(cands)
    assert res is not None
