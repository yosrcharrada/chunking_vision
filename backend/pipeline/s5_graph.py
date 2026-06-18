"""
S5 — Entity Intelligence & Graph Enrichment
=============================================
Pipeline position : runs AFTER S4 (boundary quality filter) and BEFORE S6 (embedding).
Responsibility    : extract named entities from each chunk, build a typed-edge
                    knowledge graph linking chunks that share entities, enrich
                    each chunk's embedding with its graph neighbourhood, and
                    persist entity co-occurrences to the KG store for warm-start
                    on future document runs.

Architecture overview
─────────────────────
1. NER extraction  — spaCy (en_core_web_sm) or regex fallback.
2. Entity linking  — canonical form + head token for coreference grouping.
3. Relation extraction — lightweight subject–verb–object pattern matching.
4. Graph construction  — chunks are NODES; edges connect chunks that share
                         entities or relations, weighted by co-occurrence count.
   Three edge types:
     shared_entity  — both chunks mention the same canonical entity
     relation_bridge — both chunks mention the same (subject, object) pair
     kg_prior       — entity pair has historical co-occurrence in KGStore

5. GNN-style enrichment — each chunk's graph_vector is the weighted mean
   embedding of its graph neighbours (message-passing step):
       h_graph(Cᵢ) = Σⱼ∈N(i) (wᵢⱼ / Σwᵢⱼ) · e(Cⱼ)
   This lets a chunk "borrow" information from topically related chunks
   even if they are far apart in the document.

6. KGStore write — new entity co-occurrences are persisted to disk so that
   the next document run benefits from accumulated prior knowledge.

Known limitation — NER model
─────────────────────────────
The default spaCy model (en_core_web_sm) was trained on English newswire.
For French legal/financial documents (Tunisian regulations, tax conventions)
it will mislabel generic French phrases as PERSON/DATE/ORG.
Recommended fix: install fr_core_news_sm and auto-detect language in _get_nlp().
Until then, the _link_entities() canonical form at least deduplicates surface
variants, and the KGStore accumulates useful signal across multiple runs.
"""

import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Path to the persistent knowledge graph store on disk
KG_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "kg_store.json")


# ─────────────────────────────────────────────────────────────────────────────
# KGStore — persistent entity co-occurrence graph
# ─────────────────────────────────────────────────────────────────────────────

class KGStore:
    """
    Persistent knowledge graph store.

    Stores entity co-occurrence counts across ALL pipeline runs.
    S5 reads prior edge weights BEFORE building the current document's graph,
    then writes new co-occurrences AFTER processing — making the KG cumulative.

    This means:
    - First run on document A: KGStore is empty; graph built from doc A only.
    - Second run on document B: KGStore contains doc A's entity pairs; edges
      between entities that co-occurred in doc A are BOOSTED in doc B's graph.
    - Subsequent runs accumulate more knowledge.

    Storage format (kg_store.json):
    {
        "cooccurrence": {"entity_a": {"entity_b": count, ...}, ...},
        "chunk_index":  {"entity_text": ["job_id::C0", "job_id::C3", ...], ...}
    }
    """

    def __init__(self, path: str = KG_STORE_PATH):
        # Resolve to absolute path to avoid working-directory confusion
        self.path = os.path.abspath(path)

        # entity_cooccurrence[a][b] = number of times a and b appeared together
        self.entity_cooccurrence: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # entity_chunk_index[entity_text] = list of "job_id::C{i}" strings
        self.entity_chunk_index: Dict[str, List[str]] = defaultdict(list)

        # Load existing data from disk (if the file exists)
        self._load()

    def _load(self) -> None:
        """Load co-occurrence data from kg_store.json into memory."""
        if not os.path.exists(self.path):
            return   # no prior runs — start with empty store
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for a, neighbors in data.get("cooccurrence", {}).items():
                for b, w in neighbors.items():
                    self.entity_cooccurrence[a][b] = int(w)
            self.entity_chunk_index = defaultdict(list, data.get("chunk_index", {}))
        except Exception:
            pass   # silently ignore corrupt file — start fresh

    def save(self) -> None:
        """
        Persist the current state to kg_store.json.

        Eviction policy
        ───────────────
        To prevent the store from growing unboundedly across thousands of
        document runs, we cap the entity_chunk_index at _KG_MAX_TRIALS
        entries per entity (keeping only the most recent chunk IDs).
        Co-occurrence counts are NOT evicted — they accumulate as intended.
        """
        try:
            # Evict oldest chunk-index entries beyond the cap
            evicted_index = {
                ent: ids[-_KG_MAX_TRIALS:]
                for ent, ids in self.entity_chunk_index.items()
            }
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "cooccurrence": {
                            k: dict(v) for k, v in self.entity_cooccurrence.items()
                        },
                        "chunk_index": evicted_index,
                    },
                    fh,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            pass   # silently ignore write failures (e.g. read-only filesystem)

    def add_chunk_entities(self, chunk_id: str, entities: List[str]) -> None:
        """
        Register entity co-occurrences from one chunk.

        For each pair (a, b) of entities within the chunk, increment both
        entity_cooccurrence[a][b] and entity_cooccurrence[b][a] (symmetric).
        Also records which chunk IDs each entity appeared in.
        """
        # Record that this entity appeared in this chunk
        for ent in entities:
            if chunk_id not in self.entity_chunk_index[ent]:
                self.entity_chunk_index[ent].append(chunk_id)

        # Update pairwise co-occurrence counts
        for i, a in enumerate(entities):
            for b in entities[i + 1:]:
                self.entity_cooccurrence[a][b] += 1
                self.entity_cooccurrence[b][a] += 1   # symmetric

    def get_prior_weight(self, a: str, b: str) -> int:
        """
        Return historical co-occurrence count for entity pair (a, b).
        Returns 0 if the pair has never been seen together.
        """
        return int(self.entity_cooccurrence.get(a, {}).get(b, 0))


# Module-level KGStore singleton (loaded once, persists across the request lifetime)
_kg_store = KGStore()


# Supported spaCy model names per language code.
# Add entries here to support additional languages without changing any other code.
_SPACY_MODELS: Dict[str, str] = {
    "en": "en_core_web_sm",   # English — newswire + web text
    "fr": "fr_core_news_sm",  # French  — news + legal corpus
    "de": "de_core_news_sm",  # German
    "es": "es_core_news_sm",  # Spanish
    "it": "it_core_news_sm",  # Italian
    "ar": "ar_core_news_sm",  # Arabic  (if installed)
}

# Maximum number of entity co-occurrence records kept in KGStore.
# Prevents unbounded disk growth across many document runs.
_KG_MAX_TRIALS = 500

# Per-language spaCy model cache: lang_code → loaded model (or None)
_nlp_cache: Dict[str, Any] = {}


def _detect_language(text: str) -> str:
    """
    Detect the primary language of the document text.

    Strategy (priority order):
    1. langdetect library (if installed) — probabilistic n-gram model.
    2. langid library (if installed) — fast Naive Bayes classifier.
    3. Heuristic: count French-specific function words vs English ones.
       This covers the most common case (French legal documents) without
       any external dependency.

    Returns a 2-letter ISO 639-1 language code (e.g. "fr", "en", "de").
    Defaults to "en" if detection fails.
    """
    # ── Option 1: langdetect ─────────────────────────────────────────────────
    try:
        from langdetect import detect  # type: ignore
        # Sample the first 2000 chars to keep it fast
        lang = detect(text[:2000])
        # langdetect returns codes like "fr", "en", "zh-cn" — normalise to 2-char
        return lang.split("-")[0].lower()
    except Exception:
        pass

    # ── Option 2: langid ─────────────────────────────────────────────────────
    try:
        import langid  # type: ignore
        lang, _ = langid.classify(text[:2000])
        return lang.lower()
    except Exception:
        pass

    # ── Option 3: heuristic word-count approach ───────────────────────────────
    # Count occurrences of high-frequency words that are unambiguous markers
    # of their language.  We sample 5000 chars to keep the heuristic fast.
    sample = text[:5000].lower()
    words  = re.findall(r"\b\w+\b", sample)
    word_set = set(words)

    # French marker words (very high frequency in French, rare in English)
    fr_markers = {"le","la","les","de","des","du","et","en","un","une",
                  "dans","pour","que","est","sur","par","avec","au","aux",
                  "cette","leur","leurs","dont","ainsi","également"}
    # English marker words
    en_markers = {"the","a","an","and","or","but","is","are","was","were",
                  "have","has","had","will","would","could","should","may",
                  "this","that","these","those","with","from","their","they"}

    fr_count = sum(1 for w in words if w in fr_markers)
    en_count = sum(1 for w in words if w in en_markers)

    if fr_count > en_count * 1.5:
        return "fr"
    return "en"   # safe default


def _get_nlp(text: str = ""):
    """
    Lazy-load the spaCy NLP model appropriate for the document's language.

    Algorithm
    ─────────
    1. Detect the document language (see _detect_language).
    2. Look up the matching spaCy model name in _SPACY_MODELS.
    3. Try to load it; if unavailable, try the English fallback.
    4. If all spaCy models fail, return None → triggers regex NER fallback.

    Results are cached per language code so that multiple chunks from the
    same document reuse the already-loaded model.

    Raises
    ──────
    Never raises — always returns a model or None.
    """
    lang = _detect_language(text) if text else "en"
    model_name = _SPACY_MODELS.get(lang, "en_core_web_sm")

    # Return cached model if already loaded for this language
    if lang in _nlp_cache:
        return _nlp_cache[lang]

    # Try to load the language-specific model
    try:
        import spacy  # noqa: E402
        nlp = spacy.load(model_name, disable=["parser", "tagger", "lemmatizer"])
        _nlp_cache[lang] = nlp
        return nlp
    except Exception:
        pass

    # Fallback: try English model (widely installed)
    if model_name != "en_core_web_sm":
        try:
            import spacy  # noqa: E402
            nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
            _nlp_cache[lang] = nlp   # cache under the detected lang to avoid retrying
            return nlp
        except Exception:
            pass

    # No spaCy available — regex NER fallback will be used
    _nlp_cache[lang] = None
    return None



# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def enrich_graph(
    chunks: List[Dict],
    embeddings: Optional[List[List[float]]],
    config: Dict[str, Any],
) -> List[Dict]:
    """
    Enrich each chunk with entity information and graph-based vectors.

    Processing steps:
    1. Run NER on each chunk → raw entity list.
    2. Link/canonicalise entities → deduplicated linked entities.
    3. Extract lightweight SRO relations.
    4. Build typed-edge graph: nodes=chunks, edges=shared entity/relation + KG prior.
    5. Assign graph_neighbors, typed_edges, graph_vector to each chunk.
    6. Write new entity data to KGStore → persist for future runs.

    Output fields added to each chunk:
      entities       — list of {text, label, canonical, entity_id, parent_id}
      relations      — list of {subject, relation, object, type}
      graph_neighbors — list of chunk indices sharing at least one entity
      typed_edges    — list of {target, weight, types} dicts
      graph_vector   — 256-dim weighted-mean neighbour embedding (list[float])
    """
    if not chunks:
        return chunks

    enriched = [dict(c) for c in chunks]
    job_id   = config.get("job_id", "unknown")   # used to build unique chunk IDs

    # Detect language from the full document text (passed via config) so we
    # load the correct spaCy model.  Fall back to a sample of the first 3 chunks
    # if the full text was not stored in config (keeps memory usage bounded).
    full_text_sample = str(config.get("_full_text_sample", "")) or (
        " ".join(c.get("text", "") for c in chunks[:3])
    )
    nlp = _get_nlp(full_text_sample)

    # ── Step 1 + 2 + 3: NER, linking, and relation extraction ────────────
    for i, chunk in enumerate(enriched):
        raw_entities = _extract_entities(chunk.get("text", ""), nlp)
        linked       = _link_entities(raw_entities)
        relations    = _extract_relations(chunk.get("text", ""), linked)
        enriched[i]["entities"]  = linked
        enriched[i]["relations"] = relations

    # ── Step 4: Build the entity graph ───────────────────────────────────
    # edge_counts[(i,j)] = total edge weight between chunk i and chunk j
    # edge_types[(i,j)]  = list of edge type strings for those chunks
    edge_counts: Dict[Tuple[int, int], int]        = defaultdict(int)
    edge_types:  Dict[Tuple[int, int], List[str]]  = defaultdict(list)

    for i in range(len(enriched)):
        for j in range(i + 1, len(enriched)):
            # Shared entities and shared relations
            shared_ent, shared_rel = _shared_signals(enriched[i], enriched[j])

            if shared_ent > 0:
                pair = (i, j)
                edge_counts[pair] += shared_ent
                edge_types[pair].append("shared_entity")

            if shared_rel > 0:
                pair = (i, j)
                edge_counts[pair] += shared_rel
                edge_types[pair].append("relation_bridge")

            # KG prior: boost edges between entities with historical co-occurrence
            # This makes the graph richer on subsequent runs of the same domain
            for ei in {e["canonical"] for e in enriched[i].get("entities", [])}:
                for ej in {e["canonical"] for e in enriched[j].get("entities", [])}:
                    prior = _kg_store.get_prior_weight(ei, ej)
                    if prior > 0:
                        pair = (i, j)
                        edge_counts[pair] += prior
                        edge_types[pair].append("kg_prior")

    # ── Step 5: Assign graph metadata and compute graph vectors ──────────
    for i, chunk in enumerate(enriched):
        neighbors: List[int]          = []
        weights:   List[int]          = []
        typed:     List[Dict[str, Any]] = []

        # Collect all neighbours of chunk i from the edge map
        for (a, b), cnt in edge_counts.items():
            if a == i or b == i:
                nb = b if a == i else a
                neighbors.append(nb)
                weights.append(cnt)
                typed.append({
                    "target": nb,
                    "weight": cnt,
                    "types":  sorted(set(edge_types.get((a, b), []))),
                })

        chunk["graph_neighbors"] = neighbors
        chunk["typed_edges"]     = typed

        # GNN message-passing step: graph_vector = weighted mean of neighbour embeddings
        chunk["graph_vector"] = _graph_vector_for_chunk(i, neighbors, weights, embeddings)

    # ── Step 6: Write new entity data to KGStore ─────────────────────────
    for i, chunk in enumerate(enriched):
        cid           = f"{job_id}::C{i}"   # unique ID: job_id + chunk index
        entity_texts  = [e["canonical"] for e in chunk.get("entities", [])]
        _kg_store.add_chunk_entities(cid, entity_texts)

    _kg_store.save()   # persist to disk for future runs

    return enriched


def build_entity_graph_data(chunks: List[Dict]) -> Dict[str, Any]:
    """
    Build a JSON-serialisable graph data structure for the frontend entity graph.

    Returns:
      {
        "nodes": [{"id": int, "label": str, "entity_count": int, ...}],
        "edges": [{"source": int, "target": int, "weight": int, "types": [...]}]
      }
    """
    # Nodes: one per chunk
    nodes = []
    for i, chunk in enumerate(chunks):
        nodes.append({
            "id":             i,
            "label":          f"C{i}",
            "entity_count":   len(chunk.get("entities",  [])),
            "relation_count": len(chunk.get("relations", [])),
        })

    # Edges: deduplicated typed edges
    edges: List[Dict[str, Any]] = []
    seen  = set()
    for i, chunk in enumerate(chunks):
        for edge in chunk.get("typed_edges", []):
            pair = (min(i, edge["target"]), max(i, edge["target"]))
            if pair in seen:
                continue
            seen.add(pair)
            edges.append({
                "source": pair[0],
                "target": pair[1],
                "weight": edge.get("weight", 1),
                "types":  edge.get("types", []),
            })

    return {"nodes": nodes, "edges": edges}


# ─────────────────────────────────────────────────────────────────────────────
# NER, linking, and relation extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_entities(text: str, nlp) -> List[Dict[str, str]]:
    """
    Extract named entities from text using spaCy if available, else regex.

    spaCy path: runs the NER component and returns up to 40 entities as
                [{text: str, label: str}] dicts.
    Regex fallback: matches Capitalised phrases (up to 5 words) as generic ENTITY.
    """
    if nlp is not None:
        try:
            # Truncate to 10k chars to keep inference time bounded
            doc  = nlp(text[:10_000])
            ents = [
                {"text": ent.text.strip(), "label": ent.label_}
                for ent in doc.ents
                if ent.text.strip()
            ]
            return ents[:40]   # cap to avoid noise
        except Exception:
            pass   # fall through to regex
    return _regex_ner(text)


def _link_entities(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Canonical form and deduplication.

    For each entity:
    - canonical: lowercase, whitespace-normalised, with common suffix abbreviations
      resolved (e.g. "inc." → "inc").
    - head: last word of the canonical form (used as a loose coreference key).
    - entity_id:  "LABEL::canonical"
    - parent_id:  "LABEL::head"  (groups variants like "Smith Inc." and "Smith")

    Deduplication: only the first occurrence of each entity_id is kept.
    """
    linked = []
    seen   = set()

    for ent in entities:
        txt = ent.get("text", "").strip()
        if not txt:
            continue

        # Normalise to canonical form
        canonical = re.sub(r"\s+", " ", txt.lower())
        canonical = canonical.replace("inc.", "inc").replace("corp.", "corp")

        # Head word (last token) for loose coreference grouping
        head = canonical.split()[-1] if canonical.split() else canonical

        entity_id = f"{ent.get('label', 'ENTITY')}::{canonical}"
        parent_id = f"{ent.get('label', 'ENTITY')}::{head}"

        if entity_id in seen:
            continue   # skip duplicate
        seen.add(entity_id)

        linked.append({
            "text":      txt,
            "label":     ent.get("label", "ENTITY"),
            "canonical": canonical,
            "entity_id": entity_id,
            "parent_id": parent_id,
        })

    return linked


def _extract_relations(text: str, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Lightweight Subject–Relation–Object (SRO) triple extraction.

    For each sentence, checks if two distinct entities appear and a
    relation verb (is, has, uses, owns, etc.) appears between them.
    Returns up to 30 unique (subject, relation, object) triples.

    This is intentionally simple — for a production system, replace with
    a dedicated relation extraction model (e.g. REBEL, SPN4RE).
    """
    ent_values = [e["text"] for e in entities]
    if len(ent_values) < 2:
        return []   # need at least 2 entities to form a relation

    rels: List[Dict[str, str]] = []

    # Split text into sentences
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

    for sent in sentences:
        low = sent.lower()

        for subj in ent_values:
            if subj.lower() not in low:
                continue   # subject not in this sentence

            for obj in ent_values:
                if subj == obj or obj.lower() not in low:
                    continue   # object not in sentence or same as subject

                # Look for a relation verb between the two entities
                m = re.search(
                    r"\b(is|has|uses|owns|acquired|manages|contains|supports|reports|causes)\b",
                    low,
                )
                if m:
                    rels.append({
                        "subject":  subj,
                        "relation": m.group(1),
                        "object":   obj,
                        "type":     "sro",
                    })
                    break   # one relation per (subj, sentence) pair

    # Deduplicate (subject, relation, object) triples
    uniq = []
    seen = set()
    for r in rels:
        key = (r["subject"], r["relation"], r["object"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)

    return uniq[:30]   # cap to avoid bloating the JSON output


def _shared_signals(
    a: Dict[str, Any],
    b: Dict[str, Any],
) -> Tuple[int, int]:
    """
    Count shared entities and shared relations between two chunk dicts.

    Returns (shared_entity_count, shared_relation_count).
    Both counts are used to set edge weights in the graph.
    """
    # Shared canonical entities
    ents_a    = {e["canonical"] for e in a.get("entities", [])}
    ents_b    = {e["canonical"] for e in b.get("entities", [])}
    shared_ent = len(ents_a & ents_b)

    # Shared (subject, object) pairs from the SRO triples
    rel_a     = {(r["subject"].lower(), r["object"].lower()) for r in a.get("relations", [])}
    rel_b     = {(r["subject"].lower(), r["object"].lower()) for r in b.get("relations", [])}
    shared_rel = len(rel_a & rel_b)

    return shared_ent, shared_rel


# ─────────────────────────────────────────────────────────────────────────────
# GNN message-passing step
# ─────────────────────────────────────────────────────────────────────────────

def _graph_vector_for_chunk(
    idx: int,
    neighbors: List[int],
    weights:   List[int],
    embeddings: Optional[List[List[float]]],
) -> List[float]:
    """
    Compute the graph-enriched vector for chunk idx.

    h_graph(Cᵢ) = Σⱼ∈N(i) (wᵢⱼ / Σwᵢⱼ) · e(Cⱼ)

    where N(i) = set of neighbour chunk indices,
          wᵢⱼ  = edge weight (shared entity/relation count + KG prior),
          e(Cⱼ) = embedding vector of chunk j from S6 (or empty list).

    If chunk i has no neighbours, its own embedding is returned unchanged.
    If embeddings are not yet available (RL loop), returns an empty list.
    """
    # No embeddings available (e.g. RL trial loop passes [] for speed)
    if not embeddings or idx >= len(embeddings):
        return []

    # No neighbours: return own embedding unchanged
    if not neighbors:
        return list(embeddings[idx]) if embeddings[idx] else []

    # Collect valid neighbour embeddings and their weights
    valid_vecs:    List[np.ndarray] = []
    valid_weights: List[float]      = []

    for nb, w in zip(neighbors, weights):
        if nb < len(embeddings) and embeddings[nb]:
            valid_vecs.append(np.array(embeddings[nb], dtype=np.float32))
            valid_weights.append(float(w))

    if not valid_vecs:
        # No valid neighbour embeddings — fall back to own embedding
        return list(embeddings[idx]) if embeddings[idx] else []

    # Weighted mean of neighbour embeddings (message aggregation)
    total = sum(valid_weights)
    vec   = sum(
        (v * (wt / total) for v, wt in zip(valid_vecs, valid_weights)),
        np.zeros_like(valid_vecs[0]),
    )
    return vec.tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Regex NER fallback
# ─────────────────────────────────────────────────────────────────────────────

def _regex_ner(text: str) -> List[Dict[str, str]]:
    """
    Language-aware regex NER fallback used when spaCy is not installed.

    Matches Capitalised multi-word phrases (1–5 words) as generic ENTITYs.

    The key problem with the original: in French (and most Romance languages),
    ALL sentence-starters are capitalised, so naive Capitalised-phrase matching
    produces enormous noise ("Sont", "Personnes", "Etat", "Lorsque", …).

    Fix: we apply three filters before accepting an entity:
      1. Minimum length: the phrase must be > 3 characters.
      2. Blocklist: a union of English AND French common sentence-starters
         and grammatical words that are never true named entities.
      3. Position filter: if the match starts exactly at a sentence boundary
         (preceded by ". " or newline), it is likely a sentence-starter
         capitalised by grammar — skip it.

    Returns up to 30 unique entities labelled "ENTITY".
    """
    # Combined English + French false-positive sentence-starters and function words.
    # These are words that appear capitalised in text but are never named entities.
    _BLOCKLIST = {
        # English
        "The","This","That","These","Those","When","Where","What","Which",
        "However","Therefore","Moreover","Furthermore","Although","Because",
        "Since","While","Both","Each","Some","Many","Most","Such",
        "Here","There","Thus","Hence","Also","But","And","Or","Yet",
        # French sentence-starters and function words that get capitalised
        "Les","Des","Une","Dans","Pour","Que","Est","Sur","Par","Avec",
        "Aux","Cette","Leur","Leurs","Dont","Ainsi","Lorsque","Lorsqu",
        "Sont","Ont","Peut","Doit","Doivent","Toute","Tout","Tous",
        "Selon","Sous","Entre","Après","Avant","Sans","Même","Plus",
        "Lors","Afin","Dont","Sauf","Quant","Comme","Bien","Soit",
        "Seul","Seule","Seuls","Seules","Tel","Tels","Telle","Telles",
        "Aucun","Aucune","Chaque","Plusieurs","Elles","Nous","Vous",
        "Notamment","Respectivement","Également",
        # French common nouns that appear capitalised in headings/article titles
        # but are NOT named entities — they name legal concepts, not proper nouns
        "Personnes","Personne","Revenus","Revenu","Impôts","Impôt",
        "Résidents","Résident","Sociétés","Société","Bénéfices","Bénéfice",
        "Dividendes","Dividende","Intérêts","Intérêt","Redevances","Redevance",
        "Salaires","Salaire","Pensions","Pension","Rémunérations","Rémunération",
        "Éléments","Élément","Définitions","Définition","Dispositions","Disposition",
        "Entreprises","Entreprise","Établissements","Établissement","Services",
        "Champ","Application","Objet","Portée","Effets","Entrée","Vigueur",
        "Dénonciation","Ratification","Protocole","Annexe","Amendement",
        # English legal common nouns capitalised in headings
        "Persons","Person","Income","Taxes","Tax","Residents","Resident",
        "Companies","Company","Dividends","Dividend","Interest","Royalties",
        "Salaries","Salary","Pensions","Pension","Benefits","Benefit",
        "Provisions","Provision","Definitions","Definition","Services","Service",
        "Application","Object","Scope","Effects","Entry","Force","Termination",
    }

    entities: List[Dict[str, str]] = []

    # Build a set of sentence-start character offsets to detect grammar capitals.
    # A character is a sentence-start if preceded by ". ", "? ", "! ", or a newline.
    sentence_starts: set = set()
    for m in re.finditer(r"(?:^|(?<=[.!?\n])\s+)", text):
        sentence_starts.add(m.end())

    for m in re.finditer(r"\b([A-ZÀ-Ÿ][a-zà-ÿ]+(?:\s+[A-ZÀ-Ÿ][a-zà-ÿ]+){0,4})\b", text):
        word   = m.group(1)
        offset = m.start()

        # Filter 1: minimum meaningful length
        if len(word) <= 3:
            continue

        # Filter 2: blocklist of known non-entity words
        first_token = word.split()[0]
        if first_token in _BLOCKLIST:
            continue

        # Filter 3: sentence-start position with only one word → grammar capital
        # (Multi-word capitalised phrases at sentence start are still kept because
        # "European Union" or "Tunisian Republic" legitimately start sentences.)
        if offset in sentence_starts and len(word.split()) == 1:
            continue

        entities.append({"text": word, "label": "ENTITY"})

    # Deduplicate by lowercase text
    seen: set = set()
    uniq: List[Dict[str, str]] = []
    for e in entities:
        key = e["text"].lower()
        if key not in seen:
            seen.add(key)
            uniq.append(e)

    return uniq[:30]   # cap to avoid bloating downstream processing