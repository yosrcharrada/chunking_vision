"""
S5 — Entity Intelligence
Hierarchical linking/disambiguation, lightweight relation extraction,
and typed-edge graph construction with persistent KG support.
"""

import json
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

KG_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "kg_store.json")


class KGStore:
    def __init__(self, path: str = KG_STORE_PATH):
        self.path = os.path.abspath(path)
        self.entity_cooccurrence: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.entity_chunk_index: Dict[str, List[str]] = defaultdict(list)
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.path):
            return
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            for a, neighbors in data.get("cooccurrence", {}).items():
                for b, w in neighbors.items():
                    self.entity_cooccurrence[a][b] = int(w)
            self.entity_chunk_index = defaultdict(list, data.get("chunk_index", {}))
        except Exception:
            pass

    def save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "cooccurrence": {k: dict(v) for k, v in self.entity_cooccurrence.items()},
                        "chunk_index": dict(self.entity_chunk_index),
                    },
                    fh,
                    ensure_ascii=False,
                    indent=2,
                )
        except Exception:
            pass

    def add_chunk_entities(self, chunk_id: str, entities: List[str]) -> None:
        for ent in entities:
            if chunk_id not in self.entity_chunk_index[ent]:
                self.entity_chunk_index[ent].append(chunk_id)
        for i, a in enumerate(entities):
            for b in entities[i + 1:]:
                self.entity_cooccurrence[a][b] += 1
                self.entity_cooccurrence[b][a] += 1

    def get_prior_weight(self, a: str, b: str) -> int:
        return int(self.entity_cooccurrence.get(a, {}).get(b, 0))


_kg_store = KGStore()
_nlp = None


def _get_nlp():
    global _nlp
    if _nlp is not None:
        return _nlp
    try:
        import spacy  # noqa: E402
        _nlp = spacy.load("en_core_web_sm", disable=["parser", "tagger", "lemmatizer"])
    except Exception:
        _nlp = None
    return _nlp


def enrich_graph(chunks: List[Dict], embeddings: Optional[List[List[float]]], config: Dict[str, Any]) -> List[Dict]:
    if not chunks:
        return chunks
    enriched = [dict(c) for c in chunks]
    job_id = config.get("job_id", "unknown")
    nlp = _get_nlp()

    for i, chunk in enumerate(enriched):
        raw_entities = _extract_entities(chunk.get("text", ""), nlp)
        linked = _link_entities(raw_entities)
        relations = _extract_relations(chunk.get("text", ""), linked)
        enriched[i]["entities"] = linked
        enriched[i]["relations"] = relations

    edge_counts: Dict[Tuple[int, int], int] = defaultdict(int)
    edge_types: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for i in range(len(enriched)):
        for j in range(i + 1, len(enriched)):
            shared, relation_shared = _shared_signals(enriched[i], enriched[j])
            if shared > 0:
                pair = (i, j)
                edge_counts[pair] += shared
                edge_types[pair].append("shared_entity")
            if relation_shared > 0:
                pair = (i, j)
                edge_counts[pair] += relation_shared
                edge_types[pair].append("relation_bridge")
            for ei in {e["canonical"] for e in enriched[i].get("entities", [])}:
                for ej in {e["canonical"] for e in enriched[j].get("entities", [])}:
                    prior = _kg_store.get_prior_weight(ei, ej)
                    if prior > 0:
                        pair = (i, j)
                        edge_counts[pair] += prior
                        edge_types[pair].append("kg_prior")

    for i, chunk in enumerate(enriched):
        neighbors: List[int] = []
        weights: List[int] = []
        typed: List[Dict[str, Any]] = []
        for (a, b), cnt in edge_counts.items():
            if a == i or b == i:
                nb = b if a == i else a
                neighbors.append(nb)
                weights.append(cnt)
                typed.append({"target": nb, "weight": cnt, "types": sorted(set(edge_types.get((a, b), [])))})
        chunk["graph_neighbors"] = neighbors
        chunk["typed_edges"] = typed
        chunk["graph_vector"] = _graph_vector_for_chunk(i, neighbors, weights, embeddings)

    for i, chunk in enumerate(enriched):
        cid = f"{job_id}::C{i}"
        entity_texts = [e["canonical"] for e in chunk.get("entities", [])]
        _kg_store.add_chunk_entities(cid, entity_texts)
    _kg_store.save()
    return enriched


def build_entity_graph_data(chunks: List[Dict]) -> Dict[str, Any]:
    nodes = []
    for i, chunk in enumerate(chunks):
        nodes.append({
            "id": i,
            "label": f"C{i}",
            "entity_count": len(chunk.get("entities", [])),
            "relation_count": len(chunk.get("relations", [])),
        })
    edges: List[Dict[str, Any]] = []
    seen = set()
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
                "types": edge.get("types", []),
            })
    return {"nodes": nodes, "edges": edges}


def _extract_entities(text: str, nlp) -> List[Dict[str, str]]:
    if nlp is not None:
        try:
            doc = nlp(text[:10_000])
            ents = [{"text": ent.text.strip(), "label": ent.label_} for ent in doc.ents if ent.text.strip()]
            return ents[:40]
        except Exception:
            pass
    return _regex_ner(text)


def _link_entities(entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    linked = []
    seen = set()
    for ent in entities:
        txt = ent.get("text", "").strip()
        if not txt:
            continue
        canonical = re.sub(r"\s+", " ", txt.lower())
        canonical = canonical.replace("inc.", "inc").replace("corp.", "corp")
        head = canonical.split()[-1] if canonical.split() else canonical
        entity_id = f"{ent.get('label', 'ENTITY')}::{canonical}"
        parent_id = f"{ent.get('label', 'ENTITY')}::{head}"
        if entity_id in seen:
            continue
        seen.add(entity_id)
        linked.append({
            "text": txt,
            "label": ent.get("label", "ENTITY"),
            "canonical": canonical,
            "entity_id": entity_id,
            "parent_id": parent_id,
        })
    return linked


def _extract_relations(text: str, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    ent_values = [e["text"] for e in entities]
    if len(ent_values) < 2:
        return []
    rels: List[Dict[str, str]] = []
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    for sent in sentences:
        low = sent.lower()
        for subj in ent_values:
            if subj.lower() not in low:
                continue
            for obj in ent_values:
                if subj == obj or obj.lower() not in low:
                    continue
                m = re.search(r"\b(is|has|uses|owns|acquired|manages|contains|supports|reports|causes)\b", low)
                if m:
                    rels.append({"subject": subj, "relation": m.group(1), "object": obj, "type": "sro"})
                    break
    uniq = []
    seen = set()
    for r in rels:
        key = (r["subject"], r["relation"], r["object"])
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    return uniq[:30]


def _shared_signals(a: Dict[str, Any], b: Dict[str, Any]) -> Tuple[int, int]:
    ents_a = {e["canonical"] for e in a.get("entities", [])}
    ents_b = {e["canonical"] for e in b.get("entities", [])}
    shared_ent = len(ents_a & ents_b)
    rel_a = {(r["subject"].lower(), r["object"].lower()) for r in a.get("relations", [])}
    rel_b = {(r["subject"].lower(), r["object"].lower()) for r in b.get("relations", [])}
    shared_rel = len(rel_a & rel_b)
    return shared_ent, shared_rel


def _graph_vector_for_chunk(
    idx: int,
    neighbors: List[int],
    weights: List[int],
    embeddings: Optional[List[List[float]]],
) -> List[float]:
    if not embeddings or idx >= len(embeddings):
        return []
    if not neighbors:
        return list(embeddings[idx]) if embeddings[idx] else []
    valid_vecs: List[np.ndarray] = []
    valid_weights: List[float] = []
    for nb, w in zip(neighbors, weights):
        if nb < len(embeddings) and embeddings[nb]:
            valid_vecs.append(np.array(embeddings[nb], dtype=np.float32))
            valid_weights.append(float(w))
    if not valid_vecs:
        return list(embeddings[idx]) if embeddings[idx] else []
    total = sum(valid_weights)
    vec = sum((v * (wt / total) for v, wt in zip(valid_vecs, valid_weights)), np.zeros_like(valid_vecs[0]))
    return vec.tolist()


def _regex_ner(text: str) -> List[Dict[str, str]]:
    entities: List[Dict[str, str]] = []
    for m in re.finditer(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4})\b", text):
        word = m.group(1)
        if len(word) > 3 and word not in {"The", "This", "That", "These", "Those", "When", "Where", "What", "Which"}:
            entities.append({"text": word, "label": "ENTITY"})
    seen = set()
    uniq = []
    for e in entities:
        key = e["text"].lower()
        if key not in seen:
            seen.add(key)
            uniq.append(e)
    return uniq[:30]
