"""Voyage AI embeddings: caching, the code pre-screen, and semantic search."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Literal

import numpy as np

from .config import Codebook, Project, get_voyage_client, has_voyage_key
from .io import read_jsonl

# Voyage accepts up to 1,000 texts per request, subject to a per-request token cap.
_BATCH_SIZE = 64


def _hash(text: str, model: str, input_type: str | None) -> str:
    return hashlib.sha256(f"{model}|{input_type}|{text}".encode()).hexdigest()[:16]


def embed_texts(
    texts: list[str],
    model: str,
    input_type: Literal["query", "document"] | None,
    cache_path: Path | None = None,
    client=None,
) -> np.ndarray:
    """Embed texts with Voyage, reusing cached vectors for unchanged texts.

    input_type: "document" for the things being searched (segments),
    "query" for what you search with (code definitions, questions).
    Voyage vectors are normalized, so dot product = cosine similarity.
    """
    keys = [_hash(t, model, input_type) for t in texts]
    cache: dict[str, np.ndarray] = {}
    if cache_path and cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        cache = dict(zip(data["keys"].tolist(), data["vectors"], strict=True))

    missing = [i for i, k in enumerate(keys) if k not in cache]
    if missing:
        client = client or get_voyage_client()
        for start in range(0, len(missing), _BATCH_SIZE):
            idx = missing[start : start + _BATCH_SIZE]
            result = client.embed(
                [texts[i] for i in idx], model=model, input_type=input_type
            )
            for i, vector in zip(idx, result.embeddings, strict=True):
                cache[keys[i]] = np.asarray(vector, dtype=np.float32)
        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                cache_path,
                keys=np.array(list(cache.keys())),
                vectors=np.stack(list(cache.values())),
            )

    return np.stack([cache[k] for k in keys])


def segment_embeddings(project: Project, client=None) -> tuple[list[dict], np.ndarray]:
    segments = read_jsonl(project.segments_path)
    if not segments:
        raise FileNotFoundError("No segments found. Run `qual ingest` first.")
    vectors = embed_texts(
        [s["text"] for s in segments],
        model=project.config.models.embedding_model,
        input_type="document",
        cache_path=project.outputs / "embeddings" / "segments.npz",
        client=client,
    )
    return segments, vectors


def code_embeddings(project: Project, codebook: Codebook, client=None) -> np.ndarray:
    return embed_texts(
        [c.embedding_text() for c in codebook.codes],
        model=project.config.models.embedding_model,
        input_type="query",
        cache_path=project.outputs / "embeddings" / "codes.npz",
        client=client,
    )


def select_candidates(
    similarity: np.ndarray,
    code_ids: list[str],
    top_k: int,
    min_similarity: float | None,
) -> list[list[str]]:
    """For each segment (row), return the code ids Claude should evaluate.

    Keeps the top_k most similar codes, plus any code at or above min_similarity.
    Returned ids keep codebook order.
    """
    candidates: list[list[str]] = []
    k = min(top_k, len(code_ids))
    for row in similarity:
        keep = set(np.argsort(-row)[:k].tolist())
        if min_similarity is not None:
            keep |= set(np.flatnonzero(row >= min_similarity).tolist())
        candidates.append([code_ids[j] for j in sorted(keep)])
    return candidates


def prescreen(
    project: Project, codebook: Codebook, client=None
) -> tuple[list[dict], dict[str, list[str]], dict[str, dict[str, float]]]:
    """Return segments, candidate code ids per segment, and similarity scores.

    Without a Voyage key (or with the pre-screen disabled) no embeddings are
    computed: Claude evaluates every code for every segment, and the similarity
    scores are empty.
    """
    cfg = project.config.prefilter
    if not cfg.enabled or not (client is not None or has_voyage_key()):
        segments = read_jsonl(project.segments_path)
        if not segments:
            raise FileNotFoundError("No segments found. Run `qual ingest` first.")
        if cfg.enabled:
            print(
                "Note: VOYAGE_API_KEY is not set, so the embedding pre-screen is "
                "skipped and Claude evaluates every code for every segment.",
                file=sys.stderr,
            )
        all_ids = [c.id for c in codebook.codes]
        return segments, {s["segment_id"]: list(all_ids) for s in segments}, {}

    segments, seg_vectors = segment_embeddings(project, client)
    code_vectors = code_embeddings(project, codebook, client)
    similarity = seg_vectors @ code_vectors.T
    code_ids = [c.id for c in codebook.codes]

    candidate_lists = select_candidates(
        similarity, code_ids, cfg.top_k, cfg.min_similarity
    )

    candidates = {
        s["segment_id"]: c for s, c in zip(segments, candidate_lists, strict=True)
    }
    scores = {
        s["segment_id"]: {
            cid: round(float(v), 4) for cid, v in zip(code_ids, row, strict=True)
        }
        for s, row in zip(segments, similarity, strict=True)
    }
    return segments, candidates, scores


def search(project: Project, query: str, top_n: int = 10) -> list[tuple[dict, float]]:
    """Return the segments most similar to a free-text query."""
    segments, seg_vectors = segment_embeddings(project)
    query_vector = embed_texts(
        [query], project.config.models.embedding_model, input_type="query"
    )[0]
    scores = seg_vectors @ query_vector
    order = np.argsort(-scores)[:top_n]
    return [(segments[i], float(scores[i])) for i in order]
