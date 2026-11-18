"""Inductive step: Claude reads the transcripts and proposes a draft codebook."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from pydantic import BaseModel, Field

from . import llm
from .config import Code, Codebook, Project, get_claude_client, has_voyage_key
from .embed import segment_embeddings
from .io import read_jsonl

# Rough budget for transcript text in one request (Claude's context is 1M tokens;
# leave room for instructions and the response). ~3 characters per token.
_MAX_TRANSCRIPT_CHARS = 1_500_000


class DraftCodebook(BaseModel):
    """Claude's proposed codebook."""

    codes: list[Code]
    notes: str = Field(
        description="Notes for the research team: overlaps between codes, "
        "judgment calls, and themes that were too thin to become codes."
    )


def _group_by_transcript(segments: list[dict]) -> str:
    grouped: dict[str, list[str]] = defaultdict(list)
    for s in segments:
        grouped[s["transcript_id"]].append(s["text"])
    return "\n\n".join(
        f'<transcript id="{tid}">\n' + "\n\n".join(texts) + "\n</transcript>"
        for tid, texts in grouped.items()
    )


def representative_sample(project: Project, budget_chars: int) -> list[dict]:
    """Pick segments that cover the range of content when the corpus is too large.

    Clusters segment embeddings and takes the segments closest to each cluster
    centre, round-robin, until the character budget is used. Without a Voyage
    key, takes evenly spaced segments across all transcripts instead.
    """
    if not has_voyage_key():
        all_segments = read_jsonl(project.segments_path)
        total = sum(len(s["text"]) for s in all_segments)
        step = max(1, -(-total // budget_chars))
        return all_segments[::step]

    from sklearn.cluster import KMeans

    segments, vectors = segment_embeddings(project)
    n_clusters = min(len(segments), max(10, len(segments) // 20))
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=0).fit(vectors)
    distances = np.linalg.norm(vectors - km.cluster_centers_[km.labels_], axis=1)

    queues = [
        sorted(np.flatnonzero(km.labels_ == c), key=lambda i: distances[i])
        for c in range(n_clusters)
    ]
    chosen: list[int] = []
    used = 0
    while any(queues) and used < budget_chars:
        for queue in queues:
            if queue and used < budget_chars:
                i = queue.pop(0)
                chosen.append(i)
                used += len(segments[i]["text"])
    return [segments[i] for i in sorted(chosen)]


def draft_codebook(project: Project) -> tuple[DraftCodebook, bool]:
    """Ask Claude for a draft codebook. Returns (draft, whether the corpus was sampled)."""
    cfg = project.config
    segments = read_jsonl(project.segments_path)
    if not segments:
        raise FileNotFoundError("No segments found. Run `qual ingest` first.")

    sampled = sum(len(s["text"]) for s in segments) > _MAX_TRANSCRIPT_CHARS
    if sampled:
        segments = representative_sample(project, _MAX_TRANSCRIPT_CHARS)

    existing = ""
    # The untouched template from `qual init` is not a real codebook: start fresh.
    if project.codebook_path.exists() and not project.load_codebook().is_placeholder():
        existing = (
            "The team already has a codebook, below. Keep its codes and ids unless "
            "the transcripts clearly show a code should be split, merged or redefined, "
            "and add codes for important themes it misses.\n\n"
            f"<existing_codebook>\n{project.load_codebook().as_prompt()}\n</existing_codebook>"
        )

    questions = "\n".join(f"- {q}" for q in cfg.research_questions) or "(none given)"
    system = (
        "You are an experienced qualitative researcher helping a research team "
        "build a codebook from interview and focus group transcripts. The team will "
        "review and edit your draft before any coding happens, so be explicit about "
        "judgment calls."
    )
    user = f"""<study>
Name: {cfg.name}
{cfg.description}
Context: {cfg.context or "(none given)"}
Research questions:
{questions}
</study>

{existing}

{_group_by_transcript(segments)}

Propose a codebook for analyzing these transcripts against the research questions.
- Ground every code in what participants actually said; aim for 10-25 codes.
- Each code must be distinguishable from the others: use include/exclude rules to
  draw the boundaries between codes that could be confused.
- `id` is short snake_case; `name` and `definition` are in English.
- `examples` are 1-3 short verbatim quotes, in the transcript's original language.
{"- You are seeing a representative sample of segments, not every transcript in full." if sampled else ""}"""

    params = llm.build_params(
        model=cfg.models.drafting_model,
        system=system,
        user=user,
        effort=cfg.models.drafting_effort,
        max_tokens=64000,
        output_model=DraftCodebook,
    )
    draft, _ = llm.call_json(get_claude_client(), params, DraftCodebook)
    return draft, sampled


def save_draft(project: Project, draft: DraftCodebook, sampled: bool) -> None:
    Codebook(codes=draft.codes).save(project.outputs / "codebook_draft.yaml")
    note = draft.notes
    if sampled:
        note = (
            "NOTE: the corpus was too large to read in full; Claude saw a "
            "representative sample of segments.\n\n" + note
        )
    (project.outputs / "codebook_draft_notes.md").write_text(note, encoding="utf-8")
