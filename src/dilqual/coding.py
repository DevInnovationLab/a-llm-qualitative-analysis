"""Deductive step: apply the codebook to every segment."""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from pydantic import BaseModel, Field
from tqdm import tqdm

from . import llm
from .config import Codebook, Project, get_claude_client
from .embed import prescreen
from .io import append_jsonl, read_jsonl

MAX_TOKENS = 16000


class CodeDecision(BaseModel):
    """Claude's decision for one code on one segment."""

    code_id: str
    applies: bool
    evidence: str = Field(
        description="Verbatim quote from the segment supporting the code; empty if it does not apply."
    )
    rationale: str = Field(description="One sentence explaining the decision.")


class SegmentCoding(BaseModel):
    """All code decisions for one segment."""

    decisions: list[CodeDecision]


def system_prompt(project: Project, codebook: Codebook) -> list[dict[str, Any]]:
    cfg = project.config
    questions = "\n".join(f"- {q}" for q in cfg.research_questions) or "(none given)"
    instructions = f"""You are an experienced qualitative researcher coding transcript segments \
for a research team. Apply the codebook exactly as written: the team will compare your \
coding against human coders.

<study>
Name: {cfg.name}
{cfg.description}
Context: {cfg.context or "(none given)"}
Research questions:
{questions}
</study>

How to code:
- You will receive one segment and a list of code ids to evaluate. Return exactly one \
decision per listed code id, and no others.
- Apply a code only when the segment substantively expresses it, following each code's \
definition and its apply/do-not-apply rules. A passing mention is not enough.
- Codes are not mutually exclusive; a segment can have several or none.
- Moderator/interviewer questions provide context but are not evidence on their own: code \
what participants say.
- `evidence` is a short verbatim quote copied exactly from the segment (original language). \
Leave it empty when the code does not apply.
- `rationale` is one sentence in English."""
    return llm.cached_system(
        instructions, f"<codebook>\n{codebook.as_prompt()}\n</codebook>"
    )


def user_prompt(segment: dict, candidate_ids: list[str]) -> str:
    return (
        f'<segment id="{segment["segment_id"]}" transcript="{segment["transcript_id"]}">\n'
        f"{segment['text']}\n</segment>\n\n"
        f"Evaluate these codes: {', '.join(candidate_ids)}"
    )


def _normalize(text: str) -> str:
    text = re.sub(r"[“”«»\"]", '"', text)
    text = re.sub(r"[‘’']", "'", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def is_verbatim(evidence: str, segment_text: str) -> bool:
    """Whether the quoted evidence really appears in the segment."""
    if not evidence.strip():
        return True
    quote = _normalize(evidence).strip(" .…\"'")
    return all(
        part.strip() in _normalize(segment_text)
        for part in re.split(r"\.\.\.|…", quote)
        if part.strip()
    )


def make_record(
    segment: dict,
    candidate_ids: list[str],
    coding: SegmentCoding | None,
    *,
    codebook_hash: str,
    model: str,
    error: str = "",
) -> dict[str, Any]:
    """Normalize Claude's answer into the record stored in codings.jsonl."""
    decisions: dict[str, dict[str, Any]] = {}
    if coding is not None:
        for d in coding.decisions:
            if d.code_id in candidate_ids and d.code_id not in decisions:
                decisions[d.code_id] = {
                    "applies": d.applies,
                    "evidence": d.evidence if d.applies else "",
                    "rationale": d.rationale,
                    "evidence_verbatim": is_verbatim(d.evidence, segment["text"])
                    if d.applies
                    else True,
                }
        missing = [c for c in candidate_ids if c not in decisions]
        if missing and not error:
            error = f"No decision returned for: {', '.join(missing)}"
    return {
        "segment_id": segment["segment_id"],
        "transcript_id": segment["transcript_id"],
        "codebook_hash": codebook_hash,
        "model": model,
        "candidates": candidate_ids,
        "decisions": decisions,
        "status": "error" if error else "ok",
        "error": error,
    }


def current_codings(project: Project, codebook_hash: str) -> dict[str, dict]:
    """Latest successful coding per segment for this codebook version."""
    latest: dict[str, dict] = {}
    for record in read_jsonl(project.codings_path):
        if record["codebook_hash"] != codebook_hash:
            continue
        previous = latest.get(record["segment_id"])
        if previous is None or previous["status"] != "ok" or record["status"] == "ok":
            latest[record["segment_id"]] = record
    return latest


def plan(
    project: Project,
) -> tuple[Codebook, list[dict], dict[str, list[str]], list[dict]]:
    """Work out which segments still need coding.

    Returns codebook, all segments, candidates per segment, and segments to code.
    """
    codebook = project.load_codebook()
    segments, candidates, scores = prescreen(project, codebook)
    (project.outputs / "prescreen.json").write_text(
        json.dumps({"candidates": candidates, "similarity": scores}), encoding="utf-8"
    )
    done = current_codings(project, codebook.fingerprint())
    todo = [
        s
        for s in segments
        if candidates[s["segment_id"]]
        and done.get(s["segment_id"], {}).get("status") != "ok"
    ]
    return codebook, segments, candidates, todo


def estimate(project: Project) -> dict[str, Any]:
    """Rough size of a coding run (characters / 3 ≈ tokens)."""
    codebook, segments, candidates, todo = plan(project)
    system_chars = sum(len(b["text"]) for b in system_prompt(project, codebook))
    segment_chars = sum(len(s["text"]) for s in todo)
    n_pairs = sum(len(candidates[s["segment_id"]]) for s in todo)
    return {
        "segments_total": len(segments),
        "segments_to_code": len(todo),
        "codes": len(codebook.codes),
        "code_decisions_requested": n_pairs,
        "code_decisions_screened_out": len(todo) * len(codebook.codes) - n_pairs,
        "approx_input_tokens_uncached": (segment_chars + 100 * len(todo)) // 3,
        "approx_system_tokens_per_request (cached after first)": system_chars // 3,
        "approx_output_tokens": n_pairs * 60,
    }


def code_live(project: Project, workers: int = 4) -> dict[str, int]:
    """Code segments with concurrent live requests; results append to codings.jsonl."""
    codebook, _, candidates, todo = plan(project)
    if not todo:
        return {"coded": 0, "errors": 0}

    client = get_claude_client()
    models = project.config.models
    system = system_prompt(project, codebook)
    codebook_hash = codebook.fingerprint()

    def work(segment: dict) -> dict:
        cands = candidates[segment["segment_id"]]
        params = llm.build_params(
            model=models.coding_model,
            system=system,
            user=user_prompt(segment, cands),
            effort=models.coding_effort,
            max_tokens=MAX_TOKENS,
            output_model=SegmentCoding,
        )
        try:
            coding, _ = llm.call_json(client, params, SegmentCoding)
            return make_record(
                segment,
                cands,
                coding,
                codebook_hash=codebook_hash,
                model=models.coding_model,
            )
        except Exception as exc:  # recorded and retried on the next run
            return make_record(
                segment,
                cands,
                None,
                codebook_hash=codebook_hash,
                model=models.coding_model,
                error=f"{type(exc).__name__}: {exc}",
            )

    counts = {"coded": 0, "errors": 0}
    # Send the first request alone so the cached system prompt is written once
    # before the concurrent requests start reading it.
    first, rest = todo[0], todo[1:]
    with tqdm(total=len(todo), desc="Coding segments") as bar:
        for record in [work(first)]:
            append_jsonl(project.codings_path, record)
            counts["coded" if record["status"] == "ok" else "errors"] += 1
            bar.update()
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(work, s) for s in rest]
            for future in as_completed(futures):
                record = future.result()
                append_jsonl(project.codings_path, record)
                counts["coded" if record["status"] == "ok" else "errors"] += 1
                bar.update()
    return counts


# ---------------------------------------------------------------------------
# Batch API: half price, results within 24h (usually under an hour)
# ---------------------------------------------------------------------------


def _batches_file(project: Project):
    return project.outputs / "batches.json"


def submit_batch(project: Project) -> str | None:
    """Submit all segments that still need coding as one Message Batch."""
    codebook, _, candidates, todo = plan(project)
    if not todo:
        return None
    models = project.config.models
    system = system_prompt(project, codebook)
    requests = [
        {
            "custom_id": s["segment_id"],
            "params": llm.build_params(
                model=models.coding_model,
                system=system,
                user=user_prompt(s, candidates[s["segment_id"]]),
                effort=models.coding_effort,
                max_tokens=MAX_TOKENS,
                output_model=SegmentCoding,
            ),
        }
        for s in todo
    ]
    batch = get_claude_client().messages.batches.create(requests=requests)
    state = (
        json.loads(_batches_file(project).read_text())
        if _batches_file(project).exists()
        else []
    )
    state.append(
        {
            "batch_id": batch.id,
            "codebook_hash": codebook.fingerprint(),
            "model": models.coding_model,
            "candidates": {s["segment_id"]: candidates[s["segment_id"]] for s in todo},
            "collected": False,
        }
    )
    _batches_file(project).write_text(json.dumps(state, indent=1), encoding="utf-8")
    return batch.id


def collect_batches(project: Project) -> list[dict[str, Any]]:
    """Collect results from finished batches. Returns a status line per batch."""
    path = _batches_file(project)
    if not path.exists():
        return []
    state = json.loads(path.read_text())
    client = get_claude_client()
    segments = {s["segment_id"]: s for s in read_jsonl(project.segments_path)}
    report = []
    for entry in state:
        if entry["collected"]:
            continue
        batch = client.messages.batches.retrieve(entry["batch_id"])
        if batch.processing_status != "ended":
            report.append(
                {
                    "batch_id": entry["batch_id"],
                    "status": batch.processing_status,
                    "processing": batch.request_counts.processing,
                }
            )
            continue
        ok = errors = 0
        for result in client.messages.batches.results(entry["batch_id"]):
            segment = segments.get(result.custom_id)
            if segment is None:
                continue
            cands = entry["candidates"][result.custom_id]
            kwargs = {"codebook_hash": entry["codebook_hash"], "model": entry["model"]}
            coding, error = None, ""
            if result.result.type == "succeeded":
                message = result.result.message
                try:
                    llm.check_refusal(message)
                    coding = SegmentCoding.model_validate_json(llm.text_of(message))
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
            else:
                error = f"batch result: {result.result.type}"
            record = make_record(segment, cands, coding, error=error, **kwargs)
            append_jsonl(project.codings_path, record)
            ok += record["status"] == "ok"
            errors += record["status"] != "ok"
        entry["collected"] = True
        report.append(
            {
                "batch_id": entry["batch_id"],
                "status": "collected",
                "ok": ok,
                "errors": errors,
            }
        )
    path.write_text(json.dumps(state, indent=1), encoding="utf-8")
    return report
