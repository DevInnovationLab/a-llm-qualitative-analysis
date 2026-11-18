"""Turn codings.jsonl into tables: the segment x code matrix and the evidence list."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .coding import current_codings
from .config import Codebook, Project
from .io import read_jsonl


@dataclass
class Results:
    """Coded results for the current codebook."""

    codebook: Codebook
    segments: pd.DataFrame  # one row per segment, with transcript metadata
    matrix: (
        pd.DataFrame
    )  # segment_id x code_id: 1 applied, 0 not applied, NaN not coded
    screened: pd.DataFrame  # segment_id x code_id: True if the pre-screen skipped it
    evidence: pd.DataFrame  # one row per applied code: quote + rationale
    errors: pd.DataFrame  # segments whose latest coding failed


def load_metadata(project: Project) -> pd.DataFrame | None:
    path = project.metadata_path
    if not path.exists():
        return None
    meta = pd.read_csv(path, dtype=str).fillna("")
    if "transcript_id" not in meta.columns:
        raise ValueError(
            f"{path} needs a 'transcript_id' column (the file name without extension)."
        )
    return meta


def load_results(project: Project) -> Results:
    codebook = project.load_codebook()
    code_ids = [c.id for c in codebook.codes]
    segments = pd.DataFrame(read_jsonl(project.segments_path))
    if segments.empty:
        raise FileNotFoundError("No segments found. Run `qual ingest` first.")
    segments["speakers"] = segments["speakers"].apply(lambda s: ", ".join(s))
    meta = load_metadata(project)
    if meta is not None:
        segments = segments.merge(meta, on="transcript_id", how="left")

    codings = current_codings(project, codebook.fingerprint())
    matrix = pd.DataFrame(np.nan, index=segments["segment_id"], columns=code_ids)
    screened = pd.DataFrame(False, index=segments["segment_id"], columns=code_ids)
    evidence_rows, error_rows = [], []
    for seg_id, record in codings.items():
        if seg_id not in matrix.index:
            continue
        if record["status"] != "ok":
            error_rows.append({"segment_id": seg_id, "error": record["error"]})
            continue
        for code_id in code_ids:
            decision = record["decisions"].get(code_id)
            if decision is None:
                matrix.loc[seg_id, code_id] = 0
                screened.loc[seg_id, code_id] = code_id not in record["candidates"]
                continue
            matrix.loc[seg_id, code_id] = int(decision["applies"])
            if decision["applies"]:
                evidence_rows.append(
                    {
                        "code_id": code_id,
                        "segment_id": seg_id,
                        "transcript_id": record["transcript_id"],
                        "evidence": decision["evidence"],
                        "rationale": decision["rationale"],
                        "evidence_verbatim": decision["evidence_verbatim"],
                    }
                )

    # Segments with no candidate codes at all were never sent: all screened out.
    candidates_path = project.outputs / "prescreen.json"
    if candidates_path.exists():
        import json

        candidates = json.loads(candidates_path.read_text())["candidates"]
        for seg_id, cands in candidates.items():
            if not cands and seg_id in matrix.index:
                matrix.loc[seg_id] = 0
                screened.loc[seg_id] = True

    evidence = pd.DataFrame(
        evidence_rows,
        columns=[
            "code_id",
            "segment_id",
            "transcript_id",
            "evidence",
            "rationale",
            "evidence_verbatim",
        ],
    )
    return Results(
        codebook=codebook,
        segments=segments,
        matrix=matrix,
        screened=screened,
        evidence=evidence,
        errors=pd.DataFrame(error_rows, columns=["segment_id", "error"]),
    )
