"""Check Claude's coding against human coders before trusting the full results."""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score

from .config import Project
from .io import read_jsonl
from .results import load_results

CODING_SHEET = "coding"
INSTRUCTIONS = [
    "HOW TO FILL THIS IN",
    "",
    "1. Each person codes their own copy of this file without looking at Claude's results.",
    "2. Put your name or initials in the 'coded_by' column of every row you finish.",
    "   Rows without 'coded_by' are ignored, so you can stop partway.",
    "3. For each code column, enter 1 if the code applies to the segment. Leave it blank (or 0) if not.",
    "4. Use the 'codebook' sheet for definitions. Use 'notes' for anything unclear.",
    "5. Save the file and run: qual validate --human <file> [<file> ...]",
]


def make_sample(project: Project, n: int, seed: int = 42, enrich: bool = True) -> Path:
    """Write a blank human-coding workbook for a sample of segments.

    With enrich=True, half the sample is drawn from the segments most similar to
    each code (so rare codes get enough positives to measure agreement) and half
    at random. The workbook does not show Claude's codes.
    """
    codebook = project.load_codebook()
    segments = pd.DataFrame(read_jsonl(project.segments_path))
    if segments.empty:
        raise FileNotFoundError("No segments found. Run `qual ingest` first.")
    n = min(n, len(segments))
    rng = np.random.default_rng(seed)
    chosen: list[str] = []

    prescreen_path = project.outputs / "prescreen.json"
    if enrich and prescreen_path.exists():
        similarity = json.loads(prescreen_path.read_text())["similarity"]
        sim = pd.DataFrame(similarity).T  # segment_id x code_id
        per_code = max(1, math.ceil((n // 2) / len(codebook.codes)))
        for code in codebook.codes:
            if code.id not in sim.columns:
                continue
            taken = 0
            for seg_id in sim[code.id].sort_values(ascending=False).index:
                if taken >= per_code:
                    break
                if seg_id not in chosen:
                    chosen.append(seg_id)
                    taken += 1
        chosen = chosen[: n // 2]

    remaining = [s for s in segments["segment_id"] if s not in chosen]
    chosen += list(rng.choice(remaining, size=n - len(chosen), replace=False))
    sample = segments.set_index("segment_id").loc[chosen].reset_index()
    sample = sample.sample(frac=1, random_state=seed)  # hide how rows were picked

    sheet = sample[["segment_id", "transcript_id", "text"]].copy()
    for code in codebook.codes:
        sheet[code.id] = ""
    sheet["coded_by"] = ""
    sheet["notes"] = ""

    out = project.validation_dir / "human_coding_template.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pd.DataFrame({"instructions": INSTRUCTIONS}).to_excel(
            writer, sheet_name="instructions", index=False
        )
        sheet.to_excel(writer, sheet_name=CODING_SHEET, index=False)
        pd.DataFrame([c.model_dump() for c in codebook.codes]).assign(
            examples=lambda d: d["examples"].apply(" | ".join)
        ).to_excel(writer, sheet_name="codebook", index=False)
        ws = writer.sheets[CODING_SHEET]
        ws.column_dimensions["C"].width = 90
        ws.freeze_panes = "D2"
    return out


def _to_binary(value) -> float:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return 0.0
    text = str(value).strip().lower()
    if text in {"", "0", "0.0", "no", "n", "false"}:
        return 0.0
    if text in {"1", "1.0", "yes", "y", "x", "true", "si", "sí"}:
        return 1.0
    raise ValueError(f"Unrecognized coding value: {value!r} (use 1 or leave blank)")


def read_human_file(path: Path, code_ids: list[str]) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=CODING_SHEET, dtype=object)
    missing = [c for c in code_ids if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing code columns: {missing}")
    df = df[df["coded_by"].notna() & (df["coded_by"].astype(str).str.strip() != "")]
    out = df[["segment_id", "coded_by"]].copy()
    out["coded_by"] = out["coded_by"].astype(str).str.strip()
    for code_id in code_ids:
        out[code_id] = df[code_id].map(_to_binary)
    return out


def _kappa(a: pd.Series, b: pd.Series) -> float:
    if a.nunique() == 1 and b.nunique() == 1:
        return 1.0 if a.iloc[0] == b.iloc[0] else 0.0
    return float(cohen_kappa_score(a, b))


def compare(human: pd.DataFrame, claude: pd.Series, screened: pd.Series) -> dict:
    """Agreement statistics for one code: human (0/1) vs Claude (0/1)."""
    tp = int(((human == 1) & (claude == 1)).sum())
    fp = int(((human == 0) & (claude == 1)).sum())
    fn = int(((human == 1) & (claude == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return {
        "n": len(human),
        "human_positive": int(human.sum()),
        "claude_positive": int(claude.sum()),
        "percent_agreement": round(float((human == claude).mean()) * 100, 1),
        "kappa": round(_kappa(human, claude), 3),
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "missed_by_prescreen": int(((human == 1) & screened).sum()),
    }


def flag(row: dict) -> str:
    notes = []
    if row["human_positive"] < 5:
        notes.append("too few human positives to judge; code more segments")
    elif row["kappa"] < 0.6:
        notes.append("low agreement: tighten the definition / include-exclude rules")
    if row["missed_by_prescreen"] > 0:
        notes.append("pre-screen dropped human positives: raise prefilter.top_k")
    return "; ".join(notes) or "ok"


def run_validation(
    project: Project, human_paths: list[Path]
) -> tuple[pd.DataFrame, Path]:
    """Compare Claude with human coders; write validation_report.xlsx."""
    results = load_results(project)
    code_ids = [c.id for c in results.codebook.codes]
    humans = pd.concat(
        [read_human_file(p, code_ids) for p in human_paths], ignore_index=True
    )
    if humans.empty:
        raise ValueError("No coded rows found (fill in the 'coded_by' column).")

    coded = humans[humans["segment_id"].isin(results.matrix.dropna(how="all").index)]
    not_coded = sorted(set(humans["segment_id"]) - set(coded["segment_id"]))

    rows, disagreements = [], []
    for code_id in code_ids:
        claude = results.matrix.loc[coded["segment_id"], code_id].to_numpy()
        screened = results.screened.loc[coded["segment_id"], code_id].to_numpy()
        stats = compare(
            coded[code_id].reset_index(drop=True),
            pd.Series(claude),
            pd.Series(screened),
        )
        stats = {"code_id": code_id, **stats}
        stats["flag"] = flag(stats)
        rows.append(stats)

        evidence = results.evidence.set_index(["segment_id", "code_id"])
        for (_, h), c in zip(coded.iterrows(), claude, strict=True):
            if h[code_id] != c:
                key = (h["segment_id"], code_id)
                ev = evidence.loc[key] if key in evidence.index else None
                disagreements.append(
                    {
                        "code_id": code_id,
                        "segment_id": h["segment_id"],
                        "coded_by": h["coded_by"],
                        "human": int(h[code_id]),
                        "claude": int(c),
                        "claude_evidence": "" if ev is None else ev["evidence"],
                        "claude_rationale": "" if ev is None else ev["rationale"],
                        "screened_out": bool(
                            results.screened.loc[h["segment_id"], code_id]
                        ),
                        "text": results.segments.set_index("segment_id").loc[
                            h["segment_id"], "text"
                        ],
                    }
                )

    summary = pd.DataFrame(rows)
    human_vs_human = _inter_human(humans, code_ids)

    out = project.validation_dir / "validation_report.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="claude_vs_human", index=False)
        if human_vs_human is not None:
            human_vs_human.to_excel(writer, sheet_name="human_vs_human", index=False)
        pd.DataFrame(disagreements).to_excel(
            writer, sheet_name="disagreements", index=False
        )
        if not_coded:
            pd.DataFrame({"segment_id": not_coded}).to_excel(
                writer, sheet_name="not_coded_by_claude", index=False
            )
    return summary, out


def _inter_human(humans: pd.DataFrame, code_ids: list[str]) -> pd.DataFrame | None:
    """Kappa between each pair of human coders on the segments both coded."""
    coders = sorted(humans["coded_by"].unique())
    if len(coders) < 2:
        return None
    rows = []
    for a, b in itertools.combinations(coders, 2):
        ha = (
            humans[humans["coded_by"] == a]
            .drop_duplicates("segment_id")
            .set_index("segment_id")
        )
        hb = (
            humans[humans["coded_by"] == b]
            .drop_duplicates("segment_id")
            .set_index("segment_id")
        )
        shared = ha.index.intersection(hb.index)
        if len(shared) == 0:
            continue
        for code_id in code_ids:
            rows.append(
                {
                    "coder_a": a,
                    "coder_b": b,
                    "code_id": code_id,
                    "n": len(shared),
                    "kappa": round(
                        _kappa(ha.loc[shared, code_id], hb.loc[shared, code_id]), 3
                    ),
                }
            )
    return pd.DataFrame(rows) if rows else None
