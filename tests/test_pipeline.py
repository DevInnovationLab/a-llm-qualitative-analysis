from __future__ import annotations

from html import escape

import numpy as np
import pandas as pd
import pytest

from dilqual import coding, llm
from dilqual.config import Project
from dilqual.embed import select_candidates
from dilqual.ingest import ingest
from dilqual.io import read_jsonl
from dilqual.report import build_reports
from dilqual.validate import CODING_SHEET, compare, make_sample, run_validation


def test_select_candidates_top_k_and_threshold():
    sim = np.array([[0.9, 0.1, 0.5, 0.45], [0.1, 0.2, 0.3, 0.4]])
    ids = ["a", "b", "c", "d"]
    assert select_candidates(sim, ids, top_k=2, min_similarity=None) == [
        ["a", "c"],
        ["c", "d"],
    ]
    assert select_candidates(sim, ids, top_k=1, min_similarity=0.44) == [
        ["a", "c", "d"],
        ["d"],
    ]


def test_is_verbatim():
    text = "PARTICIPANT 1: We are like sisters now. When my child was sick…"
    assert coding.is_verbatim("we are  like sisters now", text)
    assert coding.is_verbatim("We are like sisters ... my child was sick", text)
    assert not coding.is_verbatim("We are like brothers", text)


def test_build_params_structured_output_and_cache():
    system = llm.cached_system("instructions", "codebook")
    params = llm.build_params(
        model="claude-opus-5-5",
        system=system,
        user="u",
        effort="medium",
        max_tokens=100,
        output_model=coding.SegmentCoding,
    )
    assert params["output_config"]["effort"] == "medium"
    assert params["output_config"]["format"]["type"] == "json_schema"
    assert params["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert "cache_control" not in params["system"][0]


def test_make_record_drops_unrequested_and_flags_missing():
    seg = {"segment_id": "s1", "transcript_id": "t", "text": "hello world"}
    result = coding.SegmentCoding(
        decisions=[
            coding.CodeDecision(
                code_id="a", applies=True, evidence="hello", rationale="r"
            ),
            coding.CodeDecision(
                code_id="zzz", applies=True, evidence="x", rationale="r"
            ),
        ]
    )
    record = coding.make_record(seg, ["a", "b"], result, codebook_hash="h", model="m")
    assert set(record["decisions"]) == {"a"}
    assert record["status"] == "error"
    assert "b" in record["error"]


def test_full_pipeline_offline(project_dir, fakes):
    project = Project(project_dir)
    ingest(project)

    counts = coding.code_live(project, workers=2)
    assert counts == {"coded": 8, "errors": 0}
    # Opus 5.5 requests opt into server-side refusal fallbacks
    assert all(r["fallbacks"] == "default" for r in fakes.claude.messages.requests)
    # top_k=3 of 5 codes: Claude only sees 3 codes per segment
    assert all(
        len(r["messages"][0]["content"].rsplit(": ", 1)[1].split(", ")) == 3
        for r in fakes.claude.messages.requests
    )

    # Re-running codes nothing new; embeddings come from cache.
    voyage_calls = fakes.voyage.calls
    assert coding.code_live(project) == {"coded": 0, "errors": 0}
    assert fakes.voyage.calls == voyage_calls

    xlsx, html, results = build_reports(project)
    assert xlsx.exists() and html.exists()
    sheets = pd.read_excel(xlsx, sheet_name=None)
    assert {
        "summary",
        "by_transcript",
        "by_village",
        "evidence",
        "segments",
        "codebook",
    } <= set(sheets)
    assert results.matrix.notna().all().all()
    assert results.evidence["evidence_verbatim"].all()
    assert not results.evidence.empty
    page = html.read_text()
    assert all(escape(q) in page for q in results.evidence["evidence"])

    # Validation: one human agrees with Claude, one human disagrees on one code.
    template = make_sample(project, n=8)
    sheet = pd.read_excel(template, sheet_name=CODING_SHEET)
    code_ids = [c.id for c in results.codebook.codes]
    for code_id in code_ids:
        sheet[code_id] = results.matrix.loc[sheet["segment_id"], code_id].to_numpy()
    sheet["coded_by"] = "AB"
    agree_path = project.validation_dir / "ab.xlsx"
    sheet.to_excel(agree_path, sheet_name=CODING_SHEET, index=False)

    other = sheet.copy()
    other["coded_by"] = "CD"
    other["social_support"] = 1 - other["social_support"]
    other_path = project.validation_dir / "cd.xlsx"
    other.to_excel(other_path, sheet_name=CODING_SHEET, index=False)

    summary, report_path = run_validation(project, [agree_path])
    assert (summary["percent_agreement"] == 100).all()

    summary, report_path = run_validation(project, [agree_path, other_path])
    report = pd.read_excel(report_path, sheet_name=None)
    assert "human_vs_human" in report
    social = summary.set_index("code_id").loc["social_support"]
    assert social["percent_agreement"] == 50.0
    assert len(report["disagreements"]) == 8


def test_codebook_change_triggers_recoding(project_dir, fakes):
    project = Project(project_dir)
    ingest(project)
    coding.code_live(project)
    cb = project.codebook_path
    text = cb.read_text().replace("Irregular or seasonal income", "Irregular income")
    cb.write_text(text, encoding="utf-8")
    assert coding.code_live(project)["coded"] == 8
    assert len(read_jsonl(project.codings_path)) == 16


def test_compare_metrics():
    human = pd.Series([1, 1, 0, 0])
    claude = pd.Series([1, 0, 0, 1])
    stats = compare(human, claude, pd.Series([False, True, False, False]))
    assert stats["percent_agreement"] == 50.0
    assert stats["precision"] == 0.5
    assert stats["recall"] == 0.5
    assert stats["missed_by_prescreen"] == 1
    assert stats["kappa"] == pytest.approx(0.0)


def test_runs_with_anthropic_key_only(project_dir, fakes, monkeypatch):
    """No Voyage key: no embedding calls, Claude evaluates every code."""
    monkeypatch.setattr("dilqual.embed.has_voyage_key", lambda: False)
    project = Project(project_dir)
    ingest(project)

    assert coding.estimate(project)["code_decisions_screened_out"] == 0
    assert coding.code_live(project, workers=2) == {"coded": 8, "errors": 0}
    assert fakes.voyage.calls == 0
    assert all(
        len(r["messages"][0]["content"].rsplit(": ", 1)[1].split(", ")) == 5
        for r in fakes.claude.messages.requests
    )

    xlsx, html, results = build_reports(project)
    assert results.matrix.notna().all().all()
    assert not results.screened.any().any()
    # The validation sample falls back to a purely random draw.
    sheet = pd.read_excel(make_sample(project, n=6), sheet_name=CODING_SHEET)
    assert len(sheet) == 6


def test_placeholder_codebook_is_not_coded(project_dir, fakes):
    from dilqual import templates

    project = Project(project_dir)
    ingest(project)
    project.codebook_path.write_text(templates.CODEBOOK_YAML, encoding="utf-8")
    assert project.load_codebook().is_placeholder()
    with pytest.raises(ValueError, match="placeholder"):
        coding.code_live(project)
