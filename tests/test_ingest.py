from __future__ import annotations

import docx

from dilqual.config import Project, SegmentationConfig
from dilqual.ingest import (
    Transcript,
    Turn,
    ingest,
    parse_turns,
    read_docx,
    segment_transcript,
)
from dilqual.io import read_jsonl


def test_parse_turns_handles_timestamps_and_continuations():
    text = (
        "[00:01:02] MODERATOR: Hello everyone\n"
        "Participant 1: First answer\n"
        "continues here\n"
        "Participant 1: same speaker again\n"
        "https://example.org is not a speaker\n"
        "Participant 2: Second answer"
    )
    turns = parse_turns(text, SegmentationConfig().speaker_pattern)
    assert [t.speaker for t in turns] == ["MODERATOR", "Participant 1", "Participant 2"]
    assert "continues here" in turns[1].text
    assert "same speaker again" in turns[1].text
    assert "https://example.org" in turns[1].text


def test_question_segmentation_groups_answers_under_moderator():
    transcript = Transcript(
        "t1",
        "t1.md",
        turns=[
            Turn("MODERATOR", "Q1?"),
            Turn("P1", "a"),
            Turn("P2", "b"),
            Turn("Moderator 2", "Q2?"),
            Turn("P1", "c"),
        ],
    )
    segments = segment_transcript(transcript, SegmentationConfig(unit="question"))
    assert len(segments) == 2
    assert segments[0].speakers == ["MODERATOR", "P1", "P2"]
    assert segments[1].segment_id == "t1-0002"


def test_long_groups_are_split():
    turns = [Turn("MODERATOR", "Q?")] + [Turn(f"P{i}", "x" * 100) for i in range(10)]
    segments = segment_transcript(
        Transcript("t", "t.md", turns=turns), SegmentationConfig(max_chars=350)
    )
    assert len(segments) > 1
    assert all(len(s.text) <= 400 for s in segments)


def test_read_docx_paragraphs_and_tables(tmp_path):
    document = docx.Document()
    document.add_paragraph("MODERATOR: What helps?")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Participant 1"
    table.rows[0].cells[1].text = "The group meetings."
    path = tmp_path / "t.docx"
    document.save(path)
    text = read_docx(path)
    assert "MODERATOR: What helps?" in text
    assert "Participant 1: The group meetings." in text


def test_ingest_example_project(project_dir):
    project = Project(project_dir)
    summary = ingest(project)
    assert set(summary["transcript_id"]) == {"fg_riverside", "fg_hilltop"}
    assert (summary["warnings"] == "").all()
    segments = read_jsonl(project.segments_path)
    # 4 moderator questions in each transcript
    assert sum(s["transcript_id"] == "fg_riverside" for s in segments) == 4
    assert sum(s["transcript_id"] == "fg_hilltop" for s in segments) == 4
    assert all(s["text"] for s in segments)


def test_no_speakers_falls_back_to_paragraphs(tmp_path, project_dir):
    (project_dir / "transcripts" / "notes.txt").write_text(
        "First paragraph with no speakers.\n\nSecond paragraph.", encoding="utf-8"
    )
    summary = ingest(Project(project_dir)).set_index("transcript_id")
    assert "paragraph" in summary.loc["notes", "warnings"]
    assert summary.loc["notes", "segments"] == 1
