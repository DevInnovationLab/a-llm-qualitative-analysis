"""Read transcripts (Word, PDF, Markdown/text, Excel/CSV) and split them into segments."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .config import Project, SegmentationConfig, TabularConfig
from .io import write_jsonl

TEXT_SUFFIXES = {".md", ".txt"}
WORD_SUFFIXES = {".docx"}
PDF_SUFFIXES = {".pdf"}
TABLE_SUFFIXES = {".xlsx", ".xls", ".csv"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | WORD_SUFFIXES | PDF_SUFFIXES | TABLE_SUFFIXES


@dataclass
class Turn:
    """One speaker turn."""

    speaker: str
    text: str


@dataclass
class Transcript:
    """A transcript as a list of turns (or raw text when no speakers were found)."""

    transcript_id: str
    source_file: str
    turns: list[Turn] = field(default_factory=list)
    raw_text: str = ""
    warnings: list[str] = field(default_factory=list)


@dataclass
class Segment:
    """The unit Claude codes."""

    segment_id: str
    transcript_id: str
    position: int
    speakers: list[str]
    text: str


# ---------------------------------------------------------------------------
# Readers
# ---------------------------------------------------------------------------


def read_docx(path: Path) -> str:
    """Return the text of a Word document, including two-column speaker tables."""
    import docx

    document = docx.Document(str(path))
    lines = [p.text for p in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells]
            cells = [c for i, c in enumerate(cells) if c and c not in cells[:i]]
            if len(cells) >= 2:
                lines.append(f"{cells[0]}: {' '.join(cells[1:])}")
            elif cells:
                lines.append(cells[0])
            lines.append("")
    return "\n".join(lines)


def read_pdf(path: Path) -> str:
    """Return the text of a text-based PDF (scanned PDFs need OCR first)."""
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_table(path: Path, cfg: TabularConfig) -> list[Transcript]:
    """Read an Excel/CSV file with one row per speaker turn."""
    if path.suffix.lower() == ".csv":
        frames = {path.stem: pd.read_csv(path, dtype=str)}
    else:
        sheets = pd.read_excel(path, sheet_name=None, dtype=str)
        frames = (
            {path.stem: sheets[next(iter(sheets))]}
            if len(sheets) == 1
            else {f"{path.stem}__{name}": df for name, df in sheets.items()}
        )

    transcripts: list[Transcript] = []
    for base_id, df in frames.items():
        df.columns = [str(c).strip() for c in df.columns]
        missing = {cfg.speaker_column, cfg.text_column} - set(df.columns)
        if missing:
            raise ValueError(
                f"{path.name}: missing column(s) {sorted(missing)}. Found: {list(df.columns)}. "
                "Set tabular.speaker_column / tabular.text_column in config.yaml."
            )
        df = df.fillna("")
        groups = (
            df.groupby(cfg.transcript_column, sort=False)
            if cfg.transcript_column
            else [(base_id, df)]
        )
        for group_id, group in groups:
            transcript_id = (
                f"{base_id}__{group_id}" if cfg.transcript_column else base_id
            )
            turns = [
                Turn(speaker=str(r[cfg.speaker_column]).strip() or "UNKNOWN", text=t)
                for _, r in group.iterrows()
                if (t := str(r[cfg.text_column]).strip())
            ]
            transcripts.append(
                Transcript(
                    transcript_id=slugify(str(transcript_id)),
                    source_file=path.name,
                    turns=merge_consecutive(turns),
                )
            )
    return transcripts


# ---------------------------------------------------------------------------
# Parsing text into turns
# ---------------------------------------------------------------------------


def slugify(value: str) -> str:
    value = re.sub(r"[^\w\-]+", "_", value.strip(), flags=re.UNICODE)
    return value.strip("_") or "transcript"


def parse_turns(text: str, pattern: str) -> list[Turn]:
    """Split text into speaker turns using a regex with (speaker, text) groups."""
    regex = re.compile(pattern)
    turns: list[Turn] = []
    for line in text.replace("\r\n", "\n").split("\n"):
        match = regex.match(line)
        if match and not _looks_like_url(line):
            turns.append(
                Turn(speaker=match.group(1).strip(), text=match.group(2).strip())
            )
        elif line.strip() and turns:
            turns[-1].text = f"{turns[-1].text}\n{line.strip()}".strip()
    return merge_consecutive([t for t in turns if t.text])


def _looks_like_url(line: str) -> bool:
    return bool(re.match(r"^\s*https?:", line))


def merge_consecutive(turns: list[Turn]) -> list[Turn]:
    merged: list[Turn] = []
    for turn in turns:
        if merged and merged[-1].speaker.lower() == turn.speaker.lower():
            merged[-1].text = f"{merged[-1].text}\n{turn.text}"
        else:
            merged.append(Turn(turn.speaker, turn.text))
    return merged


def read_transcripts(path: Path, project: Project) -> list[Transcript]:
    """Read one transcript file into one or more Transcript objects."""
    suffix = path.suffix.lower()
    seg_cfg = project.config.segmentation
    if suffix in TABLE_SUFFIXES:
        return read_table(path, project.config.tabular)
    if suffix in WORD_SUFFIXES:
        text = read_docx(path)
    elif suffix in PDF_SUFFIXES:
        text = read_pdf(path)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")

    transcript = Transcript(
        transcript_id=slugify(path.stem), source_file=path.name, raw_text=text
    )
    if not text.strip():
        transcript.warnings.append(
            "No text extracted (scanned PDF? run OCR first, or export as .docx)."
        )
        return [transcript]
    if seg_cfg.unit != "paragraph":
        transcript.turns = parse_turns(text, seg_cfg.speaker_pattern)
        if len(transcript.turns) < 2:
            transcript.warnings.append(
                "Fewer than 2 speaker turns detected; segmented by paragraph instead. "
                "Check segmentation.speaker_pattern in config.yaml."
            )
            transcript.turns = []
    return [transcript]


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------


def _is_moderator(speaker: str, cfg: SegmentationConfig) -> bool:
    label = re.sub(r"[\d\s]+$", "", speaker.strip().lower())
    return label in {m.lower() for m in cfg.moderator_labels}


def _format_turns(turns: list[Turn]) -> str:
    return "\n\n".join(f"{t.speaker}: {t.text}" for t in turns)


def _split_long(groups: list[list[Turn]], max_chars: int) -> list[list[Turn]]:
    """Split groups whose text exceeds max_chars, on turn boundaries."""
    out: list[list[Turn]] = []
    for group in groups:
        current: list[Turn] = []
        size = 0
        for turn in group:
            length = len(turn.speaker) + len(turn.text) + 4
            if current and size + length > max_chars:
                out.append(current)
                current, size = [], 0
            current.append(turn)
            size += length
        if current:
            out.append(current)
    return out


def paragraph_chunks(text: str, min_chars: int) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        buffer = paragraph if not buffer else f"{buffer}\n\n{paragraph}"
        if len(buffer) >= min_chars:
            chunks.append(buffer)
            buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks


def segment_transcript(
    transcript: Transcript, cfg: SegmentationConfig
) -> list[Segment]:
    """Split a transcript into segments according to the configured unit."""
    if not transcript.turns:
        texts = paragraph_chunks(transcript.raw_text, cfg.min_chars)
        return [
            Segment(
                segment_id=f"{transcript.transcript_id}-{i:04d}",
                transcript_id=transcript.transcript_id,
                position=i,
                speakers=[],
                text=text,
            )
            for i, text in enumerate(texts, start=1)
        ]

    if cfg.unit == "turn":
        groups = [[t] for t in transcript.turns]
    else:  # question
        groups = []
        for turn in transcript.turns:
            if not groups or _is_moderator(turn.speaker, cfg):
                groups.append([turn])
            else:
                groups[-1].append(turn)
        if not any(_is_moderator(t.speaker, cfg) for t in transcript.turns):
            transcript.warnings.append(
                "No moderator turns found; each turn is its own segment. "
                "Add the moderator's label to segmentation.moderator_labels."
            )
            groups = [[t] for t in transcript.turns]

    groups = _split_long(groups, cfg.max_chars)
    return [
        Segment(
            segment_id=f"{transcript.transcript_id}-{i:04d}",
            transcript_id=transcript.transcript_id,
            position=i,
            speakers=list(dict.fromkeys(t.speaker for t in group)),
            text=_format_turns(group),
        )
        for i, group in enumerate(groups, start=1)
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def find_transcript_files(directory: Path, metadata_path: Path) -> list[Path]:
    files = [
        p
        for p in sorted(directory.rglob("*"))
        if p.is_file()
        and p.suffix.lower() in SUPPORTED_SUFFIXES
        and not p.name.startswith(("~$", "."))
        and p.resolve() != metadata_path.resolve()
    ]
    return files


def ingest(project: Project) -> pd.DataFrame:
    """Read every transcript, segment it, and write outputs/segments.jsonl.

    Returns a per-transcript summary (also written to outputs/ingest_summary.csv).
    """
    files = find_transcript_files(project.transcripts_dir, project.metadata_path)
    if not files:
        raise FileNotFoundError(
            f"No transcripts found in {project.transcripts_dir} "
            f"(supported: {', '.join(sorted(SUPPORTED_SUFFIXES))})."
        )

    segments: list[Segment] = []
    summary: list[dict] = []
    seen_ids: set[str] = set()
    for path in files:
        for transcript in read_transcripts(path, project):
            if transcript.transcript_id in seen_ids:
                raise ValueError(
                    f"Two transcripts map to the id '{transcript.transcript_id}'. "
                    "Rename one of the files."
                )
            seen_ids.add(transcript.transcript_id)
            transcript_segments = segment_transcript(
                transcript, project.config.segmentation
            )
            segments.extend(transcript_segments)
            summary.append(
                {
                    "transcript_id": transcript.transcript_id,
                    "source_file": transcript.source_file,
                    "turns": len(transcript.turns),
                    "segments": len(transcript_segments),
                    "speakers": ", ".join(
                        sorted({t.speaker for t in transcript.turns})[:15]
                    ),
                    "warnings": " | ".join(transcript.warnings),
                }
            )

    write_jsonl(project.segments_path, (asdict(s) for s in segments))
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(project.outputs / "ingest_summary.csv", index=False)
    return summary_df
