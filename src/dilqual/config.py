"""Project configuration: config.yaml, codebook.yaml, and API clients."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# config.yaml
# ---------------------------------------------------------------------------


class SegmentationConfig(BaseModel):
    """How transcripts are split into segments for coding."""

    # question: a moderator/interviewer turn plus all responses until the next one
    # turn:     each speaker turn on its own (consecutive turns by one speaker merged)
    # paragraph: blank-line separated paragraphs (for transcripts without speakers)
    unit: Literal["question", "turn", "paragraph"] = "question"
    # Speaker labels (case-insensitive) that mark the moderator/interviewer.
    moderator_labels: list[str] = Field(
        default_factory=lambda: [
            "moderator",
            "moderador",
            "moderadora",
            "interviewer",
            "entrevistador",
            "entrevistadora",
            "facilitator",
        ]
    )
    # Regex with two groups (speaker, text) that matches a line starting a turn.
    # The default matches lines like "MODERATOR: ...", "Participant 3: ...",
    # "[00:01:02] Speaker 1: ...".
    speaker_pattern: str = (
        r"^\s*(?:\[?\(?\d{1,2}:\d{2}(?::\d{2})?\)?\]?\s*)?"
        r"\**([A-Za-zÁÉÍÓÚÜÑáéíóúüñ][\w .'’\-ÁÉÍÓÚÜÑáéíóúüñ]{0,40}?)\**\s*:\s*(.*)$"
    )
    # Paragraph mode: merge short paragraphs until a segment has at least this many characters.
    min_chars: int = 300
    # Turn/question mode: segments longer than this are split on turn boundaries.
    max_chars: int = 6000


class TabularConfig(BaseModel):
    """Column names for transcripts delivered as Excel/CSV (one row per turn)."""

    speaker_column: str = "speaker"
    text_column: str = "text"
    # Optional: if a single sheet holds several transcripts, the column that identifies them.
    transcript_column: str | None = None


class ModelsConfig(BaseModel):
    """Which models to use and how hard they should think."""

    coding_model: str = "claude-opus-5-5"
    coding_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    drafting_model: str = "claude-opus-5-5"
    drafting_effort: Literal["low", "medium", "high", "xhigh", "max"] = "high"
    embedding_model: str = "voyage-4"


class PrefilterConfig(BaseModel):
    """Embedding-based screen that decides which codes Claude evaluates per segment.

    For every segment, the `top_k` most similar codes are always sent to Claude,
    plus any other code with similarity >= `min_similarity` (if set). All other
    codes are recorded as "not applied (screened out)". Validation reports how
    many human-coded positives the screen dropped, so the team can tune this.
    """

    enabled: bool = True
    top_k: int = 5
    min_similarity: float | None = None


class ProjectConfig(BaseModel):
    """Contents of a project's config.yaml."""

    name: str
    description: str = ""
    research_questions: list[str] = Field(default_factory=list)
    # Free-text context Claude should know (program, population, setting).
    context: str = ""
    transcript_language: str = "en"
    transcripts_dir: str = "transcripts"
    metadata_file: str = "transcripts/metadata.csv"
    codebook_file: str = "codebook.yaml"
    outputs_dir: str = "outputs"
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    tabular: TabularConfig = Field(default_factory=TabularConfig)
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    prefilter: PrefilterConfig = Field(default_factory=PrefilterConfig)


class Project:
    """A project folder: config, codebook, transcripts, and outputs."""

    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        config_path = self.root / "config.yaml"
        if not config_path.exists():
            raise FileNotFoundError(
                f"No config.yaml in {self.root}. Create a project with: qual init <path>"
            )
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        self.config = ProjectConfig.model_validate(raw)
        self.outputs.mkdir(parents=True, exist_ok=True)

    @property
    def transcripts_dir(self) -> Path:
        return self.root / self.config.transcripts_dir

    @property
    def metadata_path(self) -> Path:
        return self.root / self.config.metadata_file

    @property
    def codebook_path(self) -> Path:
        return self.root / self.config.codebook_file

    @property
    def outputs(self) -> Path:
        return self.root / self.config.outputs_dir

    @property
    def segments_path(self) -> Path:
        return self.outputs / "segments.jsonl"

    @property
    def codings_path(self) -> Path:
        return self.outputs / "codings.jsonl"

    @property
    def validation_dir(self) -> Path:
        path = self.outputs / "validation"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def load_codebook(self) -> Codebook:
        return Codebook.load(self.codebook_path)


# ---------------------------------------------------------------------------
# codebook.yaml
# ---------------------------------------------------------------------------


class Code(BaseModel):
    """One code in the codebook."""

    id: str = Field(description="Short snake_case identifier, unique in the codebook")
    name: str = Field(description="Human-readable code name")
    definition: str = Field(description="What the code captures, in 1-3 sentences")
    include: str = Field(default="", description="When to apply the code")
    exclude: str = Field(default="", description="When NOT to apply the code")
    examples: list[str] = Field(
        default_factory=list, description="Short example quotes that fit the code"
    )

    def as_prompt(self) -> str:
        lines = [f'<code id="{self.id}">', f"Name: {self.name}"]
        lines.append(f"Definition: {self.definition}")
        if self.include:
            lines.append(f"Apply when: {self.include}")
        if self.exclude:
            lines.append(f"Do not apply when: {self.exclude}")
        for example in self.examples:
            lines.append(f"Example: {example}")
        lines.append("</code>")
        return "\n".join(lines)

    def embedding_text(self) -> str:
        parts = [self.name, self.definition, self.include, *self.examples]
        return "\n".join(p for p in parts if p)


class Codebook(BaseModel):
    """The full codebook."""

    codes: list[Code]

    @classmethod
    def load(cls, path: Path) -> Codebook:
        if not path.exists():
            raise FileNotFoundError(
                f"Codebook not found: {path}. Write one by hand, or run "
                "`qual draft-codebook` and review the draft."
            )
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        codebook = cls.model_validate(raw)
        ids = [c.id for c in codebook.codes]
        duplicates = {i for i in ids if ids.count(i) > 1}
        if duplicates:
            raise ValueError(f"Duplicate code ids in {path}: {sorted(duplicates)}")
        if not codebook.codes:
            raise ValueError(f"Codebook {path} has no codes.")
        return codebook

    def save(self, path: Path) -> None:
        data = {"codes": [c.model_dump() for c in self.codes]}
        path.write_text(
            yaml.safe_dump(data, sort_keys=False, allow_unicode=True, width=88),
            encoding="utf-8",
        )

    def fingerprint(self) -> str:
        """Hash of the codebook; codings made with a different codebook are redone."""
        payload = json.dumps([c.model_dump() for c in self.codes], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def as_prompt(self) -> str:
        return "\n\n".join(c.as_prompt() for c in self.codes)

    def by_id(self) -> dict[str, Code]:
        return {c.id: c for c in self.codes}


# ---------------------------------------------------------------------------
# API clients
# ---------------------------------------------------------------------------


def get_claude_client():
    """Create a Claude client from ANTHROPIC_API_KEY (loaded from .env if present)."""
    import anthropic

    load_dotenv()
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to a .env file (see .env.example)."
        )
    return anthropic.Anthropic()


def get_voyage_client():
    """Create a Voyage AI client from VOYAGE_API_KEY (loaded from .env if present)."""
    import voyageai

    load_dotenv()
    if not os.getenv("VOYAGE_API_KEY"):
        raise RuntimeError(
            "VOYAGE_API_KEY is not set. Add it to a .env file (see .env.example)."
        )
    return voyageai.Client()
