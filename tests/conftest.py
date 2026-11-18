"""Fake Claude and Voyage clients so the pipeline runs offline in tests."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / "projects" / "example"

# Keyword rules the fake Claude uses to "code" segments.
RULES = {
    "peer_accountability": ["everyone will see", "fine", "ashamed", "remind"],
    "income_volatility": ["dry season", "no work", "price", "flooded"],
    "loan_productive_use": ["sewing machine", "chickens", "repaired the motorcycle"],
    "loan_consumption_emergency": ["clinic", "school fees", "funeral"],
    "social_support": ["like sisters", "helped us", "advice", "learned"],
}


class FakeVoyage:
    """Deterministic bag-of-words embeddings."""

    def __init__(self):
        self.calls = 0

    def embed(self, texts, model, input_type=None):
        self.calls += 1
        vectors = []
        for text in texts:
            v = np.zeros(256)
            for word in re.findall(r"\w+", text.lower()):
                v[int(hashlib.md5(word.encode()).hexdigest(), 16) % 256] += 1
            vectors.append((v / (np.linalg.norm(v) or 1)).tolist())
        return SimpleNamespace(embeddings=vectors)


class _Stream:
    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.message


class FakeMessages:
    def __init__(self):
        self.requests = []

    def stream(self, **params):
        self.requests.append(params)
        user = params["messages"][0]["content"]
        segment = re.search(r"<segment[^>]*>\n(.*)\n</segment>", user, re.S).group(1)
        code_ids = user.rsplit("Evaluate these codes: ", 1)[1].split(", ")
        decisions = []
        for code_id in code_ids:
            hit = next((k for k in RULES[code_id] if k in segment.lower()), None)
            evidence = ""
            if hit:
                start = segment.lower().index(hit)
                evidence = segment[start : start + len(hit) + 20]
            decisions.append(
                {
                    "code_id": code_id,
                    "applies": bool(hit),
                    "evidence": evidence,
                    "rationale": "keyword rule",
                }
            )
        text = json.dumps({"decisions": decisions})
        return _Stream(
            SimpleNamespace(
                stop_reason="end_turn",
                content=[SimpleNamespace(type="text", text=text)],
            )
        )


class FakeClaude:
    def __init__(self):
        self.messages = FakeMessages()
        self.beta = SimpleNamespace(messages=self.messages)


@pytest.fixture
def project_dir(tmp_path):
    dest = tmp_path / "example"
    shutil.copytree(EXAMPLE, dest, ignore=shutil.ignore_patterns("outputs"))
    return dest


@pytest.fixture
def fakes(monkeypatch):
    voyage, claude = FakeVoyage(), FakeClaude()
    monkeypatch.setenv("VOYAGE_API_KEY", "test-key")
    monkeypatch.setattr("dilqual.embed.get_voyage_client", lambda: voyage)
    monkeypatch.setattr("dilqual.coding.get_claude_client", lambda: claude)
    return SimpleNamespace(voyage=voyage, claude=claude)
