"""Export coded results: an Excel workbook and an HTML quote report."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

from .config import Project
from .results import Results, load_results


def code_summary(results: Results) -> pd.DataFrame:
    matrix = results.matrix
    coded = matrix.dropna(how="all")
    seg = results.segments.set_index("segment_id")
    rows = []
    for code in results.codebook.codes:
        applied = coded.index[coded[code.id] == 1]
        transcripts = seg.loc[applied, "transcript_id"].unique()
        rows.append(
            {
                "code_id": code.id,
                "name": code.name,
                "segments": len(applied),
                "percent_of_segments": round(
                    100 * len(applied) / max(len(coded), 1), 1
                ),
                "transcripts": len(transcripts),
                "percent_of_transcripts": round(
                    100
                    * len(transcripts)
                    / max(seg.loc[coded.index, "transcript_id"].nunique(), 1),
                    1,
                ),
            }
        )
    return pd.DataFrame(rows)


def by_group(results: Results, column: str) -> pd.DataFrame:
    """Share of segments with each code, by a segment/metadata column."""
    coded = results.matrix.dropna(how="all")
    groups = results.segments.set_index("segment_id").loc[coded.index, column]
    grouped = coded.groupby(groups)
    table = grouped.mean().mul(100).round(1).T
    table.loc["(segments)"] = grouped.size().reindex(table.columns).to_numpy()
    table.columns = [f"% {c}" for c in table.columns]
    return table.reset_index(names="code_id")


def cooccurrence(results: Results) -> pd.DataFrame:
    coded = results.matrix.dropna(how="all").fillna(0)
    return (coded.T @ coded).astype(int)


def write_excel(project: Project, results: Results) -> Path:
    out = project.outputs / "results.xlsx"
    code_ids = [c.id for c in results.codebook.codes]
    segments = results.segments.set_index("segment_id")
    wide = segments.join(results.matrix).reset_index()
    meta_cols = [
        c
        for c in results.segments.columns
        if c not in {"segment_id", "transcript_id", "position", "speakers", "text"}
    ]

    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        code_summary(results).to_excel(writer, sheet_name="summary", index=False)
        by_group(results, "transcript_id").to_excel(
            writer, sheet_name="by_transcript", index=False
        )
        for col in meta_cols:
            by_group(results, col).to_excel(
                writer, sheet_name=f"by_{col}"[:31], index=False
            )
        cooccurrence(results).to_excel(writer, sheet_name="co_occurrence")
        results.evidence.merge(
            results.segments[["segment_id", "text", *meta_cols]], on="segment_id"
        ).to_excel(writer, sheet_name="evidence", index=False)
        wide[
            ["segment_id", "transcript_id", *meta_cols, "speakers", *code_ids, "text"]
        ].to_excel(writer, sheet_name="segments", index=False)
        pd.DataFrame([c.model_dump() for c in results.codebook.codes]).assign(
            examples=lambda d: d["examples"].apply(" | ".join)
        ).to_excel(writer, sheet_name="codebook", index=False)
        if not results.errors.empty:
            results.errors.to_excel(writer, sheet_name="errors", index=False)
    return out


_CSS = """
body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:980px;margin:0 auto;
padding:24px 16px;color:#1d2433;background:#fff;line-height:1.5}
h1{margin-bottom:4px} .muted{color:#5b6475}
nav a{margin-right:10px;white-space:nowrap}
section{border-top:1px solid #dde2ea;margin-top:28px;padding-top:12px}
.def{background:#f4f6fa;padding:10px 14px;border-radius:6px}
blockquote{margin:10px 0;padding:6px 12px;border-left:3px solid #3b6fd8;background:#fafbfd}
blockquote .meta{font-size:.85em;color:#5b6475}
.warn{color:#a3361b;font-size:.85em}
"""


def write_html(project: Project, results: Results, max_quotes: int = 40) -> Path:
    out = project.outputs / "quote_report.html"
    summary = code_summary(results).set_index("code_id")
    esc = html.escape
    parts = [
        f"<!doctype html><html><head><meta charset='utf-8'>"
        f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{esc(project.config.name)} quotes</title><style>{_CSS}</style></head><body>",
        f"<h1>{esc(project.config.name)}</h1>",
        f"<p class='muted'>{len(results.matrix.dropna(how='all'))} coded segments · "
        f"{len(results.codebook.codes)} codes · coded by Claude, review before citing</p>",
        "<nav>"
        + "".join(
            f"<a href='#{esc(c.id)}'>{esc(c.name)}</a>" for c in results.codebook.codes
        )
        + "</nav>",
    ]
    for code in results.codebook.codes:
        s = summary.loc[code.id]
        quotes = results.evidence[results.evidence["code_id"] == code.id]
        parts.append(f"<section id='{esc(code.id)}'><h2>{esc(code.name)}</h2>")
        parts.append(
            f"<p class='muted'>{s['segments']} segments ({s['percent_of_segments']}%) · "
            f"{s['transcripts']} transcripts</p><div class='def'>{esc(code.definition)}</div>"
        )
        for _, q in quotes.head(max_quotes).iterrows():
            warn = (
                ""
                if q["evidence_verbatim"]
                else "<div class='warn'>Quote not found verbatim in segment; check the source.</div>"
            )
            parts.append(
                f"<blockquote>{esc(q['evidence'])}<div class='meta'>{esc(q['transcript_id'])} · "
                f"{esc(q['segment_id'])} · {esc(q['rationale'])}</div>{warn}</blockquote>"
            )
        if len(quotes) > max_quotes:
            parts.append(
                f"<p class='muted'>…and {len(quotes) - max_quotes} more in results.xlsx (evidence sheet).</p>"
            )
        parts.append("</section>")
    parts.append("</body></html>")
    out.write_text("\n".join(parts), encoding="utf-8")
    return out


def build_reports(project: Project) -> tuple[Path, Path, Results]:
    results = load_results(project)
    return write_excel(project, results), write_html(project, results), results
