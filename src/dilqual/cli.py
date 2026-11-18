"""Command-line interface: `qual <command> --project <folder>`."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import Project


def _project(args) -> Project:
    return Project(Path(args.project))


def cmd_init(args) -> None:
    from . import templates

    root = Path(args.path)
    if (root / "config.yaml").exists():
        sys.exit(f"{root} already has a config.yaml; not overwriting.")
    (root / "transcripts").mkdir(parents=True, exist_ok=True)
    (root / "outputs").mkdir(exist_ok=True)
    (root / "config.yaml").write_text(
        templates.CONFIG_YAML.format(name=args.name or root.name), encoding="utf-8"
    )
    (root / "codebook.yaml").write_text(templates.CODEBOOK_YAML, encoding="utf-8")
    meta = root / "transcripts" / "metadata.csv"
    if not meta.exists():
        meta.write_text(templates.METADATA_CSV, encoding="utf-8")
    print(f"Created project at {root}")
    print(
        "Next: fill in config.yaml, put transcripts in transcripts/, then run `qual ingest`."
    )


def cmd_ingest(args) -> None:
    from .ingest import ingest

    project = _project(args)
    summary = ingest(project)
    print(summary.to_string(index=False))
    print(f"\n{summary['segments'].sum()} segments from {len(summary)} transcripts")
    print(f"Wrote {project.segments_path}")
    if summary["warnings"].astype(bool).any():
        print("\nSome transcripts have warnings; see outputs/ingest_summary.csv.")


def cmd_draft(args) -> None:
    from .draft import draft_codebook, save_draft

    project = _project(args)
    print(
        "Claude is reading the transcripts and drafting a codebook (this can take a few minutes)..."
    )
    draft, sampled = draft_codebook(project)
    save_draft(project, draft, sampled)
    print(f"Drafted {len(draft.codes)} codes.")
    print(
        f"Review {project.outputs / 'codebook_draft.yaml'} and the notes file, "
        f"edit as a team, then save the final version as {project.codebook_path.name}."
    )


def cmd_search(args) -> None:
    from .embed import search

    for segment, score in search(_project(args), args.query, args.top):
        print(f"\n[{score:.3f}] {segment['segment_id']}")
        text = segment["text"]
        print(text[:600] + ("..." if len(text) > 600 else ""))


def cmd_code(args) -> None:
    from . import coding

    project = _project(args)
    if args.dry_run:
        for key, value in coding.estimate(project).items():
            print(f"{key:55} {value:,}")
        return
    if args.batch:
        batch_id = coding.submit_batch(project)
        if batch_id is None:
            print("Nothing to code: every segment is already coded with this codebook.")
            return
        print(
            f"Submitted batch {batch_id}. Collect results with `qual collect` "
            "(most batches finish within an hour)."
        )
        if args.wait:
            _wait_and_collect(project)
        return
    counts = coding.code_live(project, workers=args.workers)
    print(f"Coded {counts['coded']} segments, {counts['errors']} errors.")
    if counts["errors"]:
        print("Re-run `qual code` to retry failed segments.")


def _wait_and_collect(project: Project) -> None:
    from .coding import collect_batches

    while True:
        report = collect_batches(project)
        pending = [r for r in report if r["status"] != "collected"]
        for r in report:
            print(r)
        if not pending:
            return
        time.sleep(60)


def cmd_collect(args) -> None:
    from .coding import collect_batches

    project = _project(args)
    if args.wait:
        _wait_and_collect(project)
        return
    report = collect_batches(project)
    for r in report or [{"status": "no pending batches"}]:
        print(r)


def cmd_validate(args) -> None:
    from .validate import make_sample, run_validation

    project = _project(args)
    if args.make_sample:
        out = make_sample(
            project, args.make_sample, seed=args.seed, enrich=not args.no_enrich
        )
        print(f"Wrote {out}")
        print(
            "Each human coder fills in a copy, then run: qual validate --human <file> ..."
        )
        return
    if not args.human:
        sys.exit(
            "Pass --make-sample N to create the human coding sheet, or --human FILE ... to compare."
        )
    summary, out = run_validation(project, [Path(p) for p in args.human])
    print(summary.to_string(index=False))
    print(f"\nWrote {out}")


def cmd_report(args) -> None:
    from .report import build_reports

    project = _project(args)
    xlsx, html, results = build_reports(project)
    n_coded = len(results.matrix.dropna(how="all"))
    print(f"{n_coded}/{len(results.matrix)} segments coded with the current codebook.")
    if not results.errors.empty:
        print(f"{len(results.errors)} segments failed; re-run `qual code` to retry.")
    print(f"Wrote {xlsx}\nWrote {html}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="qual", description="Qualitative transcript analysis with Claude."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, func, help_text: str, project: bool = True):
        p = sub.add_parser(name, help=help_text, description=help_text)
        if project:
            p.add_argument(
                "--project",
                "-p",
                default=".",
                help="Project folder (default: current folder)",
            )
        p.set_defaults(func=func)
        return p

    p = add("init", cmd_init, "Create a new project folder.", project=False)
    p.add_argument("path")
    p.add_argument("--name", default=None)

    add("ingest", cmd_ingest, "Read transcripts and split them into segments.")
    add(
        "draft-codebook",
        cmd_draft,
        "Have Claude propose a codebook from the transcripts.",
    )

    p = add(
        "search",
        cmd_search,
        "Find segments related to a question (needs VOYAGE_API_KEY).",
    )
    p.add_argument("query")
    p.add_argument("--top", type=int, default=10)

    p = add("code", cmd_code, "Apply the codebook to every segment.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show how much work a run would be, without calling Claude.",
    )
    p.add_argument(
        "--batch",
        action="store_true",
        help="Use the Batch API (half price, results within 24h).",
    )
    p.add_argument(
        "--wait", action="store_true", help="With --batch: wait and collect results."
    )
    p.add_argument(
        "--workers", type=int, default=4, help="Concurrent requests for live coding."
    )

    p = add("collect", cmd_collect, "Collect results from submitted batches.")
    p.add_argument("--wait", action="store_true")

    p = add("validate", cmd_validate, "Compare Claude's coding with human coders.")
    p.add_argument(
        "--make-sample",
        type=int,
        metavar="N",
        help="Create a human coding sheet with N segments.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--no-enrich",
        action="store_true",
        help="Pure random sample (default enriches for rare codes).",
    )
    p.add_argument(
        "--human", nargs="+", metavar="FILE", help="Filled human coding sheets."
    )

    add("report", cmd_report, "Export results.xlsx and quote_report.html.")

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
