# a-llm-qualitative-analysis

Workflow for DIL teams to analyze final qualitative transcripts with Claude: segment
transcripts, draft and refine a codebook, code at scale, check agreement with human
coders, and export coded data and quote reports.

```text
transcripts (.docx .pdf .md .txt .xlsx .csv)
        │
   qual ingest ────────────► segments (one per moderator question or speaker turn)
        │
   qual draft-codebook ────► Claude proposes codes ──► team reviews ──► codebook.yaml
        │
   qual code ──────────────► Voyage embeddings pre-screen which codes fit each segment,
        │                    Claude decides each one with a verbatim quote + rationale
   qual validate ──────────► agreement with human coders (kappa per code)
        │
   qual report ────────────► results.xlsx + quote_report.html
```

## Setup (once)

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Clone the repo and install:

   ```bash
   git clone https://github.com/DevInnovationLab/a-llm-qualitative-analysis.git
   cd a-llm-qualitative-analysis
   uv sync
   ```

3. Copy `.env.example` to `.env` and add your keys:
   - `ANTHROPIC_API_KEY` for Claude
   - `VOYAGE_API_KEY` for embeddings (Anthropic doesn't offer an embeddings model;
     Voyage AI is the provider it recommends)

Run commands with `uv run qual ...` from the repo folder.

## Running a project

```bash
uv run qual init /secure/drive/my-study        # creates config.yaml, codebook.yaml, transcripts/
# edit config.yaml, copy final transcripts into transcripts/, fill transcripts/metadata.csv

uv run qual ingest -p /secure/drive/my-study
uv run qual draft-codebook -p /secure/drive/my-study   # optional: inductive draft
uv run qual code -p /secure/drive/my-study --dry-run   # size of the run, no API calls
uv run qual code -p /secure/drive/my-study             # or --batch for half price
uv run qual validate -p /secure/drive/my-study --make-sample 60
uv run qual validate -p /secure/drive/my-study --human coder1.xlsx coder2.xlsx
uv run qual report -p /secure/drive/my-study
```

The full step-by-step guide, including how to review the codebook and interpret
validation, is in [docs/workflow.md](docs/workflow.md).

To try the workflow without real data, use the fictional example project:
`uv run qual ingest -p projects/example`, and so on.

## Data protection

- **Keep project folders outside this repo**, on approved secure storage. `projects/`
  is git-ignored except for the fictional example, but don't rely on that.
- Transcript text is sent to Anthropic (Claude) and Voyage AI (embeddings). Confirm your
  IRB protocol and data agreements allow this, and de-identify transcripts first.
- Never commit `.env` or anything in a project's `outputs/`.

## Development

```bash
uv run pytest      # offline tests with fake API clients; no keys needed
uv run ruff check
```

## License

MIT. Adapted from training materials originally developed at Innovations for
Poverty Action.
