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
   qual code ──────────────► Claude decides each code for each segment, with a verbatim
        │                    quote + rationale (optional: Voyage embeddings pre-screen)
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

3. Copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY`.

That is all a team needs: **an Anthropic API key and its transcripts.**

`VOYAGE_API_KEY` is optional. With it, embeddings pre-screen which codes Claude evaluates
for each segment (cheaper for codebooks with many codes) and `qual search` works. Without
it, Claude evaluates every code for every segment, which is the more thorough option and
fine for most projects.

Run commands with `uv run qual ...` from the repo folder.

## Running a project

```bash
uv run qual init /secure/drive/my-study        # creates config.yaml, codebook.yaml, transcripts/
# edit config.yaml, copy final transcripts into transcripts/, fill transcripts/metadata.csv

uv run qual ingest -p /secure/drive/my-study
uv run qual draft-codebook -p /secure/drive/my-study   # skip if you already have a codebook
# review outputs/codebook_draft.yaml as a team and save the final version as codebook.yaml
uv run qual code -p /secure/drive/my-study --dry-run   # size of the run, no Claude calls
uv run qual code -p /secure/drive/my-study             # or --batch for half price
uv run qual validate -p /secure/drive/my-study --make-sample 60
uv run qual validate -p /secure/drive/my-study --human coder1.xlsx coder2.xlsx
uv run qual report -p /secure/drive/my-study
```

The full step-by-step guide, including how to review the codebook and interpret
validation, is in [docs/workflow.md](docs/workflow.md).

## Where your files go

Each study is one **project folder**, created by `qual init` and passed to every command
with `-p`. Everything the workflow reads or writes is inside it:

```text
my-study/                  <- the folder you pass with -p
├── config.yaml            <- study context, research questions, settings
├── codebook.yaml          <- YOUR CODEBOOK goes here
├── transcripts/           <- YOUR TRANSCRIPTS go here
│   ├── interview_01.docx
│   ├── fg_riverside.md
│   └── metadata.csv       <- optional: site, group type... per transcript
└── outputs/               <- created by the commands; results land here
```

### Transcripts

Copy the files into the project's `transcripts/` folder (subfolders are fine). One file
is one transcript, and its **file name without the extension becomes its id** in every
output (`interview_01.docx` is `interview_01`), so use short, distinct names.

| Format | What it must look like |
|---|---|
| `.docx`, `.pdf`, `.md`, `.txt` | Each turn starts on a new line with the speaker and a colon: `MODERATOR: ...`, `Participant 3: ...`. A leading timestamp such as `[00:01:02]` is fine. PDFs must contain text, not scanned images. |
| `.xlsx`, `.csv` | One row per speaker turn, with a `speaker` column and a `text` column (other column names can be set under `tabular:` in `config.yaml`). |

```text
MODERATOR: What makes it easier for you to save every week?

PARTICIPANT 1: For me it is the meeting itself. Everyone will see whether you
brought your contribution.

PARTICIPANT 2: Yes, and the treasurer calls your name out loud.
```

By default a segment is one moderator question plus the answers to it. The moderator is
recognized by label (`Moderator`, `Interviewer`, `Facilitator` and Spanish equivalents);
if yours is labelled differently, add the label to `segmentation.moderator_labels` in
`config.yaml`. Transcripts with no speaker labels still work: they are split by
paragraph. `qual ingest` prints the speakers it found, so you can check this at once.

### Codebook

The codebook is the file `codebook.yaml` in the project folder (not inside
`transcripts/`). It is a YAML file with a list of codes:

```yaml
codes:
  - id: peer_accountability
    name: Peer accountability
    definition: Members describe other members' expectations, reminders or social
      pressure as a reason they keep saving or repaying.
    include: Mentions of shame, encouragement, or being watched by the group.
    exclude: General friendship not linked to saving (use social_support).
    examples:
      - "Nobody wants to be the one who comes with empty hands."
  - id: income_volatility
    name: Income volatility
    definition: Irregular or seasonal income makes it hard to save the agreed amount.
```

| Field | Required | What to put |
|---|---|---|
| `id` | yes | Short, unique, no spaces (`snake_case`). It becomes the column name in the results. |
| `name` | yes | The readable name shown in reports. |
| `definition` | yes | What the code captures, in 1-3 sentences. |
| `include` | no | When to apply it. |
| `exclude` | no | When not to apply it, especially versus similar codes. |
| `examples` | no | Short quotes that fit the code. |

There are two ways to get one:

- **You already have a codebook** (Word, Excel...): retype it in this format in
  `codebook.yaml`, replacing the placeholder code that `qual init` wrote.
- **You don't have one:** run `qual draft-codebook`. Claude writes a proposal in the same
  format to `outputs/codebook_draft.yaml`. Review and edit it as a team, then save it as
  `codebook.yaml` in the project folder.

A complete working project, with transcripts and a codebook, is in
[projects/example](projects/example).

To try the workflow without real data, use the fictional example project:
`uv run qual ingest -p projects/example`, and so on.

## Data protection

- **Keep project folders outside this repo**, on approved secure storage. `projects/`
  is git-ignored except for the fictional example, but don't rely on that.
- Transcript text is sent to Anthropic (Claude) and, only if you set `VOYAGE_API_KEY`,
  to Voyage AI (embeddings). Confirm your IRB protocol and data agreements allow this,
  and de-identify transcripts first.
- Never commit `.env` or anything in a project's `outputs/`.

## Development

```bash
uv run pytest      # offline tests with fake API clients; no keys needed
uv run ruff check
```

## License

MIT. Adapted from training materials originally developed at Innovations for
Poverty Action.
