# Analysis workflow

This guide walks a team through one project, from final transcripts to coded results.
Every command takes `-p <project folder>`; examples below use `-p $P`.

Claude does the reading and coding. The team stays responsible for the codebook, the
validation, and every quote that ends up in a paper.

## 0. Before you start

- [ ] Transcripts are **final** and **de-identified** (names, places, phone numbers).
- [ ] Your IRB protocol and data agreements allow sending transcript text to Anthropic
      (and to Voyage AI, if you use the optional embeddings).
- [ ] The project folder lives on approved secure storage, **not** inside this repo.
- [ ] `ANTHROPIC_API_KEY` is in your `.env` (`VOYAGE_API_KEY` is optional).

## 1. Create the project

```bash
uv run qual init $P --name "My study"
```

Edit `$P/config.yaml`:

| Setting | What to put |
|---|---|
| `context` | Program, population, setting: what a new RA would need to know |
| `research_questions` | What the analysis must answer. Claude uses them in every step |
| `segmentation.unit` | `question` for focus groups, `turn` for long interviews, `paragraph` if there are no speaker labels |
| `segmentation.moderator_labels` | How the moderator/interviewer is labelled in your transcripts |

Copy transcripts into `$P/transcripts/`. Supported: `.docx`, `.pdf` (text-based, not
scanned), `.md`, `.txt`, and `.xlsx`/`.csv` with one row per speaker turn (set the
column names under `tabular:`).

Fill `$P/transcripts/metadata.csv` with one row per transcript: `transcript_id` is the
file name without extension; add any columns you want results broken down by (site,
group type, round...). This is optional: columns left empty are simply not reported.

## 2. Ingest

```bash
uv run qual ingest -p $P
```

Check the table it prints (also in `outputs/ingest_summary.csv`):

- **speakers** should list real speaker labels. If you see things like `Note` or
  `Time`, adjust `segmentation.speaker_pattern`.
- **warnings** flag transcripts where no speakers or no moderator were found.
- Open `outputs/segments.jsonl` and read a few segments. Is each one a sensible unit to
  code? If not, change `unit`, `max_chars` or the moderator labels and re-run.

## 3. Draft the codebook (optional)

Skip this if the team already has a codebook; write it in `codebook.yaml`, replacing
the placeholder code that `qual init` put there (`qual code` refuses to run on it).

```bash
uv run qual draft-codebook -p $P
```

Claude reads all transcripts (or, for very large projects, a sample: chosen with
embeddings if you have a Voyage key, evenly spaced otherwise) and writes:

- `outputs/codebook_draft.yaml`: proposed codes with definitions, include/exclude rules
  and example quotes
- `outputs/codebook_draft_notes.md`: overlaps, judgment calls, thin themes

**Review it as a team.** Merge, split, rename and rewrite codes, and sharpen the
include/exclude rules: they are what Claude follows when coding. Save the final version
as `$P/codebook.yaml`. If a `codebook.yaml` already exists, the draft refines it instead
of starting from scratch.

With a Voyage key, `uv run qual search -p $P "query"` finds segments about a topic,
which helps when deciding whether a code is needed.

## 4. Code

```bash
uv run qual code -p $P --dry-run   # how many requests and roughly how many tokens
uv run qual code -p $P             # live, a few minutes for a typical project
uv run qual code -p $P --batch --wait   # Batch API: half price, usually under an hour
```

For each segment, **Claude decides** each code: applies yes/no, a verbatim quote as
evidence, and a one-sentence rationale. The codebook is cached between requests, so
repeated instructions cost little.

**Optional pre-screen (needs `VOYAGE_API_KEY`).** With a Voyage key, each segment is
first compared with every code definition and only the `prefilter.top_k` closest codes go
to Claude; the rest are recorded as "not applied (screened out)". This keeps codebooks
with many codes affordable. Without the key the step is skipped (the command says so)
and Claude evaluates every code.

If the key or model name is wrong, the run stops at the first request with the reason.

Coding is resumable: re-running `qual code` only codes segments that are missing or
failed. **Changing `codebook.yaml` means everything is recoded** with the new version on
the next run; old results stay in `codings.jsonl` but reports use the current codebook.

## 5. Validate against human coders

Do this before trusting or reporting the results.

```bash
uv run qual validate -p $P --make-sample 60
```

This writes `outputs/validation/human_coding_template.xlsx`, a random sample of segments
that doesn't show Claude's codes. (With the Voyage pre-screen, half the sample is
weighted toward segments likely to contain each code, so rare codes get enough examples.)

Two or more team members each fill in their own copy (1 = applies, blank = no, and their
initials in `coded_by`), then:

```bash
uv run qual validate -p $P --human coder_ab.xlsx coder_cd.xlsx
```

`outputs/validation/validation_report.xlsx` contains:

| Sheet | Use it to |
|---|---|
| `claude_vs_human` | Per code: kappa, % agreement, precision, recall, and a `flag` column |
| `human_vs_human` | Kappa between your coders. This is the benchmark: Claude should be in the same range |
| `disagreements` | Every disagreement with Claude's quote and rationale. Read these first |

How to act on it:

- **Low kappa on a code:** read its disagreements. Usually the definition is ambiguous;
  tighten the include/exclude rules, then recode and re-validate.
- **`missed_by_prescreen` > 0:** the embedding screen dropped segments humans coded.
  Raise `prefilter.top_k` (or set `prefilter.enabled: false` for small projects).
- **Too few human positives:** code a larger sample before judging that code.
- Common rule of thumb: kappa ≥ 0.6 is acceptable, ≥ 0.8 is strong. Compare with your
  human-vs-human kappa rather than treating these as fixed cut-offs.

## 6. Report

```bash
uv run qual report -p $P
```

- `outputs/results.xlsx`: summary per code, breakdowns by transcript and by each metadata
  column, code co-occurrence, every quote with its rationale, and the full segment × code
  matrix for your own analysis in Stata/R.
- `outputs/quote_report.html`: quotes grouped by code, for reading and writing up.
  Quotes Claude didn't copy exactly are flagged; **check every quote against the
  transcript before citing it.**

## Settings reference

| Setting | Default | Notes |
|---|---|---|
| `models.coding_model` | `claude-opus-5-5` | Model for coding |
| `models.coding_effort` | `medium` | `low` is faster and cheaper; raise it if validation shows careless errors |
| `models.drafting_model` / `drafting_effort` | `claude-opus-5-5` / `high` | One request per project, so higher effort is cheap |
| `models.embedding_model` | `voyage-4` | Only used with `VOYAGE_API_KEY`. `voyage-4-large` for best quality, `voyage-4-lite` for cheapest |
| `prefilter.top_k` | `5` | Codes sent to Claude per segment (pre-screen only) |
| `prefilter.min_similarity` | none | Also send any code above this similarity |

On models that support it, coding requests opt into Anthropic's server-side refusal
fallback: if a safety filter declines a segment, the API retries it on a fallback model.
Batch requests can't use this; declined segments show up as errors in the report.
