"""Files written by `qual init` for a new project."""

CONFIG_YAML = """\
# Project configuration. Paths are relative to this folder.
name: "{name}"
description: ""

# Background Claude should know: program, population, setting, data collection.
context: ""

# What the analysis needs to answer. Used when drafting the codebook and coding.
research_questions:
  - ""

transcript_language: en
transcripts_dir: transcripts
# Optional CSV with a `transcript_id` column (file name without extension) plus any
# grouping columns (site, group type, round...). Results get broken down by each column.
metadata_file: transcripts/metadata.csv
codebook_file: codebook.yaml
outputs_dir: outputs

segmentation:
  # question: moderator question + all responses until the next question (focus groups)
  # turn:     each speaker turn (in-depth interviews with long answers)
  # paragraph: blank-line paragraphs (transcripts without speaker labels)
  unit: question
  # Labels that identify the moderator/interviewer (case-insensitive, trailing numbers ignored)
  moderator_labels: [moderator, moderador, moderadora, interviewer, entrevistador, entrevistadora, facilitator]
  min_chars: 300
  max_chars: 6000

# For Excel/CSV transcripts with one row per speaker turn
tabular:
  speaker_column: speaker
  text_column: text
  transcript_column: null

models:
  coding_model: claude-opus-5-5
  coding_effort: medium      # low | medium | high | xhigh | max
  drafting_model: claude-opus-5-5
  drafting_effort: high
  embedding_model: voyage-4  # Voyage AI; only used when VOYAGE_API_KEY is set

# Optional embedding pre-screen (needs VOYAGE_API_KEY; skipped without it, and Claude
# then evaluates every code for every segment). With it, Claude only evaluates the
# top_k most similar codes per segment. Check `missed_by_prescreen` in the validation
# report; raise top_k if it is above 0.
prefilter:
  enabled: true
  top_k: 5
  min_similarity: null
"""

CODEBOOK_YAML = """\
# Codebook. Write it by hand, or run `qual draft-codebook` and copy the reviewed
# draft from outputs/codebook_draft.yaml.
codes:
  - id: example_code
    name: Example code
    definition: Replace with what this code captures, in 1-3 sentences.
    include: When to apply it.
    exclude: When NOT to apply it, especially versus similar codes.
    examples: []
"""

METADATA_CSV = "transcript_id,site,group_type\n"
