# AI use record

This file documents the use of generative AI tools during the preparation of the
manuscript and this reproducibility package. It is provided so that readers and
reviewers can judge which parts of the work involved AI assistance.

## Tools and models

| Tool | Model | Provider | Access |
|---|---|---|---|
| ChatGPT | Codex based on GPT-6 | OpenAI | https://chatgpt.com/ |

No other generative AI service was used for the work reported here.

## Period of use

20–22 November 2026.

## User

Kairui Wei (first author, corresponding author). All prompts, decisions, and
acceptance of edits were made by this author.

## Scope of use

AI assistance was limited to editorial and mechanical tasks:

1. **Language editing** — grammar, article usage, and phrasing in the English
   manuscript. No new scientific claims were introduced.
2. **Reference checking** — verifying that reference numbers in the text match the
   reference list, and that DOIs resolve to the cited records.
3. **Document formatting** — section ordering, heading styles, table and figure
   caption layout, and export settings.
4. **Package assembly** — generating the file manifest, rebuilding two
   supplementary tables (S31, S33) from existing analysis outputs, and
   standardising the license files.

## What was not done by AI

- Study design, data selection, and analysis choices.
- Model training, scoring, bootstrap procedures, and all numerical results. Every
  reported value was produced by the analysis scripts in `github/analysis/` and
  was recomputed during a final check.
- Interpretation of results, limitations, and conclusions.
- Any decision about what to report or how strongly to state it.

## Outputs affected

| Item | AI involvement |
|---|---|
| Main text (English) | Language editing and section reorganisation under author direction |
| Supplementary Table S31 | Split from the existing two-scheme comparison table (S28); no new computation |
| Supplementary Table S33 | Derived view of the existing survey table (S30); no new computation |
| MANIFEST.csv | Generated file listing, verified against the files on disk |
| LICENSE, LICENSE_CODE.txt, LICENSE_DATA.txt | Text drafted, license choice made by the author |
| Figure alt text | Drafted from figure captions, checked against the panels |

## Verification

Every numerical statement in the manuscript was checked against
`EXPECTED_OUTPUTS.json` with `github/analysis/stability_checks/_audit_numbers.py`
(tolerance 5×10⁻⁴ for point estimates, 1×10⁻³ for interval endpoints). Reference
numbering and figure/table cross-references were checked programmatically after the
final edit. The author reviewed and approved all text that AI touched.

## Statement

AI tools were not used to generate scientific content, data, or conclusions. They
were used as editorial aids, and the author remains solely responsible for the
content of the manuscript and this package.
