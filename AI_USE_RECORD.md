# AI use record

This file documents the use of generative AI tools during the preparation of the
manuscript and this reproducibility package. It is provided so that readers and
reviewers can judge which parts of the work involved AI assistance.

## Tools and models

| Tool | Model | Provider | Access |
|---|---|---|---|
| ChatGPT | [model/version: to be confirmed by author] | OpenAI | https://chatgpt.com/ |

No other generative AI service was used for the work reported here.

## Period of use

20–22 November 2026.

## User

Kairui Wei (first author). All prompts, decisions, and
acceptance of edits were made by this author.

## Scope of use

AI assistance was limited to language editing, document preparation, and consistency
review — editorial and mechanical tasks matching the `Use of AI tools` statement in the
manuscript:

1. **Language editing** — grammar, article usage, and phrasing in the English
   manuscript. No new scientific claims were introduced.
2. **Reference formatting** — checking that reference numbers in the text match the
   reference list, and that DOIs resolve to the cited records.
3. **Document preparation** — section ordering, heading styles, table and figure
   caption layout, export settings, assembly of the submission files, manifest
   generation, standardisation of the license files, and reformatting of two
   supplementary tables (S31, S33) from existing analysis outputs.

## What was not done by AI

- Study design, data selection, and analysis choices.
- Model training, scoring, bootstrap procedures, and numerical results; these were
  produced by the author and the analysis scripts in `github/analysis/`.
- Interpretation of results, limitations, and conclusions.
- Interpretive decisions and the wording of conclusions; AI suggestions were reviewed
  and accepted or rejected by the author.

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

A numerical audit of the key reported values was run with
`github/analysis/stability_checks/_audit_numbers.py` against `EXPECTED_OUTPUTS.json`
(tolerance 5×10⁻⁴ for point estimates, 1×10⁻³ for interval endpoints); a verification
run reported 74 checks with no mismatches. This checks the key conclusions, not every
number in the manuscript. Reference numbering and figure/table cross-references were
checked after the final edit. The author reviewed and approved all text that AI touched.

## Statement

AI tools were not used to generate scientific content, data, or conclusions. They
were used as editorial aids, and the author remains solely responsible for the
content of the manuscript and this package.
