"""Build Supplementary Table S33.

Focused view of the LINCS model-comparison survey (S30) on the single question
the manuscript cites it for: did a study test whether changing the public
evaluation matrix altered model ordering while predictions and evaluation
records were fixed?

Input : zenodo/supplementary/S30_lincs_model_comparison_practices.csv
Output: zenodo/supplementary/S33_matrix_swap_reporting_survey.csv
"""
import csv
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC = os.path.join(ROOT, 'zenodo', 'supplementary', 'S30_lincs_model_comparison_practices.csv')
OUT = os.path.join(ROOT, 'zenodo', 'supplementary', 'S33_matrix_swap_reporting_survey.csv')

CODE = {'NR': 'No (not reported)', 'NC': 'No (single reference only)'}


def main():
    with open(SRC, newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))

    out = []
    for r in rows:
        raw = (r['reference_swap_model_reranking'] or '').strip()
        tested = 'Yes' if raw not in ('NR', 'NC', '') else 'No'
        out.append({
            'no': r['no'],
            'study': r['study_year'],
            'prediction_endpoint': r['prediction_endpoint'],
            'lincs_evaluation_reference': r['lincs_evaluation_reference'],
            'reported_model_comparisons': r['reported_model_comparisons'],
            'reference_swap_tested': tested,
            'reason_if_not': CODE.get(raw, raw),
            'source_url': r['source_url'],
        })

    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)

    yes = sum(1 for r in out if r['reference_swap_tested'] == 'Yes')
    print('studies:', len(out), ' tested matrix swap and reported model reranking:', yes)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
