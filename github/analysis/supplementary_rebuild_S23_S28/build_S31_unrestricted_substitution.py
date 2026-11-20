"""Build Supplementary Table S31.

The unrestricted drug-substitution analysis is scheme A of the two-scheme
comparison already stored in S28: drug labels are shuffled without the
same-cell-line restriction. S31 is therefore the scheme-A view of that table,
kept as a separate file because the manuscript cites it separately.

Input : zenodo/supplementary/S28_permutation_scheme_comparison.csv
Output: zenodo/supplementary/S31_unrestricted_substitution.csv
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SRC = os.path.join(ROOT, 'zenodo', 'supplementary', 'S28_permutation_scheme_comparison.csv')
OUT = os.path.join(ROOT, 'zenodo', 'supplementary', 'S31_unrestricted_substitution.csv')


def main():
    with open(SRC, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames)
        rows = [r for r in reader if r['scheme'].upper().startswith('A')]

    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print('scheme A rows:', len(rows))
    for r in rows:
        print('  %-9s %-10s observed=%s null=%s (%s to %s) exceeds=%s'
              % (r['evaluation_matrix'], r['model'], r['observed_pearson'],
                 r['null_mean'], r['null_ci95_low'], r['null_ci95_high'],
                 r['observed_exceeds_ci']))
    print('wrote', OUT)


if __name__ == '__main__':
    main()
