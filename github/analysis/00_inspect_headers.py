# 00 Inspect the headers and dimensions of the two raw gz files, check against config.expected; writes metadata/headers.json
import os, sys, zlib, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, log, p

cfg = load_config()
exp = cfg["expected"]


def head_lines(gz_path, nbytes=4_000_000, n_lines=12):
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    with open(gz_path, "rb") as f:
        txt = d.decompress(f.read(nbytes)).decode("utf-8", "replace")
    return txt.split("\n")[:n_lines]


log("checking the gene_tpm GCT header")
gct_lines = head_lines(cfg["paths"]["gene_tpm_gz"])
log("  line0: %r" % gct_lines[0])
log("  line1: %r" % gct_lines[1])          # "74628\t19788"
gct_rows, gct_cols = (int(x) for x in gct_lines[1].split("\t"))
colnames = gct_lines[2].split("\t")
log("  header row (line 2), first cols: %s ...  %d columns total" % (colnames[:3], len(colnames)))
assert gct_rows == exp["gct_rows"], "GCT gene rows %d != expected %d" % (gct_rows, exp["gct_rows"])
assert gct_cols == exp["gct_cols"], "GCT sample columns %d != expected %d" % (gct_cols, exp["gct_cols"])
assert len(colnames) == gct_cols + 2, "GCT header columns %d != %d + 2 (Name/Description)" % (len(colnames), gct_cols)

log("checking the transcripts_tpm header")
tr_lines = head_lines(cfg["paths"]["transcripts_tpm_gz"])
tr_cols = tr_lines[0].split("\t")
log("  first row, %d columns: %s ..." % (len(tr_cols), tr_cols[:5]))
assert tr_cols[0] == "transcript_id" and tr_cols[1] == "gene_id", \
    "transcripts first two columns are not transcript_id/gene_id: %s" % tr_cols[:2]
n_tr_samples = len(tr_cols) - 2
assert n_tr_samples == exp["transcripts_sample_cols"], \
    "transcripts sample columns %d != expected %d" % (n_tr_samples, exp["transcripts_sample_cols"])

# whether the sample order matches GCT (already validated once; here re-check the first 50)
gct_samples = colnames[2:]
if gct_samples == tr_cols[2:]:
    sample_order_identical = True
else:
    sample_order_identical = set(gct_samples) == set(tr_cols[2:])
    log("  WARNING: sample order differs but the sets match; 03 will reorder to the GCT order")
assert sample_order_identical, "the two files' sample sets differ; aborting"

out = {"gct_rows": gct_rows, "gct_cols": gct_cols,
       "gct_sample_columns": gct_samples,
       "transcripts_rows_expected": exp["transcripts_rows"],
       "sample_order_identical_to_gct": bool(gct_samples == tr_cols[2:])}
with open(p("metadata", "headers.json"), "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2)
log("PASS: all header/dimension checks passed -> metadata/headers.json")
