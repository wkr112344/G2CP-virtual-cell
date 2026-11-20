# 01 Sample/annotation join check: donor parsing, tissue labels, coverage -> metadata/sample_tissue_map.json, donors.json
import os, sys, csv, json, collections
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, log, donor_id, p

cfg = load_config()
lab = cfg["analysis"]["tissue_label"]

samples = [l.strip() for l in open(cfg["paths"]["sample_ids_txt"], encoding="utf-8") if l.strip()]
log("%d samples in the expression matrix" % len(samples))
assert len(samples) == cfg["expected"]["gct_cols"]

with open(cfg["paths"]["sample_attributes"], encoding="utf-8-sig") as f:
    ann = {r["SAMPID"]: {"SMTS": r.get("SMTS", ""), lab: r.get(lab, "")} for r in csv.DictReader(f, delimiter="\t")}
log("%d SAMPIDs in the annotation table" % len(ann))

missing = [s for s in samples if s not in ann]
log("expression samples with no annotation match: %d" % len(missing))
assert not missing, "samples with no matching annotation: %s" % missing[:5]

tmap = {s: {"donor": donor_id(s), "SMTS": ann[s]["SMTS"], lab: ann[s][lab]} for s in samples}
donors = sorted({v["donor"] for v in tmap.values()})
tcnt = collections.Counter(v[lab] for v in tmap.values())
dcnt = collections.Counter(v["donor"] for v in tmap.values())
log("%d donors (%.1f samples each on average); %s has %d classes" % (len(donors), len(samples)/len(donors), lab, len(tcnt)))

json.dump(tmap, open(p("metadata", "sample_tissue_map.json"), "w", encoding="utf-8"), indent=1)
json.dump({"donors": donors, "n_samples": len(samples), "tissue_label": lab,
           "tissue_counts": dict(tcnt), "donor_sample_counts": dict(dcnt)},
          open(p("metadata", "donors.json"), "w", encoding="utf-8"), indent=1)
log("PASS -> metadata/sample_tissue_map.json + donors.json")
