# 04 Donor-level split (seed locked) -> metadata/splits.json
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_config, log, p

cfg = load_config()
assert cfg["analysis"]["split_unit"] == "donor", "the split unit must be donor"

d = json.load(open(p("metadata", "donors.json")))
donors = np.array(d["donors"])
rng = np.random.default_rng(cfg["analysis"]["seed"])
perm = rng.permutation(len(donors))
n_test = int(round(len(donors) * cfg["analysis"]["test_fraction"]))
test = sorted(donors[perm[:n_test]].tolist())
train = sorted(donors[perm[n_test:]].tolist())
assert len(set(test) & set(train)) == 0 and len(test) + len(train) == len(donors)

tmap = json.load(open(p("metadata", "sample_tissue_map.json")))
tr_s = [s for s in tmap if tmap[s]["donor"] in set(train)]
te_s = [s for s in tmap if tmap[s]["donor"] in set(test)]
out = {"seed": cfg["analysis"]["seed"], "train_donors": train, "test_donors": test,
       "train_samples": tr_s, "test_samples": te_s}
json.dump(out, open(p("metadata", "splits.json"), "w"), indent=1)
log("donors %d = train %d + test %d; samples train %d / test %d -> metadata/splits.json"
    % (len(donors), len(train), len(test), len(tr_s), len(te_s)))
