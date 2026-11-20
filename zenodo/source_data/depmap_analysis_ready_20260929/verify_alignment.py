#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Post-delivery self-check: confirm the row/column calibers and the split agree (read-only)."""
import json, os, re
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
cell = open(os.path.join(HERE, "index", "cell_order.txt")).read().splitlines()
genes = open(os.path.join(HERE, "index", "target_genes.txt")).read().splitlines()
idx = json.load(open(os.path.join(HERE, "index", "depmap_index.json")))
test = [l for l in open(os.path.join(HERE, "index", "test_cell_order.txt")).read().splitlines() if l.strip()]
print("cells", len(cell), "genes", len(genes), "test", len(test))
print("test order == split test_cells:", test == idx["test_cells"])
Yc = np.load(os.path.join(HERE, "reference", "Y_CERES.npy"), mmap_mode="r")
Yh = np.load(os.path.join(HERE, "reference", "Y_Chronos.npy"), mmap_mode="r")
print("Y_CERES", Yc.shape, "Y_Chronos", Yh.shape)
tr = idx["train_rows_in_cell_order"]; te = idx["test_rows_in_cell_order"]
gm_c = np.nanmean(np.asarray(Yc[tr], dtype=np.float64), axis=0)
gm_h = np.nanmean(np.asarray(Yh[tr], dtype=np.float64), axis=0)
print("train-only gene mean: CERES finite=%d/%d, Chronos finite=%d/%d"
      % (np.isfinite(gm_c).sum(), len(gm_c), np.isfinite(gm_h).sum(), len(gm_h)))
for k in range(5):
    for arm in ("ceres", "chronos"):
        a = np.load(os.path.join(HERE, "predictions_test", "seed%d_pred_%s_arm_test.npy" % (k, arm)), mmap_mode="r")
        assert a.shape == (len(test), len(genes)), (k, arm, a.shape)
print("all 10 test prediction matrices: shape OK")
print("OK")
