import sys, os, numpy as np, h5py, torch
sys.path.insert(0, r'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/experiments')
import PRnet_commonX_train_eval_multiseed as M

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA = r'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/data/prnet_commonX/PRnet_commonX_dualY.h5'
sd = torch.load(r'C:/Users/wkr20/WorkBuddy/Claw/gigascience_v5/data/prnet_commonX/multiseed_seed0_arm_beta.pt', map_location=DEV)
with h5py.File(DATA, 'r') as f:
    split = f['split'][:].astype(int)
    dose = f['dose'][:].astype(np.float64); time = f['time'][:].astype(np.float64)
dm, ds = dose.mean(), dose.std() or 1.0; tm, ts = time.mean(), time.std() or 1.0
dose_z = ((dose-dm)/ds).astype(np.float32); time_z = ((time-tm)/ts).astype(np.float32)
eval_idx = np.where(split == 3)[0][:2000]   # small subset for speed

model = M.make_model(DEV, 0); model.load_state_dict(sd); model.eval()
preds = []
with h5py.File(DATA, 'r') as f:
    with torch.no_grad():
        for xb, cond in M.iter_batches(f, eval_idx, 'Y_beta', dose_z, time_z, 1024, False, DEV, False):
            out = model(xb, cond, torch.randn(xb.size(0), 10, device=DEV))
            preds.append(out[:, :out.size(1)//2].detach().cpu().numpy())
pred = np.concatenate(preds, 0)
with h5py.File(DATA, 'r') as f:
    Yb = f['Y_beta'][eval_idx]; Yd = f['Y_dcic'][eval_idx]
print('pred shape:', pred.shape)
print('PRED  std over samples (per-gene mean std):', round(float(pred.std(0).mean()), 4))
print('Y_beta std over samples (per-gene mean std):', round(float(Yb.std(0).mean()), 4))
print('=> if pred std ~0 the model outputs a near-CONSTANT (bug); if pred std>0 but small, weak learner')
# quick per-sample spearman vs Yb
sp = M.corr_rows(pred, Yb, 'spearman')
print('mean per-sample spearman(pred, Y_beta):', round(float(np.nanmean(sp)), 4))
spd = M.corr_rows(pred, Yd, 'spearman')
print('mean per-sample spearman(pred, Y_dcic):', round(float(np.nanmean(spd)), 4))
