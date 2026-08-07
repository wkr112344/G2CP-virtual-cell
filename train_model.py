#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
虚拟细胞 — 扰动→表型 预测模型 (PyTorch, RTX 3050Ti 4GB)
==================================================================
多模态扰动编码器 + 剂量条件 + 12维表型回归头 (对标 G2CP 思路)

输入模态:
  - 药物/小分子/代谢物 (smiles) : 字符级 CNN
  - 肽/蛋白激素 (seq)            : 氨基酸 one-hot CNN
  - 基因扰动 (gene+kind)         : 基因符号 embedding + 扰动类型 one-hot
  - 物理因素 (phys)              : 参数数值向量 + 物理子类 embedding
统一投影到 256 维潜空间 -> 拼接剂量特征 -> 共享 MLP 回归头 -> 12维签名

关键设计: 用 *结构化特征* 而非纯 id lookup, 使模型能泛化到未见过的分子/序列
          (论文卖点)。按扰动 id 分层留出验证, 防泄露。
"""
import json, re, math, random
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- 设备 ----------
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('DEVICE =', DEVICE, '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')

# ---------- 超参 ----------
EMB   = 256      # 统一潜空间维度
BATCH = 64       # 4GB 显存下安全
LR    = 1e-3
EPOCH = 120
WD    = 1e-4
SEED  = 20260802
random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

PHENO_KEYS = ['pr','ap','dn','cc','lp','gl','in','ox','sg','en','mg','vi']

# ---------- 读数据 ----------
with open('train_samples.json', encoding='utf-8') as f:
    SAMPLES = json.load(f)
print('samples:', len(SAMPLES))

# ---------- 构建 vocab ----------
SMI_CHARS = sorted({ch for s in SAMPLES if s['featKind']=='smiles' for ch in (s.get('smiles') or '')})
AA = "ACDEFGHIKLMNPQRSTVWYBXZ*"            # 20 std + 常见
GENES = sorted({s['gene'] for s in SAMPLES if s['featKind']=='gene' and s.get('gene')})
PHYS_CLS = sorted({s.get('cls') for s in SAMPLES if s['featKind']=='phys' and s.get('cls')})
smi2i = {c:i for i,c in enumerate(SMI_CHARS)}
gene2i = {g:i for i,g in enumerate(GENES)}
phys2i = {c:i for i,c in enumerate(PHYS_CLS)}
KIND_OF = {'ko':0,'oe':1,'mut':2}
print(f'vocab: smiles={len(SMI_CHARS)} genes={len(GENES)} phys_cls={len(PHYS_CLS)}')

# ---------- 物理参数解析 ----------
def parse_phys_params(params):
    """params: dict[str,str] -> 数值向量(取每值首个数字), pad 到 6"""
    nums = []
    if isinstance(params, dict):
        for v in params.values():
            if isinstance(v, str):
                m = re.findall(r'-?\d+\.?\d*', v)
                if m: nums.append(float(m[0]))
    out = nums[:6] + [0.0]*(6-len(nums[:6]))
    return out

# ---------- Dataset ----------
MAX_SMI = 128
MAX_SEQ = 256
class PertDataset(torch.utils.data.Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        s = self.items[idx]
        fk = s['featKind']
        logd = float(s['logDose'])
        label = torch.tensor(s['label'], dtype=torch.float32)  # 12
        if fk == 'smiles':
            smi = s.get('smiles') or ''
            ids = [smi2i.get(c,0) for c in smi[:MAX_SMI]]
            ids += [0]*(MAX_SMI-len(ids))
            return {'kind':0,'tok':torch.tensor(ids),'len':min(len(smi),MAX_SMI),
                    'aux':torch.zeros(3),'phys':torch.zeros(6),'pcls':0,
                    'logd':logd,'label':label}
        if fk == 'seq':
            seq = s.get('seq') or ''
            oh = torch.zeros((MAX_SEQ, len(AA)))
            for i,ch in enumerate(seq[:MAX_SEQ]):
                if ch in AA: oh[i, AA.index(ch)] = 1.0
            return {'kind':1,'tok':oh,'len':min(len(seq),MAX_SEQ),
                    'aux':torch.zeros(3),'phys':torch.zeros(6),'pcls':0,
                    'logd':logd,'label':label}
        if fk == 'gene':
            gi = gene2i.get(s.get('gene'),0)
            ko = KIND_OF.get(s.get('kindOf'),0)
            aux = torch.zeros(3); aux[ko]=1.0
            return {'kind':2,'tok':torch.tensor(gi),'len':1,
                    'aux':aux,'phys':torch.zeros(6),'pcls':0,
                    'logd':logd,'label':label}
        # phys
        pcls = phys2i.get(s.get('cls'),0)
        pv = torch.tensor(parse_phys_params(s.get('params')), dtype=torch.float32)
        return {'kind':3,'tok':torch.tensor(0),'len':1,
                'aux':torch.zeros(3),'phys':pv,'pcls':pcls,
                'logd':logd,'label':label}

def collate(b):
    return {
        'kind': torch.tensor([x['kind'] for x in b]),
        'tok' : torch.stack([x['tok'] for x in b]),
        'len' : torch.tensor([x['len'] for x in b]),
        'aux' : torch.stack([x['aux'] for x in b]),
        'phys': torch.stack([x['phys'] for x in b]),
        'pcls': torch.tensor([x['pcls'] for x in b]),
        'logd': torch.tensor([x['logd'] for x in b], dtype=torch.float32),
        'label':torch.stack([x['label'] for x in b]),
    }

# ---------- 编码器 ----------
class SmilesEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.e = nn.Embedding(max(len(SMI_CHARS),1)+1, 32, padding_idx=0)
        self.c1 = nn.Conv1d(32, 64, 3, padding=1)
        self.c2 = nn.Conv1d(64, 128, 5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.proj = nn.Linear(128, EMB)
    def forward(self, tok, ln):
        x = self.e(tok).transpose(1,2)        # B,32,L
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = self.pool(x).squeeze(-1)          # B,128
        return self.proj(x)

class SeqEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv1d(len(AA), 64, 3, padding=1)
        self.c2 = nn.Conv1d(64, 128, 5, padding=2)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.proj = nn.Linear(128, EMB)
    def forward(self, tok, ln):               # tok: B,L,A one-hot
        x = tok.transpose(1,2)
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = self.pool(x).squeeze(-1)
        return self.proj(x)

class GeneEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.e = nn.Embedding(max(len(GENES),1)+1, EMB)
        self.fc = nn.Linear(EMB+3, EMB)
    def forward(self, tok, aux, ln):
        g = self.e(tok)                       # B,EMB
        return self.fc(torch.cat([g, aux], -1))

class PhysEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.e = nn.Embedding(max(len(PHYS_CLS),1)+1, 32)
        self.fc = nn.Linear(6+32, EMB)
    def forward(self, phys, pcls, ln):
        c = self.e(pcls)
        return self.fc(torch.cat([phys, c], -1))

class PertNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.smi = SmilesEncoder()
        self.seq = SeqEncoder()
        self.gene = GeneEncoder()
        self.phys = PhysEncoder()
        self.head = nn.Sequential(
            nn.Linear(EMB+4, EMB), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(EMB, EMB//2), nn.ReLU(),
            nn.Linear(EMB//2, 12)
        )
    def forward(self, b):
        kind = b['kind']
        z = torch.zeros(kind.size(0), EMB, device=kind.device)
        m0 = kind==0; m1 = kind==1; m2 = kind==2; m3 = kind==3
        if m0.any(): z[m0] = self.smi(b['tok'][m0], b['len'][m0])
        if m1.any(): z[m1] = self.seq(b['tok'][m1], b['len'][m1])
        if m2.any(): z[m2] = self.gene(b['tok'][m2], b['aux'][m2], b['len'][m2])
        if m3.any(): z[m3] = self.phys(b['phys'][m3], b['pcls'][m3], b['len'][m3])
        # 模态 one-hot(4) + logdose(1) 作为条件
        koh = F.one_hot(kind, num_classes=4).float()
        cond = torch.cat([z, koh, b['logd'].unsqueeze(-1)], -1)
        return self.head(cond)

# ---------- 划分 (按 id 防泄露) ----------
ids = sorted({s['id'] for s in SAMPLES})
random.shuffle(ids)
n_val = max(1, int(len(ids)*0.2))
val_ids = set(ids[:n_val])
train_items = [s for s in SAMPLES if s['id'] not in val_ids]
val_items   = [s for s in SAMPLES if s['id'] in val_ids]
print(f'split: train={len(train_items)} val={len(val_items)} perturbations train/val = {len(ids)-n_val}/{n_val}')

# 模态平衡采样 (少数模态过采样)
train_ds = PertDataset(train_items)
weights = []
for s in train_items:
    w = {'gene':1.0,'smiles':2.0,'seq':3.0,'phys':3.0}[s['featKind']]
    weights.append(w)
sampler = torch.utils.data.WeightedRandomSampler(weights, num_samples=len(train_items), replacement=True)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH, sampler=sampler, collate_fn=collate)
val_ds = PertDataset(val_items)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH, shuffle=False, collate_fn=collate)

# ---------- 模型/优化 ----------
model = PertNet().to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
crit = nn.MSELoss()

def eval_mse(dl):
    model.eval(); tot=0.0; n=0
    with torch.no_grad():
        for b in dl:
            b = {k:(v.to(DEVICE) if isinstance(v,torch.Tensor) else v) for k,v in b.items()}
            out = model(b)
            loss = crit(out, b['label'])
            tot += loss.item()*b['label'].size(0); n += b['label'].size(0)
    return tot/n

# 基线: 预测训练集均值
tr_labels = torch.stack([torch.tensor(s['label'],dtype=torch.float32) for s in train_items])
mean_label = tr_labels.mean(0)
base_mse = ((tr_labels-mean_label)**2).mean().item()
print(f'baseline (predict train-mean) MSE = {base_mse:.4f}')

best_val = 1e9; best_state=None
for ep in range(EPOCH):
    model.train(); tot=0; n=0
    for b in train_dl:
        b = {k:(v.to(DEVICE) if isinstance(v,torch.Tensor) else v) for k,v in b.items()}
        out = model(b)
        loss = crit(out, b['label'])
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        tot += loss.item()*b['label'].size(0); n += b['label'].size(0)
    tr_mse = tot/n
    val_mse = eval_mse(val_dl)
    if val_mse < best_val:
        best_val = val_mse; best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if (ep+1)%10==0 or ep==0:
        print(f'ep {ep+1:3d}/{EPOCH} trainMSE={tr_mse:.4f} valMSE={val_mse:.4f} (vs base {base_mse:.4f})')

# 保存
if best_state: model.load_state_dict(best_state)
torch.save({'state':model.state_dict(),
            'smi2i':smi2i,'gene2i':gene2i,'phys2i':phys2i,
            'AA':AA,'EMB':EMB,'PHENO_KEYS':PHENO_KEYS}, 'pert_model.pt')
print('=== SAVED pert_model.pt | best valMSE =', round(best_val,4),
      '| improvement vs baseline =', round((base_mse-best_val)/base_mse*100,1),'% ===')
