"""
UniPert 官方接口兼容门面（interface facade）
============================================
目的：让本骨架对外暴露与官方 UniPert 包**完全一致**的 API，
实现"国标三孔"对齐 —— 这样：

  1) 我们的向量能直接喂给 GEARS / CPA 等成熟下游
     （官方下游读 adata.uns['UniPert_reps']，我们也写同一个 key）；
  2) 将来在更大显存机器上装上官方 UniPert（ESM2 + 预训练权重），
     把 backend='official' 一行切换即可，对比学习 + G2CP 两阶段代码不用改
     （这就是"可换芯"：轻量本地编码器 ↔ 官方 ESM2 编码器互换）。

官方 API（来自 github.com/TencentAILabHealthcare/UniPert，论文配套包）：
    from unipert import UniPert
    unipert = UniPert()
    out_embs, invalid = unipert.encode_genes(gene_names=[...], uniprot_ids=[...], fasta_file=..., save_path=...)
    out_embs, invalid = unipert.encode_compounds(compound_names=[...], compound_dict={...}, csv_file=..., smiles_list=[...], save_path=...)
    unipert.encode_anndata_perturbations(adata, perturbation_columns=[...], perturbation_types=[...], control_key=..., return_results=...)
    -> 每个扰动向量维度 = 256（OFFICIAL_EMBED_DIM）

本门面在 backend='local'（默认）时复用我们自己的轻量编码器，
输出 256 维向量，I/O 契约与官方逐一对齐。
"""
import os
import pickle
import warnings

import numpy as np
import torch

from .config import DEVICE, OFFICIAL_EMBED_DIM
from .model import UniPert as _UniPertModule
from .compound_encoder import smiles_to_ecfp4
from . import io_adapters as io


class UniPertClient:
    """
    官方兼容门面。用法与官方完全一致：

        from unipret import UniPert
        unipert = UniPert()                       # backend='local'，自动选 GPU
        embs, invalid = unipert.encode_genes(gene_names=["ESR1", "BRCA1"])
        embs, invalid = unipert.encode_compounds(compound_dict={"Aspirin": "CC(=O)Oc1ccccc1C(=O)O"})
        unipert.encode_anndata_perturbations(adata, return_results=False)
    """

    def __init__(self, backend="local", device=None, data_json=None, checkpoint=None):
        self.backend = backend
        self.device = device or DEVICE
        self.checkpoint = checkpoint
        self.out_dim = OFFICIAL_EMBED_DIM

        # 本地解析器：名→序列 / 名→SMILES（来自 dataset.json 的真实数据）
        self.gene_to_seq, self.name_to_smiles, _ = io.load_local_maps(data_json)

        # 基因身份词汇表（本地后端用，决定 GeneEncoder 的 num_genes）
        self.gene_to_idx = {sym: i for i, sym in enumerate(self.gene_to_seq.keys())}
        self._trained = checkpoint is not None and os.path.isfile(checkpoint or "")
        self.model = None

        # 若指定官方后端，尝试加载官方包（本机未装则自动回退本地）
        if backend == "official":
            try:
                import unipert as _up
                self._official = _up.UniPert()
                self.out_dim = 256
                print("✅ 已加载官方 UniPert 编码器（ESM2 + 预训练权重）。")
            except Exception as e:
                warnings.warn(
                    f"未检测到官方 unipert 包（{e}），自动回退到本地轻量编码器。"
                    "在更大显存机器上 `pip install git+https://github.com/lynn-1998/UniPert.git` 后即可用 backend='official'。"
                )
                self.backend = "local"

    # ---------------------------------------------------------- 模型管理
    def _ensure_model(self):
        if self.model is None:
            self.model = _UniPertModule(num_genes=max(len(self.gene_to_idx), 1))
            if self.checkpoint and os.path.isfile(self.checkpoint):
                try:
                    self.model.load_state_dict(torch.load(self.checkpoint, map_location=self.device))
                    self._trained = True
                except Exception as e:
                    warnings.warn(f"加载 checkpoint 失败（{e}），使用随机初始化权重。")
            self.model.to(self.device)
            self.model.eval()

    def _rebuild_if_untrained(self):
        """仅在未加载训练权重时，按当前词汇表重建编码器（可容纳新基因）。"""
        if self._trained:
            return
        self.model = _UniPertModule(num_genes=max(len(self.gene_to_idx), 1))
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _save(out, path):
        with open(path, "wb") as f:
            pickle.dump(out, f)

    # ---------------------------------------------------------- 基因编码
    def encode_genes(self, gene_names=None, uniprot_ids=None, fasta_file=None, save_path=None):
        if self.backend == "official":
            return self._official.encode_genes(
                gene_names=gene_names, uniprot_ids=uniprot_ids, fasta_file=fasta_file, save_path=save_path)

        if fasta_file:
            self.gene_to_seq.update(io.parse_fasta(fasta_file))
            dirty = False
            for lab in self.gene_to_seq:
                if lab not in self.gene_to_idx:
                    self.gene_to_idx[lab] = len(self.gene_to_idx)
                    dirty = True
            if dirty:
                self._rebuild_if_untrained()

        resolved, invalid = io.resolve_gene_inputs(
            gene_names=gene_names, uniprot_ids=uniprot_ids, fasta_file=None, gene_to_seq=self.gene_to_seq)

        if not resolved:
            return {}, invalid

        # 未训练时把新基因并入词汇表，否则标记为 invalid（与官方 OOV 行为一致）
        for lab in list(resolved.keys()):
            if lab not in self.gene_to_idx and not self._trained:
                self.gene_to_idx[lab] = len(self.gene_to_idx)
                self._rebuild_if_untrained()

        self._ensure_model()
        seqs, idxs, labels = [], [], []
        for lab, seq in resolved.items():
            if lab in self.gene_to_idx:
                seqs.append(seq)
                idxs.append(self.gene_to_idx[lab])
                labels.append(lab)
            else:
                invalid.append(lab)
        if not seqs:
            return {}, invalid

        with torch.no_grad():
            g = self.model.encode_gene(
                seqs, torch.tensor(idxs, dtype=torch.long, device=self.device))
        emb = g.detach().cpu().numpy().astype("float32")
        out = {labels[i]: emb[i] for i in range(len(labels))}
        if save_path:
            self._save(out, save_path)
        return out, invalid

    # ---------------------------------------------------------- 化合物编码
    def encode_compounds(self, compound_names=None, compound_dict=None,
                          csv_file=None, smiles_list=None, save_path=None):
        if self.backend == "official":
            return self._official.encode_compounds(
                compound_names=compound_names, compound_dict=compound_dict,
                csv_file=csv_file, smiles_list=smiles_list, save_path=save_path)

        resolved, invalid = io.resolve_compound_inputs(
            compound_names=compound_names, compound_dict=compound_dict,
            csv_file=csv_file, smiles_list=smiles_list, name_to_smiles=self.name_to_smiles)

        if not resolved:
            return {}, invalid

        self._ensure_model()
        graphs, labels = [], []
        for nm, sm in resolved.items():
            try:
                graphs.append(smiles_to_ecfp4(sm))
                labels.append(nm)
            except Exception:
                invalid.append(nm)
        if not graphs:
            return {}, invalid

        with torch.no_grad():
            c = self.model.encode_compound(graphs)
        emb = c.detach().cpu().numpy().astype("float32")
        out = {labels[i]: emb[i] for i in range(len(labels))}
        if save_path:
            self._save(out, save_path)
        return out, invalid

    # ---------------------------------------------------------- AnnData 编码
    def encode_anndata_perturbations(self, adata, perturbation_columns=None,
                                      perturbation_types=None, control_key="control",
                                      return_results=False):
        """
        把扰动 AnnData（.h5ad）里的扰动试剂编码成 UniPert 向量，写入
        adata.uns['UniPert_reps']（官方下游 GEARS/CPA 读取的同一个 key）。

        参数（与官方一致）：
          adata               : anndata.AnnData 对象
          perturbation_columns: 存放扰动名的 obs 列名，默认 ['perturbation']
          perturbation_types  : 扰动类型，默认 ['genetic']（可含 'chemical'）
          control_key         : 对照（不参与编码），默认 'control'
          return_results      : True 时额外返回 {'UniPert_reps':..., 'invalid_ptbgs':...}
        """
        try:
            import anndata  # noqa: F401  仅在用到时才要求该依赖
        except ImportError:
            raise ImportError(
                "处理 AnnData 需要安装 anndata（pip install anndata）。"
                "本机网络受限时可只使用 encode_genes / encode_compounds。")

        perturbation_columns = perturbation_columns or ["perturbation"]
        perturbation_types = perturbation_types or ["genetic"]

        reps, invalid = {}, []
        for col in perturbation_columns:
            if col not in adata.obs:
                continue
            uniq = [str(u) for u in adata.obs[col].dropna().unique().tolist()]
            targets = [u for u in uniq if u != control_key]
            if "genetic" in perturbation_types:
                ge, gi = self.encode_genes(gene_names=targets)
                reps.update(ge)
                invalid += gi
            if "chemical" in perturbation_types:
                ce, ci = self.encode_compounds(compound_names=targets)
                reps.update(ce)
                invalid += ci

        adata.uns["UniPert_reps"] = reps
        adata.uns["invalid_ptbgs"] = invalid

        if return_results:
            return {"UniPert_reps": reps, "invalid_ptbgs": invalid}
        return None
