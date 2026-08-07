"""
predict.py —— 阶段 B 模型推理 + 可视化报告生成
================================================
加载 stageB.pt，用任意药物 SMILES 预测它在细胞里的「基因表达变化向量」
（2000 个高变基因，perturbed − control），即 UniPert-G2CP 的「扰动→表型」预测。

两种用法：
  1) 默认（生成报告）：
       python predict.py
     -> 重新跑一遍 51 个训练化合物预测，写出 report.html（可直接双击打开看）。

  2) 预测一个新药物（你给 SMILES）：
       python predict.py --smiles "CC(=O)Oc1ccccc1C(=O)O" --cell_line K562
     -> 打印该药最上调 / 最下调的 top 基因（按 HVG 的 Ensembl ID）。
     说明：化合物编码器是通用的（吃 SMILES），所以能预测训练集之外的新药；
           但本模型只在 51 个药上微调过，对全新药的绝对精度未经独立验证。

注意：本机网络受限，基因用 Ensembl ID 标注（sciPlex3 的 var_names 已损坏为 nan，
但 ensembl_id 列完整）。需要基因符号可后续补 Ensembl→symbol 映射。
"""
import sys, os, json, argparse, re
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from unipret.data_bridge import (PerturbationReader, load_local_dataset,
                                 build_samples, select_hvg)
from unipret.compound_encoder import smiles_to_graph
from unipret.model import UniPert
from unipret.effect_model import PerturbationEffectModel
from unipret.config import DEVICE, CELL_LINE_NAMES

CKPT = "stageB.pt"
ROOT = os.path.dirname(os.path.abspath(__file__))
SCIPLEX3 = "C:/Users/wkr20/Desktop/virtual_cell_real_data/sciPlex3/SrivatsanTrapnell2020_sciplex3.h5ad"
LOCAL = os.path.join(ROOT, "dataset.json")


# ----------------------------------------------------------------- 模型加载
def load_effect(ckpt=CKPT):
    sa = torch.load(ckpt, map_location=DEVICE)
    gene_vocab = sa["gene_vocab"]
    hvg_n = int(sa.get("hvg_dim", 2000))
    unipert = UniPert(num_genes=len(gene_vocab) + 1).to(DEVICE)
    unipert.load_state_dict(sa["unipert"])
    effect = PerturbationEffectModel(unipert, hvg_n,
                                     with_compound_head=True).to(DEVICE)
    effect.load_state_dict(sa["effect"])
    effect.eval()
    return effect, hvg_n, gene_vocab


def predict_one(effect, smiles, cl_idx):
    """单个药物 SMILES -> 2000 维表达变化向量 (np.float32)。"""
    g = smiles_to_graph(smiles)
    if g is None or not g[0]:
        return None
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
            pred = effect.forward_compound(
                [g], torch.tensor([cl_idx], dtype=torch.long, device=DEVICE))
    return pred[0].detach().cpu().numpy().astype(float)


# ----------------------------------------------------------------- HVG -> Ensembl
def hvg_ensembl_ids(reader, hvg):
    vals = np.array(reader.ad.var["ensembl_id"].values, dtype=object)
    out = []
    for i in hvg:
        v = str(vals[i])
        if v in ("nan", "id gene_short_name", "") or v.startswith("nan"):
            out.append(f"gene#{i}")
        else:
            out.append(v)
    return out


# ----------------------------------------------------------------- 解析 loss 日志
def parse_loss_log(path="trainB.log"):
    loss = []
    if not os.path.exists(path):
        return loss
    pat = re.compile(r"epoch\s+(\d+)/(\d+)\s+loss=([\d.]+)")
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = pat.search(line)
            if m:
                loss.append([int(m.group(1)), float(m.group(3))])
    return loss


# ----------------------------------------------------------------- 报告数据
def build_report_data(effect, hvg_n, gene_vocab):
    reader = PerturbationReader(SCIPLEX3, backed=True)
    # 不整块预加载：走 backed 磁盘路径（与训练一致，且更快、不占 8GB 内存）
    hvg = select_hvg(reader, n=hvg_n, max_cells=50000)
    ens = hvg_ensembl_ids(reader, hvg)
    local = load_local_dataset(LOCAL)
    samples = build_samples(reader, local, kind="compound", hvg=hvg)
    for s in samples:
        s["smiles"] = local["drugs"][s["local_idx"]].get("smiles", "")
    samples = [s for s in samples if s.get("smiles")]
    reader.close()

    compounds = []
    for s in samples:
        delta = predict_one(effect, s["smiles"], s["cell_line_idx"])
        if delta is None:
            continue
        compounds.append({
            "name": s["name"],
            "cell_line": CELL_LINE_NAMES[s["cell_line_idx"]] if s["cell_line_idx"] < len(CELL_LINE_NAMES) else "?",
            "cl_idx": int(s["cell_line_idx"]),
            "delta": delta.tolist(),
        })
    loss = parse_loss_log()
    return {"hvg_ensembl": ens, "compounds": compounds, "loss": loss,
            "hvg_n": hvg_n, "n_train": len(compounds)}


# ----------------------------------------------------------------- 单药预测打印
def predict_smiles_cli(smiles, cell_line="K562"):
    effect, hvg_n, gv = load_effect()
    cl_idx = CELL_LINE_NAMES.index(cell_line) if cell_line in CELL_LINE_NAMES else 0
    reader = PerturbationReader(SCIPLEX3, backed=True)
    # 不整块预加载：走 backed 磁盘路径（与训练一致，且更快、不占 8GB 内存）
    hvg = select_hvg(reader, n=hvg_n, max_cells=50000)
    ens = hvg_ensembl_ids(reader, hvg)
    reader.close()
    delta = predict_one(effect, smiles, cl_idx)
    if delta is None:
        print("!! SMILES 解析失败（可能含不支持的字符），无法构图。")
        return
    order = np.argsort(delta)
    up = order[::-1][:20]
    down = order[:20]
    print(f"\n药物 SMILES: {smiles}")
    print(f"细胞系: {cell_line}   预测维度: {delta.shape[0]} 个高变基因\n")
    print("↑ 最上调 (top up):")
    for i in up:
        print(f"   {ens[i]:20s}  Δ={delta[i]:+.4f}")
    print("↓ 最下调 (top down):")
    for i in down:
        print(f"   {ens[i]:20s}  Δ={delta[i]:+.4f}")


# ----------------------------------------------------------------- HTML 报告
def write_report(data, out="report.html"):
    js = json.dumps(data, ensure_ascii=False)
    html = HTML_TEMPLATE.replace("__DATA__", js)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"报告已写出: {out}  ({len(data['compounds'])} 个化合物预测)")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>虚拟细胞 · UniPert-G2CP 阶段B 成果报告</title>
<style>
:root{--bg:#0f1420;--panel:#1a2233;--ink:#e8edf6;--mut:#9fb0c8;--up:#ff5d6c;--dn:#36d399;--acc:#5b8cff;--line:#2a3650}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif;line-height:1.6}
.wrap{max-width:980px;margin:0 auto;padding:28px 20px 80px}
h1{font-size:24px;margin:0 0 4px}
h2{font-size:19px;margin:34px 0 12px;padding-left:10px;border-left:4px solid var(--acc)}
.sub{color:var(--mut);font-size:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:18px 20px;margin:14px 0}
.kpis{display:flex;gap:12px;flex-wrap:wrap}
.kpi{flex:1;min-width:150px;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 16px}
.kpi .v{font-size:26px;font-weight:700;color:var(--acc)}
.kpi .l{font-size:13px;color:var(--mut)}
select{background:#0c1322;color:var(--ink);border:1px solid var(--line);border-radius:8px;padding:8px 10px;font-size:14px;width:100%}
.col2{display:flex;gap:18px;flex-wrap:wrap}
.col2>div{flex:1;min-width:280px}
.bars{font-size:13px}
.row{display:flex;align-items:center;gap:8px;margin:3px 0}
.row .nm{width:150px;color:var(--mut);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.row .track{flex:1;background:#0c1322;border-radius:5px;height:16px;position:relative;overflow:hidden}
.row .fill{position:absolute;top:0;bottom:0;left:50%}
.row .val{width:62px;text-align:right;font-variant-numeric:tabular-nums}
.legend{font-size:13px;color:var(--mut);margin:6px 0}
svg{width:100%;height:auto;display:block}
.note{font-size:13px;color:var(--mut);background:#0c1322;border-left:3px solid var(--acc);padding:10px 14px;border-radius:8px;margin:10px 0}
table.hm{border-collapse:collapse;margin:0 auto}
table.hm td{width:13px;height:13px;padding:0}
.hmwrap{overflow:auto}
.tag{display:inline-block;background:#0c1322;border:1px solid var(--line);border-radius:20px;padding:2px 10px;font-size:12px;color:var(--mut);margin:2px}
code{background:#0c1322;padding:1px 6px;border-radius:5px;color:var(--acc)}
</style></head>
<body><div class="wrap">
<h1>虚拟细胞 · UniPert-G2CP 阶段 B 成果报告</h1>
<div class="sub">在 RTX 3050Ti (4GB) 上用 sciPlex3 真实单细胞转录组微调完成 · 生成时间 <span id="t"></span></div>

<div class="card">
  <h2 style="margin-top:6px">你现在能干什么？能预测吗？</h2>
  <p><b>能预测。</b> 这个模型干的事是：<b>输入一种药 + 一个细胞系 → 输出它在细胞里会「让哪些基因升高、哪些降低」</b>（2000 个高变基因的表达变化向量）。这正是论文 UniPert-G2CP 的核心——「扰动 → 表型」预测。</p>
  <ul>
    <li><b>① 看已训练药物的预测</b>：下面「预测探索器」里选一种药，立刻看到它最上调/最下调的 top 基因（红=升，绿=降）。</li>
    <li><b>② 预测新药（命令行）</b>：<code>python predict.py --smiles "..." --cell_line K562</code> 给任意 SMILES 就能预测（化合物编码器是通用的，不吃药物名）。</li>
    <li><b>③ 比较药物</b>：下面「药物相似度热图」看哪些药作用机制相似（效应向量越像，说明它们可能打同一个通路）。</li>
  </ul>
  <div class="note">诚实说明：本模型只在 sciPlex3 里 <b>51 个对齐上的药物</b> 上微调过，所以「预测新药」是<b>能力可行但绝对精度未经独立验证</b>。它学的是「结构 → 转录效应」的映射趋势，适合做假设生成/排序，不是临床剂量级预测。</div>
</div>

<div class="kpis" id="kpis"></div>

<h2>训练收敛曲线（loss）</h2>
<div class="card"><svg id="loss" viewBox="0 0 800 240"></svg>
<div class="legend">横轴=训练轮次(epoch)，纵轴=均方误差(loss)。从 ~1.15 降到 ~0.68，平稳收敛，说明模型学到了「药物结构 → 表达变化」的映射。</div></div>

<h2>预测探索器</h2>
<div class="card">
  <select id="sel"></select>
  <div class="legend" id="meta"></div>
  <div class="col2">
    <div><div class="legend" style="color:var(--up)">↑ 最上调基因 (top up)</div><div class="bars" id="up"></div></div>
    <div><div class="legend" style="color:var(--dn)">↓ 最下调基因 (top down)</div><div class="bars" id="dn"></div></div>
  </div>
  <div class="note">每条 = 一个高变基因（用 Ensembl ID 标注）。Δ 是「用药后 − 对照」的表达变化：正=上调、负=下调。数值越大，该药对这个基因的影响越强。</div>
</div>

<h2>药物相似度热图（机制聚类）</h2>
<div class="card">
  <div class="hmwrap"><table class="hm" id="hm"></table></div>
  <div class="legend">颜色越红=两药预测效应越相似（cosine 相似度越接近 1）；越蓝=越不同。同一类机制的药（如 HDAC 抑制剂）应当聚成一团——这能验证模型不是瞎猜，而是抓住了药物机制。</div>
  <div id="groups"></div>
</div>

<h2>已知机制分组（供对照热图）</h2>
<div class="card" id="known"></div>

<div class="note">数据来源：sciPlex3（Srivatsan &amp; Trapnell 2020）真实单细胞药物扰动；模型：UniPert-G2CP 两阶段（阶段A 基因预训练 → 阶段B 化合物微调）。基因以 Ensembl ID 标注，符号映射可后续补。</div>
</div>

<script>
const DATA = __DATA__;
document.getElementById('t').textContent = new Date().toLocaleString();

// KPIs
const kp = [
  ['训练化合物数', DATA.n_train],
  ['预测维度(高变基因)', DATA.hvg_n],
  ['训练轮次', DATA.loss.length ? DATA.loss[DATA.loss.length-1][0] : '-'],
  ['最终 loss', DATA.loss.length ? DATA.loss[DATA.loss.length-1][1].toFixed(3) : '-'],
];
document.getElementById('kpis').innerHTML = kp.map(([l,v])=>
  `<div class="kpi"><div class="v">${v}</div><div class="l">${l}</div></div>`).join('');

// Loss curve
(function(){
  const W=800,H=240,pad=36;
  const L=DATA.loss; if(!L.length) return;
  const xs=L.map(d=>d[0]), ys=L.map(d=>d[1]);
  const xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
  const sx=x=>pad+(x-xmin)/(xmax-xmin||1)*(W-2*pad);
  const sy=y=>H-pad-(y-ymin)/(ymax-ymin||1)*(H-2*pad);
  let pts=L.map(d=>`${sx(d[0]).toFixed(1)},${sy(d[1]).toFixed(1)}`).join(' ');
  let grid='';
  for(let i=0;i<=4;i++){const y=pad+i*(H-2*pad)/4;const val=(ymax-(ymax-ymin)*i/4).toFixed(2);
    grid+=`<line x1="${pad}" y1="${y}" x2="${W-pad}" y2="${y}" stroke="#2a3650"/><text x="4" y="${y+4}" fill="#9fb0c8" font-size="11">${val}</text>`;}
  const svg=document.getElementById('loss');
  svg.innerHTML=`<line x1="${pad}" y1="${H-pad}" x2="${W-pad}" y2="${H-pad}" stroke="#2a3650"/>
    ${grid}<polyline points="${pts}" fill="none" stroke="#5b8cff" stroke-width="2.5"/>
    <text x="${W-pad}" y="${H-10}" fill="#9fb0c8" font-size="11" text-anchor="end">epoch</text>`;
})();

// Explorer
const sel=document.getElementById('sel');
sel.innerHTML=DATA.compounds.map((c,i)=>`<option value="${i}">${c.name} · ${c.cell_line}</option>`).join('');
const ENS=DATA.hvg_ensembl;
function bars(arr,color){return arr.map(([i,v])=>{
  const pct=Math.min(50,Math.abs(v)/0.6*50);
  return `<div class="row"><div class="nm" title="${ENS[i]}">${ENS[i]}</div>
    <div class="track"><div class="fill" style="width:${pct}%;background:${color};${v<0?'left:0;transform:translateX(-100%)':''}"></div></div>
    <div class="val">${v>=0?'+':''}${v.toFixed(3)}</div></div>`;}).join('');}
function render(){
  const c=DATA.compounds[+sel.value];
  document.getElementById('meta').textContent=`药物：${c.name}　细胞系：${c.cell_line}　预测维度：${c.delta.length}`;
  const d=c.delta;
  const order=[...d.keys()].sort((a,b)=>d[b]-d[a]);
  const up=order.slice(0,20).map(i=>[i,d[i]]);
  const dn=order.slice(-20).map(i=>[i,d[i]]);
  document.getElementById('up').innerHTML=bars(up,'var(--up)');
  document.getElementById('dn').innerHTML=bars(dn,'var(--dn)');
}
sel.onchange=render; render();

// Heatmap (cosine)
(function(){
  const C=DATA.compounds; const n=C.length;
  const vecs=C.map(c=>c.delta);
  function cos(a,b){let s=0,na=0,nb=0;for(let i=0;i<a.length;i++){s+=a[i]*b[i];na+=a[i]*a[i];nb+=b[i]*b[i];}
    return s/(Math.sqrt(na)*Math.sqrt(nb)+1e-9);}
  const tbl=document.getElementById('hm');
  let html='<tr><td></td>'+C.map((c,i)=>`<td style="font-size:8px;color:#9fb0c8">${i+1}</td>`).join('')+'</tr>';
  for(let i=0;i<n;i++){
    let row=`<td style="font-size:8px;color:#9fb0c8">${i+1}</td>`;
    for(let j=0;j<n;j++){
      const v=cos(vecs[i],vecs[j]); const t=(v+1)/2;
      const r=Math.round(255*(1-t)+t*255), g=Math.round(93*(1-t)+211*t), b=Math.round(153*(1-t)+120*t);
      row+=`<td style="background:rgb(${r},${g},${b})" title="${C[i].name} × ${C[j].name}: ${v.toFixed(2)}"></td>`;
    }
    html+=`<tr>${row}</tr>`;
  }
  tbl.innerHTML=html;
})();

// Known mechanism groups
const known={
  'HDAC 抑制剂(表观遗传)':['Panobinostat','Belinostat','Entinostat','Mocetinostat','Abexinostat','Givinostat','Quisinostat','Tacedinaline'],
  'EGFR/HER 抑制剂':['Lapatinib','Pelitinib'],
  '多激酶抑制剂':['Sorafenib','Regorafenib'],
  '雌激素受体调节剂':['Toremifene Citrate','Fulvestrant'],
};
document.getElementById('known').innerHTML=Object.entries(known).map(([k,v])=>
  `<div class="tag"><b style="color:#e8edf6">${k}</b>: ${v.join('、')}</div>`).join('');
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", help="预测一个新药物（SMILES）")
    ap.add_argument("--cell_line", default="K562", choices=CELL_LINE_NAMES)
    ap.add_argument("--out", default="report.html")
    args = ap.parse_args()

    if args.smiles:
        predict_smiles_cli(args.smiles, args.cell_line)
        return

    print("加载阶段 B 模型...", flush=True)
    effect, hvg_n, gv = load_effect()
    print("生成 51 化合物预测数据...", flush=True)
    data = build_report_data(effect, hvg_n, gv)
    print(f"  化合物预测数: {len(data['compounds'])}  高变基因: {data['hvg_n']}", flush=True)
    write_report(data, args.out)


if __name__ == "__main__":
    main()
