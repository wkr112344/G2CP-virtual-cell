#!/usr/bin/env python3
import argparse, gzip, json
from pathlib import Path
import numpy as np
import pandas as pd
import h5py
import matplotlib.pyplot as plt

def opener(p):
    return gzip.open(p, "rt") if str(p).endswith(".gz") else open(p, "rt")

def gct_skiprows(p):
    with opener(p) as f:
        a=f.readline().strip(); b=f.readline().strip()
    return 2 if a.startswith("#1.") and "\t" in b else 0

def header(p, skip=0):
    with opener(p) as f:
        for _ in range(skip): next(f)
        return f.readline().rstrip("\n").split("\t")

def donor(s):
    x=s.split("-")
    return "-".join(x[:2]) if len(x)>=2 else s

def read_meta(p):
    d=pd.read_csv(p, sep="\t", dtype=str, low_memory=False)
    tc="SMTSD" if "SMTSD" in d.columns else "SMTS"
    z=d[["SAMPID",tc]].rename(columns={tc:"tissue"}).drop_duplicates("SAMPID")
    z["donor"]=z["SAMPID"].map(donor)
    return z

def write_str(h5,name,vals):
    dt=h5py.string_dtype("utf-8")
    h5.create_dataset(name,data=np.asarray(vals,dtype=object),dtype=dt)

def decode(x):
    return [v.decode() if isinstance(v,(bytes,np.bytes_)) else str(v) for v in x]

def inspect(rna,rsem,meta,out):
    sr=gct_skiprows(rna)
    hr=header(rna,sr); he=header(rsem,0)
    rs=hr[2:]
    es=[x for x in he if x not in ("transcript_id","gene_id")]
    md=read_meta(meta); ms=set(md.SAMPID)
    common=[s for s in rs if s in set(es) and s in ms]
    rep={
      "rnaseqc_samples":len(rs),"rsem_samples":len(es),
      "rna_rsem_intersection":len(set(rs)&set(es)),
      "three_way_intersection":len(common),
      "rna_only":len(set(rs)-set(es)),"rsem_only":len(set(es)-set(rs))
    }
    Path(out,"input_overlap.json").write_text(json.dumps(rep,indent=2))
    print(json.dumps(rep,indent=2))
    return common,md

def rnaseqc_to_h5(src,samples,dst,chunk=128):
    sk=gct_skiprows(src); hd=header(src,sk); idcols=hd[:2]
    use=idcols+samples
    with opener(src) as f:
        for _ in range(sk+1): next(f)
        n=sum(1 for _ in f)
    with h5py.File(dst,"w") as h:
        X=h.create_dataset("tpm",(n,len(samples)),dtype="float32",
                           chunks=(min(chunk,n),min(256,len(samples))),compression="lzf")
        write_str(h,"samples",samples)
        genes=[]; desc=[]; pos=0
        for k,c in enumerate(pd.read_csv(src,sep="\t",compression="gzip" if str(src).endswith(".gz") else None,
                                         skiprows=sk,usecols=use,chunksize=chunk,low_memory=False)):
            genes += c[idcols[0]].astype(str).tolist()
            desc += c[idcols[1]].astype(str).tolist()
            v=c[samples].to_numpy(np.float32,copy=False)
            X[pos:pos+len(c),:]=v; pos+=len(c)
            if k%100==0: print("RNA rows",pos,flush=True)
        write_str(h,"genes",genes); write_str(h,"descriptions",desc)

def rsem_to_h5(src,samples,dst,chunk=128):
    use=["transcript_id","gene_id"]+samples
    g2r={}; genes=[]
    with h5py.File(dst,"w") as h:
        X=h.create_dataset("tpm",(0,len(samples)),maxshape=(None,len(samples)),dtype="float32",
                           chunks=(64,min(256,len(samples))),compression="lzf")
        write_str(h,"samples",samples)
        total=0
        for k,c in enumerate(pd.read_csv(src,sep="\t",compression="gzip" if str(src).endswith(".gz") else None,
                                         usecols=use,chunksize=chunk,low_memory=False)):
            total+=len(c)
            gids=c["gene_id"].astype(str).to_numpy()
            V=c[samples].to_numpy(np.float32,copy=False)
            ug,inv=np.unique(gids,return_inverse=True)
            sums=np.zeros((len(ug),len(samples)),np.float32)
            np.add.at(sums,inv,V)
            new=[g for g in ug if g not in g2r]
            if new:
                old=len(genes); X.resize((old+len(new),len(samples)))
                X[old:old+len(new),:]=0
                for j,g in enumerate(new,old): g2r[g]=j; genes.append(g)
            for j,g in enumerate(ug):
                r=g2r[g]
                X[r,:]=X[r,:]+sums[j,:]
            if k%100==0: print("RSEM transcripts",total,"genes",len(genes),flush=True)
        write_str(h,"genes",genes)

class Corr:
    def __init__(self,n):
        self.n=np.zeros(n,np.int64); self.sx=np.zeros(n); self.sy=np.zeros(n)
        self.sxx=np.zeros(n); self.syy=np.zeros(n); self.sxy=np.zeros(n)
    def add(self,x,y):
        m=np.isfinite(x)&np.isfinite(y); a=np.where(m,x,0.); b=np.where(m,y,0.)
        self.n+=m.sum(0); self.sx+=a.sum(0); self.sy+=b.sum(0)
        self.sxx+=(a*a).sum(0); self.syy+=(b*b).sum(0); self.sxy+=(a*b).sum(0)
    def r(self):
        n=self.n.astype(float); num=self.sxy-self.sx*self.sy/n
        den=np.sqrt(np.maximum(self.sxx-self.sx*self.sx/n,0)*np.maximum(self.syy-self.sy*self.sy/n,0))
        z=np.full(len(n),np.nan); ok=(n>2)&(den>0); z[ok]=num[ok]/den[ok]; return z

def align_split(rnah,rsemh,meta,out,test_fraction,seed):
    with h5py.File(rnah,"r") as a,h5py.File(rsemh,"r") as b:
        ag=decode(a["genes"][:]); bg=decode(b["genes"][:]); sm=decode(a["samples"][:])
        if sm!=decode(b["samples"][:]): raise ValueError("sample axes differ")
    bi={g:i for i,g in enumerate(bg)}; ai={g:i for i,g in enumerate(ag)}
    cg=[g for g in ag if g in bi]
    gm=pd.DataFrame({"gene_id":cg,"rna_row":[ai[g] for g in cg],"rsem_row":[bi[g] for g in cg]})
    gm.to_csv(Path(out,"common_genes.csv"),index=False)
    m=meta[meta.SAMPID.isin(sm)].set_index("SAMPID").loc[sm].reset_index()
    ds=np.array(sorted(m.donor.unique())); rng=np.random.default_rng(seed); rng.shuffle(ds)
    nt=max(1,round(len(ds)*test_fraction)); td=set(ds[:nt])
    m["split"]=np.where(m.donor.isin(td),"test","train")
    trt=set(m.loc[m.split=="train","tissue"]); m["usable"]=m.tissue.isin(trt)
    m.to_csv(Path(out,"sample_split.csv"),index=False)
    return gm,m

def score(rnah,rsemh,gm,m,out,block=128):
    sm=m.SAMPID.tolist(); s2i={s:i for i,s in enumerate(sm)}
    tr=m[(m.split=="train")&m.usable]; te=m[(m.split=="test")&m.usable]
    tissues=sorted(tr.tissue.unique()); t2i={t:i for i,t in enumerate(tissues)}
    cols={t:np.array([s2i[s] for s in tr.loc[tr.tissue==t,"SAMPID"]],int) for t in tissues}
    tc=np.array([s2i[s] for s in te.SAMPID],int); ti=np.array([t2i[t] for t in te.tissue],int)
    A=[Corr(len(te)) for _ in range(5)]
    ar=gm.rna_row.to_numpy(int); br=gm.rsem_row.to_numpy(int)
    with h5py.File(rnah,"r") as ah,h5py.File(rsemh,"r") as bh:
        X=ah["tpm"]; Y=bh["tpm"]
        for st in range(0,len(gm),block):
            en=min(st+block,len(gm))
            xa=np.vstack([X[i,:] for i in ar[st:en]]).astype(np.float32)
            xb=np.vstack([Y[i,:] for i in br[st:en]]).astype(np.float32)
            xa=np.log2(xa+1); xb=np.log2(xb+1)
            ma=np.column_stack([xa[:,cols[t]].mean(1) for t in tissues])
            mb=np.column_stack([xb[:,cols[t]].mean(1) for t in tissues])
            pa=ma[:,ti]; pb=mb[:,ti]; ya=xa[:,tc]; yb=xb[:,tc]
            A[0].add(pa,ya); A[1].add(pa,yb); A[2].add(pb,ya); A[3].add(pb,yb); A[4].add(ya,yb)
            if st%(block*50)==0: print("score genes",st,en,len(gm),flush=True)
    z=pd.DataFrame({"sample_id":te.SAMPID.values,"donor":te.donor.values,"tissue":te.tissue.values,
                    "RR":A[0].r(),"RE":A[1].r(),"ER":A[2].r(),"EE":A[3].r(),
                    "reference_agreement":A[4].r()})
    z.to_csv(Path(out,"per_sample_scores.csv"),index=False)
    return z

def bootstrap(z,out,nboot,seed):
    rng=np.random.default_rng(seed); groups={d:g.index.to_numpy() for d,g in z.groupby("donor")}
    ds=np.array(sorted(groups)); mets=["RR","RE","ER","EE","reference_agreement"]
    def calc(idx):
        q={m:float(np.nanmean(z.loc[idx,m])) for m in mets}
        q["interaction"]=(q["RR"]-q["RE"])-(q["ER"]-q["EE"])
        q["rna_matching_advantage"]=q["RR"]-q["RE"]
        q["rsem_matching_advantage"]=q["EE"]-q["ER"]
        return q
    pt=calc(z.index.to_numpy()); rows=[]
    for b in range(nboot):
        idx=np.concatenate([groups[d] for d in rng.choice(ds,len(ds),replace=True)])
        q=calc(idx); q["bootstrap"]=b; rows.append(q)
        if b%500==0: print("bootstrap",b,nboot,flush=True)
    B=pd.DataFrame(rows); B.to_csv(Path(out,"bootstrap_scores.csv"),index=False)
    sm={}
    for k,v in pt.items():
        a=B[k].to_numpy(); sm[k]={"point":v,"ci95_low":float(np.nanpercentile(a,2.5)),
                                  "ci95_high":float(np.nanpercentile(a,97.5))}
    Path(out,"benchmark_summary.json").write_text(json.dumps(sm,indent=2))
    pd.DataFrame([{"metric":k,**v} for k,v in sm.items()]).to_csv(Path(out,"benchmark_summary.csv"),index=False)
    return sm

def figures(z,sm,out):
    fig,ax=plt.subplots(figsize=(6,4))
    ax.hist(z.reference_agreement.dropna(),bins=40)
    ax.set_xlabel("RNASeQC vs RSEM Pearson r across genes"); ax.set_ylabel("Held-out samples")
    ax.set_title("GTEx reference-product agreement"); fig.tight_layout()
    fig.savefig(Path(out,"fig_reference_agreement.png"),dpi=300); plt.close(fig)

    M=np.array([[sm["RR"]["point"],sm["RE"]["point"]],[sm["ER"]["point"],sm["EE"]["point"]]])
    fig,ax=plt.subplots(figsize=(5.5,4.5)); im=ax.imshow(M,aspect="auto")
    ax.set_xticks([0,1],["RNASeQC eval","RSEM eval"]); ax.set_yticks([0,1],["RNASeQC-trained","RSEM-trained"])
    for i in range(2):
        for j in range(2): ax.text(j,i,f"{M[i,j]:.4f}",ha="center",va="center")
    ax.set_title(f"GTEx 2×2 benchmark\ninteraction={sm['interaction']['point']:.4f}")
    fig.colorbar(im,ax=ax,label="Mean per-sample Pearson r"); fig.tight_layout()
    fig.savefig(Path(out,"fig_2x2_heatmap.png"),dpi=300); plt.close(fig)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--rnaseqc",required=True); p.add_argument("--rsem",required=True); p.add_argument("--metadata",required=True)
    p.add_argument("--outdir",required=True); p.add_argument("--test-fraction",type=float,default=.2)
    p.add_argument("--seed",type=int,default=20260929); p.add_argument("--n-bootstrap",type=int,default=5000)
    p.add_argument("--chunk-rows",type=int,default=128); p.add_argument("--gene-block",type=int,default=128)
    p.add_argument("--resume",action="store_true")
    a=p.parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True); inter=out/"intermediate"; inter.mkdir(exist_ok=True)
    common,meta=inspect(a.rnaseqc,a.rsem,a.metadata,out)
    rh=inter/"rnaseqc_gene_tpm.h5"; eh=inter/"rsem_gene_tpm.h5"
    if not (a.resume and rh.exists()): rnaseqc_to_h5(a.rnaseqc,common,rh,a.chunk_rows)
    if not (a.resume and eh.exists()): rsem_to_h5(a.rsem,common,eh,a.chunk_rows)
    gm,m=align_split(rh,eh,meta,out,a.test_fraction,a.seed)
    z=score(rh,eh,gm,m,out,a.gene_block)
    sm=bootstrap(z,out,a.n_bootstrap,a.seed+1)
    figures(z,sm,out)
    print(json.dumps(sm,indent=2))

if __name__=="__main__":
    main()
