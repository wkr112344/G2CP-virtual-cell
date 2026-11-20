import os, json, re
import numpy as np
import pandas as pd
from scipy.stats import rankdata
BASE='/mnt/data/depmap_work'; OUT='/mnt/data/depmap_background_control_results_fast'; os.makedirs(OUT,exist_ok=True)
Yc=np.load(f'{BASE}/pkg3/reference/Y_CERES.npy').astype(np.float64)
Yh=np.load(f'{BASE}/pkg4/reference/Y_Chronos.npy').astype(np.float64)
mask=np.load(f'{BASE}/pkg4/reference/common_observed_mask.npy').astype(bool)
idx=json.load(open(f'{BASE}/pkg2/index/depmap_index.json'))
train=np.array(idx['train_rows_in_cell_order']); test=np.array(idx['test_rows_in_cell_order']); test_cells=idx['test_pred_row_order']
all_cells=[x.strip() for x in open(f'{BASE}/pkg2/index/cell_order.txt')]; genes_full=[x.strip() for x in open(f'{BASE}/pkg2/index/target_genes.txt')]
genes_symbol=[re.sub(r'\s*\(\d+\)$','',x) for x in genes_full]
Yc[~mask]=np.nan; Yh[~mask]=np.nan
mu_c=np.nanmean(Yc[train],axis=0); mu_h=np.nanmean(Yh[train],axis=0)
np.save(f'{OUT}/train_gene_mean_CERES.npy',mu_c.astype('float32')); np.save(f'{OUT}/train_gene_mean_Chronos.npy',mu_h.astype('float32'))
Yct=Yc[test]; Yht=Yh[test]; Mt=np.isfinite(Yct)&np.isfinite(Yht)&mask[test]

def masked_corr_axis(A,B,M,axis):
    A2=np.where(M,A,np.nan); B2=np.where(M,B,np.nan)
    ma=np.nanmean(A2,axis=axis,keepdims=True); mb=np.nanmean(B2,axis=axis,keepdims=True)
    da=np.where(M,A-ma,0.0); db=np.where(M,B-mb,0.0)
    num=np.sum(da*db,axis=axis); den=np.sqrt(np.sum(da*da,axis=axis)*np.sum(db*db,axis=axis)); n=np.sum(M,axis=axis)
    return np.divide(num,den,out=np.full_like(num,np.nan,dtype=float),where=(den>0)&(n>=3))

def row_pearson(A,B,M): return masked_corr_axis(A,B,M,1)
def col_pearson(A,B,M): return masked_corr_axis(A,B,M,0)
def row_spearman(A,B,M):
    A2=np.where(M,A,np.nan); B2=np.where(M,B,np.nan)
    RA=rankdata(A2,axis=1,nan_policy='omit'); RB=rankdata(B2,axis=1,nan_policy='omit')
    MM=np.isfinite(RA)&np.isfinite(RB)
    return masked_corr_axis(RA,RB,MM,1)
def col_spearman(A,B,M):
    A2=np.where(M,A,np.nan); B2=np.where(M,B,np.nan)
    RA=rankdata(A2,axis=0,nan_policy='omit'); RB=rankdata(B2,axis=0,nan_policy='omit')
    MM=np.isfinite(RA)&np.isfinite(RB)
    return masked_corr_axis(RA,RB,MM,0)
def summ(x):
    x=np.asarray(x); x=x[np.isfinite(x)]
    return dict(n=int(len(x)),mean=float(np.mean(x)) if len(x) else np.nan,median=float(np.median(x)) if len(x) else np.nan,q25=float(np.quantile(x,.25)) if len(x) else np.nan,q75=float(np.quantile(x,.75)) if len(x) else np.nan,min=float(np.min(x)) if len(x) else np.nan,max=float(np.max(x)) if len(x) else np.nan)
def pred(seed,arm):
    d='pkg1' if seed<=2 else 'pkg2'; return np.load(f'{BASE}/{d}/predictions_test/seed{seed}_pred_{arm}_arm_test.npy').astype(np.float64)

# T1 baseline
base_rows=[]
for pname,P in [('CERES_train_gene_mean',np.broadcast_to(mu_c,Yct.shape)),('Chronos_train_gene_mean',np.broadcast_to(mu_h,Yct.shape))]:
  for rname,R in [('CERES',Yct),('Chronos',Yht)]:
    for metric,fn in [('pearson',row_pearson)]:
      s=summ(fn(P,R,Mt)); s.update(predictor=pname,reference=rname,metric=metric,axis='cell-wise across genes'); base_rows.append(s)
pd.DataFrame(base_rows).to_csv(f'{OUT}/task1_train_gene_mean_baseline_summary.csv',index=False)

# Raw + residual all model scores; keep gene-wise full per gene summaries compact (one row per gene per seed/quad)
cell_raw=[]; cell_res=[]; gene_raw=[]; gene_res=[]; model_summary=[]; raw_int=np.empty((5,len(test_cells))); res_int=np.empty_like(raw_int); quad_raw=[]; quad_res=[]
for s in range(5):
  Pc=pred(s,'ceres'); Ph=pred(s,'chronos'); Pcr=Pc-mu_c; Phr=Ph-mu_h
  qraw={('ceres','CERES'):(Pc,Yct),('ceres','Chronos'):(Pc,Yht),('chronos','CERES'):(Ph,Yct),('chronos','Chronos'):(Ph,Yht)}
  qres={('ceres','CERES'):(Pcr,Yct-mu_c),('ceres','Chronos'):(Pcr,Yht-mu_h),('chronos','CERES'):(Phr,Yct-mu_c),('chronos','Chronos'):(Phr,Yht-mu_h)}
  rawp={}; resp={}
  for resid,Q,cellout,geneout,qout in [(False,qraw,cell_raw,gene_raw,quad_raw),(True,qres,cell_res,gene_res,quad_res)]:
    for (arm,rname),(P,R) in Q.items():
      rp=row_pearson(P,R,Mt); rawp[(arm,rname)]=rp if not resid else rawp.get((arm,rname)); resp[(arm,rname)]=rp if resid else resp.get((arm,rname))
      for metric,vals in [('pearson',rp)]:
        for cell,v in zip(test_cells,vals): cellout.append([s,arm,rname,metric,cell,v])
        z=summ(vals); z.update(seed=s,arm=arm,reference=rname,metric=metric,axis='cell-wise across genes',residualized=resid); model_summary.append(z)
      gp=col_pearson(P,R,Mt); nobs=np.sum(Mt,axis=0)
      # compact full gene table
      for j in range(len(genes_full)): geneout.append([s,arm,rname,genes_full[j],genes_symbol[j],int(nobs[j]),gp[j]])
      for metric,vals in [('pearson',gp)]:
        z=summ(vals); z.update(seed=s,arm=arm,reference=rname,metric=metric,axis='gene-wise across cells',residualized=resid); model_summary.append(z)
      qout.append([s,arm,rname,float(np.nanmedian(rp)),float(np.nanmean(rp))])
  # interactions recompute explicitly to avoid dict assignment quirks
  cc=row_pearson(Pc,Yct,Mt); ch=row_pearson(Pc,Yht,Mt); hc=row_pearson(Ph,Yct,Mt); hh=row_pearson(Ph,Yht,Mt)
  raw_int[s]=cc-ch-hc+hh
  cc=row_pearson(Pcr,Yct-mu_c,Mt); ch=row_pearson(Pcr,Yht-mu_h,Mt); hc=row_pearson(Phr,Yct-mu_c,Mt); hh=row_pearson(Phr,Yht-mu_h,Mt)
  res_int[s]=cc-ch-hc+hh
  print('seed',s,'raw/res med',np.nanmedian(raw_int[s]),np.nanmedian(res_int[s]),flush=True)

cols=['seed','arm','reference','metric','cell','score']
pd.DataFrame(cell_raw,columns=cols).to_csv(f'{OUT}/task2_cellwise_raw_scores.csv',index=False)
pd.DataFrame(cell_res,columns=cols).to_csv(f'{OUT}/task4_cellwise_residualized_scores.csv',index=False)
gcols=['seed','arm','reference','gene','gene_symbol','n_cells','pearson']
pd.DataFrame(gene_raw,columns=gcols).to_csv(f'{OUT}/task3_genewise_raw_scores.csv',index=False)
pd.DataFrame(gene_res,columns=gcols).to_csv(f'{OUT}/task4_genewise_residualized_scores.csv',index=False)
pd.DataFrame(model_summary).to_csv(f'{OUT}/task2_4_model_score_summaries.csv',index=False)
pd.DataFrame(quad_raw,columns=['seed','arm','reference','median_cellwise_pearson','mean_cellwise_pearson']).to_csv(f'{OUT}/task5_raw_2x2_quadrants.csv',index=False)
pd.DataFrame(quad_res,columns=['seed','arm','reference','median_cellwise_pearson','mean_cellwise_pearson']).to_csv(f'{OUT}/task5_residualized_2x2_quadrants.csv',index=False)
# interaction details
rows=[]
for s in range(5):
  for i,c in enumerate(test_cells): rows.append([s,c,raw_int[s,i],res_int[s,i]])
pd.DataFrame(rows,columns=['seed','cell','raw_interaction','residualized_interaction']).to_csv(f'{OUT}/task5_interaction_per_cell_seed.csv',index=False)

# bootstrap function exact style
def boot_interaction(mat,B=10000,seed=12345):
  rng=np.random.default_rng(seed); out=np.empty(B)
  for b in range(B):
    s=int(rng.integers(0,5)); ids=rng.integers(0,mat.shape[1],mat.shape[1]); out[b]=np.nanmedian(mat[s,ids])
  return out
br=boot_interaction(raw_int); bz=boot_interaction(res_int)
rawsum=dict(mean=float(np.mean(br)),median=float(np.median(br)),ci95_low=float(np.quantile(br,.025)),ci95_high=float(np.quantile(br,.975)),per_seed=[float(np.nanmedian(raw_int[s])) for s in range(5)])
ressum=dict(mean=float(np.mean(bz)),median=float(np.median(bz)),ci95_low=float(np.quantile(bz,.025)),ci95_high=float(np.quantile(bz,.975)),per_seed=[float(np.nanmedian(res_int[s])) for s in range(5)],ci_excludes_zero=bool(np.quantile(bz,.025)>0))
json.dump(rawsum,open(f'{OUT}/task6_raw_interaction_bootstrap.json','w'),indent=2); json.dump(ressum,open(f'{OUT}/task6_residualized_interaction_bootstrap.json','w'),indent=2)
pd.DataFrame({'raw':br,'residualized':bz}).to_csv(f'{OUT}/task6_interaction_bootstrap_distributions.csv',index=False)

# DeepDEP 43x1244 strict
D=pd.read_csv(f'{BASE}/pkg2/first_predictor_DeepDEP/DeepDEP_predictor_strict.csv').set_index('Unnamed: 0')
common_cells=[c for c in test_cells if c in D.index]; gs=set(genes_symbol); common_genes=[g for g in D.columns if g in gs]
gmap={g:i for i,g in enumerate(genes_symbol)}; rmap={c:i for i,c in enumerate(all_cells)}
gi=np.array([gmap[g] for g in common_genes]); ri=np.array([rmap[c] for c in common_cells])
Pd=D.loc[common_cells,common_genes].to_numpy(float); Rc=Yc[ri][:,gi]; Rh=Yh[ri][:,gi]; Md=np.isfinite(Pd)&np.isfinite(Rc)&np.isfinite(Rh)&mask[ri][:,gi]
mc=mu_c[gi]; mh=mu_h[gi]
drec=[]; ddetail={'cell':common_cells}
for analysis,P,RR in [('raw',Pd,{'CERES':Rc,'Chronos':Rh}),('residualized_pred_minus_CERES_train_mean',Pd-mc,{'CERES':Rc-mc,'Chronos':Rh-mh})]:
  for rname,R in RR.items():
    for metric,fn in [('pearson',row_pearson)]:
      vals=fn(P,R,Md); z=summ(vals); z.update(analysis=analysis,reference=rname,metric=metric,axis='cell-wise across genes',n_cells=len(common_cells),n_genes=len(common_genes)); drec.append(z); ddetail[f'{analysis}_{rname}_{metric}']=vals
    gp=col_pearson(P,R,Md)
    for metric,vals in [('pearson',gp)]:
      z=summ(vals); z.update(analysis=analysis,reference=rname,metric=metric,axis='gene-wise across cells',n_cells=len(common_cells),n_genes=len(common_genes)); drec.append(z)
pd.DataFrame(drec).to_csv(f'{OUT}/task7_DeepDEP_43x1244_summary.csv',index=False); pd.DataFrame(ddetail).to_csv(f'{OUT}/task7_DeepDEP_43x1244_cellwise_detail.csv',index=False)
json.dump({'n_cells':len(common_cells),'n_genes':len(common_genes),'cells':common_cells,'residualization_note':'DeepDEP diagnostic residualization subtracts CERES train-only gene mean from fixed prediction; raw fixed-output comparison is primary.'},open(f'{OUT}/task7_DeepDEP_grid.json','w'),indent=2)

# overall
base_df=pd.DataFrame(base_rows); ms=pd.DataFrame(model_summary)
overall={'grid':{'total_cells':908,'train':636,'val':136,'test':136,'genes':17393},'baseline':base_rows,'raw_interaction':rawsum,'residualized_interaction':ressum,'retention_ratio':float(ressum['mean']/rawsum['mean']),'absolute_change':float(ressum['mean']-rawsum['mean']),'DeepDEP_grid':{'cells':len(common_cells),'genes':len(common_genes)}}
json.dump(overall,open(f'{OUT}/OVERALL_RESULTS.json','w'),indent=2)
with open(f'{OUT}/REPORT.md','w',encoding='utf-8') as f:
 f.write('# DepMap common gene-background control (tasks 1–7)\n\n')
 f.write(f"Raw interaction: {rawsum['mean']:.6f} (95% CI {rawsum['ci95_low']:.6f}–{rawsum['ci95_high']:.6f}).\n\n")
 f.write(f"Residualized interaction: {ressum['mean']:.6f} (95% CI {ressum['ci95_low']:.6f}–{ressum['ci95_high']:.6f}).\n\n")
 f.write(f"Retention: {100*overall['retention_ratio']:.1f}%.\n")
print(json.dumps(overall,indent=2))
