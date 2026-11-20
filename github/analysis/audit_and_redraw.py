"""Audit supplied score tables; rebuild statistics and publication figures.
No raw predictions, experimental measurements, or reference scores are altered.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
R=Path(__file__).resolve().parents[1];S=R/'source_data'; O=R/'supplementary';F=R/'figures'; A=R/'author_notes'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'lines.linewidth':1.15,'savefig.dpi':600})
BLUE='#216A94'; ORANGE='#C65C27'; GRAY='#687681'; GREEN='#337F66';W=170/25.4
report={}; plotdata=[]
def save(n,fig):
 fig.savefig(F/(n+'.pdf'));fig.savefig(F/(n+'.png'),dpi=600);plt.close(fig)
def panel(ax,l,title):ax.set_title(l+'  '+title,loc='left',fontweight='bold',pad=10)
def write(d,name):d.to_csv(O/name,index=False)
def rec(fig,panel,series,val,**kw):plotdata.append(dict(figure=fig,panel=panel,series=series,value=float(val),**kw))
# LINCS aggregation and source checks.
d=pd.read_csv(S/'frozen_prediction_matrices/frozen_perrow_three_metrics.csv')
assert not d.duplicated(['cell','drug']).any()
met=pd.read_csv(O/'S1_weighting_and_metrics.csv');c=pd.read_csv(S/'C_perdrug_stability.csv');ext=pd.read_csv(S/'B_sevenmodel.csv');dep=pd.read_csv(S/'L_heterogeneity_272cell.csv')
for m in ['pearson','spearman','cosine']:
 d[m+'_delta']=d[m+'_A_beta2020']-d[m+'_B_dcic2021']
 for unit in ['drug','cell']:
  q=d.groupby(unit)[[m+'_A_beta2020',m+'_B_dcic2021',m+'_delta']].mean().mean()
  old=met[(met.metric==m)&(met.weighting==unit)].iloc[0]
  assert np.allclose(q,[old.beta_mean,old.dcic_mean,old.delta],atol=1e-12)
rank=pd.read_csv(O/'S3_topk_performance_ranking.csv')
for row in rank.itertuples():
 aa=set(c.nsmallest(row.k,'rank_beta').drug_id);bb=set(c.nsmallest(row.k,'rank_dcic').drug_id)
 assert len(aa&bb)==row.overlap
report['lincs']={'rows':len(d),'cells':d.cell.nunique(),'drugs':d.drug.nunique(),'duplicate_pairs':0,'rank_rho':spearmanr(c.pcc_beta,c.pcc_dcic).statistic}
# DepMap: verify cell scores -> quadrants and paired interaction; common cell IDs across all seeds.
D=S/'depmap_background_control';dr=pd.read_csv(D/'task2_cellwise_raw_scores.csv');dz=pd.read_csv(D/'task4_cellwise_residualized_scores.csv');di=pd.read_csv(D/'task5_interaction_per_cell_seed.csv');base=pd.read_csv(D/'task1_train_gene_mean_baseline_summary.csv');base=base[base.metric=='pearson'];paired=pd.read_csv(D/'task2_model_vs_trainmean_baseline_paired.csv')
quad=[]
for label,df in [('raw',dr),('residualized',dz)]:
 p=df.pivot(index=['seed','cell'],columns=['arm','reference'],values='score').sort_index()
 calc=p[('ceres','CERES')]-p[('ceres','Chronos')]-p[('chronos','CERES')]+p[('chronos','Chronos')]
 target=di.set_index(['seed','cell'])[label+'_interaction'].sort_index()
 assert np.allclose(calc,target,atol=1e-12)
 for (seed,arm,ref),v in df.groupby(['seed','arm','reference']):quad.append(dict(analysis=label,seed=seed,arm=arm,reference=ref,median=v.score.median(),mean=v.score.mean()))
q=pd.DataFrame(quad);write(q,'S5_depmap_raw_residualized_2x2.csv')
raw=di.pivot(index='seed',columns='cell',values='raw_interaction').sort_index(axis=1).to_numpy();res=di.pivot(index='seed',columns='cell',values='residualized_interaction').sort_index(axis=1).to_numpy();assert raw.shape==res.shape==(5,136)
rng=np.random.default_rng(20260930);boots=np.empty((10000,2))
for i in range(len(boots)):
 si=rng.integers(0,5,5);ci=rng.integers(0,136,136)
 for j,mat in enumerate([raw,res]):boots[i,j]=np.median(mat[np.ix_(si,ci)],axis=1).mean()
depstats=[]
for j,(label,mat) in enumerate([('raw',raw),('residualized',res)]):
 z=dict(analysis=label,estimate=np.median(mat,axis=1).mean(),ci_low=np.quantile(boots[:,j],.025),ci_high=np.quantile(boots[:,j],.975),seed_min=np.median(mat,axis=1).min(),seed_max=np.median(mat,axis=1).max());depstats.append(z)
write(pd.DataFrame(depstats),'S6_depmap_interaction_bootstrap_summary.csv');write(pd.DataFrame(boots,columns=['raw','residualized']),'S9_depmap_crossed_bootstrap.csv');report['depmap_interaction']=depstats
# Verify gene-wise residual invariance, a mathematical property, not independent evidence.
graw=pd.read_csv(D/'task3_genewise_raw_scores.csv');gres=pd.read_csv(D/'task4_genewise_residualized_scores.csv');assert graw.iloc[:,:-1].equals(gres.iloc[:,:-1]);err=float(np.nanmax(np.abs(graw.pearson-gres.pearson)))
report['gene_wise_residual_max_abs_diff']=err
report['gene_wise_median_range']=[graw.groupby(['seed','arm','reference']).pearson.median().min(),graw.groupby(['seed','arm','reference']).pearson.median().max()]
# GTEx: donor-median estimand recovered from supplied individual score table.
G=S/'gtex_reference_sensitivity';g=pd.read_csv(G/'per_sample_scores.csv');split=pd.read_csv(G/'sample_split.csv');assert not g['sample'].duplicated().any();assert set(g['sample'])==set(split.loc[split.split=='test','sample']);assert not set(split.loc[split.split=='test','donor'])&set(split.loc[split.split=='train','donor']);assert not g.isna().any().any()
for pref,col in [('r','interaction_pearson'),('sp_r','interaction_spearman')]:assert np.allclose(g[pref+'RR']-g[pref+'RE']-g[pref+'ER']+g[pref+'EE'],g[col],atol=1e-12)
g['adv_R']=g.rRR-g.rRE;g['adv_E']=g.rEE-g.rER
donor=g.groupby('donor')[['interaction_pearson','interaction_spearman','adv_R','adv_E']].median();rng=np.random.default_rng(20260930);ids=rng.integers(0,len(donor),(5000,len(donor)));gb=np.median(donor.to_numpy()[ids],axis=1)
gstats=[]
for j,col in enumerate(donor):gstats.append(dict(statistic=col,estimate=donor[col].median(),ci_low=np.quantile(gb[:,j],.025),ci_high=np.quantile(gb[:,j],.975)))
write(pd.DataFrame(gstats),'S8_gtex_reference_product_2x2.csv');write(donor.reset_index(),'S10_gtex_donor_medians.csv');write(pd.DataFrame(gb,columns=donor.columns),'S11_gtex_donor_bootstrap.csv')
tissue=g.groupby('tissue').agg(n_samples=('sample','size'),n_donors=('donor','nunique'),interaction_pearson=('interaction_pearson','median'),interaction_spearman=('interaction_spearman','median'),adv_R=('adv_R','median'),adv_E=('adv_E','median')).reset_index();write(tissue,'S12_gtex_tissue_sensitivity.csv')
report['gtex']={'samples':len(g),'donors':g.donor.nunique(),'tissues':g.tissue.nunique(),'statistics':gstats,'pearson_quadrant_medians':g[['rRR','rRE','rER','rEE']].median().to_dict(),'spearman_quadrant_medians':g[['sp_rRR','sp_rRE','sp_rER','sp_rEE']].median().to_dict(),'sample_median_interaction':g.interaction_pearson.median(),'difference_of_medians':g.rRR.median()-g.rRE.median()-g.rER.median()+g.rEE.median(),'sample_adv_E_negative':int((g.adv_E<0).sum()),'tissue_positive':int((tissue.interaction_pearson>0).sum()),'tissue_range':[tissue.interaction_pearson.min(),tissue.interaction_pearson.max()],'low_coverage_tissues':int((tissue.n_donors<10).sum())}
# Source data for all figure summaries.
# Fig 1: comparison matrix, separate independent experimental branches.
fig=plt.figure(figsize=(W,3.45));ax=fig.add_axes([.01,.02,.98,.96]);ax.axis('off')
cols=[('LINCS L1000',BLUE,['beta2020 / dcic2021','11,275 pairs; 2,037 drugs\n63 cells; 978 genes','Frozen predictions\nTwo fixed training states','Scores; model ordering\nPerformance / disease lists']),('DepMap 21Q2',ORANGE,['CERES / Chronos','272 cells × 1,244 genes\nFrozen DeepDEP','636 train / 136 test cells\n17,393 genes; 5 seeds','Train-only gene baseline\nRaw / residual / gene-wise']),('GTEx v11',GREEN,['RNASeQC / RSEM','19,788 samples; 946 donors\n74,628 shared genes','757 train / 189 test donors\nTissue-mean predictors','Matched-reference scores\nDonor / tissue summaries'])]
for k,(title,color,lines) in enumerate(cols):
 x=.17+k*.33;ax.text(x,.93,title,ha='center',weight='bold',fontsize=10,color=color)
 for j,line in enumerate(lines):
  y=.77-j*.20;ax.text(x,y,line,ha='center',va='center',fontsize=8,bbox=dict(boxstyle='round,pad=.65',facecolor='#F4F7F9',edgecolor=color,lw=.8))
  if j<3 and not (k==1 and j==1):ax.annotate('',(x,y-.11),(x,y-.055),arrowprops=dict(arrowstyle='->',color=GRAY,lw=.8))
save('Figure1_design',fig)
# Fig 2: ordered by identity, not a cross-model leaderboard.
fig,axs=plt.subplots(1,2,figsize=(W,2.8));fig.subplots_adjust(left=.12,right=.98,bottom=.20,top=.84,wspace=.61)
for j,m in enumerate(['pearson','spearman','cosine']):
 rr=met[(met.metric==m)&(met.weighting=='drug')].iloc[0];axs[0].errorbar(rr.delta,j,xerr=[[rr.delta-rr.ci_low],[rr.ci_high-rr.delta]],fmt='o',color=BLUE,capsize=3);rec(2,'A',m,rr.delta,ci_low=rr.ci_low,ci_high=rr.ci_high)
axs[0].set(yticks=range(3),yticklabels=['Pearson','Spearman','Cosine'],xlabel='Drug-weighted score difference',xlim=(.25,.31),ylim=(-.6,2.6));panel(axs[0],'A','Frozen output')
names={'ciger':'CIGER','deepce':'DeepCE','multidcp':'MultiDCP','pertd it':'PertDiT','pertd it':'PertDiT','pertdit':'PertDiT','prnet':'PRnet','transigen':'TranSiGen','xpert':'XPert'}
e=ext.sort_values('model',ascending=False)
axs[1].errorbar(e.delta,range(len(e)),xerr=[e.delta-e.ci_low,e.ci_high-e.delta],fmt='o',color=BLUE,capsize=3);axs[1].set(yticks=range(7),yticklabels=[names.get(x,x) for x in e.model],xlabel='Within-model score difference',xlim=(0,.11));panel(axs[1],'B','Prediction sources')
for row in e.itertuples():rec(2,'B',row.model,row.delta,ci_low=row.ci_low,ci_high=row.ci_high)
save('Figure2_fixed_output',fig)
# Fig 3: LINCS scores and change in ordering; DepMap interaction shown only with its controls in Fig 6.
fig,axs=plt.subplots(1,2,figsize=(W,2.8));fig.subplots_adjust(left=.10,right=.98,bottom=.21,top=.84,wspace=.48)
vals=np.array([[.3679865,.07238622],[.1705,.1534]])
for v,col,mark,lab in zip(vals,[BLUE,ORANGE],['o','s'],['beta-trained','dcic-trained']):axs[0].plot([0,1],v,marker=mark,color=col,label=lab)
axs[0].set(xticks=[0,1],xticklabels=['beta2020','dcic2021'],ylabel='Drug-weighted Pearson',ylim=(0,.42));axs[0].legend(frameon=False,loc='upper right');panel(axs[0],'A','Fixed training states')
v=vals[0]-vals[1];axs[1].axvline(0,color=GRAY,lw=.8)
for j,(x,col) in enumerate(zip(v,[BLUE,ORANGE])):axs[1].plot(x,1-j,'o',color=col);axs[1].text(x,1-j+.14,f'{x:+.4f}',ha='center',fontsize=8);rec(3,'B',['beta2020','dcic2021'][j],x)
axs[1].set(yticks=[0,1],yticklabels=['dcic2021','beta2020'],xlabel='beta-trained − dcic-trained',xlim=(-.14,.26),ylim=(-.55,1.55));panel(axs[1],'B','Model difference')
save('Figure3_training_evaluation',fig)
# Fig 4: avoid mixing rank association and retention on a shared axis.
fig,axs=plt.subplots(1,3,figsize=(W,2.75),gridspec_kw={'width_ratios':[1.3,1,1]});fig.subplots_adjust(left=.09,right=.98,bottom=.22,top=.80,wspace=.59)
axs[0].plot(rank.k,rank.retention,'o-',color=BLUE,ms=3);axs[0].plot(rank.k,rank.chance_expected_retention,'--',color=GRAY);axs[0].set(xscale='log',xlabel='List size, k',ylabel='Top-k retention',ylim=(0,1));axs[0].text(.03,.95,'Solid: observed\nDashed: random lists',transform=axs[0].transAxes,va='top',fontsize=7);panel(axs[0],'A','Drug lists')
vv=pd.read_csv(S/'J_disease_panels.csv').iloc[1:4];labels=['Liver','Thyroid','APL']
for j,(r,ret) in enumerate(zip(vv.spearman,[.1,.3,.2])):
 axs[1].plot(r,2-j,'o',color=BLUE);axs[1].text(r+.065,2-j,f'{r:.3f}',va='center',fontsize=7);axs[2].plot(ret,2-j,'s',color=ORANGE);axs[2].text(ret+.08,2-j,f'{int(ret*10)}/10',va='center',fontsize=7)
 rec(4,'B',labels[j],r);rec(4,'C',labels[j],ret)
axs[1].set(yticks=[2,1,0],yticklabels=labels,xlabel='Rank Spearman',xlim=(0,.77),ylim=(-.6,2.6));axs[2].set(yticks=[2,1,0],yticklabels=labels,xlabel='Top-10 retention',xlim=(0,1),ylim=(-.6,2.6));panel(axs[1],'B','Disease ranks');panel(axs[2],'C','Disease lists');save('Figure4_selection',fig)
# Fig 5: heterogeneity
fig,axs=plt.subplots(1,2,figsize=(W,2.8));fig.subplots_adjust(left=.10,right=.98,bottom=.21,top=.84,wspace=.4)
cell=d.groupby('cell').agg(delta=('pearson_delta','mean'),n=('pearson_delta','size'));axs[0].scatter(cell.n,cell.delta,c=BLUE,s=13,alpha=.8);axs[0].axhline(cell.delta.mean(),ls='--',c=ORANGE,lw=1);axs[0].set(xscale='log',xlabel='Responses per cell line',ylabel='Mean Pearson difference',ylim=(0,.68));panel(axs[0],'A','LINCS coverage')
axs[1].scatter(dep.product_concordance,dep.delta_pearson,c=BLUE,s=9,alpha=.55);axs[1].axhline(0,c=GRAY,lw=.8);axs[1].set(xlabel='CERES–Chronos Pearson',ylabel='DeepDEP Pearson difference');panel(axs[1],'B','DepMap: 272 cells');save('Figure5_heterogeneity',fig)
# Fig 6: explicit small differences; seed points, no falsely identical bars.
fig,axs=plt.subplots(2,2,figsize=(W,5.1));fig.subplots_adjust(left=.12,right=.98,bottom=.11,top=.92,wspace=.48,hspace=.69)
keys=[('ceres','CERES'),('ceres','Chronos'),('chronos','CERES'),('chronos','Chronos')];labs=['C → C','C → H','H → C','H → H']
for j,((arm,ref),lab) in enumerate(zip(keys,labs)):
 y=3-j;bv=float(base[(base.predictor.str.lower().str.startswith(arm))&(base.reference==ref)]['median'].iloc[0]);mv=q[(q.analysis=='raw')&(q.arm==arm)&(q.reference==ref)]['median'].to_numpy();axs[0,0].plot([mv.mean(),bv],[y,y],c=GRAY,lw=1.3);axs[0,0].plot(bv,y,'D',color=ORANGE,ms=4,label='Train-mean baseline' if j==0 else None);axs[0,0].plot(mv,y+np.linspace(-.08,.08,5),'o',color=BLUE,ms=2.7,label='Model: five seeds' if j==0 else None);axs[0,0].text(.946,y,f'{bv:.4f}',fontsize=7,va='center')
 pv=paired[(paired.arm==arm)&(paired.reference==ref)].paired_median_delta_model_minus_baseline.to_numpy();axs[0,1].plot(pv*1000,y+np.linspace(-.1,.1,5),'o',color=BLUE,ms=3)
 rv=q[(q.analysis=='residualized')&(q.arm==arm)&(q.reference==ref)]['median'].to_numpy();axs[1,0].plot(rv,y+np.linspace(-.08,.08,5),'o',color=GREEN,ms=3)
 rec(6,'A',lab+' baseline',bv);rec(6,'A',lab+' model mean',mv.mean());rec(6,'C',lab+' residual mean',rv.mean())
axs[0,0].set(yticks=range(4),yticklabels=labs[::-1],xlim=(.898,.957),ylim=(-.6,4.15),xlabel='Median cell-wise Pearson');axs[0,0].legend(frameon=False,loc='upper left',fontsize=7.2,handletextpad=.35);panel(axs[0,0],'A','Models and baseline')
axs[0,1].axvline(0,c=GRAY,lw=.8);axs[0,1].set(yticks=range(4),yticklabels=labs[::-1],xlabel='Paired model − baseline (×10⁻³)',xlim=(-1.25,.12),ylim=(-.6,3.6));panel(axs[0,1],'B','Small paired differences')
axs[1,0].axvline(0,c=GRAY,lw=.8);axs[1,0].set(yticks=range(4),yticklabels=labs[::-1],xlabel='Median residual Pearson',xlim=(-.008,.004),ylim=(-.6,3.6));axs[1,0].tick_params(axis='x',labelsize=7);panel(axs[1,0],'C','After background removal')
for j,(st,col,mat) in enumerate(zip(depstats,[BLUE,GREEN],[raw,res])):
 y=1-j;axs[1,1].errorbar(st['estimate'],y,xerr=[[st['estimate']-st['ci_low']],[st['ci_high']-st['estimate']]],fmt='D',color=col,capsize=4,ms=4);axs[1,1].plot(np.median(mat,axis=1),np.full(5,y-.15),'o',color=col,ms=2.5);rec(6,'D',st['analysis'],st['estimate'],ci_low=st['ci_low'],ci_high=st['ci_high'])
axs[1,1].axvline(0,c=GRAY,lw=.8);axs[1,1].set(yticks=[1,0],yticklabels=['Raw','Residual'],xlabel='Training × evaluation interaction',xlim=(-.013,.068),ylim=(-.55,1.5));panel(axs[1,1],'D','Interaction with 95% CI');save('Figure6_depmap_gene_background_control',fig)
# Fig 7: all input values obtained from actual supplied rRR fields, not stale plotting schema.
fig,axs=plt.subplots(2,2,figsize=(W,5.2));fig.subplots_adjust(left=.12,right=.96,bottom=.11,top=.93,wspace=.52,hspace=.73)
mat=g[['rRR','rRE','rER','rEE']].median().to_numpy().reshape(2,2);im=axs[0,0].imshow(mat,vmin=.94,vmax=.99,cmap='Blues',aspect='auto')
for i in range(2):
 for j in range(2):axs[0,0].text(j,i,f'{mat[i,j]:.4f}',ha='center',va='center',color='white' if mat[i,j]>.975 else '#18242E',fontsize=10);rec(7,'A',str(i)+str(j),mat[i,j])
axs[0,0].set(xticks=[0,1],xticklabels=['RNASeQC','RSEM'],yticks=[0,1],yticklabels=['RNASeQC','RSEM'],xlabel='Evaluation product',ylabel='Tissue-mean source');panel(axs[0,0],'A','Median sample Pearson');cb=fig.colorbar(im,ax=axs[0,0],fraction=.047,pad=.05);cb.ax.tick_params(labelsize=7.2)
for j,col in enumerate(['adv_R','adv_E']):
 yy=donor[col].to_numpy();jitter=np.linspace(-.14,.14,len(yy));np.random.default_rng(1).shuffle(jitter);axs[0,1].scatter(j+jitter,yy,s=4,c=[BLUE,ORANGE][j],alpha=.45);axs[0,1].plot([j-.17,j+.17],[np.median(yy)]*2,color='#172A38',lw=1.5)
axs[0,1].axhline(0,color=GRAY,lw=.8);axs[0,1].set(xticks=[0,1],xticklabels=['RNASeQC','RSEM'],ylabel='Matched-reference advantage',xlabel='Tissue-mean source',ylim=(-.003,.056));panel(axs[0,1],'B','189 donor medians')
for j,st in enumerate(gstats[:2]):
 y=1-j;axs[1,0].errorbar(st['estimate'],y,xerr=[[st['estimate']-st['ci_low']],[st['ci_high']-st['estimate']]],fmt='o',color=[BLUE,ORANGE][j],capsize=4);axs[1,0].text(st['estimate'],y+.22,f"{st['estimate']:.5f}",ha='center',fontsize=8);rec(7,'C',st['statistic'],st['estimate'],ci_low=st['ci_low'],ci_high=st['ci_high'])
axs[1,0].set(yticks=[1,0],yticklabels=['Pearson','Spearman'],xlabel='Donor-median interaction',xlim=(.057,.076),ylim=(-.55,1.6));panel(axs[1,0],'C','Interaction with 95% CI')
hi=tissue.n_donors>=10
axs[1,1].scatter(tissue.loc[hi,'n_donors'],tissue.loc[hi,'interaction_pearson'],s=13,color=GREEN,alpha=.8,label='≥10 donors');axs[1,1].scatter(tissue.loc[~hi,'n_donors'],tissue.loc[~hi,'interaction_pearson'],s=15,facecolor='white',edgecolor=GRAY,label='<10 donors');axs[1,1].axhline(0,c=GRAY,lw=.8);axs[1,1].set(xscale='log',xlabel='Test donors per tissue',ylabel='Median sample interaction',ylim=(-.004,.118));axs[1,1].legend(frameon=False,loc='lower left',fontsize=7);panel(axs[1,1],'D','68 tissue labels');save('Figure7_gtex_reference_matching',fig)
write(pd.DataFrame(plotdata),'Figure_summary_values.csv')
(A/'numerical_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
(R/'supplementary'/'analysis_summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False))
print(json.dumps(report,indent=2,ensure_ascii=False))
