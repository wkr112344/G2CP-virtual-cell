from pathlib import Path
import numpy as np, pandas as pd, json
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]; SRC=ROOT/'source_data'; OUT=ROOT/'supplementary'; FIG=ROOT/'figures'
rng=np.random.default_rng(20260929); B=5000
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'savefig.dpi':220})
blue='#26678D'; orange='#C56B35'; grey='#6B7785'
def ci(x):
 x=np.asarray(x); bs=np.array([rng.choice(x,len(x),replace=True).mean() for _ in range(B)])
 return [float(x.mean()),*map(float,np.quantile(bs,[.025,.975]))]
def savefig(n):
 plt.savefig(FIG/f'{n}.png',bbox_inches='tight');plt.savefig(FIG/f'{n}.pdf',bbox_inches='tight');plt.close()
d=pd.read_csv(SRC/'frozen_prediction_matrices/frozen_perrow_three_metrics.csv'); c=pd.read_csv(SRC/'C_perdrug_stability.csv'); ext=pd.read_csv(SRC/'B_sevenmodel.csv'); dep=pd.read_csv(SRC/'L_heterogeneity_272cell.csv'); seeds=pd.read_csv(SRC/'G_depmap_2x2_5seed.csv'); boot=pd.read_csv(SRC/'H_depmap_interaction_bootstrap5000.csv'); disease=pd.read_csv(SRC/'J_disease_panels.csv'); neg=pd.read_csv(SRC/'M_negative_controls.csv')
res={'rows':len(d),'drugs':d.drug.nunique(),'cells':d.cell.nunique()}; metrics=[]; groups=[]
for m in ['pearson','spearman','cosine']:
 a,b=f'{m}_A_beta2020',f'{m}_B_dcic2021'; d[m+'_delta']=d[a]-d[b]
 for axis in ['drug','cell']:
  v=d.groupby(axis)[[a,b,m+'_delta']].mean(); v['n_rows']=d.groupby(axis).size();v.reset_index().to_csv(OUT/f'S_{m}_by_{axis}.csv',index=False)
  est,lo,hi=ci(v[m+'_delta']);metrics.append({'metric':m,'weighting':axis,'n_units':len(v),'beta_mean':v[a].mean(),'dcic_mean':v[b].mean(),'delta':est,'ci_low':lo,'ci_high':hi,'n_positive':int((v[m+'_delta']>0).sum()),'min_delta':v[m+'_delta'].min(),'max_delta':v[m+'_delta'].max()})
 metrics.append({'metric':m,'weighting':'row','n_units':len(d),'beta_mean':d[a].mean(),'dcic_mean':d[b].mean(),'delta':d[m+'_delta'].mean()})
met=pd.DataFrame(metrics);met.to_csv(OUT/'S1_weighting_and_metrics.csv',index=False)
loo=[]
for cell in sorted(d.cell.unique()):
 keep=d[d.cell!=cell]; v=keep.groupby('drug').pearson_delta.mean();loo.append({'omitted_cell':cell,'remaining_rows':len(keep),'remaining_drugs':len(v),'drug_weighted_delta':v.mean()})
loo=pd.DataFrame(loo);loo.to_csv(OUT/'S2_leave_one_cell_out.csv',index=False);res['loo_range']=[loo.drug_weighted_delta.min(),loo.drug_weighted_delta.max()]
rank=[]
for k in [10,20,50,100,204,500,1000]:
 aa=set(c.nsmallest(k,'rank_beta').drug_id);bb=set(c.nsmallest(k,'rank_dcic').drug_id);inter=len(aa&bb)
 rank.append({'k':k,'overlap':inter,'retention':inter/k,'jaccard':inter/(2*k-inter),'chance_expected_overlap':k*k/len(c),'chance_expected_retention':k/len(c)})
rank=pd.DataFrame(rank);rank.to_csv(OUT/'S3_topk_performance_ranking.csv',index=False)
res['rank_spearman']=float(spearmanr(c.pcc_beta,c.pcc_dcic).statistic);res['rank_abs_shift_median']=float((c.rank_beta-c.rank_dcic).abs().median());res['rank_abs_shift_mean']=float((c.rank_beta-c.rank_dcic).abs().mean())
res['depmap_delta']=ci(dep.delta_pearson);res['depmap_positive']=int((dep.delta_pearson>0).sum());res['depmap_range']=[dep.delta_pearson.min(),dep.delta_pearson.max()];res['depmap_concordance']=dep.product_concordance.mean();res['depmap_rho']=float(spearmanr(dep.delta_pearson,dep.product_concordance).statistic)
res['seed_interaction_mean']=seeds.interaction.mean();res['seed_interaction_range']=[seeds.interaction.min(),seeds.interaction.max()];res['bootstrap_median']=boot.interaction.median();res['bootstrap_mean']=boot.interaction.mean();res['bootstrap_ci']=boot.interaction.quantile([.025,.975]).tolist();res['seed_cell_means']=seeds[['CC','CH','HC','HH']].mean().to_dict()
# Existing cell-level diagnostic table is descriptive only: provenance of the mean-profile construction is not available.
n=pd.read_csv(SRC/'N_depmap_reference_structure_controls_272cell.csv')
res['reference_structure_diagnostic']={'mean_profile_ceres_own':n.ceres_mean_pred_vs_ceres.mean(),'mean_profile_ceres_cross':n.ceres_mean_pred_vs_chronos.mean(),'mean_profile_chronos_own':n.chronos_mean_pred_vs_chronos.mean(),'mean_profile_chronos_cross':n.chronos_mean_pred_vs_ceres.mean(),'delta_association':float(spearmanr(n.deepdep_delta_pearson,n.ceres_mean_pred_delta_ceres_minus_chronos).statistic),'status':'exploratory existing-table diagnostic; not an out-of-sample baseline'}
res['positive_drug_fraction']=float((c.delta>0).mean());res['positive_cell_fraction']=float((d.groupby('cell').pearson_delta.mean()>0).mean());res['threshold_drug_fractions']={str(t):float((c.delta>t).mean()) for t in [.01,.05,.1,.2]}
(OUT/'analysis_summary.json').write_text(json.dumps(res,indent=2))
# Figure 1: explicit comparison dimensions (schematic, no estimated quantities).
fig,axs=plt.subplots(1,2,figsize=(10,4.4));
for ax,title,lines in [(axs[0],'LINCS L1000',['beta2020  /  dcic2021','11,275 pairs | 2,037 drugs | 63 observed cells','978 landmark genes','Frozen output + two training states','Scores -> model contrast -> top-k lists']),(axs[1],'DepMap 21Q2',['CERES  /  Chronos','272 cells x 1,244 genes: frozen DeepDEP','136 test cells: controlled training comparison','One fixed split | five training seeds','Scores -> top-k recovery + interaction'])]:
 ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off');ax.set_title(title,weight='bold',pad=12)
 for j,line in enumerate(lines):
  y=.88-j*.18; ax.text(.5,y,line,ha='center',va='center',fontsize=10,bbox=dict(boxstyle='round,pad=.5',fc='#F0F4F6',ec='#9CAEB9'))
  if j<4:ax.annotate('',xy=(.5,y-.125),xytext=(.5,y-.05),arrowprops=dict(arrowstyle='->',color=grey))
fig.tight_layout();savefig('Figure1_design')
fig,axs=plt.subplots(1,2,figsize=(10,4));
for j,m in enumerate(['pearson','spearman','cosine']):
 rr=met[(met.metric==m)&(met.weighting=='drug')].iloc[0];axs[0].errorbar(rr.delta,j,xerr=[[rr.delta-rr.ci_low],[rr.ci_high-rr.delta]],fmt='o',color=blue,capsize=4)
axs[0].set(yticks=range(3),yticklabels=['Pearson','Spearman','Cosine'],xlabel='Drug-weighted reference contrast',xlim=(0,.34));axs[0].set_title('A  Same frozen prediction',loc='left',weight='bold');axs[0].axvline(0,c=grey,lw=.7)
e=ext.sort_values('delta');axs[1].errorbar(e.delta,range(7),xerr=[e.delta-e.ci_low,e.ci_high-e.delta],fmt='o',capsize=3,color=blue);axs[1].set(yticks=range(7),yticklabels=e.model,xlabel='Within-model PCC contrast',xlim=(0,.11));axs[1].set_title('B  Seven prediction sources',loc='left',weight='bold');fig.tight_layout();savefig('Figure2_fixed_output')
fig,axs=plt.subplots(1,2,figsize=(10,4)); x=[0,1]
axs[0].plot(x,[.3679865,.07238622],'o-',c=blue,label='beta-trained');axs[0].plot(x,[.1705,.1534],'s-',c=orange,label='dcic-trained');axs[0].set(xticks=x,xticklabels=['beta2020','dcic2021'],ylabel='Drug-weighted PCC',ylim=(0,.42));axs[0].legend(frameon=False);axs[0].set_title('A  LINCS: fixed training states',loc='left',weight='bold')
axs[1].plot(seeds.seed,seeds.interaction,'o',color=blue,markersize=7);axs[1].axhline(seeds.interaction.mean(),color=orange,ls='--',label='Mean of seed summaries');axs[1].set(xticks=seeds.seed,xlabel='Training seed',ylabel='Archived interaction summary',ylim=(0,.065));axs[1].legend(frameon=False,loc='lower center');axs[1].set_title('B  DepMap: five training seeds',loc='left',weight='bold');fig.tight_layout();savefig('Figure3_training_evaluation')
fig,axs=plt.subplots(1,2,figsize=(10,4));axs[0].plot(rank.k,rank.retention,'o-',c=blue,label='Observed');axs[0].plot(rank.k,rank.chance_expected_retention,'--',c=grey,label='Random-list expectation');axs[0].set(xscale='log',xlabel='k (2,037 drugs)',ylabel='Top-k retention',ylim=(0,1));axs[0].legend(frameon=False);axs[0].set_title('A  Drug performance lists',loc='left',weight='bold')
vv=disease.iloc[1:4];axs[1].bar(np.arange(3)-.17,vv.spearman,.34,label='Global rank Spearman',color=blue);axs[1].bar(np.arange(3)+.17,[.1,.3,.2],.34,label='Top-10 retention',color=orange);axs[1].set(xticks=range(3),xticklabels=['Liver','Thyroid','APL'],ylim=(0,.65),ylabel='Rank association / list retention');axs[1].legend(frameon=False,fontsize=9);axs[1].set_title('B  Disease-signature lists (n=1,399)',loc='left',weight='bold');fig.tight_layout();savefig('Figure4_selection')
fig,axs=plt.subplots(1,2,figsize=(10,4));cell=d.groupby('cell').agg(delta=('pearson_delta','mean'),n=('pearson_delta','size'));axs[0].scatter(cell.n,cell.delta,c=blue,s=26,alpha=.8);axs[0].axhline(0,c=grey,lw=.8);axs[0].axhline(met[(met.metric=='pearson')&(met.weighting=='cell')].delta.iloc[0],c=orange,ls='--',label='Equal-cell mean');axs[0].set(xscale='log',xlabel='Responses per cell line',ylabel='Mean within-cell PCC contrast');axs[0].legend(frameon=False);axs[0].set_title('A  LINCS: unequal cell coverage',loc='left',weight='bold')
axs[1].scatter(dep.product_concordance,dep.delta_pearson,c=blue,s=18,alpha=.6);axs[1].axhline(0,c=grey,lw=.8);axs[1].set(xlabel='CERES-Chronos Pearson',ylabel='DeepDEP PCC contrast');axs[1].set_title('B  DepMap: 272 cell lines',loc='left',weight='bold');fig.tight_layout();savefig('Figure5_heterogeneity')
print(json.dumps(res,indent=2));print(met.to_string(index=False));print(rank.to_string(index=False))
