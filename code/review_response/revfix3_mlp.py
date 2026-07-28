"""Reviewer-requested MLP experiment (Information round 1, R1 comment 1).
Replicates the Phase-1 comparison (specific vs pooled-intercept vs pooled-interaction)
with a neural network (sklearn MLPClassifier) at two capacities:
  MLP16  = one hidden layer of 16 units  (width-constrained)
  MLP100 = one hidden layer of 100 units (sklearn default width)
Hypothesis (stated in advance): the artifact tracks the model's ability to synthesize
task-conditional feature reweighting internally. A sufficiently wide MLP given the
one-hot can approximate per-task interactions (like trees), so its intercept-pooled
gap should be small; a width-constrained MLP should sit between linear and tree.

Protocol matches revfix.py: matched per-instance folds shared by all arms,
PR-AUC (average precision), seeds [42,1,7,13,99]. NF=10 code-smell, NF=5 multi-class.
The two large datasets use the established 15k subsample (subsample~full agreement
shown in Section 'Robustness Checks'; MLP at 373k full scale is not tractable).
NOTE: MLPClassifier has no class_weight; all three arms are equally unweighted,
so the within-model comparison remains fair (stated in the manuscript).
Resumable: skips (dataset,classifier,seed) rows already in revfix3_mlp.csv.
"""
import csv,time,warnings,numpy as np
warnings.filterwarnings('ignore')
from sklearn.neural_network import MLPClassifier
import _loaders as L
from revfix import fold_assign,specific_matched,pooled_matched,_done

DATA=L.DATA; SEEDS=[42,1,7,13,99]

def facs(seed):
    mk=lambda h:(lambda:MLPClassifier(hidden_layer_sizes=h,max_iter=300,early_stopping=True,
                                      n_iter_no_change=10,random_state=seed))
    return {'MLP16':mk((16,)),'MLP100':mk((100,))}

# smallest-first so results accumulate early; letter (26 tasks) last
ALL=[('vehicle',(lambda s:L.L_mc('vehicle',s,5)),5),
     ('digits',(lambda s:L.L_mc('digits',s,5)),5),
     ('segment',(lambda s:L.L_mc('segment',s,5)),5),
     ('IST2021',L.L_ist,10),
     ('Crowdsmelling',L.L_crowd,10),
     ('ImprovMLCQ',L.L_improv,10),
     ('SmellyCode++',L.L_smelly,10),      # 15k subsample
     ('ml-Codesmell',L.L_mlcs,10),        # 15k subsample
     ('letter',(lambda s:L.L_mc('letter',s,5)),5)]

def cell(loader,NF,cn,seed):
    pf=loader(seed); fa=fold_assign(pf,seed,NF); fac=facs(seed)[cn]
    sp=specific_matched(pf,fa,fac,NF); b0=pooled_matched(pf,fa,fac,NF,False); b1=pooled_matched(pf,fa,fac,NF,True)
    ks=[k for k in pf if k in sp and not np.isnan(sp[k]) and not np.isnan(b0.get(k,np.nan)) and not np.isnan(b1.get(k,np.nan))]
    return np.mean([sp[k] for k in ks]),np.mean([b0[k] for k in ks]),np.mean([b1[k] for k in ks])

def main():
    out=DATA/'revfix3_mlp.csv'
    if not out.exists():
        with open(out,'w',newline='') as f:
            csv.writer(f).writerow(['dataset','classifier','seed','specAP','interceptAP','interactionAP','d_intercept','d_interaction'])
    done=_done(out,3)
    for ds,ld,NF in ALL:
        for cn in ['MLP16','MLP100']:
            for seed in SEEDS:
                if (ds,cn,str(seed)) in done: print(f'[MLP] skip {ds} {cn} {seed}',flush=True); continue
                t=time.time()
                try:
                    sA,i0,i1=cell(ld,NF,cn,seed)
                    with open(out,'a',newline='') as f:
                        csv.writer(f).writerow([ds,cn,seed,round(sA,4),round(i0,4),round(i1,4),round(sA-i0,4),round(sA-i1,4)])
                    print(f'[MLP] {ds:14s} {cn:7s} seed={seed} d_int={sA-i0:+.3f} d_itr={sA-i1:+.3f} ({time.time()-t:.0f}s)',flush=True)
                except Exception as e:
                    print(f'[MLP] {ds} {cn} {seed} ERR {e}',flush=True)
    print('### MLP ALL DONE',flush=True)

if __name__=='__main__':
    print('### MLP START',flush=True)
    main()
