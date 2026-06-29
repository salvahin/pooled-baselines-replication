"""Phase 2: does the pooled-baseline artifact generalize beyond code smells?
Reframe standard multi-class tabular datasets as per-task one-vs-rest, run the
same fairness 2x2: specific (OvR) vs pooled-intercept vs pooled-interaction,
x {LogReg,LinearSVC,RF,HistGB} x seeds. Incremental CSV."""
import csv,time,warnings,numpy as np
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
NF=5; HERE=Path(__file__).resolve().parent; DATA=HERE.parent.parent/'data'; ML=HERE/'mlbench'; SEEDS=[42,1,7,13,99]
CAP={'letter':1500}  # cap instances so K x n stays tractable
def facs(seed):
    return {'LogReg':lambda:LogisticRegression(class_weight='balanced',max_iter=2000,random_state=seed),
            'LinearSVC':lambda:LinearSVC(class_weight='balanced',max_iter=5000,random_state=seed),
            'RF':lambda:RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=seed,n_jobs=-1),
            'HistGB':lambda:HistGradientBoostingClassifier(class_weight='balanced',random_state=seed)}
def sco(c,X): return c.predict_proba(X)[:,1] if hasattr(c,'predict_proba') else c.decision_function(X)
def specific_ap(pf,fac,seed):
    out={}
    for k,(X,y) in pf.items():
        if y.sum()<NF: out[k]=np.nan;continue
        ap=[]
        for tr,te in StratifiedKFold(NF,shuffle=True,random_state=seed).split(X,y):
            s=StandardScaler();c=fac();c.fit(s.fit_transform(X[tr]),y[tr]);ap.append(average_precision_score(y[te],sco(c,s.transform(X[te]))))
        out[k]=np.mean(ap)
    return out
def pooled_ap(pf,fac,seed,inter):
    order=list(pf);ns=len(order);Xs=[];ys=[];idx=[]
    for si,k in enumerate(order):
        X,y=pf[k];Xs.append(X);ys.append(y);idx.append(np.full(len(X),si))
    Xa=np.vstack(Xs);ya=np.concatenate(ys);sidx=np.concatenate(idx);out={k:[] for k in order}
    for tr,te in StratifiedKFold(NF,shuffle=True,random_state=seed).split(Xa,sidx):
        sc=StandardScaler();Ftr=sc.fit_transform(Xa[tr]);Fte=sc.transform(Xa[te])
        def aug(F,si):
            oh=np.zeros((len(F),ns));oh[np.arange(len(F)),si]=1
            return np.hstack([F,oh]) if not inter else np.hstack([F,oh,np.einsum('ij,ik->ijk',F,oh).reshape(len(F),-1)])
        c=fac();c.fit(aug(Ftr,sidx[tr]),ya[tr]);pr=sco(c,aug(Fte,sidx[te]))
        for si,k in enumerate(order):
            m=sidx[te]==si
            if m.sum() and ya[te][m].sum()>0: out[k].append(average_precision_score(ya[te][m],pr[m]))
    return {k:(np.mean(v) if v else np.nan) for k,v in out.items()}
def load(name,seed):
    d=np.load(ML/f'{name}.npz');X=d['X'].astype(float);y=d['y'].astype(int)
    cap=CAP.get(name)
    if cap and len(X)>cap:
        rng=np.random.RandomState(seed);ix=rng.choice(len(X),cap,replace=False);X=X[ix];y=y[ix]
    classes=[c for c in sorted(set(y)) if (y==c).sum()>=2*NF]
    return {f'c{c}':(X,(y==c).astype(int)) for c in classes}
out=DATA/'phase2_generalization_2026-06-24.csv'
with open(out,'w',newline='') as f: csv.writer(f).writerow(['dataset','classifier','seed','n_tasks','d_intercept','d_interaction'])
DSETS=['digits','segment','vehicle','letter']
for ds in DSETS:
    if not (ML/f'{ds}.npz').exists(): print('skip',ds);continue
    for cn in ['LogReg','LinearSVC','RF','HistGB']:
        for seed in SEEDS:
            t=time.time();pf=load(ds,seed);fac=facs(seed)[cn]
            try:
                sp=specific_ap(pf,fac,seed);b0=pooled_ap(pf,fac,seed,False);b1=pooled_ap(pf,fac,seed,True)
                ks=[k for k in pf if not np.isnan(sp[k]) and not np.isnan(b0[k]) and not np.isnan(b1[k])]
                di=np.mean([sp[k]-b0[k] for k in ks]);dj=np.mean([sp[k]-b1[k] for k in ks])
                with open(out,'a',newline='') as f: csv.writer(f).writerow([ds,cn,seed,len(ks),round(di,4),round(dj,4)])
                print(f'{ds:10s} {cn:10s} seed={seed} d_intercept={di:+.3f} d_interaction={dj:+.3f} ({time.time()-t:.0f}s)',flush=True)
            except Exception as e: print(f'{ds} {cn} {seed} ERR {e}',flush=True)
print('PHASE2 DONE',flush=True)
