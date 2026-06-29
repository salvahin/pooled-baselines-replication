"""STRENGTHEN A -- does class imbalance moderate the pooled-baseline gap?
Controlled within-dataset positive-rate ladder, applied to ALL datasets
(5 code-smell + 4 multi-class), for a linear (LogReg) and a tree (RF) family,
5 seeds, 5 folds. For each task we resample to a target positive rate r, then
measure the gap = PR-AUC(specific) - PR-AUC(intercept-pooled), averaged over tasks.
If imbalance moderates, the gap should trend monotonically with r consistently
across datasets. We report the per-(dataset,classifier) Spearman trend; sign
inconsistency across datasets => imbalance is NOT a consistent moderator."""
import csv,time,warnings,numpy as np
warnings.filterwarnings('ignore')
from pathlib import Path
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
import _loaders as L

NF=5; SEEDS=[42,1,7,13,99]; RATES=[0.05,0.10,0.20,0.33]; MINPOS=2*NF; MINTOT=200
def facs(seed):
    return {'LogReg':lambda:LogisticRegression(class_weight='balanced',max_iter=2000,random_state=seed),
            'RF':lambda:RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=seed,n_jobs=-1)}
def sco(c,X): return c.predict_proba(X)[:,1] if hasattr(c,'predict_proba') else c.decision_function(X)

def set_rate(X,y,r,rng):
    pos=np.where(y==1)[0]; neg=np.where(y==0)[0]; np_,nn=len(pos),len(neg)
    if np_==0 or nn==0: return None
    need_neg=int(round(np_*(1-r)/r))
    if need_neg<=nn:
        keep_pos=pos; keep_neg=rng.choice(neg,need_neg,replace=False)
    else:
        need_pos=int(round(nn*r/(1-r))); keep_pos=rng.choice(pos,need_pos,replace=False); keep_neg=neg
    idx=np.concatenate([keep_pos,keep_neg]); rng.shuffle(idx)
    Xy=(X[idx],y[idx])
    if Xy[1].sum()<MINPOS or len(idx)<MINTOT: return None
    return Xy

def specific_ap(pf,fac,seed):
    out={}
    for k,(X,y) in pf.items():
        if y.sum()<NF: out[k]=np.nan;continue
        ap=[]
        for tr,te in StratifiedKFold(NF,shuffle=True,random_state=seed).split(X,y):
            s=StandardScaler();c=fac();c.fit(s.fit_transform(X[tr]),y[tr]);ap.append(average_precision_score(y[te],sco(c,s.transform(X[te]))))
        out[k]=np.mean(ap)
    return out
def pooled_ap(pf,fac,seed):
    order=list(pf);ns=len(order);Xs=[];ys=[];idx=[]
    for si,k in enumerate(order):
        X,y=pf[k];Xs.append(X);ys.append(y);idx.append(np.full(len(X),si))
    Xa=np.vstack(Xs);ya=np.concatenate(ys);sidx=np.concatenate(idx);out={k:[] for k in order}
    for tr,te in StratifiedKFold(NF,shuffle=True,random_state=seed).split(Xa,sidx):
        sc=StandardScaler();Ftr=sc.fit_transform(Xa[tr]);Fte=sc.transform(Xa[te])
        def aug(F,si):
            oh=np.zeros((len(F),ns));oh[np.arange(len(F)),si]=1;return np.hstack([F,oh])
        c=fac();c.fit(aug(Ftr,sidx[tr]),ya[tr]);pr=sco(c,aug(Fte,sidx[te]))
        for si,k in enumerate(order):
            m=sidx[te]==si
            if m.sum() and ya[te][m].sum()>0: out[k].append(average_precision_score(ya[te][m],pr[m]))
    return {k:(np.mean(v) if v else np.nan) for k,v in out.items()}

def gap_at_rate(loader_pf,fac,seed,r):
    rng=np.random.RandomState(seed*1000+int(r*100))
    pf={}
    for k,(X,y) in loader_pf.items():
        z=set_rate(X,y,r,rng)
        if z is not None: pf[k]=z
    if len(pf)<2: return np.nan
    sp=specific_ap(pf,fac,seed);b0=pooled_ap(pf,fac,seed)
    ks=[k for k in pf if not np.isnan(sp[k]) and not np.isnan(b0[k])]
    if not ks: return np.nan
    return float(np.mean([sp[k]-b0[k] for k in ks]))

ALL=[(n,lambda s,n=n:L.CODE_LOADERS[n](s)) for n in L.CODE_LOADERS]+[(n,lambda s,n=n:L.L_mc(n,s,NF)) for n in L.MC_DSETS]
out=L.DATA/'strengthen_A_imbalance_2026-06-24.csv'
with open(out,'w',newline='') as f: csv.writer(f).writerow(['dataset','classifier','rate','seed','gap_intercept'])
rows=[]
for ds,ld in ALL:
    for cn in ['LogReg','RF']:
        for r in RATES:
            gs=[]
            for seed in SEEDS:
                t=time.time();base=ld(seed);g=gap_at_rate(base,facs(seed)[cn],seed,r)
                gs.append(g)
                with open(out,'a',newline='') as f: csv.writer(f).writerow([ds,cn,r,seed,('' if np.isnan(g) else round(g,4))])
                print(f'{ds:14s} {cn:7s} r={r:.2f} seed={seed} gap={g:+.3f} ({time.time()-t:.0f}s)',flush=True)
            rows.append((ds,cn,r,np.nanmean(gs)))
print('\n=== per-(dataset,classifier) trend: Spearman(rate, gap) over rate ladder ===',flush=True)
import collections
by=collections.defaultdict(list)
for ds,cn,r,g in rows: by[(ds,cn)].append((r,g))
for (ds,cn),v in by.items():
    v=[(r,g) for r,g in v if not np.isnan(g)]
    if len(v)>=3:
        rr=[a for a,_ in v];gg=[b for _,b in v];rho,p=stats.spearmanr(rr,gg)
        print(f'{ds:14s} {cn:7s} rho={rho:+.2f} p={p:.2f}  gaps@rates={[round(b,3) for b in gg]}',flush=True)
print('STRENGTHEN_A DONE',flush=True)
