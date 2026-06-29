"""Reviewer-fix re-runs (matched folds, full-scale, finer imbalance ladder).
- MATCHED FOLDS: specific and pooled arms share one per-instance fold assignment,
  so they test on identical instances per task per fold (a properly paired test).
- NO SUBSAMPLING: the two large datasets run at full row count (SmellyCode++ 107k,
  ml-Codesmell 373k). letter stays capped at 1500 (task-count reasons).
- Outputs: revfix_phase1.csv (Phase-1 dPR-AUC, all 9 datasets, matched, full),
           revfix_imbalance.csv (7-rate ladder, matched).
Resumable: skips (dataset,classifier,seed[,rate]) keys already in each CSV.
N=10 aggregation for the Wilcoxon (#1) is done at analysis time from these CSVs."""
import csv,time,warnings,numpy as np,pandas as pd
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
import _loaders as L

DATA=L.DATA; HERE=L.HERE; SEEDS=[42,1,7,13,99]
def facs(seed):
    return {'LogReg':lambda:LogisticRegression(class_weight='balanced',max_iter=2000,random_state=seed),
            'LinearSVC':lambda:LinearSVC(class_weight='balanced',dual=False,max_iter=5000,random_state=seed),
            'RF':lambda:RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=seed,n_jobs=-1),
            'HistGB':lambda:HistGradientBoostingClassifier(class_weight='balanced',random_state=seed)}
def sco(c,X): return c.predict_proba(X)[:,1] if hasattr(c,'predict_proba') else c.decision_function(X)
def aug(F,si,ns,inter):
    oh=np.zeros((len(F),ns));oh[np.arange(len(F)),si]=1
    if not inter: return np.hstack([F,oh])
    return np.hstack([F,oh,np.einsum('ij,ik->ijk',F,oh).reshape(len(F),-1)])
def _done(path,ncol):
    s=set()
    if path.exists():
        import csv as _c
        with open(path) as f:
            r=_c.reader(f);next(r,None)
            for row in r:
                if len(row)>=ncol: s.add(tuple(row[:ncol]))
    return s

# ---- full (no-subsample) loaders for the two big datasets ----
def L_smelly_full(seed):
    HAL=['Logical Lines','Distinct Operators','Distinct Operands','Total Operators','Total Operands','Vocabulary','Length','Calculated Length','Volume','Difficulty','Effort','Time Required','Bugs','Cyclomatic Complexity']
    SM={'god_class':'God class','long_method':'Long method','feature_envy':'Feature envy','data_class':'Data class'}
    df=pd.read_csv(DATA/'SmellyCode++.csv',usecols=HAL+list(SM.values()))
    X=np.nan_to_num(df[HAL].values,nan=0.0);return {k:(X,(df[c]==1).astype(int).values) for k,c in SM.items()}
def L_mlcs_full(seed):
    df=pd.read_csv(HERE/'mlcodesmell_class.csv')
    labels=['Brain Class','Data Class','Futile Abstract Pipeline','Futile Hierarchy','God Class','Hierarchy Duplication','Model Class','Schizofrenic Class']
    feats=[c for c in df.columns if c not in (['Address']+labels)];X=np.nan_to_num(df[feats].apply(pd.to_numeric,errors='coerce').values,nan=0.0)
    SM={'data_class':'Data Class','god_class':'God Class','schizofrenic_class':'Schizofrenic Class'}
    return {k:(X,(df[c].astype(str).str.upper()=='TRUE').astype(int).values) for k,c in SM.items()}

# dataset registry: (name, loader, nfolds)
CODE=[('IST2021',L.L_ist,10),('ImprovMLCQ',L.L_improv,10),('Crowdsmelling',L.L_crowd,10),
      ('SmellyCode++',L_smelly_full,10),('ml-Codesmell',L_mlcs_full,10)]
MC=[(n,(lambda s,n=n:L.L_mc(n,s,5)),5) for n in L.MC_DSETS]
ALL=CODE+MC

# ---- matched-fold machinery ----
def fold_assign(pf,seed,NF):
    fa={}
    for k,(X,y) in pf.items():
        f=np.full(len(y),-1,int)
        if y.sum()<NF: fa[k]=f; continue
        for fi,(tr,te) in enumerate(StratifiedKFold(NF,shuffle=True,random_state=seed).split(X,y)): f[te]=fi
        fa[k]=f
    return fa
def specific_matched(pf,fa,fac,NF):
    out={}
    for k,(X,y) in pf.items():
        if (fa[k]<0).all(): out[k]=np.nan; continue
        ap=[]
        for f in range(NF):
            te=fa[k]==f; tr=(fa[k]>=0)&(~te)
            if y[te].sum()==0 or y[tr].sum()==0: continue
            s=StandardScaler();c=fac();c.fit(s.fit_transform(X[tr]),y[tr]);ap.append(average_precision_score(y[te],sco(c,s.transform(X[te]))))
        out[k]=np.mean(ap) if ap else np.nan
    return out
def pooled_matched(pf,fa,fac,NF,inter):
    order=[k for k in pf if not (fa[k]<0).all()]; ns=len(order)
    Xs=[];ya=[];si=[];fo=[]
    for i,k in enumerate(order):
        X,y=pf[k];Xs.append(X);ya.append(y);si.append(np.full(len(X),i));fo.append(fa[k])
    Xa=np.vstack(Xs);ya=np.concatenate(ya);sidx=np.concatenate(si);fold=np.concatenate(fo)
    out={k:[] for k in order}
    for f in range(NF):
        te=fold==f; tr=(fold>=0)&(~te)
        if te.sum()==0 or tr.sum()==0: continue
        sc=StandardScaler();Ftr=sc.fit_transform(Xa[tr]);Fte=sc.transform(Xa[te])
        c=fac();c.fit(aug(Ftr,sidx[tr],ns,inter),ya[tr]);pr=sco(c,aug(Fte,sidx[te],ns,inter))
        for i,k in enumerate(order):
            m=sidx[te]==i
            if m.sum() and ya[te][m].sum()>0: out[k].append(average_precision_score(ya[te][m],pr[m]))
    return {k:(np.mean(v) if v else np.nan) for k,v in out.items()}

def seed_policy(ds,cn):
    if ds=='ml-Codesmell' and cn=='LinearSVC': return SEEDS[:3]   # solver cost at full scale
    return SEEDS

def cell_phase1(loader,NF,cn,seed):
    pf=loader(seed); fa=fold_assign(pf,seed,NF); fac=facs(seed)[cn]
    sp=specific_matched(pf,fa,fac,NF); b0=pooled_matched(pf,fa,fac,NF,False); b1=pooled_matched(pf,fa,fac,NF,True)
    ks=[k for k in pf if k in sp and not np.isnan(sp[k]) and not np.isnan(b0.get(k,np.nan)) and not np.isnan(b1.get(k,np.nan))]
    sA=np.mean([sp[k] for k in ks]); i0=np.mean([b0[k] for k in ks]); i1=np.mean([b1[k] for k in ks])
    return sA,i0,i1

def run_phase1(include_slow):
    out=DATA/'revfix_phase1.csv'
    if not out.exists():
        with open(out,'w',newline='') as f: csv.writer(f).writerow(['dataset','classifier','seed','specAP','interceptAP','interactionAP','d_intercept','d_interaction'])
    done=_done(out,3)
    for ds,ld,NF in ALL:
        for cn in ['LogReg','RF','HistGB','LinearSVC']:
            slow=(ds=='ml-Codesmell' and cn=='LinearSVC')
            if slow and not include_slow: continue
            if not slow and include_slow: continue
            for seed in seed_policy(ds,cn):
                if (ds,cn,str(seed)) in done: print(f'[P1] skip {ds} {cn} {seed}',flush=True); continue
                t=time.time()
                try:
                    sA,i0,i1=cell_phase1(ld,NF,cn,seed)
                    with open(out,'a',newline='') as f: csv.writer(f).writerow([ds,cn,seed,round(sA,4),round(i0,4),round(i1,4),round(sA-i0,4),round(sA-i1,4)])
                    print(f'[P1] {ds:14s} {cn:10s} seed={seed} d_int={sA-i0:+.3f} d_itr={sA-i1:+.3f} ({time.time()-t:.0f}s)',flush=True)
                except Exception as e: print(f'[P1] {ds} {cn} {seed} ERR {e}',flush=True)

# ---- finer-ladder imbalance (#2), matched folds ----
NFA=5; RATES7=[0.05,0.10,0.15,0.20,0.25,0.30,0.35]; MINPOS=2*NFA; MINTOT=200
def set_rate(X,y,r,rng):
    pos=np.where(y==1)[0]; neg=np.where(y==0)[0]; np_,nn=len(pos),len(neg)
    if np_==0 or nn==0: return None
    need=int(round(np_*(1-r)/r))
    if need<=nn: kp=pos; kn=rng.choice(neg,need,replace=False)
    else: npos=int(round(nn*r/(1-r))); kp=rng.choice(pos,npos,replace=False); kn=neg
    idx=np.concatenate([kp,kn]); rng.shuffle(idx); X2,y2=X[idx],y[idx]
    if y2.sum()<MINPOS or len(idx)<MINTOT: return None
    return (X2,y2)
def gap_at_rate(base,fac,seed,r):
    rng=np.random.RandomState(seed*1000+int(r*100)); pf={}
    for k,(X,y) in base.items():
        z=set_rate(X,y,r,rng)
        if z is not None: pf[k]=z
    if len(pf)<2: return np.nan
    fa=fold_assign(pf,seed,NFA); sp=specific_matched(pf,fa,fac,NFA); b0=pooled_matched(pf,fa,fac,NFA,False)
    ks=[k for k in pf if not np.isnan(sp[k]) and not np.isnan(b0.get(k,np.nan))]
    return float(np.mean([sp[k]-b0[k] for k in ks])) if ks else np.nan
def run_imbalance():
    out=DATA/'revfix_imbalance.csv'
    if not out.exists():
        with open(out,'w',newline='') as f: csv.writer(f).writerow(['dataset','classifier','rate','seed','gap_intercept'])
    done=_done(out,4)
    # imbalance uses 15k for the big two (trend analysis; keeps it tractable)
    ld_imb={'SmellyCode++':L.L_smelly,'ml-Codesmell':L.L_mlcs}
    afac=lambda seed:{'LogReg':lambda:LogisticRegression(class_weight='balanced',max_iter=2000,random_state=seed),'RF':lambda:RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=seed,n_jobs=-1)}
    for ds,ld,NF in ALL:
        loader=ld_imb.get(ds,ld)
        for cn in ['LogReg','RF']:
            for r in RATES7:
                for seed in SEEDS:
                    if (ds,cn,str(r),str(seed)) in done: continue
                    t=time.time();base=loader(seed);g=gap_at_rate(base,afac(seed)[cn],seed,r)
                    with open(out,'a',newline='') as f: csv.writer(f).writerow([ds,cn,r,seed,('' if np.isnan(g) else round(g,4))])
                    print(f'[A] {ds:14s} {cn:7s} r={r:.2f} seed={seed} gap={g:+.3f} ({time.time()-t:.0f}s)',flush=True)
    print('REVFIX_IMBALANCE DONE',flush=True)

if __name__=='__main__':
    print('### REVFIX START',flush=True)
    run_phase1(include_slow=False)   # matched Phase 1, all except ml-Codesmell LinearSVC
    print('REVFIX_PHASE1_FAST DONE',flush=True)
    run_imbalance()                  # 7-rate ladder (#2)
    run_phase1(include_slow=True)    # ml-Codesmell LinearSVC full (slow, 3 seeds), last
    print('### REVFIX ALL DONE',flush=True)
