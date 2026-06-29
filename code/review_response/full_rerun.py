"""FULL-DATA re-run of the two previously-subsampled datasets (SmellyCode++,
ml-Codesmell) at their full row counts, to verify the headline numbers are not
an artifact of the 15k subsample. Same pipeline/protocol as the main runs
(10-fold CV for code smells, StandardScaler per fold, class_weight balanced,
5 seeds). Function bodies are copied verbatim from phase1/strengthen_{A,B,C}.
Produces: full_phase1.csv, full_params.csv, full_threshold.csv, full_imbalance.csv
and prints aggregates at the end."""
import csv,time,warnings,numpy as np,pandas as pd
warnings.filterwarnings('ignore')
from pathlib import Path
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve

HERE=Path(__file__).resolve().parent; DATA=HERE.parent.parent/'data'
NF=10; SEEDS=[42,1,7,13,99]

# ---------- FULL loaders (no subsample) ----------
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
LOADERS={'SmellyCode++':L_smelly_full,'ml-Codesmell':L_mlcs_full}

def facs(seed):
    # LinearSVC: dual=False is far faster when n_samples >> n_features (our full
    # pooled stacks are ~1.1M rows x <=167 cols); same squared-hinge L2 objective.
    return {'LogReg':lambda:LogisticRegression(class_weight='balanced',max_iter=2000,random_state=seed),
            'LinearSVC':lambda:LinearSVC(class_weight='balanced',dual=False,max_iter=5000,random_state=seed),
            'RF':lambda:RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=seed,n_jobs=-1),
            'HistGB':lambda:HistGradientBoostingClassifier(class_weight='balanced',random_state=seed)}

def _done_keys(path, ncols):
    """Return set of already-computed key tuples (first ncols fields) for resume."""
    s=set()
    if path.exists():
        with open(path) as f:
            r=csv.reader(f); next(r,None)
            for row in r:
                if len(row)>=ncols: s.add(tuple(row[:ncols]))
    return s
def sco(c,X): return c.predict_proba(X)[:,1] if hasattr(c,'predict_proba') else c.decision_function(X)
def aug(F,si,ns,inter):
    oh=np.zeros((len(F),ns));oh[np.arange(len(F)),si]=1
    if not inter: return np.hstack([F,oh])
    return np.hstack([F,oh,np.einsum('ij,ik->ijk',F,oh).reshape(len(F),-1)])

# ---------- Phase 1 (ΔPR-AUC) ----------
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
        c=fac();c.fit(aug(Ftr,sidx[tr],ns,inter),ya[tr]);pr=sco(c,aug(Fte,sidx[te],ns,inter))
        for si,k in enumerate(order):
            m=sidx[te]==si
            if m.sum() and ya[te][m].sum()>0: out[k].append(average_precision_score(ya[te][m],pr[m]))
    return {k:(np.mean(v) if v else np.nan) for k,v in out.items()}

def run_phase1():
    out=DATA/'full_phase1.csv'
    if not out.exists():
        with open(out,'w',newline='') as f: csv.writer(f).writerow(['dataset','classifier','seed','specAP','interceptAP','interactionAP','d_intercept','d_interaction'])
    done=_done_keys(out,3)
    for ds,ld in LOADERS.items():
        for cn in ['LogReg','LinearSVC','RF','HistGB']:
            for seed in SEEDS:
                if (ds,cn,str(seed)) in done: print(f'[P1] skip {ds} {cn} {seed} (cached)',flush=True); continue
                t=time.time();pf=ld(seed);fac=facs(seed)[cn]
                try:
                    sp=specific_ap(pf,fac,seed);b0=pooled_ap(pf,fac,seed,False);b1=pooled_ap(pf,fac,seed,True)
                    ks=[k for k in pf if not np.isnan(sp[k]) and not np.isnan(b0[k]) and not np.isnan(b1[k])]
                    sA=np.mean([sp[k] for k in ks]);i0=np.mean([b0[k] for k in ks]);i1=np.mean([b1[k] for k in ks])
                    with open(out,'a',newline='') as f: csv.writer(f).writerow([ds,cn,seed,round(sA,4),round(i0,4),round(i1,4),round(sA-i0,4),round(sA-i1,4)])
                    print(f'[P1] {ds:14s} {cn:10s} seed={seed} d_int={sA-i0:+.3f} d_itr={sA-i1:+.3f} ({time.time()-t:.0f}s)',flush=True)
                except Exception as e: print(f'[P1] {ds} {cn} {seed} ERR {e}',flush=True)
    print('FULL_PHASE1 DONE',flush=True)

# ---------- Part C (params/parity, LogReg) ----------
def run_params():
    def lr(): return LogisticRegression(class_weight='balanced',max_iter=2000,random_state=42)
    out=DATA/'full_params.csv'
    with open(out,'w',newline='') as f: csv.writer(f).writerow(['dataset','K','spec_params','itr_params','ratio','spec_PRAUC','itr_PRAUC','dPRAUC'])
    NF5=5
    for ds,ld in LOADERS.items():
        pf=ld(42);order=[k for k,(X,y) in pf.items() if y.sum()>=NF5]
        # specific
        models=[];aps=[]
        for k in order:
            X,y=pf[k];ap=[]
            for tr,te in StratifiedKFold(NF5,shuffle=True,random_state=42).split(X,y):
                s=StandardScaler();c=lr();c.fit(s.fit_transform(X[tr]),y[tr]);ap.append(average_precision_score(y[te],c.predict_proba(s.transform(X[te]))[:,1]))
            s=StandardScaler();c=lr();c.fit(s.fit_transform(X),y);models.append(c);aps.append(np.mean(ap))
        spar=int(sum(m.coef_.size+m.intercept_.size for m in models));spa=float(np.mean(aps))
        # interaction-pooled
        ns=len(order);Xs=[];ys=[];idx=[]
        for si,k in enumerate(order):
            X,y=pf[k];Xs.append(X);ys.append(y);idx.append(np.full(len(X),si))
        Xa=np.vstack(Xs);ya=np.concatenate(ys);sidx=np.concatenate(idx);outp={k:[] for k in order}
        for tr,te in StratifiedKFold(NF5,shuffle=True,random_state=42).split(Xa,sidx):
            sc=StandardScaler();Ftr=sc.fit_transform(Xa[tr]);Fte=sc.transform(Xa[te]);c=lr();c.fit(aug(Ftr,sidx[tr],ns,True),ya[tr]);pr=c.predict_proba(aug(Fte,sidx[te],ns,True))[:,1]
            for si,k in enumerate(order):
                m=sidx[te]==si
                if m.sum() and ya[te][m].sum()>0: outp[k].append(average_precision_score(ya[te][m],pr[m]))
        sc=StandardScaler();c=lr();c.fit(aug(sc.fit_transform(Xa),sidx,ns,True),ya);ipar=int(c.coef_.size+c.intercept_.size)
        ipa=float(np.mean([np.mean(v) for v in outp.values() if v]))
        with open(out,'a',newline='') as f: csv.writer(f).writerow([ds,len(order),spar,ipar,round(ipar/max(spar,1),2),round(spa,4),round(ipa,4),round(spa-ipa,4)])
        print(f'[C] {ds:14s} K={len(order)} spec={spar} itr={ipar} ratio={ipar/max(spar,1):.2f}x dPRAUC={spa-ipa:+.3f}',flush=True)
    print('FULL_PARAMS DONE',flush=True)

# ---------- Part B (threshold) ----------
def bestf1(y,s):
    p,r,th=precision_recall_curve(y,s); f1=2*p*r/(p+r+1e-12); return float(np.nanmax(f1))
def b_specific(pf,fac,seed):
    o={k:{'f105':[],'fbest':[],'ap':[]} for k in pf}
    for k,(X,y) in pf.items():
        if y.sum()<NF: o[k]=None;continue
        for tr,te in StratifiedKFold(NF,shuffle=True,random_state=seed).split(X,y):
            s=StandardScaler();Xtr=s.fit_transform(X[tr]);Xte=s.transform(X[te]);c=fac();c.fit(Xtr,y[tr]);pr=sco(c,Xte)
            o[k]['f105'].append(f1_score(y[te],c.predict(Xte),zero_division=0));o[k]['fbest'].append(bestf1(y[te],pr));o[k]['ap'].append(average_precision_score(y[te],pr))
    return o
def b_pooled(pf,fac,seed,inter):
    order=list(pf);ns=len(order);Xs=[];ys=[];idx=[]
    for si,k in enumerate(order):
        X,y=pf[k];Xs.append(X);ys.append(y);idx.append(np.full(len(X),si))
    Xa=np.vstack(Xs);ya=np.concatenate(ys);sidx=np.concatenate(idx);o={k:{'f105':[],'fbest':[],'ap':[]} for k in order}
    for tr,te in StratifiedKFold(NF,shuffle=True,random_state=seed).split(Xa,sidx):
        sc=StandardScaler();Ftr=sc.fit_transform(Xa[tr]);Fte=sc.transform(Xa[te]);c=fac();At=aug(Fte,sidx[te],ns,inter)
        c.fit(aug(Ftr,sidx[tr],ns,inter),ya[tr]);pr=sco(c,At);pred=c.predict(At)
        for si,k in enumerate(order):
            m=sidx[te]==si
            if m.sum() and ya[te][m].sum()>0:
                o[k]['f105'].append(f1_score(ya[te][m],pred[m],zero_division=0));o[k]['fbest'].append(bestf1(ya[te][m],pr[m]));o[k]['ap'].append(average_precision_score(ya[te][m],pr[m]))
    return o
def run_threshold():
    out=DATA/'full_threshold.csv'
    if not out.exists():
        with open(out,'w',newline='') as f: csv.writer(f).writerow(['dataset','classifier','seed','dF1@0.5','dF1@best','dPR-AUC_int','dPR-AUC_itr'])
    done=_done_keys(out,3)
    for ds,ld in LOADERS.items():
        for cn in ['LogReg','LinearSVC','RF','HistGB']:
            for seed in SEEDS:
                if (ds,cn,str(seed)) in done: print(f'[B] skip {ds} {cn} {seed} (cached)',flush=True); continue
                t=time.time();pf=ld(seed);fac=facs(seed)[cn]
                try:
                    sp=b_specific(pf,fac,seed);b0=b_pooled(pf,fac,seed,False);b1=b_pooled(pf,fac,seed,True)
                    ks=[k for k in sp if sp[k] and sp[k]['ap'] and b0[k]['ap'] and b1[k]['ap']]
                    if not ks: continue
                    d105=np.mean([np.mean(sp[k]['f105'])-np.mean(b0[k]['f105']) for k in ks])
                    dbest=np.mean([np.mean(sp[k]['fbest'])-np.mean(b0[k]['fbest']) for k in ks])
                    dap0=np.mean([np.mean(sp[k]['ap'])-np.mean(b0[k]['ap']) for k in ks])
                    dap1=np.mean([np.mean(sp[k]['ap'])-np.mean(b1[k]['ap']) for k in ks])
                    with open(out,'a',newline='') as f: csv.writer(f).writerow([ds,cn,seed,round(d105,4),round(dbest,4),round(dap0,4),round(dap1,4)])
                    print(f'[B] {ds:14s} {cn:10s} seed={seed} dF1@0.5={d105:+.3f} dF1@best={dbest:+.3f} dAP_int={dap0:+.3f} dAP_itr={dap1:+.3f} ({time.time()-t:.0f}s)',flush=True)
                except Exception as e: print(f'[B] {ds} {cn} {seed} ERR {e}',flush=True)
    print('FULL_THRESHOLD DONE',flush=True)

# ---------- Part A (imbalance ladder) ----------
NFA=5; RATES=[0.05,0.10,0.20,0.33]; MINPOS=2*NFA; MINTOT=200
def set_rate(X,y,r,rng):
    pos=np.where(y==1)[0]; neg=np.where(y==0)[0]; np_,nn=len(pos),len(neg)
    if np_==0 or nn==0: return None
    need_neg=int(round(np_*(1-r)/r))
    if need_neg<=nn: keep_pos=pos; keep_neg=rng.choice(neg,need_neg,replace=False)
    else: need_pos=int(round(nn*r/(1-r))); keep_pos=rng.choice(pos,need_pos,replace=False); keep_neg=neg
    idx=np.concatenate([keep_pos,keep_neg]); rng.shuffle(idx); Xy=(X[idx],y[idx])
    if Xy[1].sum()<MINPOS or len(idx)<MINTOT: return None
    return Xy
def a_specific(pf,fac,seed):
    out={}
    for k,(X,y) in pf.items():
        if y.sum()<NFA: out[k]=np.nan;continue
        ap=[]
        for tr,te in StratifiedKFold(NFA,shuffle=True,random_state=seed).split(X,y):
            s=StandardScaler();c=fac();c.fit(s.fit_transform(X[tr]),y[tr]);ap.append(average_precision_score(y[te],sco(c,s.transform(X[te]))))
        out[k]=np.mean(ap)
    return out
def a_pooled(pf,fac,seed):
    order=list(pf);ns=len(order);Xs=[];ys=[];idx=[]
    for si,k in enumerate(order):
        X,y=pf[k];Xs.append(X);ys.append(y);idx.append(np.full(len(X),si))
    Xa=np.vstack(Xs);ya=np.concatenate(ys);sidx=np.concatenate(idx);out={k:[] for k in order}
    for tr,te in StratifiedKFold(NFA,shuffle=True,random_state=seed).split(Xa,sidx):
        sc=StandardScaler();Ftr=sc.fit_transform(Xa[tr]);Fte=sc.transform(Xa[te]);c=fac();c.fit(aug(Ftr,sidx[tr],ns,False),ya[tr]);pr=sco(c,aug(Fte,sidx[te],ns,False))
        for si,k in enumerate(order):
            m=sidx[te]==si
            if m.sum() and ya[te][m].sum()>0: out[k].append(average_precision_score(ya[te][m],pr[m]))
    return {k:(np.mean(v) if v else np.nan) for k,v in out.items()}
def gap_at_rate(base,fac,seed,r):
    rng=np.random.RandomState(seed*1000+int(r*100));pf={}
    for k,(X,y) in base.items():
        z=set_rate(X,y,r,rng)
        if z is not None: pf[k]=z
    if len(pf)<2: return np.nan
    sp=a_specific(pf,fac,seed);b0=a_pooled(pf,fac,seed)
    ks=[k for k in pf if not np.isnan(sp[k]) and not np.isnan(b0[k])]
    return float(np.mean([sp[k]-b0[k] for k in ks])) if ks else np.nan
def run_imbalance():
    out=DATA/'full_imbalance.csv'
    if not out.exists():
        with open(out,'w',newline='') as f: csv.writer(f).writerow(['dataset','classifier','rate','seed','gap_intercept'])
    done=_done_keys(out,4)
    afac=lambda seed:{'LogReg':lambda:LogisticRegression(class_weight='balanced',max_iter=2000,random_state=seed),'RF':lambda:RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=seed,n_jobs=-1)}
    for ds,ld in LOADERS.items():
        for cn in ['LogReg','RF']:
            for r in RATES:
                for seed in SEEDS:
                    if (ds,cn,str(r),str(seed)) in done: print(f'[A] skip {ds} {cn} r={r} {seed} (cached)',flush=True); continue
                    t=time.time();base=ld(seed);g=gap_at_rate(base,afac(seed)[cn],seed,r)
                    with open(out,'a',newline='') as f: csv.writer(f).writerow([ds,cn,r,seed,('' if np.isnan(g) else round(g,4))])
                    print(f'[A] {ds:14s} {cn:7s} r={r:.2f} seed={seed} gap={g:+.3f} ({time.time()-t:.0f}s)',flush=True)
    print('\n[A] per-(dataset,clf) Spearman(rate,gap):',flush=True)
    import collections; by=collections.defaultdict(list)
    dfm=pd.read_csv(out)
    for _,row in dfm.iterrows():
        if pd.notna(row['gap_intercept']): by[(row['dataset'],row['classifier'])].append((float(row['rate']),float(row['gap_intercept'])))
    for (ds,cn),v in by.items():
        agg=collections.defaultdict(list)
        for r,g in v: agg[r].append(g)
        pts=sorted((r,np.mean(gl)) for r,gl in agg.items())
        if len(pts)>=3: rho,p=stats.spearmanr([a for a,_ in pts],[b for _,b in pts]); print(f'   {ds:14s} {cn:7s} rho={rho:+.2f} gaps={[round(b,3) for _,b in pts]}',flush=True)
    print('FULL_IMBALANCE DONE',flush=True)

if __name__=='__main__':
    print('### FULL RERUN START',flush=True)
    run_phase1()    # headline first
    run_params()    # cheap
    run_threshold()
    run_imbalance()
    print('### FULL RERUN ALL DONE',flush=True)
