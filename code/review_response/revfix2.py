"""Reviewer-fix round 2:
 PART S (#M-B): add the literal softmax/multinomial multi-class baseline on the
   genuine multi-class datasets and compare specific(OvR) vs softmax vs
   intercept-pooled vs interaction-pooled. Softmax has per-class weights, so it
   should behave like specific (no artifact); only the shared-weight one-hot
   pooled baseline should be deficient. This pins down *which* 'multi-class'
   construction is unfair. Matched folds on the original multi-class target.
 PART T (#m-1): regenerate the threshold decomposition with MATCHED folds
   (15k sample for the two large datasets, like the imbalance sweep).
Outputs: revfix_softmax.csv, revfix_threshold.csv. Resumable."""
import csv,time,warnings,numpy as np,pandas as pd
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve
import revfix as R   # reuse aug, fold_assign, specific_matched, pooled_matched, facs, loaders
DATA=R.DATA; SEEDS=R.SEEDS

# ============ PART S: softmax vs OvR vs pooled (multi-class datasets) ============
def load_mc_raw(name,seed):
    d=np.load(R.L.ML/f'{name}.npz'); X=d['X'].astype(float); y=d['y'].astype(int)
    cap=R.L.MC_CAP.get(name)
    if cap and len(X)>cap:
        rng=np.random.RandomState(seed); ix=rng.choice(len(X),cap,replace=False); X=X[ix]; y=y[ix]
    return X,y
def sco(c,X): return c.predict_proba(X)[:,1]
def run_softmax():
    out=DATA/'revfix_softmax.csv'; NF=5
    if not out.exists():
        with open(out,'w',newline='') as f: csv.writer(f).writerow(['dataset','seed','specAP','softmaxAP','interceptAP','interactionAP','d_softmax','d_intercept','d_interaction'])
    done=R._done(out,2)
    for ds in R.L.MC_DSETS:
        for seed in SEEDS:
            if (ds,str(seed)) in done: print(f'[S] skip {ds} {seed}',flush=True); continue
            t=time.time(); X,y=load_mc_raw(ds,seed); classes=[c for c in sorted(set(y)) if (y==c).sum()>=2*NF]
            spec={c:[] for c in classes}; soft={c:[] for c in classes}; i0={c:[] for c in classes}; i1={c:[] for c in classes}
            for tr,te in StratifiedKFold(NF,shuffle=True,random_state=seed).split(X,y):
                s=StandardScaler(); Xtr=s.fit_transform(X[tr]); Xte=s.transform(X[te])
                # softmax (per-class weights)
                sm=LogisticRegression(multi_class='multinomial',class_weight='balanced',max_iter=2000,random_state=seed).fit(Xtr,y[tr])
                P=sm.predict_proba(Xte); smc={c:P[:,list(sm.classes_).index(c)] for c in classes}
                # specific OvR
                for c in classes:
                    ytr=(y[tr]==c).astype(int); yte=(y[te]==c).astype(int)
                    if ytr.sum()==0 or yte.sum()==0: continue
                    cl=LogisticRegression(class_weight='balanced',max_iter=2000,random_state=seed).fit(Xtr,ytr)
                    spec[c].append(average_precision_score(yte,sco(cl,Xte)))
                    soft[c].append(average_precision_score(yte,smc[c]))
                # pooled intercept/interaction (one-hot multi-task on OvR tasks)
                ns=len(classes)
                Xs=[];ya=[];si=[]
                for ci,c in enumerate(classes):
                    Xs.append(Xtr); ya.append((y[tr]==c).astype(int)); si.append(np.full(len(Xtr),ci))
                Xa=np.vstack(Xs); yaa=np.concatenate(ya); sidx=np.concatenate(si)
                for inter,dst in [(False,i0),(True,i1)]:
                    cl=LogisticRegression(class_weight='balanced',max_iter=2000,random_state=seed).fit(R.aug(Xa,sidx,ns,inter),yaa)
                    for ci,c in enumerate(classes):
                        yte=(y[te]==c).astype(int)
                        if yte.sum()==0: continue
                        oh=np.zeros((len(Xte),ns)); oh[:,ci]=1
                        Xte_aug=R.aug(Xte,np.full(len(Xte),ci),ns,inter)
                        dst[c].append(average_precision_score(yte,cl.predict_proba(Xte_aug)[:,1]))
            ks=[c for c in classes if spec[c] and soft[c] and i0[c] and i1[c]]
            sA=np.mean([np.mean(spec[c]) for c in ks]); sm_=np.mean([np.mean(soft[c]) for c in ks])
            a0=np.mean([np.mean(i0[c]) for c in ks]); a1=np.mean([np.mean(i1[c]) for c in ks])
            with open(out,'a',newline='') as f: csv.writer(f).writerow([ds,seed,round(sA,4),round(sm_,4),round(a0,4),round(a1,4),round(sA-sm_,4),round(sA-a0,4),round(sA-a1,4)])
            print(f'[S] {ds:8s} seed={seed} spec={sA:.3f} softmax={sm_:.3f} (d={sA-sm_:+.3f}) intercept d={sA-a0:+.3f} interaction d={sA-a1:+.3f} ({time.time()-t:.0f}s)',flush=True)
    print('REVFIX_SOFTMAX DONE',flush=True)

# ============ PART T: matched-fold threshold decomposition ============
def bestf1(y,s):
    p,r,th=precision_recall_curve(y,s); f1=2*p*r/(p+r+1e-12); return float(np.nanmax(f1))
def scod(c,X): return c.predict_proba(X)[:,1] if hasattr(c,'predict_proba') else c.decision_function(X)
def thresh_cell(pf,fa,fac,NF):
    # specific
    order=[k for k in pf if not (fa[k]<0).all()]
    sp={k:{'f105':[],'fbest':[],'ap':[]} for k in order}
    for k in order:
        X,y=pf[k]
        for f in range(NF):
            te=fa[k]==f; tr=(fa[k]>=0)&(~te)
            if y[te].sum()==0 or y[tr].sum()==0: continue
            s=StandardScaler(); Xtr=s.fit_transform(X[tr]); Xte=s.transform(X[te]); c=fac(); c.fit(Xtr,y[tr]); pr=scod(c,Xte)
            sp[k]['f105'].append(f1_score(y[te],c.predict(Xte),zero_division=0)); sp[k]['fbest'].append(bestf1(y[te],pr)); sp[k]['ap'].append(average_precision_score(y[te],pr))
    # pooled intercept & interaction, matched
    ns=len(order); Xs=[];ya=[];si=[];fo=[]
    for i,k in enumerate(order):
        X,y=pf[k]; Xs.append(X); ya.append(y); si.append(np.full(len(X),i)); fo.append(fa[k])
    Xa=np.vstack(Xs); yaa=np.concatenate(ya); sidx=np.concatenate(si); fold=np.concatenate(fo)
    res={}
    for inter in [False,True]:
        mp={k:{'f105':[],'fbest':[],'ap':[]} for k in order}
        for f in range(NF):
            te=fold==f; tr=(fold>=0)&(~te)
            if te.sum()==0 or tr.sum()==0: continue
            sc=StandardScaler(); Ftr=sc.fit_transform(Xa[tr]); Fte=sc.transform(Xa[te]); c=fac(); At=R.aug(Fte,sidx[te],ns,inter)
            c.fit(R.aug(Ftr,sidx[tr],ns,inter),yaa[tr]); pr=scod(c,At); pred=c.predict(At)
            for i,k in enumerate(order):
                m=sidx[te]==i
                if m.sum() and yaa[te][m].sum()>0:
                    mp[k]['f105'].append(f1_score(yaa[te][m],pred[m],zero_division=0)); mp[k]['fbest'].append(bestf1(yaa[te][m],pr[m])); mp[k]['ap'].append(average_precision_score(yaa[te][m],pr[m]))
        res[inter]=mp
    ks=[k for k in order if sp[k]['ap'] and res[False][k]['ap'] and res[True][k]['ap']]
    if not ks: return None
    d105=np.mean([np.mean(sp[k]['f105'])-np.mean(res[False][k]['f105']) for k in ks])
    dbest=np.mean([np.mean(sp[k]['fbest'])-np.mean(res[False][k]['fbest']) for k in ks])
    dap0=np.mean([np.mean(sp[k]['ap'])-np.mean(res[False][k]['ap']) for k in ks])
    dap1=np.mean([np.mean(sp[k]['ap'])-np.mean(res[True][k]['ap']) for k in ks])
    return d105,dbest,dap0,dap1
def run_threshold():
    out=DATA/'revfix_threshold.csv'; NF=10
    if not out.exists():
        with open(out,'w',newline='') as f: csv.writer(f).writerow(['dataset','classifier','seed','dF1@0.5','dF1@best','dPR-AUC_int','dPR-AUC_itr'])
    done=R._done(out,3)
    # threshold uses 15k for the two big datasets (secondary analysis, like imbalance)
    ld15={'SmellyCode++':R.L.L_smelly,'ml-Codesmell':R.L.L_mlcs}
    CODE=[('IST2021',R.L.L_ist),('ImprovMLCQ',R.L.L_improv),('Crowdsmelling',R.L.L_crowd),('SmellyCode++',R.L.L_smelly),('ml-Codesmell',R.L.L_mlcs)]
    for ds,ld in CODE:
        for cn in ['LogReg','RF','HistGB','LinearSVC']:
            for seed in SEEDS:
                if (ds,cn,str(seed)) in done: print(f'[T] skip {ds} {cn} {seed}',flush=True); continue
                t=time.time(); pf=ld(seed); fa=R.fold_assign(pf,seed,NF); fac=R.facs(seed)[cn]
                try:
                    r=thresh_cell(pf,fa,fac,NF)
                    if r is None: continue
                    with open(out,'a',newline='') as f: csv.writer(f).writerow([ds,cn,seed]+[round(x,4) for x in r])
                    print(f'[T] {ds:14s} {cn:10s} seed={seed} dF1@0.5={r[0]:+.3f} dF1@best={r[1]:+.3f} dAP_int={r[2]:+.3f} dAP_itr={r[3]:+.3f} ({time.time()-t:.0f}s)',flush=True)
                except Exception as e: print(f'[T] {ds} {cn} {seed} ERR {e}',flush=True)
    print('REVFIX_THRESHOLD DONE',flush=True)

if __name__=='__main__':
    print('### REVFIX2 START',flush=True)
    run_softmax()
    run_threshold()
    print('### REVFIX2 ALL DONE',flush=True)
