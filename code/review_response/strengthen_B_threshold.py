"""STRENGTHEN B -- are F1@0.5 architecture differences threshold artifacts?
Extends the single-seed R3 to 5 seeds x 4 classifiers on the 5 code-smell datasets.
For each (dataset,classifier) reports, averaged over tasks then over seeds (mean +/- 95% CI):
  dF1@0.5     = F1(specific) - F1(intercept-pooled) at the default 0.5 threshold
  dF1@best    = same but each arm at its own F1-optimal threshold
  dPR-AUC_int = PR-AUC(specific) - PR-AUC(intercept-pooled)
  dPR-AUC_itr = PR-AUC(specific) - PR-AUC(interaction-pooled)
If dF1@0.5 shrinks toward dF1@best, the 0.5 gap is a thresholding artifact; any
residual ranking gap (dPR-AUC_int) is the intercept-baseline gap, which dPR-AUC_itr
shows the interaction baseline removes."""
import csv,time,warnings,numpy as np
warnings.filterwarnings('ignore')
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve
import _loaders as L

NF=10; SEEDS=[42,1,7,13,99]
def facs(seed):
    return {'LogReg':lambda:LogisticRegression(class_weight='balanced',max_iter=2000,random_state=seed),
            'LinearSVC':lambda:LinearSVC(class_weight='balanced',max_iter=5000,random_state=seed),
            'RF':lambda:RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=seed,n_jobs=-1),
            'HistGB':lambda:HistGradientBoostingClassifier(class_weight='balanced',random_state=seed)}
def sco(c,X): return c.predict_proba(X)[:,1] if hasattr(c,'predict_proba') else c.decision_function(X)
def bestf1(y,s):
    p,r,th=precision_recall_curve(y,s); f1=2*p*r/(p+r+1e-12); return float(np.nanmax(f1))

def specific(pf,fac,seed):
    o={k:{'f105':[],'fbest':[],'ap':[]} for k in pf}
    for k,(X,y) in pf.items():
        if y.sum()<NF: o[k]=None;continue
        for tr,te in StratifiedKFold(NF,shuffle=True,random_state=seed).split(X,y):
            s=StandardScaler();Xtr=s.fit_transform(X[tr]);Xte=s.transform(X[te]);c=fac();c.fit(Xtr,y[tr])
            pr=sco(c,Xte);o[k]['f105'].append(f1_score(y[te],c.predict(Xte),zero_division=0));o[k]['fbest'].append(bestf1(y[te],pr));o[k]['ap'].append(average_precision_score(y[te],pr))
    return o
def pooled(pf,fac,seed,inter):
    order=list(pf);ns=len(order);Xs=[];ys=[];idx=[]
    for si,k in enumerate(order):
        X,y=pf[k];Xs.append(X);ys.append(y);idx.append(np.full(len(X),si))
    Xa=np.vstack(Xs);ya=np.concatenate(ys);sidx=np.concatenate(idx);o={k:{'f105':[],'fbest':[],'ap':[]} for k in order}
    for tr,te in StratifiedKFold(NF,shuffle=True,random_state=seed).split(Xa,sidx):
        sc=StandardScaler();Ftr=sc.fit_transform(Xa[tr]);Fte=sc.transform(Xa[te])
        def aug(F,si):
            oh=np.zeros((len(F),ns));oh[np.arange(len(F)),si]=1
            return np.hstack([F,oh]) if not inter else np.hstack([F,oh,np.einsum('ij,ik->ijk',F,oh).reshape(len(F),-1)])
        c=fac();c.fit(aug(Ftr,sidx[tr]),ya[tr]);pr=sco(c,aug(Fte,sidx[te]));pred=c.predict(aug(Fte,sidx[te]))
        for si,k in enumerate(order):
            m=sidx[te]==si
            if m.sum() and ya[te][m].sum()>0:
                o[k]['f105'].append(f1_score(ya[te][m],pred[m],zero_division=0));o[k]['fbest'].append(bestf1(ya[te][m],pr[m]));o[k]['ap'].append(average_precision_score(ya[te][m],pr[m]))
    return o

def deltas(sp,b0,b1):
    ks=[k for k in sp if sp[k] and sp[k]['ap'] and b0[k]['ap'] and b1[k]['ap']]
    if not ks: return None
    d105=np.mean([np.mean(sp[k]['f105'])-np.mean(b0[k]['f105']) for k in ks])
    dbest=np.mean([np.mean(sp[k]['fbest'])-np.mean(b0[k]['fbest']) for k in ks])
    dap0=np.mean([np.mean(sp[k]['ap'])-np.mean(b0[k]['ap']) for k in ks])
    dap1=np.mean([np.mean(sp[k]['ap'])-np.mean(b1[k]['ap']) for k in ks])
    return d105,dbest,dap0,dap1

out=L.DATA/'strengthen_B_threshold_2026-06-24.csv'
with open(out,'w',newline='') as f: csv.writer(f).writerow(['dataset','classifier','seed','dF1@0.5','dF1@best','dPR-AUC_int','dPR-AUC_itr'])
agg={}
for ds,ld in L.CODE_LOADERS.items():
    for cn in ['LogReg','LinearSVC','RF','HistGB']:
        acc=[]
        for seed in SEEDS:
            t=time.time();pf=ld(seed);fac=facs(seed)[cn]
            try:
                sp=specific(pf,fac,seed);b0=pooled(pf,fac,seed,False);b1=pooled(pf,fac,seed,True)
                d=deltas(sp,b0,b1)
                if d is None: continue
                acc.append(d)
                with open(out,'a',newline='') as f: csv.writer(f).writerow([ds,cn,seed]+[round(x,4) for x in d])
                print(f'{ds:14s} {cn:10s} seed={seed} dF1@0.5={d[0]:+.3f} dF1@best={d[1]:+.3f} dAP_int={d[2]:+.3f} dAP_itr={d[3]:+.3f} ({time.time()-t:.0f}s)',flush=True)
            except Exception as e: print(f'{ds} {cn} {seed} ERR {e}',flush=True)
        if acc: agg[(ds,cn)]=np.array(acc)
print('\n=== mean +/- 95% CI over seeds ===',flush=True)
def ci(a):
    m=a.mean();h=stats.sem(a)*stats.t.ppf(0.975,len(a)-1) if len(a)>1 else 0.0;return m,h
for (ds,cn),a in agg.items():
    labs=['dF1@0.5','dF1@best','dAP_int','dAP_itr']
    s=' '.join(f'{labs[i]}={ci(a[:,i])[0]:+.3f}+/-{ci(a[:,i])[1]:.3f}' for i in range(4))
    print(f'{ds:14s} {cn:10s} {s}',flush=True)
print('STRENGTHEN_B DONE',flush=True)
