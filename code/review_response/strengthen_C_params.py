"""STRENGTHEN C -- 'specialize for free': parameters, train time, accuracy parity.
For every dataset (5 code-smell + 4 multi-class), with LogReg, compares the K
specialized models against the single interaction-pooled model that matches them
in accuracy. Reports total parameter count, wall-clock train time, and confirms
PR-AUC parity (mean over tasks). Specialized = modular/independently deployable;
interaction-pooled only catches up by growing to ~the same parameter budget."""
import csv,time,warnings,numpy as np
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
import _loaders as L

NF=5; SEED=42
def lr(): return LogisticRegression(class_weight='balanced',max_iter=2000,random_state=SEED)
def nparams(models):  # logistic: coef_ + intercept_ per model
    return int(sum(m.coef_.size+m.intercept_.size for m in models))

def specific(pf):
    t0=time.time();models=[];aps=[]
    for k,(X,y) in pf.items():
        if y.sum()<NF: continue
        ap=[]
        for tr,te in StratifiedKFold(NF,shuffle=True,random_state=SEED).split(X,y):
            s=StandardScaler();c=lr();c.fit(s.fit_transform(X[tr]),y[tr]);ap.append(average_precision_score(y[te],c.predict_proba(s.transform(X[te]))[:,1]))
        # one final full-data model for the deployed param count
        s=StandardScaler();c=lr();c.fit(s.fit_transform(X),y);models.append(c);aps.append(np.mean(ap))
    return nparams(models),time.time()-t0,float(np.mean(aps))

def interaction_pooled(pf):
    order=list(pf);ns=len(order);Xs=[];ys=[];idx=[]
    for si,k in enumerate(order):
        X,y=pf[k];Xs.append(X);ys.append(y);idx.append(np.full(len(X),si))
    Xa=np.vstack(Xs);ya=np.concatenate(ys);sidx=np.concatenate(idx)
    def aug(F,si):
        oh=np.zeros((len(F),ns));oh[np.arange(len(F)),si]=1
        return np.hstack([F,oh,np.einsum('ij,ik->ijk',F,oh).reshape(len(F),-1)])
    t0=time.time();out={k:[] for k in order}
    for tr,te in StratifiedKFold(NF,shuffle=True,random_state=SEED).split(Xa,sidx):
        sc=StandardScaler();Ftr=sc.fit_transform(Xa[tr]);Fte=sc.transform(Xa[te]);c=lr();c.fit(aug(Ftr,sidx[tr]),ya[tr]);pr=c.predict_proba(aug(Fte,sidx[te]))[:,1]
        for si,k in enumerate(order):
            m=sidx[te]==si
            if m.sum() and ya[te][m].sum()>0: out[k].append(average_precision_score(ya[te][m],pr[m]))
    sc=StandardScaler();c=lr();c.fit(aug(sc.fit_transform(Xa),sidx),ya);npar=int(c.coef_.size+c.intercept_.size)
    aps=[np.mean(v) for v in out.values() if v]
    return npar,time.time()-t0,float(np.mean(aps))

ALL=[(n,lambda s,n=n:L.CODE_LOADERS[n](s)) for n in L.CODE_LOADERS]+[(n,lambda s,n=n:L.L_mc(n,s,NF)) for n in L.MC_DSETS]
out=L.DATA/'strengthen_C_params_2026-06-24.csv'
with open(out,'w',newline='') as f: csv.writer(f).writerow(['dataset','K_tasks','spec_params','itr_params','param_ratio','spec_train_s','itr_train_s','spec_PRAUC','itr_PRAUC','dPRAUC'])
for ds,ld in ALL:
    try:
        pf=ld(SEED);K=sum(1 for k,(X,y) in pf.items() if y.sum()>=NF)
        sp_p,sp_t,sp_a=specific(pf);it_p,it_t,it_a=interaction_pooled(pf)
        row=[ds,K,sp_p,it_p,round(it_p/max(sp_p,1),2),round(sp_t,2),round(it_t,2),round(sp_a,4),round(it_a,4),round(sp_a-it_a,4)]
        with open(out,'a',newline='') as f: csv.writer(f).writerow(row)
        print(f'{ds:14s} K={K} spec_params={sp_p} itr_params={it_p} ratio={it_p/max(sp_p,1):.2f}x  PRAUC spec={sp_a:.3f} itr={it_a:.3f} dPRAUC={sp_a-it_a:+.3f}',flush=True)
    except Exception as e: print(f'{ds} ERR {e}',flush=True)
print('STRENGTHEN_C DONE',flush=True)
