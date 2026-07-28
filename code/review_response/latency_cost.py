"""Engineering trade-off measurements (Information round 1, R1 comment 2).
For each dataset (LogReg, seed 42): train time and inference latency of the
K specialized models vs the single interaction-pooled model.

Accounting is per full task sweep: to produce scores for all K tasks,
- the specific arm runs K predict_proba calls on an n x d matrix;
- the pooled arm must score every (instance, task) pair: one predict_proba call
  on an (n*K) x (d+K+dK) augmented matrix (built once, build time included).
Latency reported per 10k instances (all K tasks), median of 5 repeats.
Output: latency_cost.csv
"""
import csv,time,warnings,numpy as np
warnings.filterwarnings('ignore')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import _loaders as L
from revfix import aug

SEED=42
mk=lambda:LogisticRegression(class_weight='balanced',max_iter=2000,random_state=SEED)

ALL=[('IST2021',L.L_ist),('ImprovMLCQ',L.L_improv),('Crowdsmelling',L.L_crowd),
     ('SmellyCode++',L.L_smelly),('ml-Codesmell',L.L_mlcs),
     ('digits',lambda s:L.L_mc('digits',s,5)),('segment',lambda s:L.L_mc('segment',s,5)),
     ('vehicle',lambda s:L.L_mc('vehicle',s,5)),('letter',lambda s:L.L_mc('letter',s,5))]

def median_time(fn,reps=5):
    ts=[]
    for _ in range(reps):
        t=time.perf_counter(); fn(); ts.append(time.perf_counter()-t)
    return float(np.median(ts))

def run(ds,loader):
    pf=loader(SEED); order=list(pf); K=len(order)
    # shared scaler per arm (fit on the arm's own training representation)
    # --- specific arm: K models ---
    models={}; t0=time.perf_counter()
    scalers={}
    for k in order:
        X,y=pf[k]; s=StandardScaler(); Xs=s.fit_transform(X)
        c=mk(); c.fit(Xs,y); models[k]=c; scalers[k]=s
    t_train_spec=time.perf_counter()-t0
    # --- pooled-interaction arm: 1 model on stacked rows ---
    Xs_=[];ya=[];si=[]
    for i,k in enumerate(order):
        X,y=pf[k]; Xs_.append(X); ya.append(y); si.append(np.full(len(X),i))
    Xa=np.vstack(Xs_); ya=np.concatenate(ya); sidx=np.concatenate(si)
    t0=time.perf_counter()
    sc=StandardScaler(); Fa=sc.fit_transform(Xa)
    cP=mk(); cP.fit(aug(Fa,sidx,K,True),ya)
    t_train_pool=time.perf_counter()-t0
    # --- inference: score all K tasks for a 10k batch (or n if smaller) ---
    n=min(10000,len(pf[order[0]][0])); Xq=pf[order[0]][0][:n]
    def infer_spec():
        for k in order: models[k].predict_proba(scalers[k].transform(Xq))
    def infer_pool():
        Fq=sc.transform(np.vstack([Xq]*K)); sq=np.repeat(np.arange(K),n)
        cP.predict_proba(aug(Fq,sq,K,True))
    l_spec=median_time(infer_spec); l_pool=median_time(infer_pool)
    d=pf[order[0]][0].shape[1]
    return K,d,n,t_train_spec,t_train_pool,l_spec,l_pool

if __name__=='__main__':
    out=L.DATA/'latency_cost.csv'
    with open(out,'w',newline='') as f:
        csv.writer(f).writerow(['dataset','K','d','batch_n','train_s_specific','train_s_pooled_itr','infer_s_specific','infer_s_pooled_itr'])
    for ds,ld in ALL:
        try:
            K,d,n,ts,tp,ls,lp=run(ds,ld)
            with open(out,'a',newline='') as f:
                csv.writer(f).writerow([ds,K,d,n,round(ts,3),round(tp,3),round(ls,4),round(lp,4)])
            print(f'{ds:14s} K={K:2d} d={d:3d} train spec={ts:6.2f}s pool={tp:6.2f}s | infer/10k spec={ls*1000:7.1f}ms pool={lp*1000:7.1f}ms',flush=True)
        except Exception as e:
            print(f'{ds} ERR {e}',flush=True)
    print('LATENCY DONE',flush=True)
