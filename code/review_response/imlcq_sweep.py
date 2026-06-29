import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score
NF,SEED,P=10,42,500
rng=np.random.RandomState(SEED)
def rf(): return RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=SEED,n_jobs=-1)
df=pd.read_csv('data/ImprovMLCQ.csv')
ck=sorted([c for c in df.columns if c.startswith('ck_')])
Xall=np.nan_to_num(df[ck].values,nan=0.0)
SM={'blob':'blob_label','data_class':'dataclass_label','feature_envy':'featureenvy_label','long_method':'longmethod_label'}
Y={k:(df[c]==1).astype(int).values for k,c in SM.items()}
for k in SM: print(f'  {k}: {Y[k].sum()} positives', flush=True)
def subsample(k,rate):
    y=Y[k];pos=np.where(y==1)[0];neg=np.where(y==0)[0]
    npos=min(P,len(pos));nneg=int(round(npos*(1-rate)/rate))
    if nneg>len(neg): nneg=len(neg)
    pi=rng.choice(pos,npos,replace=False);ni=rng.choice(neg,nneg,replace=False)
    idx=np.concatenate([pi,ni]);return Xall[idx],np.concatenate([np.ones(npos),np.zeros(nneg)]).astype(int),npos/(npos+nneg)
def evalrate(rate):
    order=list(SM);pf={k:subsample(k,rate) for k in order};actual=np.mean([pf[k][2] for k in order])
    sF={};sA={}
    for k in order:
        Xk,yk,_=pf[k];skf=StratifiedKFold(NF,shuffle=True,random_state=SEED);fs=[];ap=[]
        for tr,te in skf.split(Xk,yk):
            sc=StandardScaler();Xtr=sc.fit_transform(Xk[tr]);Xte=sc.transform(Xk[te])
            c=rf();c.fit(Xtr,yk[tr]);fs.append(f1_score(yk[te],c.predict(Xte),zero_division=0));ap.append(average_precision_score(yk[te],c.predict_proba(Xte)[:,1]))
        sF[k]=np.mean(fs);sA[k]=np.mean(ap)
    ns=len(order);Xl=[];yl=[];sl=[]
    for si,k in enumerate(order):
        Xk,yk,_=pf[k];oh=np.zeros(ns);oh[si]=1
        Xl.append(np.hstack([Xk,np.tile(oh,(len(Xk),1))]));yl.append(yk);sl.append(np.array([k]*len(Xk)))
    Xc=np.vstack(Xl);yc=np.concatenate(yl);scl=np.concatenate(sl);strat=LabelEncoder().fit_transform(scl);n=Xall.shape[1]
    mF={k:[] for k in order};mA={k:[] for k in order}
    for tr,te in StratifiedKFold(NF,shuffle=True,random_state=SEED).split(Xc,strat):
        Xtr,Xte=Xc[tr].copy(),Xc[te].copy();s=StandardScaler();Xtr[:,:n]=s.fit_transform(Xtr[:,:n]);Xte[:,:n]=s.transform(Xte[:,:n])
        c=rf();c.fit(Xtr,yc[tr]);p=c.predict(Xte);pr=c.predict_proba(Xte)[:,1];ts=scl[te]
        for k in order:
            m=ts==k
            if m.sum(): mF[k].append(f1_score(yc[te][m],p[m],zero_division=0));mA[k].append(average_precision_score(yc[te][m],pr[m]))
    dF1=np.mean([sF[k]-np.mean(mF[k]) for k in order]);dAP=np.mean([sA[k]-np.mean(mA[k]) for k in order])
    return actual,np.mean([sF[k] for k in order]),np.mean([np.mean(mF[k]) for k in order]),dF1,dAP
print('\nImprovMLCQ within-dataset sweep (CK features, 4 smells; fix P=500, vary neg):')
for r in [0.05,0.10,0.20,0.33]:
    t=time.time();a,sf,mf,dF1,dAP=evalrate(r)
    print(f'target={r:.0%} actual={a:.1%}  specF1={sf:.3f} mcF1={mf:.3f}  dF1={dF1:+.3f}  dPR-AUC={dAP:+.3f}  ({time.time()-t:.0f}s)',flush=True)
