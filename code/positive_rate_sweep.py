"""W4 de-confounder: within-SmellyCode++ positive-rate sweep.
Fix positives per smell (controls sample size); undersample negatives to hit
target positive rates; run smell-specific vs multi-class at each rate, in both
F1 and PR-AUC. Holds language/metrics/labels fixed, varies ONLY positive rate."""
import numpy as np, pandas as pd, warnings, time, csv
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score
NF,SEED,P_PER=10,42,1000
rng=np.random.RandomState(SEED)
HAL=['Logical Lines','Distinct Operators','Distinct Operands','Total Operators','Total Operands','Vocabulary','Length','Calculated Length','Volume','Difficulty','Effort','Time Required','Bugs','Cyclomatic Complexity']
SM={'god_class':'God class','long_method':'Long method','feature_envy':'Feature envy','data_class':'Data class'}
RATES=[0.03,0.05,0.10,0.20,0.33]
def rf(): return RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=SEED,n_jobs=-1)
DATA=Path(__file__).resolve().parent.parent/'data'
df=pd.read_csv(DATA/'SmellyCode++.csv',usecols=HAL+list(SM.values()))
Xall=np.nan_to_num(df[HAL].values,nan=0.0)
Y={k:(df[c]==1).astype(int).values for k,c in SM.items()}

def subsample(k,rate):
    y=Y[k];pos=np.where(y==1)[0];neg=np.where(y==0)[0]
    npos=min(P_PER,len(pos));nneg=int(round(npos*(1-rate)/rate))
    if nneg>len(neg): nneg=len(neg)
    pi=rng.choice(pos,npos,replace=False);ni=rng.choice(neg,nneg,replace=False)
    idx=np.concatenate([pi,ni]);return Xall[idx],np.concatenate([np.ones(npos),np.zeros(nneg)]).astype(int)

def evalrate(rate):
    perfile={k:subsample(k,rate) for k in SM}
    order=list(SM)
    sF={};sA={}
    for k in order:
        Xk,yk=perfile[k];skf=StratifiedKFold(NF,shuffle=True,random_state=SEED);fs=[];ap=[]
        for tr,te in skf.split(Xk,yk):
            sc=StandardScaler();Xtr=sc.fit_transform(Xk[tr]);Xte=sc.transform(Xk[te])
            c=rf();c.fit(Xtr,yk[tr]);fs.append(f1_score(yk[te],c.predict(Xte),zero_division=0));ap.append(average_precision_score(yk[te],c.predict_proba(Xte)[:,1]))
        sF[k]=np.mean(fs);sA[k]=np.mean(ap)
    ns=len(order);Xl=[];yl=[];sl=[]
    for si,k in enumerate(order):
        Xk,yk=perfile[k];oh=np.zeros(ns);oh[si]=1
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
    specF=np.mean([sF[k] for k in order]);mcF=np.mean([np.mean(mF[k]) for k in order])
    return specF,mcF,dF1,dAP

out=DATA/'positive_rate_sweep_2026-06-24.csv'
with open(out,'w',newline='') as f:
    w=csv.writer(f);w.writerow(['target_rate','specific_f1','multiclass_f1','delta_f1','delta_prauc']);f.flush()
    for r in RATES:
        t=time.time();sf,mf,dF1,dAP=evalrate(r)
        w.writerow([r,round(sf,4),round(mf,4),round(dF1,4),round(dAP,4)]);f.flush()
        print(f'rate={r:.0%}  specF1={sf:.3f} mcF1={mf:.3f}  dF1={dF1:+.3f}  dPR-AUC={dAP:+.3f}  ({time.time()-t:.0f}s)',flush=True)
print('SAVED',out,flush=True)
