import numpy as np, pandas as pd, warnings, time, os
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score
NF,SEED=10,42; rng=np.random.RandomState(SEED)
def rf(): return RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=SEED,n_jobs=-1)
IST={'god_class':('GodClass.csv','is_god_class'),'data_class':('DataClass.csv','is_data_class'),'long_method':('LongMethod.csv','is_long_method'),'feature_envy':('FeatureEnvy.csv','is_feature_envy'),'long_parameter_list':('LongParameterList.csv','is_long_parameters_list'),'switch_statements':('SwitchStatements.csv','is_switch_statements')}
ID='data/IST2021';sets=[]
for k,(fn,tg) in IST.items():
    d=pd.read_csv(os.path.join(ID,fn));sets.append({c for c in d.columns if c!=tg})
common=sorted(set.intersection(*sets))
RAW={}
for k,(fn,tg) in IST.items():
    d=pd.read_csv(os.path.join(ID,fn));X=np.nan_to_num(d[common].values,nan=0.0)
    y=d[tg].apply(lambda v:1 if v in [True,'TRUE',1,'1'] else 0).values
    RAW[k]=(X,y)
def subs(k,rate):
    X,y=RAW[k];pos=np.where(y==1)[0];neg=np.where(y==0)[0]
    npos=int(round(len(neg)*rate/(1-rate)))
    if npos>len(pos): npos=len(pos); nneg=int(round(npos*(1-rate)/rate)); neg=rng.choice(neg,min(nneg,len(neg)),replace=False)
    pi=rng.choice(pos,npos,replace=False)
    idx=np.concatenate([pi,neg]); return X[idx], np.concatenate([np.ones(npos),np.zeros(len(neg))]).astype(int)
def evalrate(rate):
    order=list(IST); pf={k:subs(k,rate) for k in order}
    sF={};sA={}
    for k in order:
        X,y=pf[k]
        if y.sum()<6: 
            sF[k]=np.nan;sA[k]=np.nan;continue
        skf=StratifiedKFold(NF,shuffle=True,random_state=SEED);fs=[];ap=[]
        for tr,te in skf.split(X,y):
            sc=StandardScaler();Xtr=sc.fit_transform(X[tr]);Xte=sc.transform(X[te])
            c=rf();c.fit(Xtr,y[tr]);fs.append(f1_score(y[te],c.predict(Xte),zero_division=0));ap.append(average_precision_score(y[te],c.predict_proba(Xte)[:,1]))
        sF[k]=np.mean(fs);sA[k]=np.mean(ap)
    ns=len(order);Xl=[];yl=[];sl=[]
    for si,k in enumerate(order):
        X,y=pf[k];oh=np.zeros(ns);oh[si]=1
        Xl.append(np.hstack([X,np.tile(oh,(len(X),1))]));yl.append(y);sl.append(np.array([k]*len(X)))
    Xc=np.vstack(Xl);yc=np.concatenate(yl);scl=np.concatenate(sl);strat=LabelEncoder().fit_transform(scl);n=len(common)
    mF={k:[] for k in order};mA={k:[] for k in order}
    for tr,te in StratifiedKFold(NF,shuffle=True,random_state=SEED).split(Xc,strat):
        Xtr,Xte=Xc[tr].copy(),Xc[te].copy();s=StandardScaler();Xtr[:,:n]=s.fit_transform(Xtr[:,:n]);Xte[:,:n]=s.transform(Xte[:,:n])
        c=rf();c.fit(Xtr,yc[tr]);p=c.predict(Xte);pr=c.predict_proba(Xte)[:,1];ts=scl[te]
        for k in order:
            m=ts==k
            if m.sum(): mF[k].append(f1_score(yc[te][m],p[m],zero_division=0));mA[k].append(average_precision_score(yc[te][m],pr[m]))
    ks=[k for k in order if not np.isnan(sF[k])]
    dF1=np.mean([sF[k]-np.mean(mF[k]) for k in ks]);dAP=np.mean([sA[k]-np.mean(mA[k]) for k in ks])
    return np.mean([sF[k] for k in ks]),np.mean([np.mean(mF[k]) for k in ks]),dF1,dAP,len(ks)
print('IST2021 DOWN-sweep (undersample positives; common CK features):')
for r in [0.33,0.20,0.10,0.05]:
    t=time.time();sf,mf,dF1,dAP,nk=evalrate(r)
    print(f'rate={r:.0%}  specF1={sf:.3f} mcF1={mf:.3f}  dF1={dF1:+.3f}  dPR-AUC={dAP:+.3f}  ({nk} smells, {time.time()-t:.0f}s)',flush=True)
