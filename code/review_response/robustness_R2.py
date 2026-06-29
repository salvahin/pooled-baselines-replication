"""R2: within-dataset positive-rate sweeps x seeds x classifiers, with CIs.
IST2021 down-sweep (undersample positives) + SmellyCode++ up-sweep (vary neg)."""
import csv,time,warnings,numpy as np,pandas as pd
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score
NF=10; HERE=Path(__file__).resolve().parent; DATA=HERE.parent.parent/'data'
SEEDS=[42,1,7,13,99]
def facs(seed): return {'RF':lambda:RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=seed,n_jobs=-1),
                        'LogReg':lambda:LogisticRegression(class_weight='balanced',max_iter=1000,random_state=seed)}
def sco(c,X): return c.predict_proba(X)[:,1] if hasattr(c,'predict_proba') else c.decision_function(X)
def specmc(fac, perfile, n_feat, seed):
    order=list(perfile);sF={};sA={}
    for k in order:
        X,y=perfile[k]
        if y.sum()<NF: sF[k]=np.nan;sA[k]=np.nan;continue
        fs=[];ap=[]
        for tr,te in StratifiedKFold(NF,shuffle=True,random_state=seed).split(X,y):
            s=StandardScaler();Xtr=s.fit_transform(X[tr]);Xte=s.transform(X[te])
            c=fac();c.fit(Xtr,y[tr]);fs.append(f1_score(y[te],c.predict(Xte),zero_division=0));ap.append(average_precision_score(y[te],sco(c,Xte)))
        sF[k]=np.mean(fs);sA[k]=np.mean(ap)
    ns=len(order);Xl=[];yl=[];sl=[]
    for si,k in enumerate(order):
        X,y=perfile[k];oh=np.zeros(ns);oh[si]=1
        Xl.append(np.hstack([X,np.tile(oh,(len(X),1))]));yl.append(y);sl.append(np.array([k]*len(X)))
    Xc=np.vstack(Xl);yc=np.concatenate(yl);scl=np.concatenate(sl);strat=LabelEncoder().fit_transform(scl)
    mF={k:[] for k in order};mA={k:[] for k in order}
    for tr,te in StratifiedKFold(NF,shuffle=True,random_state=seed).split(Xc,strat):
        Xtr,Xte=Xc[tr].copy(),Xc[te].copy();s=StandardScaler();Xtr[:,:n_feat]=s.fit_transform(Xtr[:,:n_feat]);Xte[:,:n_feat]=s.transform(Xte[:,:n_feat])
        c=fac();c.fit(Xtr,yc[tr]);p=c.predict(Xte);pr=sco(c,Xte);ts=scl[te]
        for k in order:
            m=ts==k
            if m.sum(): mF[k].append(f1_score(yc[te][m],p[m],zero_division=0));mA[k].append(average_precision_score(yc[te][m],pr[m]))
    ks=[k for k in order if not np.isnan(sF[k]) and mF[k]]
    return np.mean([sF[k]-np.mean(mF[k]) for k in ks]), np.mean([sA[k]-np.mean(mA[k]) for k in ks])
# IST2021 loaders
IST={'god_class':('GodClass.csv','is_god_class'),'data_class':('DataClass.csv','is_data_class'),'long_method':('LongMethod.csv','is_long_method'),'feature_envy':('FeatureEnvy.csv','is_feature_envy'),'long_parameter_list':('LongParameterList.csv','is_long_parameters_list'),'switch_statements':('SwitchStatements.csv','is_switch_statements')}
d=DATA/'IST2021';sets=[]
for k,(fn,tg) in IST.items():
    df=pd.read_csv(d/fn);sets.append({c for c in df.columns if c!=tg})
ISTcommon=sorted(set.intersection(*sets));ISTRAW={}
for k,(fn,tg) in IST.items():
    df=pd.read_csv(d/fn);X=np.nan_to_num(df[ISTcommon].values,nan=0.0);y=df[tg].apply(lambda v:1 if v in [True,'TRUE',1,'1'] else 0).values;ISTRAW[k]=(X,y)
HAL=['Logical Lines','Distinct Operators','Distinct Operands','Total Operators','Total Operands','Vocabulary','Length','Calculated Length','Volume','Difficulty','Effort','Time Required','Bugs','Cyclomatic Complexity']
SMSM={'god_class':'God class','long_method':'Long method','feature_envy':'Feature envy','data_class':'Data class'}
SMdf=pd.read_csv(DATA/'SmellyCode++.csv',usecols=HAL+list(SMSM.values()))
SMX=np.nan_to_num(SMdf[HAL].values,nan=0.0);SMY={k:(SMdf[c]==1).astype(int).values for k,c in SMSM.items()}
def ist_down(rate,seed):
    rng=np.random.RandomState(seed);pf={}
    for k in IST:
        X,y=ISTRAW[k];pos=np.where(y==1)[0];neg=np.where(y==0)[0]
        npos=int(round(len(neg)*rate/(1-rate)))
        if npos>len(pos):npos=len(pos)
        pi=rng.choice(pos,npos,replace=False);pf[k]=(np.vstack([X[pi],X[neg]]),np.concatenate([np.ones(npos),np.zeros(len(neg))]).astype(int))
    return pf,len(ISTcommon)
def sm_up(rate,seed,P=1000):
    rng=np.random.RandomState(seed);pf={}
    for k in SMSM:
        y=SMY[k];pos=np.where(y==1)[0];neg=np.where(y==0)[0];npos=min(P,len(pos));nneg=int(round(npos*(1-rate)/rate))
        if nneg>len(neg):nneg=len(neg)
        pi=rng.choice(pos,npos,replace=False);ni=rng.choice(neg,nneg,replace=False)
        pf[k]=(np.vstack([SMX[pi],SMX[ni]]),np.concatenate([np.ones(npos),np.zeros(nneg)]).astype(int))
    return pf,SMX.shape[1]
out=DATA/'robustness_R2_2026-06-24.csv'
w=csv.writer(open(out,'w',newline=''));w.writerow(['sweep','classifier','rate','seed','dF1','dAP'])
import sys
for name,fn,rates in [('IST2021_down',ist_down,[0.33,0.20,0.10,0.05]),('SmellyCode_up',sm_up,[0.05,0.10,0.20,0.33])]:
    for cn in ['RF','LogReg']:
        for rate in rates:
            for seed in SEEDS:
                t=time.time();pf,nf=fn(rate,seed)
                try:
                    dF1,dAP=specmc(facs(seed)[cn],pf,nf,seed)
                    with open(out,'a',newline='') as f: csv.writer(f).writerow([name,cn,rate,seed,round(dF1,4),round(dAP,4)])
                    print(f'{name} {cn} rate={rate:.0%} seed={seed} dF1={dF1:+.3f} dAP={dAP:+.3f} ({time.time()-t:.0f}s)',flush=True)
                except Exception as e: print(f'{name} {cn} {rate} {seed} ERR {e}',flush=True)
print('R2 DONE',flush=True)
