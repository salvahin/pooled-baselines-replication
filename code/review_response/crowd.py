import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score
from scipy.stats import wilcoxon, binomtest
NF,SEED=10,42
def rf(): return RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=SEED,n_jobs=-1)
base='/tmp/cs_probe/Crowdsmelling/Datasets 2020/'
files={'god_class':('god-class-2020+2019+2018.csv','is_god_class'),'long_method':('long-method-2020+2019+2018.csv','is_long_method'),'feature_envy':('feature-envy-2020+2019+2018.csv','is_feature_envy')}
META={'username','project','package','complextype','method','methodname'}
def load(fn,lab):
    df=pd.read_csv(base+fn)
    y=df[lab].apply(lambda v:1 if str(v).upper() in ('TRUE','1') else 0).values
    feats=[c for c in df.columns if c not in META and c!=lab]
    X=df[feats].apply(pd.to_numeric,errors='coerce')
    X=X.dropna(axis=1,how='any')  # keep numeric cols
    return X, y, set(X.columns)
data={}; colsets=[]
for k,(fn,lab) in files.items():
    X,y,cs=load(fn,lab); data[k]=(X,y); colsets.append(cs)
    print(f'{k}: {len(y)} rows, {len(cs)} numeric feats, {y.mean()*100:.0f}% positive', flush=True)
common=sorted(set.intersection(*colsets))
print('common features:',len(common))
order=list(files)
D={k:(np.nan_to_num(data[k][0][common].values,nan=0.0), data[k][1]) for k in order}
# specific
sF={};sA={}
for k in order:
    X,y=D[k];skf=StratifiedKFold(NF,shuffle=True,random_state=SEED);fs=[];ap=[]
    for tr,te in skf.split(X,y):
        sc=StandardScaler();Xtr=sc.fit_transform(X[tr]);Xte=sc.transform(X[te])
        c=rf();c.fit(Xtr,y[tr]);fs.append(f1_score(y[te],c.predict(Xte),zero_division=0));ap.append(average_precision_score(y[te],c.predict_proba(Xte)[:,1]))
    sF[k]=fs;sA[k]=ap
# multiclass pooled
ns=len(order);Xl=[];yl=[];sl=[]
for si,k in enumerate(order):
    X,y=D[k];oh=np.zeros(ns);oh[si]=1
    Xl.append(np.hstack([X,np.tile(oh,(len(X),1))]));yl.append(y);sl.append(np.array([k]*len(X)))
Xc=np.vstack(Xl);yc=np.concatenate(yl);scl=np.concatenate(sl);strat=LabelEncoder().fit_transform(scl);n=len(common)
mF={k:[] for k in order};mA={k:[] for k in order}
for tr,te in StratifiedKFold(NF,shuffle=True,random_state=SEED).split(Xc,strat):
    Xtr,Xte=Xc[tr].copy(),Xc[te].copy();s=StandardScaler();Xtr[:,:n]=s.fit_transform(Xtr[:,:n]);Xte[:,:n]=s.transform(Xte[:,:n])
    c=rf();c.fit(Xtr,yc[tr]);p=c.predict(Xte);pr=c.predict_proba(Xte)[:,1];ts=scl[te]
    for k in order:
        m=ts==k
        if m.sum(): mF[k].append(f1_score(yc[te][m],p[m],zero_division=0));mA[k].append(average_precision_score(yc[te][m],pr[m]))
print('\n==== Crowdsmelling (CK metrics, crowdsourced labels) ====')
dF1s=[];dAPs=[]
for k in order:
    s1,m1=np.array(sF[k]),np.array(mF[k]);sa,ma=np.array(sA[k]),np.array(mA[k])
    dF1s.append(s1.mean()-m1.mean());dAPs.append(sa.mean()-ma.mean())
    print(f'{k:14s} specF1={s1.mean():.3f} mcF1={m1.mean():.3f} dF1={s1.mean()-m1.mean():+.3f} p={wilcoxon(s1,m1)[1]:.3f} | specAP={sa.mean():.3f} mcAP={ma.mean():.3f} dAP={sa.mean()-ma.mean():+.3f}')
dF1s=np.array(dF1s);dAPs=np.array(dAPs)
print(f'  MEAN dF1={dF1s.mean():+.3f} ({int((dF1s>0).sum())}/3 pos) | MEAN dAP={dAPs.mean():+.3f} ({int((dAPs>0).sum())}/3 pos)')
