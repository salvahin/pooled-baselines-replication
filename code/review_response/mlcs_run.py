import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score
from scipy.stats import wilcoxon
NF,SEED=10,42
def rf(): return RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=SEED,n_jobs=-1)
df=pd.read_csv('class.csv').sample(n=80000,random_state=SEED).reset_index(drop=True)
SMELLS=['data_class','god_class','schizofrenic_class']
colmap={'data_class':'Data Class','god_class':'God Class','schizofrenic_class':'Schizofrenic Class'}
labelcols=['Brain Class','Data Class','Futile Abstract Pipeline','Futile Hierarchy','God Class','Hierarchy Duplication','Model Class','Schizofrenic Class']
feats=[c for c in df.columns if c not in (['Address']+labelcols)]
X=np.nan_to_num(df[feats].apply(pd.to_numeric,errors='coerce').values,nan=0.0)
Y={k:(df[colmap[k]].astype(str).str.upper()=='TRUE').astype(int).values for k in SMELLS}
for k in SMELLS: print(f'  {k}: {Y[k].sum()} pos ({Y[k].mean()*100:.2f}%)',flush=True)
print(f'  features: {len(feats)} iPlasma metrics, {len(df)} rows',flush=True)
order=SMELLS;n=X.shape[1]
sF={};sA={}
for k in order:
    y=Y[k];skf=StratifiedKFold(NF,shuffle=True,random_state=SEED);fs=[];ap=[]
    for tr,te in skf.split(X,y):
        sc=StandardScaler();Xtr=sc.fit_transform(X[tr]);Xte=sc.transform(X[te])
        c=rf();c.fit(Xtr,y[tr]);fs.append(f1_score(y[te],c.predict(Xte),zero_division=0));ap.append(average_precision_score(y[te],c.predict_proba(Xte)[:,1]))
    sF[k]=fs;sA[k]=ap;print(f'  spec {k} done',flush=True)
ns=len(order);Xl=[];yl=[];sl=[]
for si,k in enumerate(order):
    oh=np.zeros(ns);oh[si]=1
    Xl.append(np.hstack([X,np.tile(oh,(len(X),1))]));yl.append(Y[k]);sl.append(np.array([k]*len(X)))
Xc=np.vstack(Xl);yc=np.concatenate(yl);scl=np.concatenate(sl);strat=LabelEncoder().fit_transform(scl)
mF={k:[] for k in order};mA={k:[] for k in order}
t=time.time()
for fold,(tr,te) in enumerate(StratifiedKFold(NF,shuffle=True,random_state=SEED).split(Xc,strat)):
    Xtr,Xte=Xc[tr].copy(),Xc[te].copy();s=StandardScaler();Xtr[:,:n]=s.fit_transform(Xtr[:,:n]);Xte[:,:n]=s.transform(Xte[:,:n])
    c=rf();c.fit(Xtr,yc[tr]);p=c.predict(Xte);pr=c.predict_proba(Xte)[:,1];ts=scl[te]
    for k in order:
        m=ts==k
        if m.sum(): mF[k].append(f1_score(yc[te][m],p[m],zero_division=0));mA[k].append(average_precision_score(yc[te][m],pr[m]))
    print(f'  mc fold {fold} ({time.time()-t:.0f}s)',flush=True)
print('\n==== ml-Codesmell (iPlasma metrics, 8-smell Java) ====')
dF1s=[];dAPs=[]
for k in order:
    s1,m1=np.array(sF[k]),np.array(mF[k]);sa,ma=np.array(sA[k]),np.array(mA[k])
    dF1s.append(s1.mean()-m1.mean());dAPs.append(sa.mean()-ma.mean())
    print(f'{k:20s} specF1={s1.mean():.3f} mcF1={m1.mean():.3f} dF1={s1.mean()-m1.mean():+.3f} p={wilcoxon(s1,m1)[1]:.3f} | dAP={sa.mean()-ma.mean():+.3f}')
print(f'  MEAN dF1={np.mean(dF1s):+.3f} ({sum(1 for x in dF1s if x>0)}/{len(order)} pos) | MEAN dAP={np.mean(dAPs):+.3f} ({sum(1 for x in dAPs if x>0)}/{len(order)} pos)')
