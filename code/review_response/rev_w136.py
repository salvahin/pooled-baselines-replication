import numpy as np, pandas as pd, warnings, time
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score
from scipy.stats import wilcoxon, binomtest
NF,SEED=10,42
def rf(): return RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=SEED,n_jobs=-1)
def coh(a,b):
    a,b=np.array(a),np.array(b)
    s=np.sqrt(((len(a)-1)*a.std(ddof=1)**2+(len(b)-1)*b.std(ddof=1)**2)/(len(a)+len(b)-2))
    return (a.mean()-b.mean())/s if s>0 else 0.0

def run(name, X, Ydict):
    order=list(Ydict); n=X.shape[1]
    # specific: per-smell binary; collect per-fold F1 and AP
    spec_f1={k:[] for k in order}; spec_ap={k:[] for k in order}
    for k in order:
        y=Ydict[k]; skf=StratifiedKFold(NF,shuffle=True,random_state=SEED)
        for tr,te in skf.split(X,y):
            sc=StandardScaler();Xtr=sc.fit_transform(X[tr]);Xte=sc.transform(X[te])
            c=rf();c.fit(Xtr,y[tr]);p=c.predict(Xte);pr=c.predict_proba(Xte)[:,1]
            spec_f1[k].append(f1_score(y[te],p,zero_division=0)); spec_ap[k].append(average_precision_score(y[te],pr))
    # multiclass: pooled with smell-type one-hot
    ns=len(order); Xl=[];yl=[];sl=[]
    for si,k in enumerate(order):
        oh=np.zeros(ns);oh[si]=1
        Xl.append(np.hstack([X,np.tile(oh,(len(X),1))]));yl.append(Ydict[k]);sl.append(np.array([k]*len(X)))
    Xc=np.vstack(Xl);yc=np.concatenate(yl);scl=np.concatenate(sl);strat=LabelEncoder().fit_transform(scl)
    mc_f1={k:[] for k in order}; mc_ap={k:[] for k in order}
    skf=StratifiedKFold(NF,shuffle=True,random_state=SEED)
    for tr,te in skf.split(Xc,strat):
        Xtr,Xte=Xc[tr].copy(),Xc[te].copy()
        s=StandardScaler();Xtr[:,:n]=s.fit_transform(Xtr[:,:n]);Xte[:,:n]=s.transform(Xte[:,:n])
        c=rf();c.fit(Xtr,yc[tr]);p=c.predict(Xte);pr=c.predict_proba(Xte)[:,1];ts=scl[te]
        for k in order:
            m=ts==k
            if m.sum(): mc_f1[k].append(f1_score(yc[te][m],p[m],zero_division=0)); mc_ap[k].append(average_precision_score(yc[te][m],pr[m]))
    print(f'\n==== {name} ====')
    print('smell, specF1±sd, mcF1±sd, dF1, pF1, d, specAP±sd, mcAP±sd, dAP')
    dF1s=[]; dAPs=[]
    for k in order:
        s1,m1=np.array(spec_f1[k]),np.array(mc_f1[k]); sa,ma=np.array(spec_ap[k]),np.array(mc_ap[k])
        try: pf=wilcoxon(s1,m1)[1]
        except: pf=float('nan')
        dF1s.append(s1.mean()-m1.mean()); dAPs.append(sa.mean()-ma.mean())
        print(f'{k:20s} {s1.mean():.3f}±{s1.std():.3f}  {m1.mean():.3f}±{m1.std():.3f}  {s1.mean()-m1.mean():+.3f}  p={pf:.3f}  d={coh(s1,m1):.2f}  AP {sa.mean():.3f}±{sa.std():.3f} {ma.mean():.3f}±{ma.std():.3f}  dAP={sa.mean()-ma.mean():+.3f}')
    dF1s=np.array(dF1s); dAPs=np.array(dAPs)
    pos=int((dF1s>0).sum()); nn=len(dF1s)
    sign=binomtest(pos,nn,0.5,alternative='two-sided').pvalue
    try: wj=wilcoxon(dF1s)[1]
    except: wj=float('nan')
    print(f'  AGG F1: {pos}/{nn} positive | sign-test p={sign:.4f} | one-sample Wilcoxon p={wj:.4f} | mean dF1={dF1s.mean():+.3f}')
    print(f'  AGG AP: {int((dAPs>0).sum())}/{nn} positive | mean dAP={dAPs.mean():+.3f}')

# IST2021
IST={'god_class':('GodClass.csv','is_god_class'),'data_class':('DataClass.csv','is_data_class'),'long_method':('LongMethod.csv','is_long_method'),'feature_envy':('FeatureEnvy.csv','is_feature_envy'),'long_parameter_list':('LongParameterList.csv','is_long_parameters_list'),'switch_statements':('SwitchStatements.csv','is_switch_statements')}
import os
ID='data/IST2021'
sets=[]
for k,(fn,tg) in IST.items():
    d=pd.read_csv(os.path.join(ID,fn)); sets.append({c for c in d.columns if c!=tg})
common=sorted(set.intersection(*sets))
d0=pd.read_csv(os.path.join(ID,list(IST.values())[0][0]))
# load X per smell? IST uses same common features; build per-smell X,y but X differs per file -> use common features, but rows differ per file. For specific we need per-file. Build Ydict won't share X. Handle IST specially:
print('IST2021: per-file; computing separately is already in ist2021_results_pr; here we only need the aggregate sign test on the 6 dF1:')
istd=[0.033,0.033,0.087,0.100,0.045,0.059]
import numpy as np
pos=sum(1 for x in istd if x>0)
print(f'  IST2021 dF1 = {istd} | {pos}/6 positive | sign-test p={binomtest(pos,6,0.5).pvalue:.4f} | one-sample Wilcoxon p={wilcoxon(istd)[1]:.4f}')

# ImprovMLCQ
df=pd.read_csv('data/ImprovMLCQ.csv')
ck=sorted([c for c in df.columns if c.startswith('ck_')])
X=np.nan_to_num(df[ck].values,nan=0.0)
IM={'blob':'blob_label','data_class':'dataclass_label','feature_envy':'featureenvy_label','long_method':'longmethod_label'}
Y={k:(df[c]==1).astype(int).values for k,c in IM.items()}
t=time.time(); run('ImprovMLCQ', X, Y); print('  (%.0fs)'%(time.time()-t))
