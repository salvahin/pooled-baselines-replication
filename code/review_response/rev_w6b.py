import numpy as np, pandas as pd, warnings, time, os
warnings.filterwarnings('ignore')
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, average_precision_score
NF,SEED=10,42
def rf(): return RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=SEED,n_jobs=-1)

def specmc(name, perfile=None, X=None, Ydict=None, feats_common=None):
    # returns per-smell mean specF1, mcF1, specAP, mcAP
    if perfile:  # IST2021: dict smell-> (Xk, yk) plus combined build
        order=list(perfile)
        spec_f1={};spec_ap={}
        for k,(Xk,yk) in perfile.items():
            skf=StratifiedKFold(NF,shuffle=True,random_state=SEED);fs=[];ap=[]
            for tr,te in skf.split(Xk,yk):
                sc=StandardScaler();Xtr=sc.fit_transform(Xk[tr]);Xte=sc.transform(Xk[te])
                c=rf();c.fit(Xtr,yk[tr]);fs.append(f1_score(yk[te],c.predict(Xte),zero_division=0));ap.append(average_precision_score(yk[te],c.predict_proba(Xte)[:,1]))
            spec_f1[k]=np.mean(fs);spec_ap[k]=np.mean(ap)
        ns=len(order);Xl=[];yl=[];sl=[]
        for si,k in enumerate(order):
            Xk,yk=perfile[k];oh=np.zeros(ns);oh[si]=1
            Xl.append(np.hstack([Xk,np.tile(oh,(len(Xk),1))]));yl.append(yk);sl.append(np.array([k]*len(Xk)))
        Xc=np.vstack(Xl);yc=np.concatenate(yl);scl=np.concatenate(sl);strat=LabelEncoder().fit_transform(scl);n=Xl[0].shape[1]-ns
        mc_f1={k:[] for k in order};mc_ap={k:[] for k in order}
        skf=StratifiedKFold(NF,shuffle=True,random_state=SEED)
        for tr,te in skf.split(Xc,strat):
            Xtr,Xte=Xc[tr].copy(),Xc[te].copy();s=StandardScaler();Xtr[:,:n]=s.fit_transform(Xtr[:,:n]);Xte[:,:n]=s.transform(Xte[:,:n])
            c=rf();c.fit(Xtr,yc[tr]);p=c.predict(Xte);pr=c.predict_proba(Xte)[:,1];ts=scl[te]
            for k in order:
                m=ts==k
                if m.sum(): mc_f1[k].append(f1_score(yc[te][m],p[m],zero_division=0));mc_ap[k].append(average_precision_score(yc[te][m],pr[m]))
        print(f'\n==== {name} ====')
        sF=[];mF=[];sA=[];mA=[]
        for k in order:
            mf,ma=np.mean(mc_f1[k]),np.mean(mc_ap[k]);sF.append(spec_f1[k]);mF.append(mf);sA.append(spec_ap[k]);mA.append(ma)
            print(f'{k:20s} F1 {spec_f1[k]:.3f}/{mf:.3f} dF1={spec_f1[k]-mf:+.3f} | AP {spec_ap[k]:.3f}/{ma:.3f} dAP={spec_ap[k]-ma:+.3f}')
        print(f'  MEAN dF1={np.mean(sF)-np.mean(mF):+.3f} | MEAN dAP={np.mean(sA)-np.mean(mA):+.3f} | AP pos {sum(1 for a,b in zip(sA,mA) if a>b)}/{len(order)}')

# IST2021
IST={'god_class':('GodClass.csv','is_god_class'),'data_class':('DataClass.csv','is_data_class'),'long_method':('LongMethod.csv','is_long_method'),'feature_envy':('FeatureEnvy.csv','is_feature_envy'),'long_parameter_list':('LongParameterList.csv','is_long_parameters_list'),'switch_statements':('SwitchStatements.csv','is_switch_statements')}
ID='data/IST2021';sets=[]
for k,(fn,tg) in IST.items():
    d=pd.read_csv(os.path.join(ID,fn));sets.append({c for c in d.columns if c!=tg})
common=sorted(set.intersection(*sets))
perfile={}
for k,(fn,tg) in IST.items():
    d=pd.read_csv(os.path.join(ID,fn));X=np.nan_to_num(d[common].values,nan=0.0)
    y=d[tg].apply(lambda v:1 if v in [True,'TRUE',1,'1'] else 0).values
    perfile[k]=(X,y)
t=time.time();specmc('IST2021', perfile=perfile);print('  (%.0fs)'%(time.time()-t))

# SmellyCode++
HAL=['Logical Lines','Distinct Operators','Distinct Operands','Total Operators','Total Operands','Vocabulary','Length','Calculated Length','Volume','Difficulty','Effort','Time Required','Bugs','Cyclomatic Complexity']
SM={'god_class':'God class','long_method':'Long method','feature_envy':'Feature envy','data_class':'Data class'}
df=pd.read_csv('data/SmellyCode++.csv',usecols=HAL+list(SM.values()))
X=np.nan_to_num(df[HAL].values,nan=0.0)
pf={k:(X,(df[c]==1).astype(int).values) for k,c in SM.items()}
t=time.time();specmc('SmellyCode++', perfile=pf);print('  (%.0fs)'%(time.time()-t))
