"""Decisive test: is the linear-model specialization advantage an artifact of the
multi-class arm only getting per-smell INTERCEPTS? Compare specific vs two pooled
arms: (B-orig) [features, smell-onehot]; (B-fair) [features, onehot, features x onehot].
If specific - B-fair ~ 0, the advantage was a construction artifact."""
import warnings,numpy as np,pandas as pd
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
NF,SEED=10,42; HERE=Path(__file__).resolve().parent; DATA=HERE.parent.parent/'data'
def lr(): return LogisticRegression(class_weight='balanced',max_iter=2000,random_state=SEED)
def specific_ap(perfile):
    order=list(perfile);out={}
    for k in order:
        X,y=perfile[k]
        if y.sum()<NF: out[k]=np.nan;continue
        ap=[]
        for tr,te in StratifiedKFold(NF,shuffle=True,random_state=SEED).split(X,y):
            s=StandardScaler();Xtr=s.fit_transform(X[tr]);Xte=s.transform(X[te])
            c=lr();c.fit(Xtr,y[tr]);ap.append(average_precision_score(y[te],c.predict_proba(Xte)[:,1]))
        out[k]=np.mean(ap)
    return out
def pooled_ap(perfile, interactions):
    order=list(perfile);ns=len(order)
    # build combined raw feature matrix + smell index, then scale features, then augment
    Xs=[];ys=[];idx=[]
    for si,k in enumerate(order):
        X,y=perfile[k];Xs.append(X);ys.append(y);idx.append(np.full(len(X),si))
    Xall=np.vstack(Xs);yall=np.concatenate(ys);sidx=np.concatenate(idx)
    out={k:[] for k in order}
    for tr,te in StratifiedKFold(NF,shuffle=True,random_state=SEED).split(Xall,sidx):
        sc=StandardScaler();Ftr=sc.fit_transform(Xall[tr]);Fte=sc.transform(Xall[te])
        def aug(F,si):
            oh=np.zeros((len(F),ns));oh[np.arange(len(F)),si]=1
            if not interactions: return np.hstack([F,oh])
            inter=np.einsum('ij,ik->ijk',F,oh).reshape(len(F),-1)  # F x onehot
            return np.hstack([F,oh,inter])
        Atr=aug(Ftr,sidx[tr]);Ate=aug(Fte,sidx[te])
        c=lr();c.fit(Atr,yall[tr]);pr=c.predict_proba(Ate)[:,1]
        for si,k in enumerate(order):
            m=sidx[te]==si
            if m.sum() and yall[te][m].sum()>0: out[k].append(average_precision_score(yall[te][m],pr[m]))
    return {k:(np.mean(v) if v else np.nan) for k,v in out.items()}
def run(name, perfile):
    sp=specific_ap(perfile); b0=pooled_ap(perfile,False); b1=pooled_ap(perfile,True)
    ks=[k for k in perfile if not np.isnan(sp[k]) and not np.isnan(b0[k]) and not np.isnan(b1[k])]
    d0=np.mean([sp[k]-b0[k] for k in ks]); d1=np.mean([sp[k]-b1[k] for k in ks])
    print(f'{name:14s} specAP={np.mean([sp[k] for k in ks]):.3f} | pooled-intercept AP={np.mean([b0[k] for k in ks]):.3f} (dPR-AUC={d0:+.3f}) | pooled-INTERACTION AP={np.mean([b1[k] for k in ks]):.3f} (dPR-AUC={d1:+.3f})',flush=True)
# loaders
def load_ist():
    IST={'god_class':('GodClass.csv','is_god_class'),'data_class':('DataClass.csv','is_data_class'),'long_method':('LongMethod.csv','is_long_method'),'feature_envy':('FeatureEnvy.csv','is_feature_envy'),'long_parameter_list':('LongParameterList.csv','is_long_parameters_list'),'switch_statements':('SwitchStatements.csv','is_switch_statements')}
    d=DATA/'IST2021';sets=[]
    for k,(fn,tg) in IST.items():
        df=pd.read_csv(d/fn);sets.append({c for c in df.columns if c!=tg})
    common=sorted(set.intersection(*sets));pf={}
    for k,(fn,tg) in IST.items():
        df=pd.read_csv(d/fn);pf[k]=(np.nan_to_num(df[common].values,nan=0.0),df[tg].apply(lambda v:1 if v in [True,'TRUE',1,'1'] else 0).values)
    return pf
def load_improv():
    df=pd.read_csv(DATA/'ImprovMLCQ.csv');ck=sorted([c for c in df.columns if c.startswith('ck_')]);X=np.nan_to_num(df[ck].values,nan=0.0)
    IM={'blob':'blob_label','data_class':'dataclass_label','feature_envy':'featureenvy_label','long_method':'longmethod_label'}
    return {k:(X,(df[c]==1).astype(int).values) for k,c in IM.items()}
def load_smelly():
    HAL=['Logical Lines','Distinct Operators','Distinct Operands','Total Operators','Total Operands','Vocabulary','Length','Calculated Length','Volume','Difficulty','Effort','Time Required','Bugs','Cyclomatic Complexity']
    SM={'god_class':'God class','long_method':'Long method','feature_envy':'Feature envy','data_class':'Data class'}
    df=pd.read_csv(DATA/'SmellyCode++.csv',usecols=HAL+list(SM.values())).sample(n=40000,random_state=SEED).reset_index(drop=True)
    X=np.nan_to_num(df[HAL].values,nan=0.0);return {k:(X,(df[c]==1).astype(int).values) for k,c in SM.items()}
def load_crowd():
    base=HERE/'crowdsmelling_data';files={'god_class':('god-class-2020+2019+2018.csv','is_god_class'),'long_method':('long-method-2020+2019+2018.csv','is_long_method'),'feature_envy':('feature-envy-2020+2019+2018.csv','is_feature_envy')}
    META={'username','project','package','complextype','method','methodname'};data={};cs=[]
    for k,(fn,lab) in files.items():
        df=pd.read_csv(base/fn);y=df[lab].apply(lambda v:1 if str(v).upper() in ('TRUE','1') else 0).values
        Xd=df[[c for c in df.columns if c not in META and c!=lab]].apply(pd.to_numeric,errors='coerce').dropna(axis=1,how='any');data[k]=(Xd,y);cs.append(set(Xd.columns))
    common=sorted(set.intersection(*cs));return {k:(np.nan_to_num(data[k][0][common].values,nan=0.0),data[k][1]) for k in files}
print('LogReg: specific vs pooled-intercept-only vs pooled-with-interactions (ΔPR-AUC = specific − pooled)')
run('IST2021',load_ist()); run('ImprovMLCQ',load_improv()); run('Crowdsmelling',load_crowd()); run('SmellyCode++',load_smelly())
print('DONE')
