"""Phase 1: fairness 2x2. {LogReg,LinearSVC,RF,HistGB} x {specific, pooled-intercept,
pooled-interaction} x 5 datasets x 5 seeds. ΔPR-AUC = specific - pooled.
Claim: pooled-intercept is unfair to LINEAR models (large Δ) but not trees (~0);
pooled-INTERACTION is fair for all (Δ~0). Incremental CSV."""
import csv,time,warnings,numpy as np,pandas as pd
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
NF=10; HERE=Path(__file__).resolve().parent; DATA=HERE.parent.parent/'data'; SEEDS=[42,1,7,13,99]; SUB=15000
def facs(seed):
    return {'LogReg':lambda:LogisticRegression(class_weight='balanced',max_iter=2000,random_state=seed),
            'LinearSVC':lambda:LinearSVC(class_weight='balanced',max_iter=5000,random_state=seed),
            'RF':lambda:RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=seed,n_jobs=-1),
            'HistGB':lambda:HistGradientBoostingClassifier(class_weight='balanced',random_state=seed)}
def sco(c,X): return c.predict_proba(X)[:,1] if hasattr(c,'predict_proba') else c.decision_function(X)
def specific_ap(pf,fac,seed):
    out={}
    for k,(X,y) in pf.items():
        if y.sum()<NF: out[k]=np.nan;continue
        ap=[]
        for tr,te in StratifiedKFold(NF,shuffle=True,random_state=seed).split(X,y):
            s=StandardScaler();Xtr=s.fit_transform(X[tr]);Xte=s.transform(X[te])
            c=fac();c.fit(Xtr,y[tr]);ap.append(average_precision_score(y[te],sco(c,Xte)))
        out[k]=np.mean(ap)
    return out
def pooled_ap(pf,fac,seed,inter):
    order=list(pf);ns=len(order);Xs=[];ys=[];idx=[]
    for si,k in enumerate(order):
        X,y=pf[k];Xs.append(X);ys.append(y);idx.append(np.full(len(X),si))
    Xa=np.vstack(Xs);ya=np.concatenate(ys);sidx=np.concatenate(idx);out={k:[] for k in order}
    for tr,te in StratifiedKFold(NF,shuffle=True,random_state=seed).split(Xa,sidx):
        sc=StandardScaler();Ftr=sc.fit_transform(Xa[tr]);Fte=sc.transform(Xa[te])
        def aug(F,si):
            oh=np.zeros((len(F),ns));oh[np.arange(len(F)),si]=1
            if not inter: return np.hstack([F,oh])
            return np.hstack([F,oh,np.einsum('ij,ik->ijk',F,oh).reshape(len(F),-1)])
        c=fac();c.fit(aug(Ftr,sidx[tr]),ya[tr]);pr=sco(c,aug(Fte,sidx[te]))
        for si,k in enumerate(order):
            m=sidx[te]==si
            if m.sum() and ya[te][m].sum()>0: out[k].append(average_precision_score(ya[te][m],pr[m]))
    return {k:(np.mean(v) if v else np.nan) for k,v in out.items()}
# loaders
def L_ist(seed):
    IST={'god_class':('GodClass.csv','is_god_class'),'data_class':('DataClass.csv','is_data_class'),'long_method':('LongMethod.csv','is_long_method'),'feature_envy':('FeatureEnvy.csv','is_feature_envy'),'long_parameter_list':('LongParameterList.csv','is_long_parameters_list'),'switch_statements':('SwitchStatements.csv','is_switch_statements')}
    d=DATA/'IST2021';sets=[]
    for k,(fn,tg) in IST.items(): sets.append({c for c in pd.read_csv(d/fn).columns if c!=tg})
    common=sorted(set.intersection(*sets));pf={}
    for k,(fn,tg) in IST.items():
        df=pd.read_csv(d/fn);pf[k]=(np.nan_to_num(df[common].values,nan=0.0),df[tg].apply(lambda v:1 if v in [True,'TRUE',1,'1'] else 0).values)
    return pf
def L_improv(seed):
    df=pd.read_csv(DATA/'ImprovMLCQ.csv');ck=sorted([c for c in df.columns if c.startswith('ck_')]);X=np.nan_to_num(df[ck].values,nan=0.0)
    IM={'blob':'blob_label','data_class':'dataclass_label','feature_envy':'featureenvy_label','long_method':'longmethod_label'}
    return {k:(X,(df[c]==1).astype(int).values) for k,c in IM.items()}
def L_crowd(seed):
    base=HERE/'crowdsmelling_data';files={'god_class':('god-class-2020+2019+2018.csv','is_god_class'),'long_method':('long-method-2020+2019+2018.csv','is_long_method'),'feature_envy':('feature-envy-2020+2019+2018.csv','is_feature_envy')}
    META={'username','project','package','complextype','method','methodname'};data={};cs=[]
    for k,(fn,lab) in files.items():
        df=pd.read_csv(base/fn);y=df[lab].apply(lambda v:1 if str(v).upper() in ('TRUE','1') else 0).values
        Xd=df[[c for c in df.columns if c not in META and c!=lab]].apply(pd.to_numeric,errors='coerce').dropna(axis=1,how='any');data[k]=(Xd,y);cs.append(set(Xd.columns))
    common=sorted(set.intersection(*cs));return {k:(np.nan_to_num(data[k][0][common].values,nan=0.0),data[k][1]) for k in files}
def L_smelly(seed):
    HAL=['Logical Lines','Distinct Operators','Distinct Operands','Total Operators','Total Operands','Vocabulary','Length','Calculated Length','Volume','Difficulty','Effort','Time Required','Bugs','Cyclomatic Complexity']
    SM={'god_class':'God class','long_method':'Long method','feature_envy':'Feature envy','data_class':'Data class'}
    df=pd.read_csv(DATA/'SmellyCode++.csv',usecols=HAL+list(SM.values())).sample(n=SUB,random_state=seed).reset_index(drop=True)
    X=np.nan_to_num(df[HAL].values,nan=0.0);return {k:(X,(df[c]==1).astype(int).values) for k,c in SM.items()}
def L_mlcs(seed):
    f=HERE/'mlcodesmell_class.csv'
    df=pd.read_csv(f).sample(n=SUB,random_state=seed).reset_index(drop=True)
    labels=['Brain Class','Data Class','Futile Abstract Pipeline','Futile Hierarchy','God Class','Hierarchy Duplication','Model Class','Schizofrenic Class']
    feats=[c for c in df.columns if c not in (['Address']+labels)];X=np.nan_to_num(df[feats].apply(pd.to_numeric,errors='coerce').values,nan=0.0)
    SM={'data_class':'Data Class','god_class':'God Class','schizofrenic_class':'Schizofrenic Class'}
    return {k:(X,(df[c].astype(str).str.upper()=='TRUE').astype(int).values) for k,c in SM.items()}
LOADERS={'IST2021':L_ist,'ImprovMLCQ':L_improv,'Crowdsmelling':L_crowd,'SmellyCode++':L_smelly,'ml-Codesmell':L_mlcs}
out=DATA/'phase1_fairness2x2_2026-06-24.csv'
with open(out,'w',newline='') as f: csv.writer(f).writerow(['dataset','classifier','seed','specAP','interceptAP','interactionAP','d_intercept','d_interaction'])
for ds,ld in LOADERS.items():
    for cn in ['LogReg','LinearSVC','RF','HistGB']:
        for seed in SEEDS:
            t=time.time();pf=ld(seed);fac=facs(seed)[cn]
            try:
                sp=specific_ap(pf,fac,seed);b0=pooled_ap(pf,fac,seed,False);b1=pooled_ap(pf,fac,seed,True)
                ks=[k for k in pf if not np.isnan(sp[k]) and not np.isnan(b0[k]) and not np.isnan(b1[k])]
                sA=np.mean([sp[k] for k in ks]);i0=np.mean([b0[k] for k in ks]);i1=np.mean([b1[k] for k in ks])
                with open(out,'a',newline='') as f: csv.writer(f).writerow([ds,cn,seed,round(sA,4),round(i0,4),round(i1,4),round(sA-i0,4),round(sA-i1,4)])
                print(f'{ds:14s} {cn:10s} seed={seed} d_intercept={sA-i0:+.3f} d_interaction={sA-i1:+.3f} ({time.time()-t:.0f}s)',flush=True)
            except Exception as e: print(f'{ds} {cn} {seed} ERR {e}',flush=True)
print('PHASE1 DONE',flush=True)
