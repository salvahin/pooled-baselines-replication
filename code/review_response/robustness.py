"""R1 (per-dataset classifier x seed grid) + R2 (sweeps x seed x classifier).
Writes incremental CSVs. Faithful pipeline; subsamples big datasets."""
import os,csv,time,warnings,numpy as np,pandas as pd
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, average_precision_score
NF=10
HERE=Path(__file__).resolve().parent; DATA=HERE.parent.parent/'data'
SEEDS=[42,1,7,13,99]
def clf_facs(seed):
    return {
     'RF': lambda: RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=seed,n_jobs=-1),
     'HistGB': lambda: HistGradientBoostingClassifier(class_weight='balanced',random_state=seed),
     'LogReg': lambda: LogisticRegression(class_weight='balanced',max_iter=1000,random_state=seed),
     'LinearSVC': lambda: LinearSVC(class_weight='balanced',max_iter=5000,random_state=seed),
    }
def score(clf,X):
    if hasattr(clf,'predict_proba'): return clf.predict_proba(X)[:,1]
    s=clf.decision_function(X); return s
# ---------- loaders ----------
IST={'god_class':('GodClass.csv','is_god_class'),'data_class':('DataClass.csv','is_data_class'),'long_method':('LongMethod.csv','is_long_method'),'feature_envy':('FeatureEnvy.csv','is_feature_envy'),'long_parameter_list':('LongParameterList.csv','is_long_parameters_list'),'switch_statements':('SwitchStatements.csv','is_switch_statements')}
def load_ist():
    d=DATA/'IST2021';sets=[]
    for k,(fn,tg) in IST.items():
        df=pd.read_csv(d/fn);sets.append({c for c in df.columns if c!=tg})
    common=sorted(set.intersection(*sets));pf={}
    for k,(fn,tg) in IST.items():
        df=pd.read_csv(d/fn);X=np.nan_to_num(df[common].values,nan=0.0)
        y=df[tg].apply(lambda v:1 if v in [True,'TRUE',1,'1'] else 0).values;pf[k]=(X,y)
    return ('perfile',pf)
def load_smelly(seed,n=40000):
    HAL=['Logical Lines','Distinct Operators','Distinct Operands','Total Operators','Total Operands','Vocabulary','Length','Calculated Length','Volume','Difficulty','Effort','Time Required','Bugs','Cyclomatic Complexity']
    SM={'god_class':'God class','long_method':'Long method','feature_envy':'Feature envy','data_class':'Data class'}
    df=pd.read_csv(DATA/'SmellyCode++.csv',usecols=HAL+list(SM.values())).sample(n=n,random_state=seed).reset_index(drop=True)
    X=np.nan_to_num(df[HAL].values,nan=0.0);Y={k:(df[c]==1).astype(int).values for k,c in SM.items()}
    return ('shared',X,Y)
def load_improv():
    df=pd.read_csv(DATA/'ImprovMLCQ.csv');ck=sorted([c for c in df.columns if c.startswith('ck_')])
    X=np.nan_to_num(df[ck].values,nan=0.0)
    IM={'blob':'blob_label','data_class':'dataclass_label','feature_envy':'featureenvy_label','long_method':'longmethod_label'}
    return ('shared',X,{k:(df[c]==1).astype(int).values for k,c in IM.items()})
def load_crowd():
    base=HERE/'crowdsmelling_data'
    files={'god_class':('god-class-2020+2019+2018.csv','is_god_class'),'long_method':('long-method-2020+2019+2018.csv','is_long_method'),'feature_envy':('feature-envy-2020+2019+2018.csv','is_feature_envy')}
    META={'username','project','package','complextype','method','methodname'};data={};cs=[]
    for k,(fn,lab) in files.items():
        df=pd.read_csv(base/fn);y=df[lab].apply(lambda v:1 if str(v).upper() in ('TRUE','1') else 0).values
        Xd=df[[c for c in df.columns if c not in META and c!=lab]].apply(pd.to_numeric,errors='coerce').dropna(axis=1,how='any')
        data[k]=(Xd,y);cs.append(set(Xd.columns))
    common=sorted(set.intersection(*cs))
    return ('perfile',{k:(np.nan_to_num(data[k][0][common].values,nan=0.0),data[k][1]) for k in files})
def load_mlcs(seed,n=40000):
    f=HERE/'mlcodesmell_class.csv'
    if not f.exists(): return None
    df=pd.read_csv(f).sample(n=n,random_state=seed).reset_index(drop=True)
    labels=['Brain Class','Data Class','Futile Abstract Pipeline','Futile Hierarchy','God Class','Hierarchy Duplication','Model Class','Schizofrenic Class']
    feats=[c for c in df.columns if c not in (['Address']+labels)]
    X=np.nan_to_num(df[feats].apply(pd.to_numeric,errors='coerce').values,nan=0.0)
    SM={'data_class':'Data Class','god_class':'God Class','schizofrenic_class':'Schizofrenic Class'}
    return ('shared',X,{k:(df[c].astype(str).str.upper()=='TRUE').astype(int).values for k,c in SM.items()})
# ---------- core ----------
def specmc(fac, rep, seed):
    if rep[0]=='perfile':
        pf=rep[1];order=list(pf);sF={};sA={}
        for k in order:
            X,y=pf[k]
            if y.sum()<NF: sF[k]=np.nan;sA[k]=np.nan;continue
            fs=[];ap=[]
            for tr,te in StratifiedKFold(NF,shuffle=True,random_state=seed).split(X,y):
                sc=StandardScaler();Xtr=sc.fit_transform(X[tr]);Xte=sc.transform(X[te])
                c=fac();c.fit(Xtr,y[tr]);fs.append(f1_score(y[te],c.predict(Xte),zero_division=0));ap.append(average_precision_score(y[te],score(c,Xte)))
            sF[k]=np.mean(fs);sA[k]=np.mean(ap)
        ns=len(order);Xl=[];yl=[];sl=[]
        for si,k in enumerate(order):
            X,y=pf[k];oh=np.zeros(ns);oh[si]=1
            Xl.append(np.hstack([X,np.tile(oh,(len(X),1))]));yl.append(y);sl.append(np.array([k]*len(X)))
        n=pf[order[0]][0].shape[1]
    else:
        _,X0,Y=rep;order=list(Y);sF={};sA={}
        for k in order:
            y=Y[k];fs=[];ap=[]
            for tr,te in StratifiedKFold(NF,shuffle=True,random_state=seed).split(X0,y):
                sc=StandardScaler();Xtr=sc.fit_transform(X0[tr]);Xte=sc.transform(X0[te])
                c=fac();c.fit(Xtr,y[tr]);fs.append(f1_score(y[te],c.predict(Xte),zero_division=0));ap.append(average_precision_score(y[te],score(c,Xte)))
            sF[k]=np.mean(fs);sA[k]=np.mean(ap)
        ns=len(order);Xl=[];yl=[];sl=[]
        for si,k in enumerate(order):
            oh=np.zeros(ns);oh[si]=1
            Xl.append(np.hstack([X0,np.tile(oh,(len(X0),1))]));yl.append(Y[k]);sl.append(np.array([k]*len(X0)))
        n=X0.shape[1]
    Xc=np.vstack(Xl);yc=np.concatenate(yl);scl=np.concatenate(sl);strat=LabelEncoder().fit_transform(scl)
    mF={k:[] for k in order};mA={k:[] for k in order}
    for tr,te in StratifiedKFold(NF,shuffle=True,random_state=seed).split(Xc,strat):
        Xtr,Xte=Xc[tr].copy(),Xc[te].copy();s=StandardScaler();Xtr[:,:n]=s.fit_transform(Xtr[:,:n]);Xte[:,:n]=s.transform(Xte[:,:n])
        c=fac();c.fit(Xtr,yc[tr]);p=c.predict(Xte);pr=score(c,Xte);ts=scl[te]
        for k in order:
            m=ts==k
            if m.sum(): mF[k].append(f1_score(yc[te][m],p[m],zero_division=0));mA[k].append(average_precision_score(yc[te][m],pr[m]))
    ks=[k for k in order if not np.isnan(sF[k]) and mF[k]]
    dF1=np.mean([sF[k]-np.mean(mF[k]) for k in ks]);dAP=np.mean([sA[k]-np.mean(mA[k]) for k in ks])
    return dF1,dAP,len(ks)
def main():
    out=DATA/'robustness_R1_2026-06-24.csv'
    w=csv.writer(open(out,'w',newline=''));w.writerow(['dataset','classifier','seed','mean_dF1','mean_dAP','n_smells']);
    loaders={'IST2021':lambda s:load_ist(),'ImprovMLCQ':lambda s:load_improv(),'Crowdsmelling':lambda s:load_crowd(),'SmellyCode++':load_smelly,'ml-Codesmell':load_mlcs}
    for ds,ld in loaders.items():
        for seed in SEEDS:
            rep=ld(seed)
            if rep is None: print(f'skip {ds}',flush=True);break
            for cn in ['RF','HistGB','LogReg','LinearSVC']:
                t=time.time();fac=clf_facs(seed)[cn]
                try:
                    dF1,dAP,nk=specmc(fac,rep,seed)
                    w.writerow([ds,cn,seed,round(dF1,4),round(dAP,4),nk]);open(out,'a').close()
                    print(f'{ds:14s} {cn:10s} seed={seed} dF1={dF1:+.3f} dAP={dAP:+.3f} ({time.time()-t:.0f}s)',flush=True)
                except Exception as e:
                    print(f'{ds} {cn} seed={seed} ERR {e}',flush=True)
            # flush file each seed
            import sys; sys.stdout.flush()
    print('R1 DONE',flush=True)
main()
