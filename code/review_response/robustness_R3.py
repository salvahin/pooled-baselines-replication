"""R3: threshold analysis. Per dataset/classifier: ΔF1@0.5 vs ΔF1@best-threshold
(per-arm F1-optimal on the fold) vs ΔPR-AUC. If even best-threshold gap is small,
the 0.5 gap is a thresholding artifact."""
import warnings,numpy as np,pandas as pd
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, average_precision_score, precision_recall_curve
NF=10;SEED=42; HERE=Path(__file__).resolve().parent; DATA=HERE.parent.parent/'data'
def sco(c,X): return c.predict_proba(X)[:,1] if hasattr(c,'predict_proba') else c.decision_function(X)
def bestf1(y,s):
    p,r,th=precision_recall_curve(y,s); f1=2*p*r/(p+r+1e-12); return float(np.nanmax(f1))
def arms(fac, perfile, nfeat):
    order=list(perfile)
    res={'spec':{'f105':{},'fbest':{},'ap':{}}}
    sp={k:{'f105':[],'fbest':[],'ap':[]} for k in order}
    for k in order:
        X,y=perfile[k]
        if y.sum()<NF: continue
        for tr,te in StratifiedKFold(NF,shuffle=True,random_state=SEED).split(X,y):
            s=StandardScaler();Xtr=s.fit_transform(X[tr]);Xte=s.transform(X[te])
            c=fac();c.fit(Xtr,y[tr]);pr=sco(c,Xte)
            sp[k]['f105'].append(f1_score(y[te],(pr>=0.5).astype(int) if hasattr(c,'predict_proba') else c.predict(Xte),zero_division=0))
            sp[k]['fbest'].append(bestf1(y[te],pr));sp[k]['ap'].append(average_precision_score(y[te],pr))
    ns=len(order);Xl=[];yl=[];sl=[]
    for si,k in enumerate(order):
        X,y=perfile[k];oh=np.zeros(ns);oh[si]=1
        Xl.append(np.hstack([X,np.tile(oh,(len(X),1))]));yl.append(y);sl.append(np.array([k]*len(X)))
    Xc=np.vstack(Xl);yc=np.concatenate(yl);scl=np.concatenate(sl);strat=LabelEncoder().fit_transform(scl)
    mp={k:{'f105':[],'fbest':[],'ap':[]} for k in order}
    for tr,te in StratifiedKFold(NF,shuffle=True,random_state=SEED).split(Xc,strat):
        Xtr,Xte=Xc[tr].copy(),Xc[te].copy();s=StandardScaler();Xtr[:,:nfeat]=s.fit_transform(Xtr[:,:nfeat]);Xte[:,:nfeat]=s.transform(Xte[:,:nfeat])
        c=fac();c.fit(Xtr,yc[tr]);pr=sco(c,Xte);pred=c.predict(Xte);ts=scl[te]
        for k in order:
            m=ts==k
            if m.sum()and yc[te][m].sum()>0:
                mp[k]['f105'].append(f1_score(yc[te][m],pred[m],zero_division=0));mp[k]['fbest'].append(bestf1(yc[te][m],pr[m]));mp[k]['ap'].append(average_precision_score(yc[te][m],pr[m]))
    ks=[k for k in order if sp[k]['ap'] and mp[k]['ap']]
    d105=np.mean([np.mean(sp[k]['f105'])-np.mean(mp[k]['f105']) for k in ks])
    dbest=np.mean([np.mean(sp[k]['fbest'])-np.mean(mp[k]['fbest']) for k in ks])
    dap=np.mean([np.mean(sp[k]['ap'])-np.mean(mp[k]['ap']) for k in ks])
    return d105,dbest,dap
# loaders (RF+LogReg on 4 datasets)
def load_ist():
    IST={'god_class':('GodClass.csv','is_god_class'),'data_class':('DataClass.csv','is_data_class'),'long_method':('LongMethod.csv','is_long_method'),'feature_envy':('FeatureEnvy.csv','is_feature_envy'),'long_parameter_list':('LongParameterList.csv','is_long_parameters_list'),'switch_statements':('SwitchStatements.csv','is_switch_statements')}
    d=DATA/'IST2021';sets=[]
    for k,(fn,tg) in IST.items():
        df=pd.read_csv(d/fn);sets.append({c for c in df.columns if c!=tg})
    common=sorted(set.intersection(*sets));pf={}
    for k,(fn,tg) in IST.items():
        df=pd.read_csv(d/fn);pf[k]=(np.nan_to_num(df[common].values,nan=0.0),df[tg].apply(lambda v:1 if v in [True,'TRUE',1,'1'] else 0).values)
    return pf,len(common)
def load_improv():
    df=pd.read_csv(DATA/'ImprovMLCQ.csv');ck=sorted([c for c in df.columns if c.startswith('ck_')]);X=np.nan_to_num(df[ck].values,nan=0.0)
    IM={'blob':'blob_label','data_class':'dataclass_label','feature_envy':'featureenvy_label','long_method':'longmethod_label'}
    return {k:(X,(df[c]==1).astype(int).values) for k,c in IM.items()},X.shape[1]
def load_crowd():
    base=HERE/'crowdsmelling_data';files={'god_class':('god-class-2020+2019+2018.csv','is_god_class'),'long_method':('long-method-2020+2019+2018.csv','is_long_method'),'feature_envy':('feature-envy-2020+2019+2018.csv','is_feature_envy')}
    META={'username','project','package','complextype','method','methodname'};data={};cs=[]
    for k,(fn,lab) in files.items():
        df=pd.read_csv(base/fn);y=df[lab].apply(lambda v:1 if str(v).upper() in ('TRUE','1') else 0).values
        Xd=df[[c for c in df.columns if c not in META and c!=lab]].apply(pd.to_numeric,errors='coerce').dropna(axis=1,how='any');data[k]=(Xd,y);cs.append(set(Xd.columns))
    common=sorted(set.intersection(*cs));return {k:(np.nan_to_num(data[k][0][common].values,nan=0.0),data[k][1]) for k in files},len(common)
def load_smelly():
    HAL=['Logical Lines','Distinct Operators','Distinct Operands','Total Operators','Total Operands','Vocabulary','Length','Calculated Length','Volume','Difficulty','Effort','Time Required','Bugs','Cyclomatic Complexity']
    SM={'god_class':'God class','long_method':'Long method','feature_envy':'Feature envy','data_class':'Data class'}
    df=pd.read_csv(DATA/'SmellyCode++.csv',usecols=HAL+list(SM.values())).sample(n=40000,random_state=SEED).reset_index(drop=True)
    X=np.nan_to_num(df[HAL].values,nan=0.0);return {k:(X,(df[c]==1).astype(int).values) for k,c in SM.items()},X.shape[1]
fac={'RF':lambda:RandomForestClassifier(n_estimators=100,class_weight='balanced',random_state=SEED,n_jobs=-1),'LogReg':lambda:LogisticRegression(class_weight='balanced',max_iter=1000,random_state=SEED)}
print('dataset,classifier,dF1@0.5,dF1@best,dPR-AUC',flush=True)
for dn,ld in [('IST2021',load_ist),('ImprovMLCQ',load_improv),('Crowdsmelling',load_crowd),('SmellyCode++',load_smelly)]:
    pf,nf=ld()
    for cn in ['RF','LogReg']:
        d105,dbest,dap=arms(fac[cn],pf,nf)
        print(f'{dn},{cn},{d105:+.3f},{dbest:+.3f},{dap:+.3f}',flush=True)
print('R3 DONE',flush=True)
