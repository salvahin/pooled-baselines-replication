"""Shared dataset loaders for the strengthening experiments (A/B/C).
Loader bodies are copied verbatim from phase1_fairness2x2.py / phase2_generalization.py
so the strengthening runs use exactly the same data pipeline as the headline results."""
import numpy as np, pandas as pd
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE.parent.parent / 'data'
ML = HERE / 'mlbench'
SUB = 15000  # subsample cap for the two large code-smell datasets (matches phase1)

# ---------------- code-smell loaders (return {task: (X, y)}) ----------------
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

CODE_LOADERS={'IST2021':L_ist,'ImprovMLCQ':L_improv,'Crowdsmelling':L_crowd,'SmellyCode++':L_smelly,'ml-Codesmell':L_mlcs}

# ---------------- multi-class loaders (one-vs-rest tasks) ----------------
MC_CAP={'letter':1500}
def L_mc(name,seed,nf=5):
    d=np.load(ML/f'{name}.npz');X=d['X'].astype(float);y=d['y'].astype(int)
    cap=MC_CAP.get(name)
    if cap and len(X)>cap:
        rng=np.random.RandomState(seed);ix=rng.choice(len(X),cap,replace=False);X=X[ix];y=y[ix]
    classes=[c for c in sorted(set(y)) if (y==c).sum()>=2*nf]
    return {f'c{c}':(X,(y==c).astype(int)) for c in classes}
MC_DSETS=['digits','segment','vehicle','letter']
