"""Phase 3: 'specialize for free'. The interaction-pooled model matches specific
accuracy but is one monolith; K specific models are independently deployable/
updatable. Quantify params + train time (LogReg) on a few datasets."""
import time,warnings,numpy as np,pandas as pd
warnings.filterwarnings('ignore')
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
HERE=Path(__file__).resolve().parent; DATA=HERE.parent.parent/'data'; ML=HERE/'mlbench'
def lr(): return LogisticRegression(class_weight='balanced',max_iter=2000,random_state=42)
def ist():
    IST={'god_class':('GodClass.csv','is_god_class'),'data_class':('DataClass.csv','is_data_class'),'long_method':('LongMethod.csv','is_long_method'),'feature_envy':('FeatureEnvy.csv','is_feature_envy'),'long_parameter_list':('LongParameterList.csv','is_long_parameters_list'),'switch_statements':('SwitchStatements.csv','is_switch_statements')}
    d=DATA/'IST2021';sets=[]
    for k,(fn,tg) in IST.items(): sets.append({c for c in pd.read_csv(d/fn).columns if c!=tg})
    common=sorted(set.intersection(*sets));pf={}
    for k,(fn,tg) in IST.items():
        df=pd.read_csv(d/fn);pf[k]=(np.nan_to_num(df[common].values,nan=0.0),df[tg].apply(lambda v:1 if v in [True,'TRUE',1,'1'] else 0).values)
    return pf
def digits():
    d=np.load(ML/'digits.npz');X=d['X'].astype(float);y=d['y'].astype(int)
    return {f'c{c}':(X,(y==c).astype(int)) for c in sorted(set(y))}
def analyze(name,pf):
    order=list(pf);ns=len(order);f=pf[order[0]][0].shape[1]
    # specific: K binary models
    t=time.time();pspec=0
    for k in order:
        X,y=pf[k];s=StandardScaler();c=lr();c.fit(s.fit_transform(X),y);pspec+=c.coef_.size+c.intercept_.size
    tspec=time.time()-t
    # interaction-pooled: one model
    Xs=[];ys=[];idx=[]
    for si,k in enumerate(order):
        X,y=pf[k];Xs.append(X);ys.append(y);idx.append(np.full(len(X),si))
    Xa=np.vstack(Xs);ya=np.concatenate(ys);sidx=np.concatenate(idx)
    sc=StandardScaler();F=sc.fit_transform(Xa);oh=np.zeros((len(F),ns));oh[np.arange(len(F)),sidx]=1
    A=np.hstack([F,oh,np.einsum('ij,ik->ijk',F,oh).reshape(len(F),-1)])
    t=time.time();c=lr();c.fit(A,ya);tpool=time.time()-t;ppool=c.coef_.size+c.intercept_.size
    print(f'{name}: tasks={ns}, features={f}')
    print(f'  specific (K models): params={pspec}, train={tspec:.2f}s, independently deployable/updatable=YES')
    print(f'  interaction-pooled (1 model): params={ppool}, train={tpool:.2f}s, retrain-all-on-any-change=YES, needs F x task interaction engineering')
    print(f'  -> same accuracy (Phase1/2), but specific is {ppool/max(pspec,1):.1f}x fewer params and modular',flush=True)
print('PHASE 3: specialize-for-free (LogReg)')
analyze('IST2021',ist()); analyze('digits',digits())
print('PHASE3 DONE',flush=True)
