"""Decision-boundary illustration (Information round 1, R1 comment 3).
Synthetic 2D two-task example rendered as a 2x3 grid (rows = tasks, cols = arms).
Task A's true boundary depends on x1, task B's on x2. A linear model under
pooled-intercept is forced to share one weight vector across tasks (only the
bias varies), producing the same wrong slope for both tasks; specific and
pooled-interaction recover each task's boundary.
Output: ../../..(/sp_clone)/figures/fig_boundary_2d.png  (also local copy)
"""
import numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from pathlib import Path

rng=np.random.RandomState(7)
n=260
def make_task(axis):
    X=rng.uniform(-3,3,size=(n,2))
    margin=X[:,axis]+rng.normal(0,0.55,n)      # noisy boundary on one axis
    y=(margin>0).astype(int)
    return X,y
XA,yA=make_task(0)   # task A: boundary x1=0 (vertical)
XB,yB=make_task(1)   # task B: boundary x2=0 (horizontal)

def aug(F,si,ns,inter):
    oh=np.zeros((len(F),ns)); oh[np.arange(len(F)),si]=1
    if not inter: return np.hstack([F,oh])
    return np.hstack([F,oh,np.einsum('ij,ik->ijk',F,oh).reshape(len(F),-1)])

mk=lambda:LogisticRegression(max_iter=2000,random_state=0)
# arm 1: specific
cA=mk().fit(XA,yA); cB=mk().fit(XB,yB)
# arms 2-3: pooled
Xs=np.vstack([XA,XB]); ys=np.concatenate([yA,yB]); si=np.concatenate([np.zeros(n,int),np.ones(n,int)])
c0=mk().fit(aug(Xs,si,2,False),ys)
c1=mk().fit(aug(Xs,si,2,True),ys)

# scoring helpers on a grid
gx,gy=np.meshgrid(np.linspace(-3,3,400),np.linspace(-3,3,400))
G=np.c_[gx.ravel(),gy.ravel()]
def surf_specific(task):    return (cA if task==0 else cB).predict_proba(G)[:,1]
def surf_pooled(c,task,inter):
    return c.predict_proba(aug(G,np.full(len(G),task,int),2,inter))[:,1]

ARMS=[('Specific\n(one model per task)',        lambda t:surf_specific(t),          '#333333'),
      ('Pooled-intercept\n(conventional baseline)', lambda t:surf_pooled(c0,t,False), '#c0392b'),
      ('Pooled-interaction\n(fair baseline)',    lambda t:surf_pooled(c1,t,True),    '#2e6e9e')]
TASKS=[('Task A  (true boundary: $x_1=0$)',XA,yA,0),
       ('Task B  (true boundary: $x_2=0$)',XB,yB,1)]

fig,axes=plt.subplots(2,3,figsize=(11,7.2),sharex=True,sharey=True)
for r,(tname,X,y,tidx) in enumerate(TASKS):
    for c,(aname,fsurf,col) in enumerate(ARMS):
        ax=axes[r,c]
        Z=fsurf(tidx).reshape(gx.shape)
        ax.contourf(gx,gy,Z,levels=[0,0.5,1],colors=['#f5f5f5','#e8eef3'],alpha=.9)
        ax.contour(gx,gy,Z,levels=[0.5],colors=[col],linewidths=2.4)
        ax.scatter(X[y==0][:,0],X[y==0][:,1],s=9,c='#b8b8b8',edgecolors='none',label='negative')
        ax.scatter(X[y==1][:,0],X[y==1][:,1],s=11,c='#4d4d4d',edgecolors='none',label='positive')
        # true boundary, dashed
        if tidx==0: ax.axvline(0,color='#888888',ls='--',lw=1.1)
        else:       ax.axhline(0,color='#888888',ls='--',lw=1.1)
        if r==0: ax.set_title(aname,fontsize=11.5)
        if c==0: ax.set_ylabel(tname+'\n$x_2$',fontsize=10.5)
        if r==1: ax.set_xlabel('$x_1$',fontsize=10.5)
        ax.set_xlim(-3,3); ax.set_ylim(-3,3)
        ax.tick_params(labelsize=8.5)
for spine_ax in axes.ravel():
    for s in spine_ax.spines.values(): s.set_color('#cccccc')
handles=[plt.Line2D([],[],color='#888888',ls='--',lw=1.1,label='true boundary'),
         plt.Line2D([],[],color='#333333',lw=2.4,label='learned boundary (0.5)')]
fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,fontsize=10)
fig.tight_layout(rect=[0,0.045,1,1])

here=Path(__file__).resolve().parent
out_local=here/'fig_boundary_2d.png'
fig.savefig(out_local,dpi=220)
sp=here.parent.parent.parent.parent/'figures'
if sp.exists():
    fig.savefig(sp/'fig_boundary_2d.png',dpi=220)
    print('saved',sp/'fig_boundary_2d.png')
print('saved',out_local)
# print learned weights to verify the mechanism numerically
print('intercept-pooled shared weights  w =',np.round(c0.coef_[0][:2],2),' (same for both tasks)')
print('interaction-pooled task-A slope =',np.round(c1.coef_[0][:2]+c1.coef_[0][4:6],2))
print('interaction-pooled task-B slope =',np.round(c1.coef_[0][:2]+c1.coef_[0][6:8],2))
