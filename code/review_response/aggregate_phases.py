"""Aggregate Phase 1 + Phase 2 into mean +- 95% CI tables and an auto-verdict;
write PHASE_RESULTS_SUMMARY.md."""
import numpy as np,pandas as pd
from pathlib import Path
DATA=Path(__file__).resolve().parent.parent.parent/'data'
HERE=Path(__file__).resolve().parent
def ci(v): v=np.asarray(v,float); return (v.mean(), 1.96*v.std(ddof=1)/np.sqrt(len(v)) if len(v)>1 else 0.0)
lines=['# Phase 1+2 results summary (auto-generated)','']
def tbl(csvname, idcol, title):
    p=DATA/csvname
    if not p.exists(): lines.append(f'(missing {csvname})');return
    df=pd.read_csv(p)
    lines.append(f'## {title}'); lines.append('')
    lines.append('| dataset | classifier | Δ vs intercept-pooled | Δ vs interaction-pooled |')
    lines.append('|---|---|---|---|')
    for ds in df['dataset'].unique():
        for cn in ['LogReg','LinearSVC','RF','HistGB']:
            s=df[(df.dataset==ds)&(df.classifier==cn)]
            if len(s)==0: continue
            mi,ci_i=ci(s['d_intercept']); mj,ci_j=ci(s['d_interaction'])
            lines.append(f'| {ds} | {cn} | {mi:+.3f} ± {ci_i:.3f} | {mj:+.3f} ± {ci_j:.3f} |')
    lines.append('')
    # family verdict
    for fam,cls in [('LINEAR',['LogReg','LinearSVC']),('TREE',['RF','HistGB'])]:
        sub=df[df.classifier.isin(cls)]
        mi=sub['d_intercept'].mean(); mj=sub['d_interaction'].mean()
        lines.append(f'- **{fam}**: mean Δ(intercept-pooled)={mi:+.3f}, mean Δ(interaction-pooled)={mj:+.3f}')
    lines.append('')
tbl('phase1_fairness2x2_2026-06-24.csv','dataset','Phase 1 — code-smell datasets (ΔPR-AUC = specific − pooled)')
tbl('phase2_generalization_2026-06-24.csv','dataset','Phase 2 — non-code-smell multi-class datasets (OvR vs pooled)')
lines.append('## Auto-verdict')
lines.append('Artifact CONFIRMED if: LINEAR Δ(intercept) large & Δ(interaction)≈0, while TREE both ≈0 — in BOTH phases.')
open(HERE/'PHASE_RESULTS_SUMMARY.md','w').write('\n'.join(lines))
print('wrote PHASE_RESULTS_SUMMARY.md')
print('\n'.join(lines))
