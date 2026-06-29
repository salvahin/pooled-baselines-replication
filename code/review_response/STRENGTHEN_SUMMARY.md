# Strengthening results (2026-06-24/25) — hardening the 3 thin supporting claims

Runner: `strengthen_runner.log` (C,A,B all exit 0). Scripts: `strengthen_{A,B,C}_*.py`,
shared loaders `_loaders.py`. Raw CSVs in `../../data/strengthen_*_2026-06-24.csv`.
Same data pipeline as the headline Phase 1/2 runs.

## Part C — "specialize for free" (params / train time / accuracy parity)
All 9 datasets (5 code-smell + 4 multi-class), LogReg.
- Interaction-pooled parameter count / specialized parameter count: **1.04×–1.33×**
  (the K specialized models always use **equal or FEWER** parameters than the single
  pooled model that matches their accuracy).
- Accuracy parity: |ΔPR-AUC(specialized − interaction-pooled)| max **0.012**, mean **0.004**.
- Examples: IST2021 222 vs 259 params; digits 650 vs 715; letter 442 vs 459.
=> Specialization costs nothing in accuracy and no more (usually fewer) parameters;
   its only real difference is engineering (modular, independently deployable).

## Part A — does class imbalance moderate the pooled-baseline gap? (NO)
Controlled within-dataset positive-rate ladder {0.05,0.10,0.20,0.33}, all 9 datasets,
LogReg + RF, 5 seeds. Gap = PR-AUC(specific) − PR-AUC(intercept-pooled).
Linear (LogReg) per-dataset Spearman(rate, gap) sign:
- gap **grows** with imbalance (ρ<−0.2): IST2021, digits, ml-Codesmell, segment  (4)
- gap **shrinks** with imbalance (ρ>+0.2): ImprovMLCQ, SmellyCode++, vehicle       (3)
- flat (|ρ|≤0.2): letter                                                          (1)
=> The within-dataset trend is real but its **direction is dataset-specific** (4 vs 3,
   both with |ρ| up to 1.0). If imbalance were the moderator all datasets would trend
   the same way; they split. Positive rate is **not a generalizable moderator**.
   Tree gaps stay ≈0 (±0.01) at every rate (nothing to moderate).

## Part B — are F1@0.5 architecture differences threshold artifacts? (TREES: YES)
5 code-smell datasets × 4 classifiers × 5 seeds, 10-fold. Compares specific vs
intercept-pooled at default 0.5 vs each arm's F1-optimal threshold, plus PR-AUC vs
both pooled baselines.
- TREE (RF+HistGB): mean|dF1@0.5|=**0.025** → mean|dF1@best|=**0.013** (**49% smaller**
  at best threshold); ranking gap dAP_int=+0.005, dAP_itr=+0.001 (95% CI −0.001..+0.003,
  includes 0). => tree F1@0.5 "differences" are largely a thresholding artifact and the
  residual ranking difference is negligible.
- LIN (LogReg+LinearSVC): mean|dF1@0.5|=0.149 → mean|dF1@best|=0.111 (stays large);
  dAP_int=+0.148 but dAP_itr=−0.000. => linear's gap is NOT a threshold artifact — it is
  the baseline-underspecification gap, removed by the interaction baseline.
=> Two distinct mechanisms cleanly separated: trees → fixed-threshold F1 artifact;
   linear → under-specified-baseline artifact. Both inflate the apparent specialization
   advantage; PR-AUC + interaction baseline remove them respectively.

## Full-scale verification (no subsampling) — 2026-06-25
Re-ran the two previously-subsampled datasets at full row count under the identical
protocol (`full_rerun.py`; CSVs `data/full_phase1.csv` etc.). Confirms the 15k results.

| dataset | clf | d_int 15k | d_int full | d_itr 15k | d_itr full |
|---|---|---|---|---|---|
| SmellyCode++ | LogReg | +0.018 | +0.019 | -0.002 | +0.001 |
| SmellyCode++ | LinearSVC | +0.030 | +0.017 | -0.001 | +0.000 |
| SmellyCode++ | RF | -0.013 | -0.009 | -0.003 | -0.002 |
| SmellyCode++ | HistGB | -0.003 | +0.013 | +0.003 | +0.012 |
| ml-Codesmell | LogReg (5 seeds) | +0.479 | +0.500 | +0.000 | -0.007 |
| ml-Codesmell | LinearSVC (3 seeds) | +0.434 | +0.448 | -0.004 | -0.004 |

Note: ml-Codesmell LinearSVC at full is solver-bound (~2h/cell on the 1.1M-row pooled
stack even with dual=False); 3 seeds suffice (agree to +/-0.001). ml-Codesmell RF/HistGB
not re-run at full because their gap is ~0 at 15k already (no effect to be an artifact of).

## Reviewer-fix re-run (matched folds, N=10 stats, 7-rate ladder, full-scale) — 2026-06-25
External AI review flagged 4 valid issues; all addressed (`revfix.py`; CSVs revfix_phase1.csv, revfix_imbalance.csv).
- #1 independence: stats now at N=10 dataset×classifier units (seeds = repeated measures). Linear intercept gap +0.150, one-sample Wilcoxon p=0.002 (floor at N=10, all positive); trees +0.006, p=0.26 (n.s.). Interaction CI (-0.0025,+0.0012). MC: linear +0.664 p=0.008 (N=8).
- #2 Spearman: 7-point ladder (5–35%). Significant trends BOTH directions: digits/segment/ml-Codesmell -1.0 (p<0.001), IST2021 -0.93 (p=0.003) vs ImprovMLCQ +1.0 (p<0.001), SmellyCode++ +0.86 (p=0.014); letter/vehicle n.s.
- #3 matched folds: shared per-instance fold assignment for specific & pooled arms; results unchanged (linear +0.150 vs +0.148) → partitioning never the driver. Caveats deleted.
- #4b full-scale primary: Table 1/2 now run SmellyCode++ (107k) & ml-Codesmell (373k) at full size; subsampling demoted to the secondary imbalance sweep only. (ml-Codesmell LinearSVC full matched = 1 seed; +0.447, agrees with 3-seed +0.448.)
