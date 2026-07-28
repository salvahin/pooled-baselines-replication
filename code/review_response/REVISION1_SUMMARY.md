# Revision-1 results (2026-07-28) — Information round 1

Analyses added in response to the first review round at *Information*.
Scripts: `revfix3_mlp.py`, `latency_cost.py`, `fig_boundary_2d.py`.
CSVs: `../../data/revfix3_mlp.csv`, `../../data/latency_cost.csv`.
Protocol identical to the headline runs: matched per-instance folds shared by all
three arms, PR-AUC, seeds [42,1,7,13,99]; the two large datasets use the established
15k subsample (subsample~full agreement shown in the manuscript's robustness checks).

## Part A — MLP three-arm replication (reviewer 1, comment 1)

MLPClassifier, hidden widths 16 and 100, input task one-hot for the pooled-intercept
arm, explicit task×feature interaction block for the pooled-interaction arm.
Hypothesis stated in advance: an input-conditioned network sits structurally in the
intercept-pooled regime and escapes it only by spending capacity synthesizing the
missing interactions.

d = ΔPR-AUC (specific − pooled), mean ± sd over 5 seeds:

| dataset | w16 d_int | w16 d_itr | w100 d_int | w100 d_itr |
|---|---|---|---|---|
| IST2021 | -0.075 ± 0.058 | -0.165 ± 0.059 | -0.055 ± 0.017 | -0.089 ± 0.026 |
| ImprovMLCQ | -0.014 ± 0.016 | -0.055 ± 0.022 | -0.032 ± 0.011 | -0.057 ± 0.008 |
| Crowdsmelling | +0.009 ± 0.015 | -0.024 ± 0.007 | +0.001 ± 0.024 | -0.029 ± 0.010 |
| SmellyCode++ | -0.018 ± 0.006 | -0.016 ± 0.011 | -0.017 ± 0.008 | -0.029 ± 0.013 |
| ml-Codesmell | -0.036 ± 0.012 | -0.054 ± 0.015 | -0.007 ± 0.016 | -0.021 ± 0.011 |
| digits | **+0.772 ± 0.081** | -0.044 ± 0.015 | -0.010 ± 0.008 | -0.012 ± 0.006 |
| segment | -0.127 ± 0.109 | -0.203 ± 0.175 | -0.043 ± 0.008 | -0.047 ± 0.006 |
| vehicle | **+0.217 ± 0.036** | -0.263 ± 0.049 | **+0.392 ± 0.035** | -0.139 ± 0.015 |
| letter | **+0.113 ± 0.027** | -0.673 ± 0.023 | -0.599 ± 0.069 | -0.693 ± 0.029 |

Two findings:
1. The intercept-pooled artifact **re-emerges in the network** where capacity is scarce
   relative to task heterogeneity, up to linear-scale magnitude (digits w16 +0.772,
   vanishing at w100; vehicle at BOTH widths; letter w16). Capacity alone is no
   guarantee (vehicle defeats width 100). Elsewhere, including all five code-smell
   datasets, the input-conditioned network behaves like a tree ensemble (gap ≈ 0 or
   negative).
2. The interaction-pooled network matched or beat the specialized networks in **all 18
   dataset×capacity combinations** (d_itr ≤ 0 everywhere, to -0.69 on letter, whose 26
   small per-task networks fragment the training data). For neural models,
   specialization is often actively harmful; a fairly-specified pooled network is the
   safe default.

Note: MLPClassifier has no class_weight; all three arms run identically unweighted,
so the within-model comparison remains fair (stated in the manuscript).

## Part B — wall-clock costs (reviewer 1, comment 2)

LogReg, seed 42, single process. Train = all K specialized fits vs the one
interaction-pooled fit; infer = scoring ALL K tasks for a 10k batch (median of 5;
pooled time includes building its augmented matrix).

| dataset | train spec (s) | train pool (s) | infer spec (ms) | infer pool (ms) |
|---|---|---|---|---|
| IST2021 | 8.2 | 3.9 | 124 | 24 |
| ImprovMLCQ | 34.4 | 22.7 | 80 | 58 |
| Crowdsmelling | 5.0 | 4.7 | 85 | 23 |
| SmellyCode++ | 7.8 | 9.2 | 107 | 50 |
| ml-Codesmell | 6.0 | 5.9 | 64 | 88 |
| digits | 10.6 | 3.0 | 266 | 74 |
| segment | 5.7 | 3.2 | 147 | 34 |
| vehicle | 5.6 | 4.5 | 82 | 19 |
| letter | 20.3 | 4.8 | 609 | 80 |

=> Neither design dominates on speed: the pooled model trains faster in one process
(7/9) and wins all-task batch inference (8/9, up to 7.6× on letter); the specialized
set is parameter-cheaper, embarrassingly parallel across cores, and does 1/K of the
work when only one task's score is needed. Durable differences are parameters and
operational structure, not wall-clock.

## Part C — decision-boundary figure (reviewer 1, comment 3)

`fig_boundary_2d.py`: synthetic two-task 2D example (task A boundary on x1, task B on
x2). Learned weights confirm the mechanism numerically: intercept-pooled shares
w=(0.71, 0.73) across both tasks (one wrong diagonal, shifted per task); the
interaction-pooled model recovers per-task slopes (2.94, 0.08) and (0.07, 3.09).
Figure appears in the manuscript's problem-formulation section.
