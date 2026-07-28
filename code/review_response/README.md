# Review-response & robustness analysis (2026-06-24)

Supplementary analysis prompted by the Applied Sciences pre-submission review.
Findings summary: `../../../review-response-findings-2026-06-24.md`; results table:
`PHASE_RESULTS_SUMMARY.md`.

## Scripts
- `phase1_fairness2x2.py` — specific vs pooled-intercept vs pooled-interaction × 4 classifiers × 5 seeds (code-smell datasets).
- `phase2_generalization.py` — same 2×2 on multi-class tabular datasets (OvR vs pooled).
- `phase3_specialize.py` — params/train-time: specialized vs interaction-pooled.
- `robustness*.py`, `ist_down.py`, `imlcq_sweep.py`, `positive_rate_sweep.py`, `crowd.py`, `mlcs_run.py`, `fairness_test.py` — supporting experiments (R1–R6, sweeps, dataset evals).
- `aggregate_phases.py` — builds `PHASE_RESULTS_SUMMARY.md`.
- `revfix3_mlp.py` — revision 1: MLPClassifier (hidden 16/100) three-arm replication, matched folds (`revfix3_mlp.csv`).
- `latency_cost.py` — revision 1: train time + all-task batch-inference latency, specialized vs interaction-pooled (`latency_cost.csv`).
- `fig_boundary_2d.py` — revision 1: synthetic two-task 2D decision-boundary figure.
- Revision-1 results summary: `REVISION1_SUMMARY.md`.

## Data not versioned here (regenerate)
- `mlcodesmell_class.csv` — ml-Codesmell (Figshare DOI 10.6084/m9.figshare.21343299).
- `mlbench/*.npz` — multi-class datasets via `sklearn.datasets` / `fetch_openml` (digits, segment, vehicle, letter, mfeat-factors). Small ones (digits/segment/vehicle) are kept; large ones (letter, mfeat-factors) are regenerated.
- Code-smell datasets per the main package `data/README.md` + DOIs.
