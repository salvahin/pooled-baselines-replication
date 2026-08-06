# Replication Package — *An Overlooked Baseline Artifact in Comparing Specialized and Pooled Classifiers*

Code, pre-computed results, and instructions to reproduce the experiments in the paper.

## What the paper shows

Comparing a **specialized** classifier per task against a single **pooled** model that
handles all tasks is a routine decision in applied machine learning, and reported answers
disagree. Using code-smell detection as a case study, we show the disagreement is largely
an artifact of how the pooled baseline is built and scored, not a real architecture effect:

- The conventional pooled baseline (features **+ a task one-hot indicator**) is
  under-specified for **linear** models: it grants only a per-task intercept and forces all
  tasks to share one feature-weight vector, so per-task models appear better for a reason
  unrelated to specialization. **Tree ensembles** are barely affected (they can split on the
  one-hot). The gap is therefore *model-class-dependent*.
- Giving the pooled model **per-task weights** — task×feature interactions, or a genuine
  **softmax/multinomial** classifier — closes the gap. Softmax matches or exceeds the
  specialized models; only the shared-weight one-hot construction is deficient.
- The effect **reproduces on standard multi-class tabular benchmarks** (digits, segment,
  vehicle, letter) unrelated to software, so it is a property of pooled-baseline
  construction, not of code smells.
- **Class imbalance does not moderate** the choice: controlled within-dataset positive-rate
  sweeps trend in opposite directions across datasets.
- Apparent F1 differences for tree models are **fixed-threshold artifacts** that vanish
  under PR-AUC.
- The artifact **re-emerges in neural networks**: an MLP given only an input task one-hot
  shows the same construction-dependence, up to +0.77 PR-AUC where capacity is scarce
  relative to task heterogeneity (capacity- and dataset-dependent), while the
  interaction-pooled network matched or beat the specialized networks in **all 18
  dataset x capacity combinations**.
- A fairly specified pooled model matches the specialized models in accuracy, so the choice
  is an **engineering** decision with real trade-offs (the pooled model trains faster in a
  single process and serves all-task batches with lower latency; the specialized set is
  parameter-cheaper, embarrassingly parallel, and modular). We distil a short
  **fair-comparison protocol** for such studies.

## Datasets

Large datasets are **not** included; download them from their original sources (see
`data/README.md`). Standard multi-class datasets are fetched via `scikit-learn`/OpenML.

| Dataset | Metrics | Labels | Notes |
|---|---|---|---|
| IST2021 | CK (36 common) | manual | balanced (~33%) |
| ImprovMLCQ | CK | manual (MLCQ) | intermediate (5–10%) |
| Crowdsmelling | CK | crowdsourced | included (small CSVs) |
| SmellyCode++ | Halstead | hybrid | severe imbalance (1.5–4%), 107,554 rows |
| ml-Codesmell | iPlasma | tool-generated | 373,400 rows (label-provenance caveat) |
| digits / segment / vehicle / letter | — | — | sklearn/OpenML, one-vs-rest tasks |

## Key scripts (`code/review_response/`)

| Script | Produces |
|---|---|
| `phase1_fairness2x2.py` | Headline ΔPR-AUC: specific vs pooled-intercept vs pooled-interaction × 4 classifiers × 5 seeds (5 code-smell datasets) |
| `phase2_generalization.py` | Same on the 4 multi-class tabular datasets |
| `phase3_specialize.py` | Parameter / training-cost accounting |
| `strengthen_A_imbalance.py` | Controlled positive-rate sweep (imbalance-as-moderator test) |
| `strengthen_B_threshold.py` | F1@0.5 vs F1@best vs PR-AUC threshold decomposition |
| `strengthen_C_params.py` | Params, train time, accuracy parity across datasets |
| `full_rerun.py` | Full-scale (no subsampling) SmellyCode++ (107k) and ml-Codesmell (373k) |
| `revfix.py` | Matched-fold Phase 1 (shared per-instance folds) + 7-rate imbalance ladder |
| `revfix2.py` | Softmax multi-class baseline + matched-fold threshold decomposition |
| `_loaders.py` | Shared dataset loaders |
| `revfix3_mlp.py` | Neural-network (MLP, widths 16/100) replication of the three-arm comparison (revision 1) |
| `latency_cost.py` | Train-time and batch-inference latency of specialized vs interaction-pooled (revision 1) |
| `fig_boundary_2d.py` | Synthetic 2D decision-boundary figure (revision 1) |

Result summaries: `code/review_response/PHASE_RESULTS_SUMMARY.md` and `STRENGTHEN_SUMMARY.md`.
Pre-computed CSVs are in `data/` (e.g., `phase1_*`, `phase2_*`, `strengthen_*`, `revfix_*`, `revfix3_*`, `latency_*`, `full_*`).

## Requirements

```bash
pip install -r requirements.txt   # Python 3.9+, scikit-learn, pandas, numpy, scipy, matplotlib
```

## Reproduce

1. Download the datasets per `data/README.md`.
2. From `code/review_response/`, run the scripts above (each writes CSVs to `../../data/`).
   The matched-fold, full-scale, and 7-rate analyses (`revfix.py`, `revfix2.py`,
   `full_rerun.py`) are the versions reported in the final manuscript.
3. See `REPRODUCTION.md` for step-by-step details.

## Notes

- Metrics are threshold-independent (PR-AUC) by default; F1 at the default and optimal
  thresholds is reported for the threshold analysis.
- `class_weight='balanced'` and default hyperparameters are applied identically to all arms.

## Citation

```bibtex
@article{avalos2026pooledbaselines,
  title   = {An Overlooked Baseline Artifact in Comparing Specialized and Pooled Classifiers},
  author  = {Avalos, Diego and Oliva, Diego and Garcia-Ceja, Enrique and Hinojosa, Salvador},
  journal = {(under review)},
  year    = {2026}
}
```

## License

MIT License.
