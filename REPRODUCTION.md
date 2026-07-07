# Step-by-Step Reproduction Guide

This document provides detailed instructions to reproduce the experiments in *"An
Overlooked Baseline Artifact in Comparing Specialized and Pooled Classifiers, with
Evidence from Code-Smell Detection."*

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Downloading Datasets](#downloading-datasets)
4. [Running Experiments](#running-experiments)
5. [Verifying Results](#verifying-results)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

- **Python**: 3.9 or higher
- **RAM**: 8 GB minimum (16 GB recommended for the full-scale SmellyCode++ / ml-Codesmell reruns)
- **Disk Space**: ~1 GB for datasets and results
- **OS**: Linux, macOS, or Windows with WSL

## IMPORTANT: Dataset Download Required

**You must download the datasets before running any experiments.** Raw datasets are not
bundled due to size and licensing. See [data/README.md](data/README.md) for download
instructions with DOIs and verification steps. The four standard multi-class benchmarks
(digits, segment, vehicle, letter) are fetched automatically via `scikit-learn`/OpenML.

| Dataset | Size | Download Required |
|---------|------|-------------------|
| IST2021 | 6 CSV files (~420 rows each) | Yes — from GitHub |
| SmellyCode++ | ~590 MB (107,554 rows; includes a source-code column) | Yes — from Figshare |
| ImprovMLCQ | ~4 MB (13,489 rows) | Yes — from Zenodo |
| Crowdsmelling | small CSVs | Included in this package |
| ml-Codesmell | 373,400 rows, 41 features | Yes — from Figshare |
| digits / segment / vehicle / letter | small | Auto-fetched (sklearn/OpenML) |

---

## Environment Setup

### Option A: Using pip (Recommended)

```bash
git clone https://github.com/salvahin/pooled-baselines-replication.git
cd pooled-baselines-replication

python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### Option B: Using conda

```bash
conda create -n pooledbaselines python=3.10
conda activate pooledbaselines
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import sklearn, pandas, numpy, scipy, matplotlib; print('All dependencies installed successfully')"
```

### Pre-Flight Check (Verify Datasets)

```bash
cd data
python verify_data.py
```

**Do not proceed if any required datasets are missing.**

---

## Downloading Datasets

Place downloaded files in the `data/` directory. Full details, including alternate
mirrors and checksums, are in [data/README.md](data/README.md).

### 1. IST2021 (balanced, ~33% positive rate)

**Source**: https://github.com/hjamaan/IST2021-CodeSmellStackingEnsemble

```bash
cd data
git clone https://github.com/hjamaan/IST2021-CodeSmellStackingEnsemble.git
mkdir -p IST2021
cp IST2021-CodeSmellStackingEnsemble/Datasets/Original/*.csv IST2021/
```

Required files in `data/IST2021/`: `GodClass.csv`, `DataClass.csv`, `LongMethod.csv`,
`FeatureEnvy.csv`, `LongParameterList.csv`, `SwitchStatements.csv`.

### 2. SmellyCode++ (imbalanced, 1.5–4% positive rate)

**Source**: Alomari et al. (2025), *Scientific Data* — Figshare DOI
[10.6084/m9.figshare.28519385](https://doi.org/10.6084/m9.figshare.28519385), CC BY 4.0.

```bash
cd data
curl -L "https://ndownloader.figshare.com/files/52714583" -o "SmellyCode++.csv"
```

### 3. ImprovMLCQ (intermediate, 5–10% positive rate)

**Source**: Carneiro et al. (2025), SBCARS — Zenodo DOI
[10.5281/zenodo.14834187](https://doi.org/10.5281/zenodo.14834187).

```bash
cd data
curl -L "https://zenodo.org/api/records/14834187/files/out_clean.csv/content" -o ImprovMLCQ.csv
```

### 4. ml-Codesmell (tool-generated labels, 373,400 rows)

**Source**: Nguyen Thanh et al. (2022), SoICT — see `data/README.md` for the Figshare DOI
and file layout. Place the file as `data/mlcodesmell_class.csv`.

### 5. Standard multi-class benchmarks

`digits` and `segment`/`vehicle` are bundled or auto-fetched via `sklearn.datasets`;
`letter` is fetched via `fetch_openml` on first run (cached under `code/review_response/mlbench/`).

---

## Running Experiments

All current scripts live in `code/review_response/`. Each writes its CSV output to
`data/`. Run them from `code/review_response/`:

```bash
cd code/review_response
python phase1_fairness2x2.py     # headline: specific vs pooled-intercept vs pooled-interaction,
                                  # 4 classifiers x 5 seeds, 5 code-smell datasets
python phase2_generalization.py  # same 2x2 on the 4 multi-class tabular datasets
python phase3_specialize.py      # parameter count / training-cost accounting
python strengthen_A_imbalance.py # controlled positive-rate sweep (imbalance-as-moderator test)
python strengthen_B_threshold.py # F1@0.5 vs F1@best vs PR-AUC decomposition
python strengthen_C_params.py    # params, train time, accuracy parity
python full_rerun.py             # full-scale SmellyCode++ (107k) and ml-Codesmell (373k)
python revfix.py                 # matched-fold Phase 1 + 7-rate imbalance ladder
python revfix2.py                # softmax multi-class baseline + matched-fold threshold decomposition
```

`revfix.py`, `revfix2.py`, and `full_rerun.py` produce the versions of the analysis
reported in the final manuscript (matched folds, N=10/N=8 statistics, 7-point ladder,
full-scale primary tables, and the softmax comparison). The earlier phase/strengthen
scripts document the intermediate steps that led there and are kept for transparency.

**Expected runtime**: most scripts finish in minutes; `full_rerun.py`'s ml-Codesmell
pooled-stack cells are solver-bound (LinearSVC, `dual=False`) and can take on the order
of an hour per cell on the full 373,400-row / one-hot-expanded pooled stack.

### Regenerating result summaries

```bash
python aggregate_phases.py   # rebuilds PHASE_RESULTS_SUMMARY.md from the phase1/phase2 CSVs
```

---

## Verifying Results

Compare your output CSVs in `data/` to the pre-computed ones already included (same
filenames). Values should match within the stochastic tolerance of 5-seed averaging
(typically ±0.005–0.01 PR-AUC; see `code/review_response/STRENGTHEN_SUMMARY.md` for the
full-scale vs subsampled agreement figures).

Headline result: the conventional (one-hot, shared-weight) pooled baseline shows a large
linear-model gap (mean ΔPR-AUC ≈ +0.15 on the code-smell corpus, ≈ +0.66 on the multi-class
corpus) that collapses once the pooled model gets per-task weights (task×feature
interactions or softmax), while tree ensembles show almost no gap under either
construction. See `code/review_response/PHASE_RESULTS_SUMMARY.md` and
`STRENGTHEN_SUMMARY.md` for full tables.

---

## Troubleshooting

### Issue: "Dataset not found"

```bash
python -c "
import os
datasets = ['data/IST2021/GodClass.csv', 'data/SmellyCode++.csv', 'data/ImprovMLCQ.csv', 'data/mlcodesmell_class.csv']
for d in datasets:
    print(f'{d}: {\"FOUND\" if os.path.exists(d) else \"MISSING\"}')"
```

### Issue: Slow execution on ml-Codesmell full-scale runs

The LinearSVC pooled-stack cell in `full_rerun.py` is solver-bound. `dual=False` is
already set (correct when n_samples >> n_features); reducing to 3 seeds for that one
cell is an accepted, documented compromise (values agree with 5-seed runs to ±0.001–0.004).

### Issue: Different results

All scripts fix `random_state` per seed. Verify:
1. Same scikit-learn version (>= 1.0.0)
2. Same dataset file versions (see DOIs in `data/README.md`)
3. No modified hyperparameters or `class_weight='balanced'` settings

---

## Configuration Reference

| Parameter | Value | Justification |
|-----------|-------|----------------|
| `n_splits` | 10 | Standard stratified k-fold CV |
| `class_weight` | `'balanced'` | Applied identically to all arms |
| seeds | 5 (3 for the slowest ml-Codesmell full-scale LinearSVC cell) | Repeated-measures averaging |
| primary metric | PR-AUC (average precision) | Threshold-independent |

### File Structure

```
pooled-baselines-replication/
├── README.md                    # Overview and quick start
├── REPRODUCTION.md              # This file
├── requirements.txt
├── code/
│   └── review_response/         # All current analysis scripts (see README table)
├── data/
│   ├── README.md                # Dataset download instructions
│   ├── verify_data.py            # Dataset install checker
│   └── *.csv                    # Pre-computed results (phase1_*, phase2_*, strengthen_*, revfix_*, full_*)
└── notebooks/
    └── CodeSmells_Reproducibility.ipynb
```

---

## Contact

For questions or issues with reproduction, open an issue at
https://github.com/salvahin/pooled-baselines-replication/issues

---

## Citation

```bibtex
@article{avalos2026pooledbaselines,
  title   = {An Overlooked Baseline Artifact in Comparing Specialized and Pooled Classifiers, with Evidence from Code-Smell Detection},
  author  = {Avalos, Diego and Oliva, Diego and Garcia-Ceja, Enrique and Hinojosa, Salvador},
  journal = {(under review)},
  year    = {2026}
}
```
