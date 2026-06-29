# Step-by-Step Reproduction Guide

This document provides detailed instructions to reproduce all experiments from the paper *"Joint vs. Independent Learning: How Class Imbalance Dictates Classifier Architecture in Code Smell Detection"*.

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

- **Python**: 3.8 or higher
- **RAM**: 8 GB minimum (16 GB recommended for SmellyCode++ experiments)
- **Disk Space**: 500 MB for datasets and results
- **OS**: Linux, macOS, or Windows with WSL

## IMPORTANT: Dataset Download Required

**You must download the datasets before running any experiments.**

The raw datasets are not bundled due to size and licensing. See [data/README.md](data/README.md) for download instructions with DOIs and verification steps.

| Dataset | Size | Download Required |
|---------|------|-------------------|
| IST2021 | 6 CSV files (~420 rows each) | Yes - from GitHub |
| SmellyCode++ | ~590 MB (107,554 rows; includes a source-code column) | Yes - from Figshare |
| ImprovMLCQ | ~4 MB (13,489 rows) | Yes - from Zenodo |

---

## Environment Setup

### Option A: Using pip (Recommended)

```bash
# Clone the repository
git clone https://github.com/salvahin/smells-paper.git
cd smells-paper

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Option B: Using conda

```bash
conda create -n codesmells python=3.10
conda activate codesmells
pip install -r requirements.txt
```

### Verify Installation

```bash
python -c "import sklearn, pandas, numpy, scipy, imblearn; print('All dependencies installed successfully')"
```

### Pre-Flight Check (Verify Datasets)

Before running experiments, verify all datasets are correctly installed:

```bash
cd data
python verify_data.py
```

This script (included in `data/README.md`) checks:
- IST2021: 6 CSV files with ~420 rows each
- SmellyCode++: ~107,325 rows
- ImprovMLCQ: ~13,489 rows with CK metrics

**Do not proceed if any datasets are missing.**

---

## Downloading Datasets

The datasets must be downloaded separately due to licensing. Place them in the `data/` directory.

### 1. IST2021 Dataset (Balanced, ~33% positive rate)

**Source**: https://github.com/hjamaan/IST2021-CodeSmellStackingEnsemble

```bash
cd data
git clone https://github.com/hjamaan/IST2021-CodeSmellStackingEnsemble.git
mkdir -p IST2021
# CSVs are under Datasets/Original/ in the upstream repo (not Data/)
cp IST2021-CodeSmellStackingEnsemble/Datasets/Original/*.csv IST2021/
```

Required files in `data/IST2021/`:
- `GodClass.csv`
- `DataClass.csv`
- `LongMethod.csv`
- `FeatureEnvy.csv`
- `LongParameterList.csv`
- `SwitchStatements.csv`

### 2. SmellyCode++ Dataset (Imbalanced, 1.5-4% positive rate)

**Source**: Alomari et al. (2025), *Scientific Data* — Figshare DOI [10.6084/m9.figshare.28519385](https://doi.org/10.6084/m9.figshare.28519385), CC BY 4.0.

```bash
cd data
# ≈590 MB; the upstream file is multi-smell-dataset-v1_2.csv
curl -L "https://ndownloader.figshare.com/files/52714583" -o "SmellyCode++.csv"
```

(The earlier DOI `10.6084/m9.figshare.28234218` cited in prior drafts no longer resolves.) See [data/README.md](data/README.md) for details.

### 3. ImprovMLCQ Dataset (Intermediate, 5-10% positive rate)

**Source**: Silva et al. (2025), SBCARS — Zenodo DOI [10.5281/zenodo.14834187](https://doi.org/10.5281/zenodo.14834187).

This file is **not** bundled and must be downloaded (the Zenodo file is named `out_clean.csv`):

```bash
cd data
curl -L "https://zenodo.org/api/records/14834187/files/out_clean.csv/content" -o ImprovMLCQ.csv
```

### Verify Dataset Setup

```bash
ls -la data/
# Should show:
# - IST2021/          (directory with 6 CSV files)
# - SmellyCode++.csv  (107,325 rows)
# - ImprovMLCQ.csv    (13,489 rows)
```

---

## Running Experiments

### Quick Start: Run All Experiments

```bash
cd code
python run_experiments.py
```

**Expected runtime**: 30-45 minutes on a modern CPU

**Output**: Results saved to `data/` directory as timestamped CSV files.

### Individual Experiment Scripts

#### RQ1-RQ3: Core Classification Experiments

```bash
python code/run_experiments.py
```

This runs:
- **RQ1**: IST2021 smell-specific vs. unified comparison
- **RQ2a**: SmellyCode++ multi-label experiments
- **RQ2b**: ImprovMLCQ transition zone analysis
- **RQ3**: SMOTE boundary conditions

#### Cross-Classifier Validation (Table 19)

```bash
python code/run_multiclassifier_experiments.py
```

This tests the 10%/5% thresholds across four classifier families:
- RandomForest (baseline)
- LinearSVC (kernel-based)
- HistGradientBoosting (boosting ensemble)
- LogisticRegression (linear baseline)

**Expected runtime**: 15-20 minutes

#### Metaheuristic Feature Selection (RQ4)

```bash
python code/run_metaheuristic_experiments.py
```

This tests wrapper-based feature selection using PSO, SA, GWO, and WOA via the MAFESE framework.

**Requirements**:
- MAFESE library: `pip install mafese`
- SmellyCode++ dataset

**Expected runtime**: 2-3 hours (grid search over 108 configurations)

**Note**: If MAFESE is not installed or dataset is missing, the script displays pre-computed results from Table 18.

### Regenerating Figures

After running experiments:

```bash
python code/regenerate_figures.py
```

This generates:
- `fig1_ist2021_comparison.png` - RQ1 results
- `fig2_smellycode_comparison.png` - RQ2a results
- `fig3_boundary_forest_plot.png` - RQ3 effect sizes

---

## Verifying Results

### Expected Results Summary

#### IST2021 (Balanced, ~33% positive rate)
- **Expected Average ΔF1**: +0.060 (specific classifiers outperform)
- **Interpretation**: Smell-specific classifiers significantly better on balanced data

#### SmellyCode++ (Imbalanced, 1.5-4% positive rate)
- **Expected Average ΔF1**: -0.002 (essentially tied)
- **Interpretation**: No advantage to specialization on severely imbalanced data

#### ImprovMLCQ (Transition zone, 5-10% positive rate)
- **Expected findings**:
  - Blob (10.3%): ΔF1 ≈ +0.21 (specific wins)
  - Data Class (9.8%): ΔF1 ≈ +0.23 (specific wins)
  - Feature Envy (5.6%): ΔF1 ≈ -0.33 (unified wins)
  - Long Method (5.1%): ΔF1 ≈ -0.26 (unified wins)
- **Interpretation**: Transition occurs between 5% and 10% positive rate

### Comparing Your Results to Ours

Pre-computed results are included in the `data/` directory with `_pr_` suffix:

```bash
# Compare your IST2021 results to ours
diff <(cut -d',' -f1,2,5 data/ist2021_results_*.csv | head -7) \
     <(cut -d',' -f1,2,5 data/ist2021_results_pr_*.csv | head -7)
```

Results should match within ±0.01 F1 due to random seed fixing.

---

## Troubleshooting

### Issue: "Memory Error" on SmellyCode++

SmellyCode++ has 107,325 samples. If you encounter memory issues:

```python
# In run_experiments.py, reduce to 50,000 samples:
df = df.sample(n=50000, random_state=42)
```

### Issue: "Dataset not found"

Ensure datasets are in the correct location:

```bash
python -c "
import os
datasets = ['data/IST2021/GodClass.csv', 'data/SmellyCode++.csv', 'data/ImprovMLCQ.csv']
for d in datasets:
    print(f'{d}: {\"FOUND\" if os.path.exists(d) else \"MISSING\"}')"
```

### Issue: Different Results

Our experiments use `random_state=42` throughout. Verify:
1. Same scikit-learn version (>= 1.0.0)
2. Same dataset versions
3. No modifications to hyperparameters

### Issue: Slow Execution

The cross-classifier experiments can be parallelized by classifier:

```bash
# Run in parallel (4 terminals)
python code/run_multiclassifier_experiments.py --classifier RandomForest &
python code/run_multiclassifier_experiments.py --classifier LinearSVC &
python code/run_multiclassifier_experiments.py --classifier HistGradientBoosting &
python code/run_multiclassifier_experiments.py --classifier LogisticRegression &
```

---

## Configuration Reference

### Experimental Settings

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `random_state` | 42 | Reproducibility |
| `n_splits` | 10 | Standard stratified k-fold CV |
| `n_estimators` | 100 | RandomForest default, sufficient for convergence |
| `class_weight` | 'balanced' | Handles imbalanced classes |
| `SMOTE k-neighbors` | 5 | Default, validated in literature |

### File Structure

```
smells-paper/
├── README.md                    # Overview and quick start
├── REPRODUCTION.md              # This file (detailed instructions)
├── requirements.txt             # Python dependencies
├── code/
│   ├── run_experiments.py       # RQ1-RQ3 experiments
│   ├── run_multiclassifier_experiments.py  # Cross-classifier validation
│   └── regenerate_figures.py    # Figure generation
├── data/
│   ├── README.md                # Dataset download instructions
│   ├── IST2021/                 # IST2021 dataset (download required)
│   ├── SmellyCode++.csv         # SmellyCode++ (download required)
│   ├── ImprovMLCQ.csv           # ImprovMLCQ dataset
│   └── *_results_*.csv          # Pre-computed results
└── notebooks/
    └── CodeSmells_Reproducibility.ipynb  # Interactive analysis
```

---

## Contact

For questions or issues with reproduction:
1. Open an issue at https://github.com/salvahin/smells-paper/issues
2. Include your Python version, OS, and error messages

---

## Citation

If you use this code or data, please cite:

```bibtex
@article{hinojosa2026classimbalance,
  title={Joint vs. Independent Learning: How Class Imbalance Dictates
         Classifier Architecture in Code Smell Detection},
  author={Hinojosa, Salvador and others},
  journal={Applied Sciences},
  year={2026},
  publisher={MDPI}
}
```
