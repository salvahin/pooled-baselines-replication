# Replication Package: The Impact of Class Imbalance on Classifier Architecture Selection for Code Smell Detection

This repository contains all code, data, and instructions needed to reproduce the experiments described in the paper.

## Overview

We investigate when smell-specific binary classifiers outperform unified multi-class approaches for machine learning-based code smell detection. Key findings:

- **Balanced data (>10% positive rate)**: Smell-specific classifiers outperform by ΔF1 = +0.062
- **Imbalanced data (<5% positive rate)**: No advantage to specialization
- **Transition zone (5-10%)**: Mixed results requiring empirical evaluation

## Repository Structure

```
replication-package/
├── README.md                    # This file
├── REPRODUCTION.md              # Detailed step-by-step reproduction guide
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
├── code/
│   ├── run_experiments.py       # Main experiment script (RQ1-RQ3)
│   ├── run_multiclassifier_experiments.py  # Cross-classifier validation
│   ├── run_metaheuristic_experiments.py    # Metaheuristic feature selection (RQ4)
│   └── regenerate_figures.py    # Figure generation script
├── data/
│   ├── README.md                # Dataset download instructions with DOIs
│   ├── verify_data.py           # Dataset verification script
│   ├── IST2021/                 # IST2021 dataset (download required)
│   ├── SmellyCode++.csv         # SmellyCode++ dataset (download required)
│   ├── ImprovMLCQ.csv           # ImprovMLCQ dataset (download required)
│   └── *_results_*.csv          # Pre-computed results (included)
└── notebooks/
    └── CodeSmells_Reproducibility.ipynb  # Interactive analysis
```

**IMPORTANT**: Datasets must be downloaded separately. See [data/README.md](data/README.md) for instructions.

## Datasets

### IST2021 (Balanced, ~33% positive rate)
- **Source**: https://github.com/hjamaan/IST2021-CodeSmellStackingEnsemble
- **Samples**: 2,520 (420 per smell type)
- **Features**: 56-83 CK object-oriented metrics
- **Smells**: God Class, Data Class, Long Method, Feature Envy, Long Parameter List, Switch Statements

### SmellyCode++ (Imbalanced, 1.5-4% positive rate)
- **Source**: Figshare (CC BY 4.0)
- **Samples**: 107,325
- **Features**: 14 Halstead complexity metrics
- **Smells**: God Class, Long Method, Feature Envy, Data Class

### ImprovMLCQ (Intermediate, 5-10% positive rate)
- **Source**: Extended MLCQ dataset
- **Samples**: 13,489
- **Features**: 33 CK metrics
- **Smells**: Blob (10.3%), Data Class (9.8%), Feature Envy (5.6%), Long Method (5.1%)

## Requirements

- Python 3.8+
- scikit-learn >= 1.0
- pandas >= 1.3
- numpy >= 1.20
- matplotlib >= 3.4
- imbalanced-learn >= 0.9
- scipy >= 1.7

Install dependencies:
```bash
pip install -r requirements.txt
```

## Reproducing Experiments

For detailed step-by-step instructions, see [REPRODUCTION.md](REPRODUCTION.md).

### Quick Start

Run all experiments with a single command:
```bash
python code/run_experiments.py
```

This will:
1. Run IST2021 experiments (RQ1)
2. Run SmellyCode++ experiments (RQ2a)
3. Run ImprovMLCQ experiments (RQ2b)
4. Run boundary conditions with SMOTE (RQ3)
5. Run feature ablation study
6. Save results to `data/` directory

**Expected runtime**: ~30-45 minutes on a modern CPU

### Cross-Classifier Validation

To verify thresholds hold across classifier families:
```bash
python code/run_multiclassifier_experiments.py
```

This tests RandomForest, LinearSVC, HistGradientBoosting, and LogisticRegression.

**Expected runtime**: ~15-20 minutes

### Regenerating Figures

After running experiments:
```bash
python code/regenerate_figures.py
```

This generates:
- `fig1_ist2021_comparison.png` - RQ1 results
- `fig2_smellycode_comparison.png` - RQ2a results
- `fig3_boundary_forest_plot.png` - RQ3 effect sizes

## Experimental Configuration

| Parameter | Value |
|-----------|-------|
| Random State | 42 |
| Cross-Validation | 10-fold stratified |
| Classifier | RandomForestClassifier |
| Trees | 100 |
| Class Weight | balanced |
| SMOTE | Default (k=5) |

## Expected Results

### Table 1: IST2021 Results (RQ1)
| Smell Type | Specific F1 | Unified F1 | ΔF1 |
|------------|-------------|------------|-----|
| Data Class | 0.989 | 0.956 | +0.033 |
| God Class | 0.972 | 0.939 | +0.033 |
| Long Method | 0.839 | 0.752 | +0.087 |
| Feature Envy | 0.745 | 0.644 | +0.100 |
| Long Param List | 0.618 | 0.573 | +0.045 |
| Switch Statements | 0.486 | 0.426 | +0.059 |
| **Average** | **0.775** | **0.715** | **+0.060** |

### Table 2: SmellyCode++ Results (RQ2a)
| Smell Type | Pos% | Specific F1 | Multi-label F1 | ΔF1 |
|------------|------|-------------|----------------|-----|
| God Class | 4.0% | 0.538 | 0.538 | 0.000 |
| Long Method | 1.5% | 0.293 | 0.293 | -0.001 |
| Feature Envy | 1.9% | 0.302 | 0.303 | -0.001 |
| Data Class | 3.1% | 0.230 | 0.237 | -0.007 |
| **Average** | **2.6%** | **0.341** | **0.343** | **-0.002** |

### Table 3: ImprovMLCQ Results (RQ2b)
| Smell Type | Pos% | Specific F1 | Unified F1 | ΔF1 |
|------------|------|-------------|------------|-----|
| Blob | 10.3% | 0.588 | 0.375 | +0.213 |
| Data Class | 9.8% | 0.617 | 0.384 | +0.233 |
| Feature Envy | 5.6% | 0.262 | 0.592 | -0.330 |
| Long Method | 5.1% | 0.231 | 0.494 | -0.263 |

**Key insight**: At ~10% positive rate, specialization still benefits; at ~5%, unified approaches prevail.

### Table 4: Cross-Classifier Validation Summary (Smell-Specific vs Unified)
| Classifier | IST2021 ΔF1 | ImprovMLCQ ΔF1 | SmellyCode++ ΔF1 |
|------------|-------------|----------------|------------------|
| RandomForest | +0.060 | -0.037 | -0.023 |
| LinearSVC | +0.174 | +0.144 | +0.053 |
| HistGradientBoosting | +0.005 | +0.024 | +0.009 |
| LogisticRegression | +0.158 | +0.145 | +0.048 |

**Key insight**: All classifiers show positive ΔF1 on balanced data (IST2021). On imbalanced data, tree-based models (RandomForest, HistGradientBoosting) show near-zero or negative ΔF1, while linear models maintain positive ΔF1. This suggests class imbalance effects interact with classifier architecture.

## Statistical Analysis

- **Test**: Wilcoxon signed-rank test (paired, non-parametric)
- **Effect size**: Cohen's d
- **Correction**: Bonferroni for multiple comparisons
- **Significance**: α = 0.05

## Metaheuristic Feature Selection (RQ4)

We tested four algorithms via MAFESE:
- Particle Swarm Optimization (PSO)
- Grey Wolf Optimization (GWO)
- Simulated Annealing (SA)
- Whale Optimization Algorithm (WOA)

**Result**: 88% of 108 configurations degraded performance, confirming that feature selection cannot compensate for class imbalance.

## Feature Ablation Study

| Features | Avg ΔF1 | Specialization Advantage |
|----------|---------|--------------------------|
| 36 | +0.060 | YES |
| 30 | +0.057 | YES |
| 20 | +0.050 | YES |
| 14 | +0.032 | YES |

**Conclusion**: Class imbalance—not feature count—is the primary determinant.

## Citation

If you use this code or data, please cite:

```bibtex
@article{authors2026classimbalance,
  title={The Impact of Class Imbalance on Classifier Architecture Selection for Code Smell Detection},
  author={[Authors]},
  journal={Applied Sciences},
  year={2026},
  publisher={MDPI}
}
```

## License

This replication package is released under the MIT License.

## Contact

For questions about this replication package, please open an issue in this repository.
