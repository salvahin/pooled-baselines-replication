#!/usr/bin/env python3
"""
Multi-classifier validation experiments.
Tests whether the 10%/5% thresholds hold across different classifier families.

Classifiers tested:
- RandomForest (baseline, already in paper)
- LinearSVC (kernel-based, scalable)
- HistGradientBoosting (boosting ensemble)
- LogisticRegression (linear baseline)

All classifiers use class_weight='balanced' for fair comparison.

Usage:
    python run_multiclassifier_experiments.py
    python run_multiclassifier_experiments.py --classifier RandomForest  # Single classifier

Output:
    Results saved to ../data/ directory as CSV files
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import f1_score, precision_score, recall_score
import warnings
import os
from datetime import datetime
from pathlib import Path
import time
import argparse

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
RANDOM_STATE = 42
N_FOLDS = 10
N_JOBS = -1  # Use all cores

# Paths (relative to this script)
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUT_DIR = DATA_DIR

# Classifier configurations - all with balanced class weights
CLASSIFIERS = {
    'RandomForest': lambda: RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    ),
    'LinearSVC': lambda: LinearSVC(
        class_weight='balanced',
        random_state=RANDOM_STATE,
        max_iter=10000,
        dual='auto'
    ),
    'HistGradientBoosting': lambda: HistGradientBoostingClassifier(
        max_iter=100,
        class_weight='balanced',
        random_state=RANDOM_STATE
    ),
    'LogisticRegression': lambda: LogisticRegression(
        class_weight='balanced',
        random_state=RANDOM_STATE,
        max_iter=1000,
        n_jobs=N_JOBS
    ),
}

# Dataset configurations
IST2021_SMELLS = {
    'god_class': ('GodClass.csv', 'is_god_class'),
    'data_class': ('DataClass.csv', 'is_data_class'),
    'long_method': ('LongMethod.csv', 'is_long_method'),
    'feature_envy': ('FeatureEnvy.csv', 'is_feature_envy'),
    'long_parameter_list': ('LongParameterList.csv', 'is_long_parameters_list'),
    'switch_statements': ('SwitchStatements.csv', 'is_switch_statements'),
}

SMELLYCODE_SMELLS = {
    'god_class': 'God class',
    'long_method': 'Long method',
    'feature_envy': 'Feature envy',
    'data_class': 'Data class',
}

IMPROVMLCQ_SMELLS = {
    'blob': 'blob_label',
    'data_class': 'dataclass_label',
    'feature_envy': 'featureenvy_label',
    'long_method': 'longmethod_label',
}

HALSTEAD_FEATURES = [
    'Logical Lines', 'Distinct Operators', 'Distinct Operands',
    'Total Operators', 'Total Operands', 'Vocabulary', 'Length',
    'Calculated Length', 'Volume', 'Difficulty', 'Effort',
    'Time Required', 'Bugs', 'Cyclomatic Complexity'
]


# =============================================================================
# Utility Functions
# =============================================================================

def get_common_features_ist2021(ist2021_path):
    """Get common features across all IST2021 smell datasets."""
    all_feature_cols = []
    for smell_name, (file_name, target_col) in IST2021_SMELLS.items():
        df = pd.read_csv(ist2021_path / file_name)
        feature_cols = [c for c in df.columns if c != target_col]
        all_feature_cols.append(set(feature_cols))
    common = list(set.intersection(*all_feature_cols))
    common.sort()
    return common


def run_specific_classifier(X, y, clf_factory, n_folds=N_FOLDS):
    """Run smell-specific binary classifier with cross-validation."""
    fold_results = []
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = clf_factory()
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        fold_results.append({
            'f1': f1_score(y_test, y_pred, zero_division=0),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
        })

    return {
        'f1_mean': np.mean([r['f1'] for r in fold_results]),
        'f1_std': np.std([r['f1'] for r in fold_results]),
        'fold_f1': [r['f1'] for r in fold_results],
    }


def run_unified_classifier(X_combined, y_combined, smell_labels, clf_factory, n_features, n_folds=N_FOLDS):
    """Run unified classifier with smell-type encoding."""
    smell_names = list(set(smell_labels))
    le = LabelEncoder()
    stratify_labels = le.fit_transform(smell_labels)

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    results_per_smell = {smell: [] for smell in smell_names}

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_combined, stratify_labels)):
        X_train, X_test = X_combined[train_idx], X_combined[test_idx]
        y_train, y_test = y_combined[train_idx], y_combined[test_idx]
        test_smells = smell_labels[test_idx]

        # Scale only original features, not one-hot encoding
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[:, :n_features] = scaler.fit_transform(X_train[:, :n_features])
        X_test_scaled[:, :n_features] = scaler.transform(X_test[:, :n_features])

        clf = clf_factory()
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        for smell_name in smell_names:
            mask = test_smells == smell_name
            if mask.sum() > 0:
                results_per_smell[smell_name].append(
                    f1_score(y_test[mask], y_pred[mask], zero_division=0))

    return {smell: {'f1_mean': np.mean(f1s), 'f1_std': np.std(f1s), 'fold_f1': f1s}
            for smell, f1s in results_per_smell.items()}


# =============================================================================
# Dataset-Specific Experiments
# =============================================================================

def run_ist2021_experiments(clf_name, clf_factory):
    """Run IST2021 experiments for a single classifier."""
    ist2021_path = DATA_DIR / "IST2021"

    if not ist2021_path.exists():
        print(f"  WARNING: IST2021 data not found at {ist2021_path}")
        return None

    print(f"\n  IST2021 ({clf_name})...")
    start = time.time()

    common_features = get_common_features_ist2021(ist2021_path)
    n_features = len(common_features)
    n_smells = len(IST2021_SMELLS)

    # Smell-specific classifiers
    specific_results = {}
    for smell_name, (file_name, target_col) in IST2021_SMELLS.items():
        df = pd.read_csv(ist2021_path / file_name)
        X = df[common_features].values
        y = df[target_col].apply(lambda x: 1 if x in [True, 'TRUE', 1, '1'] else 0).values
        X = np.nan_to_num(X, nan=0.0)

        result = run_specific_classifier(X, y, clf_factory)
        specific_results[smell_name] = result

    # Unified classifier
    X_list, y_list, smell_labels = [], [], []
    smell_names = list(IST2021_SMELLS.keys())

    for smell_idx, (smell_name, (file_name, target_col)) in enumerate(IST2021_SMELLS.items()):
        df = pd.read_csv(ist2021_path / file_name)
        X_smell = df[common_features].values
        y_smell = df[target_col].apply(lambda x: 1 if x in [True, 'TRUE', 1, '1'] else 0).values

        for i in range(len(y_smell)):
            smell_onehot = np.zeros(n_smells)
            smell_onehot[smell_idx] = 1
            X_list.append(np.concatenate([X_smell[i], smell_onehot]))
            y_list.append(y_smell[i])
            smell_labels.append(smell_name)

    X_combined = np.nan_to_num(np.array(X_list), nan=0.0)
    y_combined = np.array(y_list)
    smell_labels = np.array(smell_labels)

    unified_results = run_unified_classifier(X_combined, y_combined, smell_labels, clf_factory, n_features)

    # Calculate deltas
    deltas = []
    for smell_name in IST2021_SMELLS.keys():
        delta = specific_results[smell_name]['f1_mean'] - unified_results[smell_name]['f1_mean']
        deltas.append(delta)

    avg_delta = np.mean(deltas)
    elapsed = time.time() - start
    print(f"    Done in {elapsed:.1f}s, Avg ΔF1 = {avg_delta:+.3f}")

    return {
        'classifier': clf_name,
        'dataset': 'IST2021',
        'positive_rate': '~33%',
        'avg_specific_f1': np.mean([r['f1_mean'] for r in specific_results.values()]),
        'avg_unified_f1': np.mean([r['f1_mean'] for r in unified_results.values()]),
        'avg_delta_f1': avg_delta,
        'per_smell_deltas': {s: specific_results[s]['f1_mean'] - unified_results[s]['f1_mean']
                            for s in IST2021_SMELLS.keys()},
    }


def run_smellycode_experiments(clf_name, clf_factory, subsample=20000):
    """Run SmellyCode++ experiments for a single classifier."""
    smellycode_path = DATA_DIR / "SmellyCode++.csv"

    if not smellycode_path.exists():
        print(f"  WARNING: SmellyCode++ data not found at {smellycode_path}")
        return None

    print(f"\n  SmellyCode++ ({clf_name}, n={subsample})...")
    start = time.time()

    df = pd.read_csv(smellycode_path)

    # Subsample for speed (especially for SVC)
    if len(df) > subsample:
        np.random.seed(RANDOM_STATE)
        indices = np.random.choice(len(df), subsample, replace=False)
        df = df.iloc[indices].reset_index(drop=True)

    X_all = df[HALSTEAD_FEATURES].values
    X_all = np.nan_to_num(X_all, nan=0.0)
    n_features = len(HALSTEAD_FEATURES)
    n_smells = len(SMELLYCODE_SMELLS)

    # Smell-specific classifiers
    specific_results = {}
    for smell_name, label_col in SMELLYCODE_SMELLS.items():
        y = (df[label_col] == 1).astype(int).values
        result = run_specific_classifier(X_all, y, clf_factory)
        specific_results[smell_name] = result
        specific_results[smell_name]['positive_rate'] = y.mean()

    # Multi-label unified (same architecture as specific but with smell encoding)
    X_list, y_list, smell_labels = [], [], []
    smell_names = list(SMELLYCODE_SMELLS.keys())

    for smell_idx, (smell_name, label_col) in enumerate(SMELLYCODE_SMELLS.items()):
        y_smell = (df[label_col] == 1).astype(int).values
        for i in range(len(df)):
            smell_onehot = np.zeros(n_smells)
            smell_onehot[smell_idx] = 1
            X_list.append(np.concatenate([X_all[i], smell_onehot]))
            y_list.append(y_smell[i])
            smell_labels.append(smell_name)

    X_combined = np.array(X_list)
    y_combined = np.array(y_list)
    smell_labels = np.array(smell_labels)

    unified_results = run_unified_classifier(X_combined, y_combined, smell_labels, clf_factory, n_features)

    # Calculate deltas
    deltas = []
    for smell_name in SMELLYCODE_SMELLS.keys():
        delta = specific_results[smell_name]['f1_mean'] - unified_results[smell_name]['f1_mean']
        deltas.append(delta)

    avg_delta = np.mean(deltas)
    avg_pos_rate = np.mean([specific_results[s]['positive_rate'] for s in SMELLYCODE_SMELLS.keys()])
    elapsed = time.time() - start
    print(f"    Done in {elapsed:.1f}s, Avg ΔF1 = {avg_delta:+.3f}")

    return {
        'classifier': clf_name,
        'dataset': 'SmellyCode++',
        'positive_rate': f'{avg_pos_rate*100:.1f}%',
        'avg_specific_f1': np.mean([r['f1_mean'] for r in specific_results.values()]),
        'avg_unified_f1': np.mean([r['f1_mean'] for r in unified_results.values()]),
        'avg_delta_f1': avg_delta,
        'per_smell_deltas': {s: specific_results[s]['f1_mean'] - unified_results[s]['f1_mean']
                            for s in SMELLYCODE_SMELLS.keys()},
    }


def run_improvmlcq_experiments(clf_name, clf_factory):
    """Run ImprovMLCQ experiments for a single classifier."""
    improvmlcq_path = DATA_DIR / "ImprovMLCQ.csv"

    if not improvmlcq_path.exists():
        print(f"  WARNING: ImprovMLCQ data not found at {improvmlcq_path}")
        return None

    print(f"\n  ImprovMLCQ ({clf_name})...")
    start = time.time()

    df = pd.read_csv(improvmlcq_path)
    available_features = sorted([c for c in df.columns if c.startswith('ck_')])
    X_all = df[available_features].values
    X_all = np.nan_to_num(X_all, nan=0.0)
    n_features = len(available_features)
    n_smells = len(IMPROVMLCQ_SMELLS)

    # Smell-specific classifiers
    specific_results = {}
    for smell_name, label_col in IMPROVMLCQ_SMELLS.items():
        y = df[label_col].values.astype(int)
        result = run_specific_classifier(X_all, y, clf_factory)
        specific_results[smell_name] = result
        specific_results[smell_name]['positive_rate'] = y.mean()

    # Unified classifier
    X_list, y_list, smell_labels = [], [], []
    smell_names = list(IMPROVMLCQ_SMELLS.keys())

    for smell_idx, (smell_name, label_col) in enumerate(IMPROVMLCQ_SMELLS.items()):
        y_smell = df[label_col].values.astype(int)
        for i in range(len(df)):
            smell_onehot = np.zeros(n_smells)
            smell_onehot[smell_idx] = 1
            X_list.append(np.concatenate([X_all[i], smell_onehot]))
            y_list.append(y_smell[i])
            smell_labels.append(smell_name)

    X_combined = np.array(X_list)
    y_combined = np.array(y_list)
    smell_labels = np.array(smell_labels)

    unified_results = run_unified_classifier(X_combined, y_combined, smell_labels, clf_factory, n_features)

    # Calculate deltas
    deltas = []
    for smell_name in IMPROVMLCQ_SMELLS.keys():
        delta = specific_results[smell_name]['f1_mean'] - unified_results[smell_name]['f1_mean']
        deltas.append(delta)

    avg_delta = np.mean(deltas)
    avg_pos_rate = np.mean([specific_results[s]['positive_rate'] for s in IMPROVMLCQ_SMELLS.keys()])
    elapsed = time.time() - start
    print(f"    Done in {elapsed:.1f}s, Avg ΔF1 = {avg_delta:+.3f}")

    return {
        'classifier': clf_name,
        'dataset': 'ImprovMLCQ',
        'positive_rate': f'{avg_pos_rate*100:.1f}%',
        'avg_specific_f1': np.mean([r['f1_mean'] for r in specific_results.values()]),
        'avg_unified_f1': np.mean([r['f1_mean'] for r in unified_results.values()]),
        'avg_delta_f1': avg_delta,
        'per_smell_deltas': {s: specific_results[s]['f1_mean'] - unified_results[s]['f1_mean']
                            for s in IMPROVMLCQ_SMELLS.keys()},
    }


# =============================================================================
# Main Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Multi-classifier validation experiments')
    parser.add_argument('--classifier', type=str, default=None,
                       help='Run only specified classifier (RandomForest, LinearSVC, etc.)')
    args = parser.parse_args()

    print("="*70)
    print("MULTI-CLASSIFIER VALIDATION EXPERIMENTS")
    print("Testing whether 10%/5% thresholds hold across classifier families")
    print("="*70)

    # Select classifiers to run
    if args.classifier:
        if args.classifier not in CLASSIFIERS:
            print(f"Unknown classifier: {args.classifier}")
            print(f"Available: {list(CLASSIFIERS.keys())}")
            return
        classifiers_to_run = {args.classifier: CLASSIFIERS[args.classifier]}
    else:
        classifiers_to_run = CLASSIFIERS

    print(f"\nClassifiers: {list(classifiers_to_run.keys())}")
    print(f"Datasets: IST2021 (~33%), ImprovMLCQ (5-10%), SmellyCode++ (2-4%)")
    print(f"CV Folds: {N_FOLDS}")
    print(f"Random State: {RANDOM_STATE}")
    print(f"Using all CPU cores (n_jobs={N_JOBS})")

    all_results = []
    total_start = time.time()

    for clf_name, clf_factory in classifiers_to_run.items():
        print(f"\n{'='*70}")
        print(f"CLASSIFIER: {clf_name}")
        print("="*70)

        # Run on all datasets
        ist_result = run_ist2021_experiments(clf_name, clf_factory)
        if ist_result:
            all_results.append(ist_result)

        imlcq_result = run_improvmlcq_experiments(clf_name, clf_factory)
        if imlcq_result:
            all_results.append(imlcq_result)

        smelly_result = run_smellycode_experiments(clf_name, clf_factory)
        if smelly_result:
            all_results.append(smelly_result)

    total_elapsed = time.time() - total_start

    if not all_results:
        print("\nNo results generated. Check that datasets are in the data/ directory.")
        return

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY: Average ΔF1 (Specific - Unified)")
    print("="*70)
    print(f"\n{'Classifier':<20} {'IST2021':>12} {'ImprovMLCQ':>12} {'SmellyCode++':>12}")
    print(f"{'':20} {'(~33%)':>12} {'(5-10%)':>12} {'(2-4%)':>12}")
    print("-"*58)

    for clf_name in classifiers_to_run.keys():
        clf_results = [r for r in all_results if r['classifier'] == clf_name]
        ist_r = [r for r in clf_results if r['dataset'] == 'IST2021']
        imlcq_r = [r for r in clf_results if r['dataset'] == 'ImprovMLCQ']
        smelly_r = [r for r in clf_results if r['dataset'] == 'SmellyCode++']

        ist_delta = ist_r[0]['avg_delta_f1'] if ist_r else float('nan')
        imlcq_delta = imlcq_r[0]['avg_delta_f1'] if imlcq_r else float('nan')
        smelly_delta = smelly_r[0]['avg_delta_f1'] if smelly_r else float('nan')

        print(f"{clf_name:<20} {ist_delta:>+12.3f} {imlcq_delta:>+12.3f} {smelly_delta:>+12.3f}")

    print("-"*58)
    print(f"\nTotal runtime: {total_elapsed/60:.1f} minutes")

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Summary CSV
    summary_data = []
    for r in all_results:
        summary_data.append({
            'classifier': r['classifier'],
            'dataset': r['dataset'],
            'positive_rate': r['positive_rate'],
            'avg_specific_f1': r['avg_specific_f1'],
            'avg_unified_f1': r['avg_unified_f1'],
            'avg_delta_f1': r['avg_delta_f1'],
        })

    df_summary = pd.DataFrame(summary_data)
    summary_path = OUTPUT_DIR / f"multiclassifier_validation_{timestamp}.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")

    # Detailed per-smell CSV
    detailed_data = []
    for r in all_results:
        for smell, delta in r['per_smell_deltas'].items():
            detailed_data.append({
                'classifier': r['classifier'],
                'dataset': r['dataset'],
                'smell': smell,
                'delta_f1': delta,
            })

    df_detailed = pd.DataFrame(detailed_data)
    detailed_path = OUTPUT_DIR / f"multiclassifier_detailed_{timestamp}.csv"
    df_detailed.to_csv(detailed_path, index=False)
    print(f"Saved: {detailed_path}")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("If all classifiers show:")
    print("  - Positive ΔF1 on IST2021 (balanced) → specific wins")
    print("  - Near-zero ΔF1 on SmellyCode++ (imbalanced) → no advantage")
    print("  - Mixed on ImprovMLCQ (transition) → confirms 5-10% zone")
    print("Then the 10%/5% thresholds are CLASSIFIER-AGNOSTIC.")
    print("="*70)


if __name__ == "__main__":
    main()
