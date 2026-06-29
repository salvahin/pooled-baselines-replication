#!/usr/bin/env python3
"""
Replication Package: Code Smell Classification Experiments

This script reproduces all experiments from the paper:
"The Impact of Class Imbalance on Classifier Architecture Selection for Code Smell Detection"

Experiments:
- RQ1: IST2021 (balanced dataset, ~33% positive rate)
- RQ2a: SmellyCode++ (imbalanced dataset, 1.5-4% positive rate)
- RQ2b: ImprovMLCQ (intermediate dataset, 5-10% positive rate)
- RQ3: Boundary conditions with SMOTE balancing
- Feature ablation study

Usage:
    python run_experiments.py

Output:
    Results saved to ../data/ directory as CSV files
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multioutput import MultiOutputClassifier
from scipy.stats import wilcoxon
from imblearn.over_sampling import SMOTE
import warnings
import os
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================
RANDOM_STATE = 42
N_FOLDS = 10
N_ESTIMATORS = 100
N_JOBS = -1  # Use all cores for RandomForest

# Paths (relative to this script)
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUT_DIR = DATA_DIR

# IST2021 smell files and target columns
IST2021_SMELLS = {
    'god_class': ('GodClass.csv', 'is_god_class'),
    'data_class': ('DataClass.csv', 'is_data_class'),
    'long_method': ('LongMethod.csv', 'is_long_method'),
    'feature_envy': ('FeatureEnvy.csv', 'is_feature_envy'),
    'long_parameter_list': ('LongParameterList.csv', 'is_long_parameters_list'),
    'switch_statements': ('SwitchStatements.csv', 'is_switch_statements'),
}

# SmellyCode++ configuration
SMELLYCODE_SMELLS = {
    'god_class': 'God class',
    'long_method': 'Long method',
    'feature_envy': 'Feature envy',
    'data_class': 'Data class',
}

HALSTEAD_FEATURES = [
    'Logical Lines', 'Distinct Operators', 'Distinct Operands',
    'Total Operators', 'Total Operands', 'Vocabulary', 'Length',
    'Calculated Length', 'Volume', 'Difficulty', 'Effort',
    'Time Required', 'Bugs', 'Cyclomatic Complexity'
]

# ImprovMLCQ configuration
IMPROVMLCQ_SMELLS = {
    'blob': 'blob_label',
    'data_class': 'dataclass_label',
    'feature_envy': 'featureenvy_label',
    'long_method': 'longmethod_label',
}


# =============================================================================
# Utility Functions
# =============================================================================

def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1-1)*var1 + (n2-1)*var2) / (n1+n2-2))
    if pooled_std == 0:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def evaluate_classifier(y_true, y_pred):
    """Calculate F1, Precision, and Recall."""
    return {
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
    }


def get_common_features(ist2021_path):
    """Get common features across all IST2021 smell datasets."""
    all_feature_cols = []
    for smell_name, (file_name, target_col) in IST2021_SMELLS.items():
        df = pd.read_csv(ist2021_path / file_name)
        feature_cols = [c for c in df.columns if c != target_col]
        all_feature_cols.append(set(feature_cols))
    common = list(set.intersection(*all_feature_cols))
    common.sort()
    return common


# =============================================================================
# RQ1: IST2021 Experiments
# =============================================================================

def run_ist2021_experiments(ist2021_path):
    """
    Run IST2021 experiments comparing smell-specific vs unified classifiers.

    Returns list of results with P, R, F1 for each smell type.
    """
    print("\n" + "="*70)
    print("EXPERIMENT: IST2021 Dataset (Balanced, ~33% positive rate)")
    print("="*70)

    common_features = get_common_features(ist2021_path)
    print(f"Using {len(common_features)} common features across smell types")

    # Load all data
    smell_data = {}
    for smell_name, (file_name, target_col) in IST2021_SMELLS.items():
        df = pd.read_csv(ist2021_path / file_name)
        X = df[common_features].values
        y = df[target_col].values.astype(int)
        smell_data[smell_name] = {'X': X, 'y': y}
        print(f"  {smell_name}: {len(y)} samples, {y.mean():.1%} positive")

    # Run smell-specific classifiers
    print("\n--- Smell-Specific Classifiers ---")
    specific_results = {}

    for smell_name, data in smell_data.items():
        X, y = data['X'], data['y']
        fold_results = []

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS
            )
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)

            metrics = evaluate_classifier(y_test, y_pred)
            fold_results.append(metrics)

        specific_results[smell_name] = {
            'f1_mean': np.mean([r['f1'] for r in fold_results]),
            'f1_std': np.std([r['f1'] for r in fold_results]),
            'precision_mean': np.mean([r['precision'] for r in fold_results]),
            'precision_std': np.std([r['precision'] for r in fold_results]),
            'recall_mean': np.mean([r['recall'] for r in fold_results]),
            'recall_std': np.std([r['recall'] for r in fold_results]),
            'fold_f1': [r['f1'] for r in fold_results],
        }
        print(f"  {smell_name}: F1={specific_results[smell_name]['f1_mean']:.3f}")

    # Run unified multi-class classifier
    print("\n--- Unified Multi-Class Classifier ---")

    # Build combined dataset with smell-type encoding
    n_smells = len(IST2021_SMELLS)
    smell_names = list(IST2021_SMELLS.keys())

    X_list, y_list, smell_labels = [], [], []
    for smell_idx, smell_name in enumerate(smell_names):
        X, y = smell_data[smell_name]['X'], smell_data[smell_name]['y']
        for i in range(len(y)):
            smell_onehot = np.zeros(n_smells)
            smell_onehot[smell_idx] = 1
            X_row = np.concatenate([X[i], smell_onehot])
            X_list.append(X_row)
            y_list.append(y[i])
            smell_labels.append(smell_name)

    X_combined = np.array(X_list)
    y_combined = np.array(y_list)
    smell_labels = np.array(smell_labels)

    print(f"  Combined: {X_combined.shape[0]} samples, {X_combined.shape[1]} features")

    # Stratified CV
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    stratify_labels = le.fit_transform(smell_labels)

    results_per_smell = {smell: {'f1': [], 'precision': [], 'recall': []}
                         for smell in smell_names}

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    n_features = len(common_features)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_combined, stratify_labels)):
        X_train, X_test = X_combined[train_idx], X_combined[test_idx]
        y_train, y_test = y_combined[train_idx], y_combined[test_idx]
        test_smells = smell_labels[test_idx]

        # Scale only feature columns, not one-hot encoding
        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[:, :n_features] = scaler.fit_transform(X_train[:, :n_features])
        X_test_scaled[:, :n_features] = scaler.transform(X_test[:, :n_features])

        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS
        )
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        # Calculate per-smell metrics
        for smell_name in smell_names:
            mask = test_smells == smell_name
            if mask.sum() > 0:
                results_per_smell[smell_name]['f1'].append(
                    f1_score(y_test[mask], y_pred[mask], zero_division=0))
                results_per_smell[smell_name]['precision'].append(
                    precision_score(y_test[mask], y_pred[mask], zero_division=0))
                results_per_smell[smell_name]['recall'].append(
                    recall_score(y_test[mask], y_pred[mask], zero_division=0))

    # Aggregate unified results
    unified_results = {}
    for smell_name, metrics in results_per_smell.items():
        unified_results[smell_name] = {
            'f1_mean': np.mean(metrics['f1']),
            'f1_std': np.std(metrics['f1']),
            'precision_mean': np.mean(metrics['precision']),
            'precision_std': np.std(metrics['precision']),
            'recall_mean': np.mean(metrics['recall']),
            'recall_std': np.std(metrics['recall']),
            'fold_f1': metrics['f1'],
        }

    # Combine results
    combined_results = []
    print("\n--- Results Summary ---")
    for smell_name in smell_names:
        spec = specific_results[smell_name]
        unif = unified_results[smell_name]
        delta_f1 = spec['f1_mean'] - unif['f1_mean']

        try:
            stat, p_value = wilcoxon(spec['fold_f1'], unif['fold_f1'])
        except:
            p_value = 1.0

        effect_size = cohens_d(spec['fold_f1'], unif['fold_f1'])

        result = {
            'smell': smell_name,
            'specific_f1': f"{spec['f1_mean']:.3f} ± {spec['f1_std']:.3f}",
            'specific_precision': f"{spec['precision_mean']:.3f} ± {spec['precision_std']:.3f}",
            'specific_recall': f"{spec['recall_mean']:.3f} ± {spec['recall_std']:.3f}",
            'multiclass_f1': f"{unif['f1_mean']:.3f} ± {unif['f1_std']:.3f}",
            'multiclass_precision': f"{unif['precision_mean']:.3f} ± {unif['precision_std']:.3f}",
            'multiclass_recall': f"{unif['recall_mean']:.3f} ± {unif['recall_std']:.3f}",
            'delta_f1': f"{delta_f1:+.3f}",
            'p_value': f"{p_value:.4f}",
            'cohens_d': f"{effect_size:.2f}",
            'spec_f1_mean': spec['f1_mean'],
            'spec_f1_std': spec['f1_std'],
            'mc_f1_mean': unif['f1_mean'],
            'mc_f1_std': unif['f1_std'],
        }
        combined_results.append(result)
        print(f"  {smell_name}: ΔF1={delta_f1:+.3f}, p={p_value:.4f}, d={effect_size:.2f}")

    avg_delta = np.mean([r['spec_f1_mean'] - r['mc_f1_mean'] for r in combined_results])
    print(f"\n  OVERALL: Average ΔF1 = {avg_delta:+.3f}")

    return combined_results


# =============================================================================
# RQ2a: SmellyCode++ Experiments
# =============================================================================

def run_smellycode_experiments(smellycode_path):
    """
    Run SmellyCode++ experiments (imbalanced, multi-label dataset).
    """
    print("\n" + "="*70)
    print("EXPERIMENT: SmellyCode++ Dataset (Imbalanced, 1.5-4% positive rate)")
    print("="*70)

    df = pd.read_csv(smellycode_path)
    print(f"Loaded {len(df):,} samples")

    X_all = df[HALSTEAD_FEATURES].values
    X_all = np.nan_to_num(X_all, nan=0.0)

    # Run smell-specific classifiers
    print("\n--- Smell-Specific Classifiers ---")
    specific_results = {}

    for smell_name, label_col in SMELLYCODE_SMELLS.items():
        y = df[label_col].values.astype(int)
        positive_rate = y.mean()
        print(f"  {smell_name}: {positive_rate:.2%} positive")

        fold_results = []
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y)):
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS
            )
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)

            metrics = evaluate_classifier(y_test, y_pred)
            fold_results.append(metrics)

        specific_results[smell_name] = {
            'positive_rate': positive_rate,
            'f1_mean': np.mean([r['f1'] for r in fold_results]),
            'f1_std': np.std([r['f1'] for r in fold_results]),
            'precision_mean': np.mean([r['precision'] for r in fold_results]),
            'recall_mean': np.mean([r['recall'] for r in fold_results]),
            'fold_f1': [r['f1'] for r in fold_results],
        }

    # Run multi-label classifier
    print("\n--- Multi-Label Classifier ---")

    smell_names = list(SMELLYCODE_SMELLS.keys())
    Y_all = np.column_stack([df[SMELLYCODE_SMELLS[s]].values for s in smell_names])

    multilabel_results = {smell: {'f1': [], 'precision': [], 'recall': []}
                          for smell in smell_names}

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    stratify_col = df[SMELLYCODE_SMELLS['god_class']].values

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, stratify_col)):
        X_train, X_test = X_all[train_idx], X_all[test_idx]
        Y_train, Y_test = Y_all[train_idx], Y_all[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        base_clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS
        )
        clf = MultiOutputClassifier(base_clf)
        clf.fit(X_train_scaled, Y_train)
        Y_pred = clf.predict(X_test_scaled)

        for i, smell_name in enumerate(smell_names):
            multilabel_results[smell_name]['f1'].append(
                f1_score(Y_test[:, i], Y_pred[:, i], zero_division=0))
            multilabel_results[smell_name]['precision'].append(
                precision_score(Y_test[:, i], Y_pred[:, i], zero_division=0))
            multilabel_results[smell_name]['recall'].append(
                recall_score(Y_test[:, i], Y_pred[:, i], zero_division=0))

    # Combine results
    combined_results = []
    print("\n--- Results Summary ---")
    for smell_name in smell_names:
        spec = specific_results[smell_name]
        ml = multilabel_results[smell_name]
        ml_f1_mean = np.mean(ml['f1'])
        delta_f1 = spec['f1_mean'] - ml_f1_mean

        result = {
            'smell': smell_name,
            'positive_rate': spec['positive_rate'],
            'specific_f1': f"{spec['f1_mean']:.3f} ± {spec['f1_std']:.3f}",
            'multilabel_f1': f"{ml_f1_mean:.3f} ± {np.std(ml['f1']):.3f}",
            'delta_f1': f"{delta_f1:+.3f}",
            'spec_f1_mean': spec['f1_mean'],
            'ml_f1_mean': ml_f1_mean,
        }
        combined_results.append(result)
        print(f"  {smell_name} ({spec['positive_rate']:.1%}): ΔF1={delta_f1:+.3f}")

    avg_delta = np.mean([r['spec_f1_mean'] - r['ml_f1_mean'] for r in combined_results])
    print(f"\n  OVERALL: Average ΔF1 = {avg_delta:+.3f}")

    return combined_results, specific_results


# =============================================================================
# RQ2b: ImprovMLCQ Experiments
# =============================================================================

def run_improvmlcq_experiments(improvmlcq_path):
    """
    Run ImprovMLCQ experiments (intermediate positive rates: 5-10%).
    """
    print("\n" + "="*70)
    print("EXPERIMENT: ImprovMLCQ Dataset (Intermediate, 5-10% positive rate)")
    print("="*70)

    df = pd.read_csv(improvmlcq_path)
    print(f"Loaded {len(df):,} samples")

    # Auto-detect CK features
    available_features = sorted([c for c in df.columns if c.startswith('ck_')])
    print(f"Using {len(available_features)} CK metric features")

    X_all = df[available_features].values
    X_all = np.nan_to_num(X_all, nan=0.0)

    # Run smell-specific classifiers
    print("\n--- Smell-Specific Classifiers ---")
    specific_results = {}

    for smell_name, label_col in IMPROVMLCQ_SMELLS.items():
        y = df[label_col].values.astype(int)
        positive_rate = y.mean()
        print(f"  {smell_name}: {positive_rate:.2%} positive")

        fold_results = []
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y)):
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS
            )
            clf.fit(X_train_scaled, y_train)
            y_pred = clf.predict(X_test_scaled)

            metrics = evaluate_classifier(y_test, y_pred)
            fold_results.append(metrics)

        specific_results[smell_name] = {
            'positive_rate': positive_rate,
            'f1_mean': np.mean([r['f1'] for r in fold_results]),
            'f1_std': np.std([r['f1'] for r in fold_results]),
            'precision_mean': np.mean([r['precision'] for r in fold_results]),
            'recall_mean': np.mean([r['recall'] for r in fold_results]),
            'fold_f1': [r['f1'] for r in fold_results],
        }

    # Run unified classifier
    print("\n--- Unified Classifier ---")

    smell_names = list(IMPROVMLCQ_SMELLS.keys())
    n_smells = len(smell_names)
    n_features = len(available_features)

    X_list, y_list, smell_labels = [], [], []
    for smell_idx, (smell_name, label_col) in enumerate(IMPROVMLCQ_SMELLS.items()):
        y_smell = df[label_col].values.astype(int)
        for i in range(len(df)):
            smell_onehot = np.zeros(n_smells)
            smell_onehot[smell_idx] = 1
            X_row = np.concatenate([X_all[i], smell_onehot])
            X_list.append(X_row)
            y_list.append(y_smell[i])
            smell_labels.append(smell_name)

    X_combined = np.array(X_list)
    y_combined = np.array(y_list)
    smell_labels = np.array(smell_labels)

    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    stratify_labels = le.fit_transform(smell_labels)

    results_per_smell = {smell: {'f1': [], 'precision': [], 'recall': []}
                         for smell in smell_names}

    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_combined, stratify_labels)):
        X_train, X_test = X_combined[train_idx], X_combined[test_idx]
        y_train, y_test = y_combined[train_idx], y_combined[test_idx]
        test_smells = smell_labels[test_idx]

        scaler = StandardScaler()
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        X_train_scaled[:, :n_features] = scaler.fit_transform(X_train[:, :n_features])
        X_test_scaled[:, :n_features] = scaler.transform(X_test[:, :n_features])

        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS
        )
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)

        for smell_name in smell_names:
            mask = test_smells == smell_name
            if mask.sum() > 0:
                results_per_smell[smell_name]['f1'].append(
                    f1_score(y_test[mask], y_pred[mask], zero_division=0))
                results_per_smell[smell_name]['precision'].append(
                    precision_score(y_test[mask], y_pred[mask], zero_division=0))
                results_per_smell[smell_name]['recall'].append(
                    recall_score(y_test[mask], y_pred[mask], zero_division=0))

    # Combine results
    combined_results = []
    print("\n--- Results Summary ---")
    for smell_name in smell_names:
        spec = specific_results[smell_name]
        unif = results_per_smell[smell_name]
        unif_f1_mean = np.mean(unif['f1'])
        delta_f1 = spec['f1_mean'] - unif_f1_mean

        result = {
            'smell': smell_name,
            'positive_rate': spec['positive_rate'],
            'specific_f1': spec['f1_mean'],
            'unified_f1': unif_f1_mean,
            'delta_f1': delta_f1,
        }
        combined_results.append(result)
        print(f"  {smell_name} ({spec['positive_rate']:.1%}): ΔF1={delta_f1:+.3f}")

    avg_delta = np.mean([r['delta_f1'] for r in combined_results])
    print(f"\n  OVERALL: Average ΔF1 = {avg_delta:+.3f}")

    return combined_results


# =============================================================================
# RQ3: Boundary Conditions (SMOTE Balancing)
# =============================================================================

def run_boundary_conditions(df_smelly, specific_results):
    """
    Test boundary conditions by applying SMOTE balancing to SmellyCode++.
    """
    print("\n" + "="*70)
    print("EXPERIMENT: Boundary Conditions (SMOTE Balancing)")
    print("="*70)

    X_all = df_smelly[HALSTEAD_FEATURES].values
    X_all = np.nan_to_num(X_all, nan=0.0)

    boundary_results = []
    smell_names = list(SMELLYCODE_SMELLS.keys())

    for smell_name in smell_names:
        label_col = SMELLYCODE_SMELLS[smell_name]
        y = df_smelly[label_col].values.astype(int)

        orig_f1 = specific_results[smell_name]['f1_mean']

        # Run with SMOTE balancing
        fold_results = []
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X_all, y)):
            X_train, X_test = X_all[train_idx], X_all[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Apply SMOTE to training data only
            try:
                smote = SMOTE(random_state=RANDOM_STATE)
                X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            except:
                X_train_resampled, y_train_resampled = X_train, y_train

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_resampled)
            X_test_scaled = scaler.transform(X_test)

            clf = RandomForestClassifier(
                n_estimators=N_ESTIMATORS,
                class_weight='balanced',
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS
            )
            clf.fit(X_train_scaled, y_train_resampled)
            y_pred = clf.predict(X_test_scaled)

            metrics = evaluate_classifier(y_test, y_pred)
            fold_results.append(metrics)

        bal_f1_mean = np.mean([r['f1'] for r in fold_results])
        bal_f1_std = np.std([r['f1'] for r in fold_results])
        delta_f1 = bal_f1_mean - orig_f1

        # Calculate Cohen's d
        orig_fold_f1 = specific_results[smell_name]['fold_f1']
        bal_fold_f1 = [r['f1'] for r in fold_results]
        effect_size = cohens_d(bal_fold_f1, orig_fold_f1)

        result = {
            'smell': smell_name,
            'orig_f1': orig_f1,
            'balanced_f1': bal_f1_mean,
            'delta_f1': delta_f1,
            'cohens_d': effect_size,
        }
        boundary_results.append(result)
        print(f"  {smell_name}: ΔF1={delta_f1:+.3f}, d={effect_size:.2f}")

    return boundary_results


# =============================================================================
# Main Execution
# =============================================================================

def save_results(ist2021_results, smellycode_results, improvmlcq_results, boundary_results):
    """Save all results to CSV files."""
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if ist2021_results:
        df = pd.DataFrame(ist2021_results)
        path = OUTPUT_DIR / f"ist2021_results_{timestamp}.csv"
        df.to_csv(path, index=False)
        print(f"Saved: {path}")

    if smellycode_results:
        df = pd.DataFrame(smellycode_results)
        path = OUTPUT_DIR / f"smellycode_results_{timestamp}.csv"
        df.to_csv(path, index=False)
        print(f"Saved: {path}")

    if improvmlcq_results:
        df = pd.DataFrame(improvmlcq_results)
        path = OUTPUT_DIR / f"improvmlcq_results_{timestamp}.csv"
        df.to_csv(path, index=False)
        print(f"Saved: {path}")

    if boundary_results:
        df = pd.DataFrame(boundary_results)
        path = OUTPUT_DIR / f"boundary_conditions_{timestamp}.csv"
        df.to_csv(path, index=False)
        print(f"Saved: {path}")


def main():
    """Main execution function."""
    print("="*70)
    print("CODE SMELL CLASSIFICATION EXPERIMENTS")
    print("Replication Package")
    print("="*70)
    print(f"Random State: {RANDOM_STATE}")
    print(f"Cross-validation Folds: {N_FOLDS}")
    print(f"RandomForest Trees: {N_ESTIMATORS}")

    # Check for data files
    ist2021_path = DATA_DIR / "IST2021"
    smellycode_path = DATA_DIR / "SmellyCode++.csv"
    improvmlcq_path = DATA_DIR / "ImprovMLCQ.csv"

    ist2021_results = None
    smellycode_results = None
    improvmlcq_results = None
    boundary_results = None

    # RQ1: IST2021
    if ist2021_path.exists():
        ist2021_results = run_ist2021_experiments(ist2021_path)
    else:
        print(f"\nWARNING: IST2021 data not found at {ist2021_path}")
        print("  Download from: https://github.com/hjamaan/IST2021-CodeSmellStackingEnsemble")

    # RQ2a: SmellyCode++
    if smellycode_path.exists():
        smellycode_results, specific_results = run_smellycode_experiments(smellycode_path)

        # RQ3: Boundary conditions
        df_smelly = pd.read_csv(smellycode_path)
        boundary_results = run_boundary_conditions(df_smelly, specific_results)
    else:
        print(f"\nWARNING: SmellyCode++ data not found at {smellycode_path}")

    # RQ2b: ImprovMLCQ
    if improvmlcq_path.exists():
        improvmlcq_results = run_improvmlcq_experiments(improvmlcq_path)
    else:
        print(f"\nWARNING: ImprovMLCQ data not found at {improvmlcq_path}")

    # Save results
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)
    save_results(ist2021_results, smellycode_results, improvmlcq_results, boundary_results)

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*70)


if __name__ == "__main__":
    main()
