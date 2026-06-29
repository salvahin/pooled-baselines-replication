#!/usr/bin/env python3
"""
Corrected Multi-Label Classification Experiments.

This script implements the CORRECT multi-label methodology:
- Smell-Specific: MultiOutputClassifier(Base) - trains K independent classifiers
- Unified RF: Native multi-label (RF.fit(X, Y_2D))
- Unified Linear/Boosting: ClassifierChain(Base) - exploits label correlations

Key fixes from previous flawed approach:
1. NO one-hot smell encoding (was handicapping linear models)
2. Uses 2D target matrix (N × K) for multi-label
3. Uses full 107K SmellyCode++ samples
4. Uses StratifiedKFold with multi-label stratification

Datasets:
- IST2021: Independent binary tasks (cannot merge - different feature sets)
- SmellyCode++: True multi-label with 4 smell types
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputClassifier, ClassifierChain
from sklearn.metrics import f1_score
import warnings
import os
from datetime import datetime
import time

warnings.filterwarnings('ignore')

# Configuration
RANDOM_STATE = 42
N_FOLDS = 10
N_JOBS = -1

# Paths
IST2021_PATH = "/Users/salvahin/Proyectos/Code-Smells-26/code-smells-experiments/0 Original"
SMELLYCODE_PATH = "/Users/salvahin/Papers/smells-paper/data/SmellyCode++.csv"
OUTPUT_PATH = "/Users/salvahin/Papers/smells-paper/data"

# IST2021 smell files (cannot be merged - different feature sets per granularity)
IST2021_SMELLS = {
    'god_class': ('GodClass.csv', 'is_god_class'),
    'data_class': ('DataClass.csv', 'is_data_class'),
    'long_method': ('LongMethod.csv', 'is_long_method'),
    'feature_envy': ('FeatureEnvy.csv', 'is_feature_envy'),
    'long_parameter_list': ('LongParameterList.csv', 'is_long_parameters_list'),
    'switch_statements': ('SwitchStatements.csv', 'is_switch_statements'),
}

# SmellyCode++ smell columns (true multi-label)
SMELLYCODE_SMELLS = ['Long method', 'God class', 'Feature envy', 'Data class']

# Halstead metrics (features)
HALSTEAD_FEATURES = [
    'Logical Lines', 'Distinct Operators', 'Distinct Operands',
    'Total Operators', 'Total Operands', 'Vocabulary', 'Length',
    'Calculated Length', 'Volume', 'Difficulty', 'Effort',
    'Time Required', 'Bugs', 'Cyclomatic Complexity'
]


def create_stratification_labels(Y):
    """Create stratification labels for multi-label data.

    Combines labels into a single string for stratification.
    """
    return ['_'.join(map(str, row)) for row in Y]


def run_ist2021_binary_experiments():
    """
    Run IST2021 experiments.

    NOTE: IST2021 has different feature sets per smell type:
    - GodClass/DataClass: 61 class-level features
    - FeatureEnvy/LongMethod: 82 method-level features
    - LongParameterList/SwitchStatements: 55 method-level features

    Therefore, we can only run independent binary classification experiments.
    Multi-label/multi-class comparison is not valid for this dataset.
    """
    print("\n" + "="*70)
    print("IST2021: INDEPENDENT BINARY CLASSIFICATION")
    print("(Cannot merge - different feature sets per smell granularity)")
    print("="*70)

    results = []

    for smell_name, (file_name, target_col) in IST2021_SMELLS.items():
        df = pd.read_csv(os.path.join(IST2021_PATH, file_name))
        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols].values
        y = df[target_col].apply(lambda x: 1 if x in [True, 'TRUE', 1, '1'] else 0).values
        X = np.nan_to_num(X, nan=0.0)

        pos_rate = y.mean() * 100

        # Run 10-fold CV with RandomForest (reference classifier)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        fold_f1s = []

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            clf = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                        random_state=RANDOM_STATE, n_jobs=N_JOBS)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            fold_f1s.append(f1_score(y_test, y_pred, zero_division=0))

        f1_mean = np.mean(fold_f1s)
        f1_std = np.std(fold_f1s)

        results.append({
            'smell': smell_name,
            'n_features': len(feature_cols),
            'n_samples': len(df),
            'positive_rate': pos_rate,
            'f1_mean': f1_mean,
            'f1_std': f1_std,
        })
        print(f"  {smell_name}: F1={f1_mean:.3f}±{f1_std:.3f} (pos_rate={pos_rate:.1f}%, n_features={len(feature_cols)})")

    return results


def run_smellycode_multilabel_experiments():
    """
    Run SmellyCode++ experiments with CORRECT multi-label methodology.

    Compares:
    - Smell-Specific: MultiOutputClassifier(Base) - K independent classifiers
    - Unified RF: Native multi-label (RF.fit(X, Y_2D))
    - Unified Linear: ClassifierChain(Base) - exploits label correlations
    """
    print("\n" + "="*70)
    print("SMELLYCODE++: TRUE MULTI-LABEL EXPERIMENTS")
    print("="*70)

    print("\nLoading full SmellyCode++ dataset (107K samples)...")
    df = pd.read_csv(SMELLYCODE_PATH)
    print(f"Loaded: {len(df)} samples")

    # Features and multi-label target
    X = df[HALSTEAD_FEATURES].values
    X = np.nan_to_num(X, nan=0.0)
    Y = df[SMELLYCODE_SMELLS].values.astype(int)  # 2D target matrix (N × K)

    print(f"\nFeature matrix X: {X.shape}")
    print(f"Target matrix Y: {Y.shape}")
    print(f"\nPer-smell positive rates:")
    for i, smell in enumerate(SMELLYCODE_SMELLS):
        pos_rate = Y[:, i].mean() * 100
        print(f"  {smell}: {pos_rate:.2f}%")

    # Multi-label co-occurrence
    smell_counts = Y.sum(axis=1)
    print(f"\nCo-occurrence stats:")
    print(f"  0 smells: {(smell_counts == 0).sum()}")
    print(f"  1 smell: {(smell_counts == 1).sum()}")
    print(f"  2+ smells: {(smell_counts >= 2).sum()}")

    # Create stratification labels
    strat_labels = create_stratification_labels(Y)

    # Classifiers to test
    classifiers = {
        'RandomForest': {
            'specific': lambda: MultiOutputClassifier(
                RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                       random_state=RANDOM_STATE, n_jobs=N_JOBS)
            ),
            'unified': lambda: RandomForestClassifier(
                n_estimators=100, class_weight='balanced',
                random_state=RANDOM_STATE, n_jobs=N_JOBS
            ),  # Native multi-label
        },
        'HistGradientBoosting': {
            'specific': lambda: MultiOutputClassifier(
                HistGradientBoostingClassifier(max_iter=100, class_weight='balanced',
                                               random_state=RANDOM_STATE)
            ),
            'unified': lambda: ClassifierChain(
                HistGradientBoostingClassifier(max_iter=100, class_weight='balanced',
                                               random_state=RANDOM_STATE),
                random_state=RANDOM_STATE
            ),
        },
        'LogisticRegression': {
            'specific': lambda: MultiOutputClassifier(
                LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE,
                                   max_iter=1000, n_jobs=N_JOBS)
            ),
            'unified': lambda: ClassifierChain(
                LogisticRegression(class_weight='balanced', random_state=RANDOM_STATE,
                                   max_iter=1000),
                random_state=RANDOM_STATE
            ),
        },
        'LinearSVC': {
            'specific': lambda: MultiOutputClassifier(
                LinearSVC(class_weight='balanced', random_state=RANDOM_STATE,
                          max_iter=10000, dual='auto')
            ),
            'unified': lambda: ClassifierChain(
                LinearSVC(class_weight='balanced', random_state=RANDOM_STATE,
                          max_iter=10000, dual='auto'),
                random_state=RANDOM_STATE
            ),
        },
    }

    all_results = []

    for clf_name, clf_factories in classifiers.items():
        print(f"\n{'='*50}")
        print(f"CLASSIFIER: {clf_name}")
        print("="*50)

        specific_fold_f1s = {smell: [] for smell in SMELLYCODE_SMELLS}
        unified_fold_f1s = {smell: [] for smell in SMELLYCODE_SMELLS}

        start_time = time.time()

        # We use LabelEncoder on stratification labels for StratifiedKFold
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        strat_encoded = le.fit_transform(strat_labels)

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, strat_encoded)):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Smell-Specific: MultiOutputClassifier (K independent classifiers)
            specific_clf = clf_factories['specific']()
            specific_clf.fit(X_train_scaled, Y_train)
            Y_pred_specific = specific_clf.predict(X_test_scaled)

            # Unified: Native multi-label (RF) or ClassifierChain (Linear/Boosting)
            unified_clf = clf_factories['unified']()
            unified_clf.fit(X_train_scaled, Y_train)
            Y_pred_unified = unified_clf.predict(X_test_scaled)

            # Per-smell F1 scores
            for i, smell in enumerate(SMELLYCODE_SMELLS):
                specific_fold_f1s[smell].append(
                    f1_score(Y_test[:, i], Y_pred_specific[:, i], zero_division=0)
                )
                unified_fold_f1s[smell].append(
                    f1_score(Y_test[:, i], Y_pred_unified[:, i], zero_division=0)
                )

            print(f"  Fold {fold+1}/{N_FOLDS} completed")

        elapsed = time.time() - start_time

        # Calculate results
        print(f"\nResults ({elapsed:.1f}s):")
        print(f"{'Smell':<20} {'Specific F1':>12} {'Unified F1':>12} {'Delta':>10}")
        print("-"*56)

        deltas = []
        for smell in SMELLYCODE_SMELLS:
            spec_mean = np.mean(specific_fold_f1s[smell])
            spec_std = np.std(specific_fold_f1s[smell])
            uni_mean = np.mean(unified_fold_f1s[smell])
            uni_std = np.std(unified_fold_f1s[smell])
            delta = spec_mean - uni_mean
            deltas.append(delta)

            print(f"{smell:<20} {spec_mean:>6.3f}±{spec_std:.3f} {uni_mean:>6.3f}±{uni_std:.3f} {delta:>+10.3f}")

            all_results.append({
                'classifier': clf_name,
                'smell': smell,
                'specific_f1_mean': spec_mean,
                'specific_f1_std': spec_std,
                'unified_f1_mean': uni_mean,
                'unified_f1_std': uni_std,
                'delta_f1': delta,
                'positive_rate': Y[:, SMELLYCODE_SMELLS.index(smell)].mean() * 100,
            })

        avg_delta = np.mean(deltas)
        print("-"*56)
        print(f"{'AVERAGE':<20} {'':<12} {'':<12} {avg_delta:>+10.3f}")
        print(f"\n{clf_name}: Avg ΔF1 = {avg_delta:+.3f}")

    return all_results


def main():
    print("="*70)
    print("CORRECTED MULTI-LABEL CLASSIFICATION EXPERIMENTS")
    print("="*70)
    print("\nKey methodology changes:")
    print("1. NO one-hot smell encoding (was handicapping linear models)")
    print("2. Uses 2D target matrix (N × K) for multi-label")
    print("3. Smell-Specific: MultiOutputClassifier (K independent classifiers)")
    print("4. Unified RF: Native multi-label support")
    print("5. Unified Linear/Boosting: ClassifierChain (exploits correlations)")
    print("6. Uses full 107K SmellyCode++ samples")
    print("="*70)

    total_start = time.time()

    # IST2021 - Independent binary (cannot merge)
    ist_results = run_ist2021_binary_experiments()

    # SmellyCode++ - True multi-label
    smelly_results = run_smellycode_multilabel_experiments()

    total_elapsed = time.time() - total_start

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: SmellyCode++ Multi-Label Results")
    print("="*70)
    print(f"\n{'Classifier':<20} {'Avg Specific F1':>15} {'Avg Unified F1':>15} {'Avg ΔF1':>10}")
    print("-"*62)

    for clf_name in ['RandomForest', 'HistGradientBoosting', 'LogisticRegression', 'LinearSVC']:
        clf_results = [r for r in smelly_results if r['classifier'] == clf_name]
        avg_spec = np.mean([r['specific_f1_mean'] for r in clf_results])
        avg_uni = np.mean([r['unified_f1_mean'] for r in clf_results])
        avg_delta = np.mean([r['delta_f1'] for r in clf_results])
        print(f"{clf_name:<20} {avg_spec:>15.3f} {avg_uni:>15.3f} {avg_delta:>+10.3f}")

    print("-"*62)
    print(f"\nTotal runtime: {total_elapsed/60:.1f} minutes")

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # SmellyCode++ results
    df_smelly = pd.DataFrame(smelly_results)
    smelly_path = os.path.join(OUTPUT_PATH, f"multilabel_smellycode_{timestamp}.csv")
    df_smelly.to_csv(smelly_path, index=False)
    print(f"\nSaved: {smelly_path}")

    # IST2021 results
    df_ist = pd.DataFrame(ist_results)
    ist_path = os.path.join(OUTPUT_PATH, f"multilabel_ist2021_{timestamp}.csv")
    df_ist.to_csv(ist_path, index=False)
    print(f"Saved: {ist_path}")

    # Summary CSV
    summary_data = []
    for clf_name in ['RandomForest', 'HistGradientBoosting', 'LogisticRegression', 'LinearSVC']:
        clf_results = [r for r in smelly_results if r['classifier'] == clf_name]
        summary_data.append({
            'classifier': clf_name,
            'dataset': 'SmellyCode++',
            'methodology': 'MultiOutputClassifier vs ClassifierChain/NativeRF',
            'avg_specific_f1': np.mean([r['specific_f1_mean'] for r in clf_results]),
            'avg_unified_f1': np.mean([r['unified_f1_mean'] for r in clf_results]),
            'avg_delta_f1': np.mean([r['delta_f1'] for r in clf_results]),
        })

    df_summary = pd.DataFrame(summary_data)
    summary_path = os.path.join(OUTPUT_PATH, f"multilabel_summary_{timestamp}.csv")
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved: {summary_path}")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("SmellyCode++ (1.5-4% positive rates) - Imbalanced dataset:")
    print("  - If ΔF1 ≈ 0: No advantage to smell-specific classifiers")
    print("  - If ΔF1 < 0: Unified (ClassifierChain/Native RF) performs better")
    print("  - This validates that imbalanced data benefits from unified approach")
    print("="*70)


if __name__ == "__main__":
    main()
