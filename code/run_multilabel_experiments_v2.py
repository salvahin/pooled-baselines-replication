#!/usr/bin/env python3
"""
Corrected Multi-Label Classification Experiments v2.

Addresses three audit gaps:
1. ClassifierChain order: Labels ordered by descending positive rate
2. Architectural consistency: RF also tested with ClassifierChain wrapper
3. ImprovMLCQ included: Validates transition zone (5-10%)

Comparison:
- Specific: MultiOutputClassifier(Base) - K independent classifiers
- Unified: ClassifierChain(Base) - exploits label correlations (ALL classifiers)
- Unified-Native: RF.fit(X, Y_2D) - native multi-label (RF only, for comparison)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
SMELLYCODE_PATH = "/Users/salvahin/Papers/smells-paper/data/SmellyCode++.csv"
IMPROVMLCQ_PATH = "/Users/salvahin/Papers/smells-paper/improvmlcq_out_clean.csv"
OUTPUT_PATH = "/Users/salvahin/Papers/smells-paper/data"

# SmellyCode++ smell columns (will be reordered by positive rate)
SMELLYCODE_SMELLS = ['Long method', 'God class', 'Feature envy', 'Data class']

# ImprovMLCQ smell columns
IMPROVMLCQ_SMELLS = ['blob_label', 'dataclass_label', 'featureenvy_label', 'longmethod_label']

# Halstead metrics (features for SmellyCode++)
HALSTEAD_FEATURES = [
    'Logical Lines', 'Distinct Operators', 'Distinct Operands',
    'Total Operators', 'Total Operands', 'Vocabulary', 'Length',
    'Calculated Length', 'Volume', 'Difficulty', 'Effort',
    'Time Required', 'Bugs', 'Cyclomatic Complexity'
]


def get_ordered_columns(Y, smell_names):
    """
    Order columns by DESCENDING positive rate for ClassifierChain.
    This ensures the most common smell is predicted first, anchoring predictions.
    """
    pos_rates = [(i, Y[:, i].mean(), name) for i, name in enumerate(smell_names)]
    pos_rates.sort(key=lambda x: -x[1])  # Descending
    order = [x[0] for x in pos_rates]
    ordered_names = [x[2] for x in pos_rates]
    print(f"  Chain order (descending pos rate): {[(n, f'{Y[:, i].mean()*100:.1f}%') for i, n in zip(order, ordered_names)]}")
    return order, ordered_names


def create_stratification_labels(Y):
    """Create stratification labels for multi-label data."""
    return ['_'.join(map(str, row)) for row in Y]


def run_multilabel_experiment(X, Y, smell_names, dataset_name):
    """
    Run multi-label experiment with proper methodology.

    Tests three configurations:
    1. Specific: MultiOutputClassifier (K independent classifiers)
    2. Unified-Chain: ClassifierChain (exploits label correlations)
    3. Unified-Native: Native RF multi-label (for comparison)
    """
    print(f"\n{'='*70}")
    print(f"DATASET: {dataset_name}")
    print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}, Labels: {Y.shape[1]}")
    print("="*70)

    # Order columns by descending positive rate for ClassifierChain
    chain_order, ordered_names = get_ordered_columns(Y, smell_names)

    # Reorder Y for ClassifierChain (most common first)
    Y_ordered = Y[:, chain_order]

    # Create stratification labels
    strat_labels = create_stratification_labels(Y)
    le = LabelEncoder()
    strat_encoded = le.fit_transform(strat_labels)

    # Classifiers to test
    classifiers = {
        'RandomForest': lambda: RandomForestClassifier(
            n_estimators=100, class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=N_JOBS
        ),
        'HistGradientBoosting': lambda: HistGradientBoostingClassifier(
            max_iter=100, class_weight='balanced', random_state=RANDOM_STATE
        ),
        'LogisticRegression': lambda: LogisticRegression(
            class_weight='balanced', random_state=RANDOM_STATE,
            max_iter=1000, n_jobs=N_JOBS
        ),
        'LinearSVC': lambda: LinearSVC(
            class_weight='balanced', random_state=RANDOM_STATE,
            max_iter=10000, dual='auto'
        ),
    }

    all_results = []

    for clf_name, clf_factory in classifiers.items():
        print(f"\n{'='*50}")
        print(f"CLASSIFIER: {clf_name}")
        print("="*50)

        # Results storage (using ORDERED names for chain, original for specific)
        specific_fold_f1s = {name: [] for name in smell_names}
        chain_fold_f1s = {name: [] for name in smell_names}
        native_fold_f1s = {name: [] for name in smell_names} if clf_name == 'RandomForest' else None

        start_time = time.time()

        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

        for fold, (train_idx, test_idx) in enumerate(skf.split(X, strat_encoded)):
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]
            Y_train_ordered, Y_test_ordered = Y_ordered[train_idx], Y_ordered[test_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # 1. SPECIFIC: MultiOutputClassifier (K independent classifiers)
            specific_clf = MultiOutputClassifier(clf_factory())
            specific_clf.fit(X_train_scaled, Y_train)
            Y_pred_specific = specific_clf.predict(X_test_scaled)

            # 2. UNIFIED-CHAIN: ClassifierChain (ordered by descending positive rate)
            chain_clf = ClassifierChain(clf_factory(), order=list(range(len(chain_order))),
                                        random_state=RANDOM_STATE)
            chain_clf.fit(X_train_scaled, Y_train_ordered)
            Y_pred_chain_ordered = chain_clf.predict(X_test_scaled)

            # Map chain predictions back to original order
            Y_pred_chain = np.zeros_like(Y_pred_chain_ordered)
            for new_idx, orig_idx in enumerate(chain_order):
                Y_pred_chain[:, orig_idx] = Y_pred_chain_ordered[:, new_idx]

            # 3. UNIFIED-NATIVE: Native RF multi-label (only for RandomForest)
            if clf_name == 'RandomForest':
                native_clf = clf_factory()
                native_clf.fit(X_train_scaled, Y_train)
                Y_pred_native = native_clf.predict(X_test_scaled)

            # Per-smell F1 scores (in original order)
            for i, smell in enumerate(smell_names):
                specific_fold_f1s[smell].append(
                    f1_score(Y_test[:, i], Y_pred_specific[:, i], zero_division=0)
                )
                chain_fold_f1s[smell].append(
                    f1_score(Y_test[:, i], Y_pred_chain[:, i], zero_division=0)
                )
                if clf_name == 'RandomForest':
                    native_fold_f1s[smell].append(
                        f1_score(Y_test[:, i], Y_pred_native[:, i], zero_division=0)
                    )

            print(f"  Fold {fold+1}/{N_FOLDS} completed")

        elapsed = time.time() - start_time

        # Calculate and display results
        print(f"\nResults ({elapsed:.1f}s):")
        if clf_name == 'RandomForest':
            print(f"{'Smell':<15} {'Specific':>10} {'Chain':>10} {'Native':>10} {'Δ(Spec-Chain)':>14} {'Δ(Spec-Native)':>15}")
            print("-"*76)
        else:
            print(f"{'Smell':<15} {'Specific':>10} {'Chain':>10} {'Δ(Spec-Chain)':>14}")
            print("-"*51)

        deltas_chain = []
        deltas_native = []

        for smell in smell_names:
            spec_mean = np.mean(specific_fold_f1s[smell])
            chain_mean = np.mean(chain_fold_f1s[smell])
            delta_chain = spec_mean - chain_mean
            deltas_chain.append(delta_chain)

            pos_rate = Y[:, smell_names.index(smell)].mean() * 100

            if clf_name == 'RandomForest':
                native_mean = np.mean(native_fold_f1s[smell])
                delta_native = spec_mean - native_mean
                deltas_native.append(delta_native)
                print(f"{smell:<15} {spec_mean:>10.3f} {chain_mean:>10.3f} {native_mean:>10.3f} {delta_chain:>+14.3f} {delta_native:>+15.3f}")
            else:
                print(f"{smell:<15} {spec_mean:>10.3f} {chain_mean:>10.3f} {delta_chain:>+14.3f}")

            # Store results
            result = {
                'dataset': dataset_name,
                'classifier': clf_name,
                'smell': smell,
                'positive_rate': pos_rate,
                'specific_f1': spec_mean,
                'chain_f1': chain_mean,
                'delta_spec_chain': delta_chain,
            }
            if clf_name == 'RandomForest':
                result['native_f1'] = native_mean
                result['delta_spec_native'] = delta_native
            all_results.append(result)

        # Averages
        avg_delta_chain = np.mean(deltas_chain)
        if clf_name == 'RandomForest':
            avg_delta_native = np.mean(deltas_native)
            print("-"*76)
            print(f"{'AVERAGE':<15} {'':>10} {'':>10} {'':>10} {avg_delta_chain:>+14.3f} {avg_delta_native:>+15.3f}")
        else:
            print("-"*51)
            print(f"{'AVERAGE':<15} {'':>10} {'':>10} {avg_delta_chain:>+14.3f}")

        print(f"\n{clf_name}: Avg Δ(Specific-Chain) = {avg_delta_chain:+.3f}")

    return all_results


def main():
    print("="*70)
    print("CORRECTED MULTI-LABEL EXPERIMENTS v2")
    print("="*70)
    print("\nAudit fixes applied:")
    print("1. ClassifierChain order: Labels ordered by DESCENDING positive rate")
    print("2. Architectural consistency: RF also tested with ClassifierChain")
    print("3. ImprovMLCQ included: Validates transition zone (5-10%)")
    print("="*70)

    total_start = time.time()
    all_results = []

    # =========================================================================
    # SMELLYCODE++ (1.5-4% positive rates - severely imbalanced)
    # =========================================================================
    print("\n" + "="*70)
    print("Loading SmellyCode++ (full 107K samples)...")
    df_smelly = pd.read_csv(SMELLYCODE_PATH)
    X_smelly = df_smelly[HALSTEAD_FEATURES].values
    X_smelly = np.nan_to_num(X_smelly, nan=0.0)
    Y_smelly = df_smelly[SMELLYCODE_SMELLS].values.astype(int)

    smelly_results = run_multilabel_experiment(
        X_smelly, Y_smelly, SMELLYCODE_SMELLS, 'SmellyCode++'
    )
    all_results.extend(smelly_results)

    # =========================================================================
    # IMPROVMLCQ (5-10% positive rates - transition zone)
    # =========================================================================
    print("\n" + "="*70)
    print("Loading ImprovMLCQ (13K samples)...")
    df_imlcq = pd.read_csv(IMPROVMLCQ_PATH)

    # Get CK features
    ck_features = sorted([c for c in df_imlcq.columns if c.startswith('ck_')])
    X_imlcq = df_imlcq[ck_features].values
    X_imlcq = np.nan_to_num(X_imlcq, nan=0.0)
    Y_imlcq = df_imlcq[IMPROVMLCQ_SMELLS].values.astype(int)

    # Rename smells for display
    imlcq_smell_names = ['Blob', 'Data Class', 'Feature Envy', 'Long Method']

    imlcq_results = run_multilabel_experiment(
        X_imlcq, Y_imlcq, imlcq_smell_names, 'ImprovMLCQ'
    )
    all_results.extend(imlcq_results)

    total_elapsed = time.time() - total_start

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("SUMMARY: Specific vs Chain (Δ = Specific - Chain)")
    print("="*70)

    df_results = pd.DataFrame(all_results)

    # Summary by classifier and dataset
    print(f"\n{'Classifier':<20} {'SmellyCode++ (1.5-4%)':>22} {'ImprovMLCQ (5-10%)':>20}")
    print("-"*64)

    for clf_name in ['RandomForest', 'HistGradientBoosting', 'LogisticRegression', 'LinearSVC']:
        smelly_delta = df_results[(df_results['classifier'] == clf_name) &
                                   (df_results['dataset'] == 'SmellyCode++')]['delta_spec_chain'].mean()
        imlcq_delta = df_results[(df_results['classifier'] == clf_name) &
                                  (df_results['dataset'] == 'ImprovMLCQ')]['delta_spec_chain'].mean()
        print(f"{clf_name:<20} {smelly_delta:>+22.3f} {imlcq_delta:>+20.3f}")

    # RF-specific: Native vs Chain comparison
    print("\n" + "="*70)
    print("RandomForest: Native Multi-Label vs ClassifierChain")
    print("="*70)
    rf_results = df_results[df_results['classifier'] == 'RandomForest']
    for dataset in ['SmellyCode++', 'ImprovMLCQ']:
        ds_results = rf_results[rf_results['dataset'] == dataset]
        avg_chain = ds_results['chain_f1'].mean()
        avg_native = ds_results['native_f1'].mean()
        delta = avg_chain - avg_native
        print(f"{dataset}: Chain F1={avg_chain:.3f}, Native F1={avg_native:.3f}, Δ(Chain-Native)={delta:+.3f}")

    print(f"\nTotal runtime: {total_elapsed/60:.1f} minutes")

    # Save results
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_path = os.path.join(OUTPUT_PATH, f"multilabel_v2_results_{timestamp}.csv")
    df_results.to_csv(results_path, index=False)
    print(f"\nSaved: {results_path}")

    print("\n" + "="*70)
    print("INTERPRETATION")
    print("="*70)
    print("Δ = Specific - Chain (positive = specific wins, negative = chain wins)")
    print("\nExpected pattern:")
    print("  - ImprovMLCQ (5-10%): Mixed results (transition zone)")
    print("  - SmellyCode++ (1.5-4%): Chain should win (negative Δ)")
    print("\nIf RF Chain > RF Native: ClassifierChain exploits label correlations better")
    print("="*70)


if __name__ == "__main__":
    main()
