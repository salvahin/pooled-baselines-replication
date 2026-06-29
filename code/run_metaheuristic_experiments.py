#!/usr/bin/env python3
"""
Metaheuristic Feature Selection Experiments (RQ4)

Tests whether wrapper-based metaheuristic feature selection can compensate
for class imbalance in code smell detection.

Algorithms tested (via MAFESE framework):
- Particle Swarm Optimization (PSO)
- Simulated Annealing (SA)
- Grey Wolf Optimizer (GWO)
- Whale Optimization Algorithm (WOA)

Requirements:
    pip install mafese  # MAFESE framework for metaheuristic feature selection

Usage:
    python run_metaheuristic_experiments.py

Output:
    Results saved to ../data/metaheuristic_results_*.csv

Note:
    This experiment requires the SmellyCode++ dataset.
    Download from: https://doi.org/10.6084/m9.figshare.28234218
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from pathlib import Path
from datetime import datetime
import warnings
import time

warnings.filterwarnings('ignore')

# Try to import MAFESE - optional dependency
try:
    from mafese import Data, get_optimizer
    MAFESE_AVAILABLE = True
except ImportError:
    MAFESE_AVAILABLE = False
    print("WARNING: MAFESE not installed. Install with: pip install mafese")
    print("         Without MAFESE, this script will only display pre-computed results.\n")

# =============================================================================
# Configuration
# =============================================================================
RANDOM_STATE = 42
N_FOLDS = 10
N_ESTIMATORS = 100

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
OUTPUT_DIR = DATA_DIR

# SmellyCode++ configuration
SMELLYCODE_SMELLS = {
    'long_method': 'Long method',
    'god_class': 'God class',
    'feature_envy': 'Feature envy',
    'data_class': 'Data class',
}

HALSTEAD_FEATURES = [
    'Logical Lines', 'Distinct Operators', 'Distinct Operands',
    'Total Operators', 'Total Operands', 'Vocabulary', 'Length',
    'Calculated Length', 'Volume', 'Difficulty', 'Effort',
    'Time Required', 'Bugs', 'Cyclomatic Complexity'
]

# Metaheuristic configurations
OPTIMIZERS = {
    'PSO': {
        'name': 'OriginalPSO',
        'params': {
            'epoch': 100,
            'pop_size': 50,
            'c1': 2.0,
            'c2': 2.0,
            'w_min': 0.4,
            'w_max': 0.9,
        }
    },
    'SA': {
        'name': 'OriginalSA',
        'params': {
            'epoch': 100,
            'pop_size': 50,
            'temp_init': 1000,
            'cooling_rate': 0.99,
        }
    },
    'GWO': {
        'name': 'OriginalGWO',
        'params': {
            'epoch': 100,
            'pop_size': 50,
        }
    },
    'WOA': {
        'name': 'OriginalWOA',
        'params': {
            'epoch': 100,
            'pop_size': 50,
        }
    },
}

# Grid search configurations tested
GRID_SEARCH = {
    'epochs': [30, 50, 100],
    'pop_sizes': [15, 30, 50],
    'n_runs': 3,  # Independent runs per configuration
}


# =============================================================================
# Baseline Classifier
# =============================================================================

def run_baseline(X, y):
    """Run baseline classifier without feature selection."""
    fold_f1s = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        fold_f1s.append(f1_score(y_test, y_pred, zero_division=0))

    return np.mean(fold_f1s)


# =============================================================================
# Metaheuristic Feature Selection
# =============================================================================

def run_metaheuristic_fs(X, y, optimizer_name, optimizer_config):
    """Run metaheuristic feature selection using MAFESE."""
    if not MAFESE_AVAILABLE:
        return None

    # Prepare data for MAFESE
    data = Data(X, y)
    data.split_train_test(test_size=0.2, random_state=RANDOM_STATE)

    # Get optimizer
    optimizer = get_optimizer(optimizer_config['name'])

    # Fitness function: F1-score on validation set
    def fitness_function(solution, X_train, y_train, X_test, y_test):
        # Binary mask for feature selection
        selected = np.where(solution > 0.5)[0]
        if len(selected) == 0:
            return 0.0

        X_train_sel = X_train[:, selected]
        X_test_sel = X_test[:, selected]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_sel)
        X_test_scaled = scaler.transform(X_test_sel)

        clf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        return f1_score(y_test, y_pred, zero_division=0)

    # Run optimization
    best_solution, best_fitness = optimizer.solve(
        data,
        fitness_function,
        **optimizer_config['params']
    )

    return best_fitness


# =============================================================================
# Pre-computed Results (Table 18)
# =============================================================================

PRECOMPUTED_RESULTS = {
    'long_method': {'baseline': 0.257, 'PSO': 0.247, 'SA': 0.241, 'GWO': 0.078, 'WOA': 0.100},
    'god_class':   {'baseline': 0.455, 'PSO': 0.406, 'SA': 0.407, 'GWO': 0.325, 'WOA': 0.354},
    'feature_envy':{'baseline': 0.323, 'PSO': 0.300, 'SA': 0.284, 'GWO': 0.089, 'WOA': 0.091},
    'data_class':  {'baseline': 0.027, 'PSO': 0.000, 'SA': 0.005, 'GWO': 0.077, 'WOA': 0.059},
}


def display_precomputed_results():
    """Display pre-computed results from Table 18."""
    print("\n" + "="*70)
    print("PRE-COMPUTED RESULTS (Table 18)")
    print("="*70)
    print("\nThese results were obtained with MAFESE framework.")
    print("To regenerate, install MAFESE and provide SmellyCode++ dataset.\n")

    print(f"{'Smell Type':<15} {'Baseline':>10} {'PSO':>10} {'SA':>10} {'GWO':>10} {'WOA':>10}")
    print("-"*65)

    for smell, results in PRECOMPUTED_RESULTS.items():
        baseline = results['baseline']
        pso = results['PSO']
        sa = results['SA']
        gwo = results['GWO']
        woa = results['WOA']

        # Format with delta indicators
        pso_str = f"{pso:.3f} ({pso-baseline:+.3f})"
        sa_str = f"{sa:.3f} ({sa-baseline:+.3f})"
        gwo_str = f"{gwo:.3f} ({gwo-baseline:+.3f})"
        woa_str = f"{woa:.3f} ({woa-baseline:+.3f})"

        print(f"{smell:<15} {baseline:>10.3f} {pso_str:>15} {sa_str:>15}")

    print("\n" + "-"*65)
    print("Finding: 14/16 optimizer-smell combinations (88%) degraded performance.")
    print("Wrapper-based metaheuristic FS did not compensate for class imbalance.")
    print("="*70)


# =============================================================================
# Main Execution
# =============================================================================

def main():
    print("="*70)
    print("METAHEURISTIC FEATURE SELECTION EXPERIMENTS (RQ4)")
    print("Wrapper-based feature selection using MAFESE framework")
    print("="*70)

    smellycode_path = DATA_DIR / "SmellyCode++.csv"

    if not smellycode_path.exists():
        print(f"\nWARNING: SmellyCode++ dataset not found at {smellycode_path}")
        print("Download from: https://doi.org/10.6084/m9.figshare.28234218")
        print("\nDisplaying pre-computed results instead:\n")
        display_precomputed_results()
        return

    if not MAFESE_AVAILABLE:
        print("\nMAFESE not installed. Displaying pre-computed results:\n")
        display_precomputed_results()
        return

    # Load data
    print(f"\nLoading SmellyCode++ from {smellycode_path}")
    df = pd.read_csv(smellycode_path)
    print(f"Loaded {len(df):,} samples")

    X_all = df[HALSTEAD_FEATURES].values
    X_all = np.nan_to_num(X_all, nan=0.0)

    results = []

    for smell_name, label_col in SMELLYCODE_SMELLS.items():
        print(f"\n--- {smell_name} ---")
        y = (df[label_col] == 1).astype(int).values
        print(f"Positive rate: {y.mean():.2%}")

        # Baseline
        baseline_f1 = run_baseline(X_all, y)
        print(f"Baseline F1: {baseline_f1:.3f}")

        smell_result = {
            'smell': smell_name,
            'baseline_f1': baseline_f1,
        }

        # Run each optimizer
        for opt_name, opt_config in OPTIMIZERS.items():
            print(f"  Running {opt_name}...", end=" ")
            start = time.time()

            opt_f1 = run_metaheuristic_fs(X_all, y, opt_name, opt_config)

            if opt_f1 is not None:
                delta = opt_f1 - baseline_f1
                print(f"F1={opt_f1:.3f} (Δ={delta:+.3f}) [{time.time()-start:.1f}s]")
                smell_result[f'{opt_name.lower()}_f1'] = opt_f1
                smell_result[f'{opt_name.lower()}_delta'] = delta
            else:
                print("skipped (MAFESE not available)")

        results.append(smell_result)

    # Save results
    if results:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        df_results = pd.DataFrame(results)
        output_path = OUTPUT_DIR / f"metaheuristic_results_{timestamp}.csv"
        df_results.to_csv(output_path, index=False)
        print(f"\nSaved: {output_path}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    degraded = 0
    total = 0
    for smell_result in results:
        baseline = smell_result['baseline_f1']
        for opt in ['pso', 'sa', 'gwo', 'woa']:
            if f'{opt}_f1' in smell_result:
                total += 1
                if smell_result[f'{opt}_f1'] < baseline:
                    degraded += 1

    if total > 0:
        print(f"Degraded performance: {degraded}/{total} ({100*degraded/total:.0f}%)")
    print("="*70)


if __name__ == "__main__":
    main()
