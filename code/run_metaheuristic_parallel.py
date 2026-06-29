"""
RQ4 — Wrapper-based metaheuristic feature selection (FAITHFUL, PARALLELIZED).

Ports the experiment to the real MAFESE 1.x API (MhaSelector) and runs the full
paper configuration (epoch=100, pop_size=50) for PSO/SA/GWO/WOA on the four
SmellyCode++ smells. The 16 (smell x optimizer) jobs are independent and run
across a process pool; each job is single-threaded (inner RF n_jobs=1) to avoid
core oversubscription. Results are appended to the output CSV as each job
finishes, so partial progress survives an interruption.

Methodology (matches the main pipeline for comparability):
  - Features: 14 Halstead metrics.
  - Fitness during search: F1 only (fit_weights=(1.0, 0.0)), per Eq. (1).
  - Estimator: RandomForest(n_estimators=100, class_weight='balanced'), seed 42.
  - Reported F1 (baseline and optimizer-selected): 10-fold StratifiedKFold,
    StandardScaler per fold -- identical protocol for both, so deltas are valid.
"""

import os
import csv
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import multiprocessing as mp

warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from mafese import MhaSelector

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'
SMELLYCODE = DATA_DIR / 'SmellyCode++.csv'

HALSTEAD = [
    'Logical Lines', 'Distinct Operators', 'Distinct Operands',
    'Total Operators', 'Total Operands', 'Vocabulary', 'Length',
    'Calculated Length', 'Volume', 'Difficulty', 'Effort',
    'Time Required', 'Bugs', 'Cyclomatic Complexity'
]
SMELLS = {
    'long_method': 'Long method',
    'god_class': 'God class',
    'feature_envy': 'Feature envy',
    'data_class': 'Data class',
}
OPTIMIZERS = {
    'PSO': 'OriginalPSO',
    'SA': 'OriginalSA',
    'GWO': 'OriginalGWO',
    'WOA': 'OriginalWOA',
}

EPOCH = 100
POP_SIZE = 50
N_ESTIMATORS = 100
N_FOLDS = 10
SEED = 42
N_WORKERS = 8

_X = None
_Y = None


def init_worker():
    """Load the feature matrix and per-smell labels once per worker process."""
    global _X, _Y
    df = pd.read_csv(SMELLYCODE, usecols=HALSTEAD + list(SMELLS.values()))
    _X = np.nan_to_num(df[HALSTEAD].values, nan=0.0)
    _Y = {k: (df[c] == 1).astype(int).values for k, c in SMELLS.items()}


def cv_f1(X, y):
    """Mean F1 over 10-fold stratified CV with a balanced RandomForest."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    scores = []
    for tr, te in skf.split(X, y):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X[tr])
        Xte = scaler.transform(X[te])
        clf = RandomForestClassifier(n_estimators=N_ESTIMATORS,
                                     class_weight='balanced',
                                     random_state=SEED, n_jobs=1)
        clf.fit(Xtr, y[tr])
        scores.append(f1_score(y[te], clf.predict(Xte), zero_division=0))
    return float(np.mean(scores))


def run_task(args):
    """args = (kind, smell, optimizer_label). Returns a result row."""
    kind, smell, opt = args
    t0 = time.time()
    X, y = _X, _Y[smell]
    if kind == 'baseline':
        f1 = cv_f1(X, y)
        return ['baseline', smell, 'baseline', f1, '', round(time.time() - t0, 1)]
    selector = MhaSelector(
        problem='classification', estimator='rf',
        estimator_paras={'n_estimators': N_ESTIMATORS, 'random_state': SEED,
                         'class_weight': 'balanced', 'n_jobs': 1},
        optimizer=OPTIMIZERS[opt],
        optimizer_paras={'epoch': EPOCH, 'pop_size': POP_SIZE},
        obj_name='F1S', seed=SEED, verbose=False)
    selector.fit(X, y, fit_weights=(1.0, 0.0))
    idx = [int(i) for i in selector.selected_feature_indexes]
    f1 = cv_f1(X[:, idx], y)
    return ['opt', smell, opt, f1, ';'.join(map(str, idx)), round(time.time() - t0, 1)]


def main():
    if not SMELLYCODE.exists():
        raise SystemExit(f"SmellyCode++ not found at {SMELLYCODE}")
    ts = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    out = DATA_DIR / f'metaheuristic_regen_{ts}.csv'
    tasks = ([('baseline', s, 'baseline') for s in SMELLS] +
             [('opt', s, o) for s in SMELLS for o in OPTIMIZERS])
    print(f"Launching {len(tasks)} tasks ({len(SMELLS)} baselines + "
          f"{len(SMELLS)*len(OPTIMIZERS)} optimizer runs) on {N_WORKERS} workers", flush=True)
    print(f"Config: epoch={EPOCH}, pop_size={POP_SIZE}, RF n_estimators={N_ESTIMATORS}", flush=True)
    print(f"Output (incremental): {out}", flush=True)
    done = 0
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['kind', 'smell', 'optimizer', 'f1', 'selected_idx', 'seconds'])
        f.flush()
        with mp.Pool(N_WORKERS, initializer=init_worker) as pool:
            for res in pool.imap_unordered(run_task, tasks):
                w.writerow(res)
                f.flush()
                done += 1
                print(f"[{done}/{len(tasks)}] {res[0]:8s} {res[1]:12s} {res[2]:8s} "
                      f"F1={res[3]:.3f}  ({res[5]}s)", flush=True)
    print(f"ALL DONE -> {out}", flush=True)


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
