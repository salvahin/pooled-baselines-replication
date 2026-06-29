"""
Clean regeneration of all data-driven figures from real results.

Design rules (for scientific-paper quality):
  - No figure titles and no free-floating annotation/explanation boxes
    (those belong in the LaTeX caption). Only axis labels, tick labels,
    legends, error bars, and data-value labels remain.
  - One consistent colorblind-safe palette (Paul Tol) across all figures.
  - Figures reflect the CURRENT manuscript: IST2021 is Specific vs Multi-Class
    (RQ1), the boundary forest plot is per-smell SMOTE (Table 17), and RQ4 is
    the regenerated performance-neutral result.

Outputs to <repo>/figures/. Run:  python code/regenerate_figures_clean.py
"""
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

CODE_DIR = Path(__file__).resolve().parent
DATA_DIR = CODE_DIR.parent / 'data'
FIG_DIR = CODE_DIR.parents[1] / 'figures'          # sp_clone/figures
IST_DIR = DATA_DIR / 'IST2021'

# Paul Tol bright palette (colorblind-safe)
BLUE, RED, GREEN, YELLOW, CYAN, PURPLE, GREY = (
    '#4477AA', '#EE6677', '#228833', '#CCBB44', '#66CCEE', '#AA3377', '#BBBBBB')

plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 12, 'axes.labelsize': 13,
    'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'grid.alpha': 0.25, 'grid.linewidth': 0.6,
    'figure.dpi': 150, 'savefig.dpi': 150, 'savefig.bbox': 'tight',
})


def latest(pattern):
    fs = sorted(DATA_DIR.glob(pattern))
    return fs[-1] if fs else None


def save(fig, name):
    out = FIG_DIR / name
    fig.savefig(out)
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------------------------------------------------------------- fig1 (RQ1)
def fig1_ist2021():
    df = pd.read_csv(latest('ist2021_results_pr_*.csv'))
    order = ['god_class', 'data_class', 'long_method', 'feature_envy',
             'long_parameter_list', 'switch_statements']
    labels = ['God Class', 'Data Class', 'Long Method', 'Feature Envy',
              'Long Param.\nList', 'Switch\nStatements']
    df = df.set_index('smell').loc[order]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - w/2, df['spec_f1_mean'], w, yerr=df['spec_f1_std'], capsize=3,
           label='Smell-specific', color=BLUE)
    ax.bar(x + w/2, df['mc_f1_mean'], w, yerr=df['mc_f1_std'], capsize=3,
           label='Multi-class', color=RED)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('F1 score (10-fold CV)'); ax.set_xlabel('Code smell')
    ax.set_ylim(0, 1.08); ax.grid(axis='x', visible=False)
    ax.legend(frameon=False, ncol=2, loc='upper right')
    save(fig, 'fig1_ist2021_comparison.png')


# ---------------------------------------------------------------- fig2 (RQ2a)
def fig2_smellycode():
    df = pd.read_csv(latest('smellycode_specialization_*.csv'))
    order = ['god_class', 'long_method', 'feature_envy', 'data_class']
    labels = ['God Class', 'Long Method', 'Feature Envy', 'Data Class']
    df = df.set_index('smell').loc[order]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.bar(x - w/2, df['specific_f1'], w, yerr=df['specific_std'], capsize=3,
           label='Smell-specific', color=BLUE)
    ax.bar(x + w/2, df['multiclass_f1'], w, yerr=df['multiclass_std'], capsize=3,
           label='Multi-class', color=RED)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('F1 score (10-fold CV)'); ax.set_xlabel('Code smell')
    ax.set_ylim(0, 0.65); ax.grid(axis='x', visible=False)
    ax.legend(frameon=False, ncol=2, loc='upper right')
    save(fig, 'fig2_smellycode_comparison.png')


# ---------------------------------------------------------------- fig2b (RQ2b)
def fig2b_improvmlcq():
    df = pd.read_csv(latest('improvmlcq_results_pr_*.csv'))
    order = ['blob', 'data_class', 'feature_envy', 'long_method']
    df = df.set_index('smell').loc[order]
    labels = [f"Blob\n({df.loc['blob','positive_rate']*100:.1f}%)",
              f"Data Class\n({df.loc['data_class','positive_rate']*100:.1f}%)",
              f"Feature Envy\n({df.loc['feature_envy','positive_rate']*100:.1f}%)",
              f"Long Method\n({df.loc['long_method','positive_rate']*100:.1f}%)"]
    x = np.arange(len(order)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(x - w/2, df['spec_f1'], w, label='Smell-specific', color=BLUE)
    ax.bar(x + w/2, df['unif_f1'], w, label='Multi-class', color=RED)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('F1 score (10-fold CV)')
    ax.set_xlabel('Code smell (positive rate)')
    ax.set_ylim(0, 0.72); ax.grid(axis='x', visible=False)
    ax.legend(frameon=False, ncol=2, loc='upper right')
    save(fig, 'fig2b_improvmlcq_comparison.png')


# ---------------------------------------------------------------- fig3 (RQ3)
def fig3_boundary():
    df = pd.read_csv(latest('boundary_conditions_pr_*.csv'))
    order = ['god_class', 'long_method', 'feature_envy', 'data_class']
    labels = ['God Class', 'Long Method', 'Feature Envy', 'Data Class']
    df = df.set_index('smell').loc[order]
    y = np.arange(len(order))[::-1]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    for yi, sm in zip(y, order):
        d = df.loc[sm, 'cohens_d']; p = df.loc[sm, 'p_value']
        sig = p < 0.05
        c = GREEN if sig else GREY
        ax.plot([0, d], [yi, yi], color=c, lw=2.5, zorder=1)
        ax.scatter([d], [yi], s=90, color=c, zorder=2,
                   label=('Significant ($p<0.05$)' if sig else 'Not significant'))
        ax.annotate(f"d = {d:+.2f}", (d, yi), textcoords='offset points',
                    xytext=(8 if d >= 0 else -8, 8),
                    ha='left' if d >= 0 else 'right', fontsize=10)
    ax.axvline(0, color='k', lw=0.8, ls='--')
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Cohen's d (SMOTE balancing effect)")
    ax.set_xlim(-2, 6.4); ax.set_ylim(-0.6, len(order) - 0.2)
    ax.grid(axis='y', visible=False)
    # de-duplicate legend; place top-right where there is empty space
    h, l = ax.get_legend_handles_labels()
    seen = dict(zip(l, h))
    ax.legend(seen.values(), seen.keys(), frameon=False, loc='upper right',
              bbox_to_anchor=(1.0, 1.02))
    save(fig, 'fig3_boundary_forest_plot.png')


# ---------------------------------------------------------------- fig9 (RQ4)
def fig9_metaheuristic():
    f = latest('metaheuristic_regen_*.csv')
    if f is None:
        print("  skip fig9: no metaheuristic_regen CSV"); return
    df = pd.read_csv(f)
    base = {r['smell']: r['f1'] for _, r in df[df.kind == 'baseline'].iterrows()}
    opt = df[df.kind == 'opt']
    order = ['long_method', 'god_class', 'feature_envy', 'data_class']
    labels = ['Long Method', 'God Class', 'Feature Envy', 'Data Class']
    series = [('Baseline', None, GREY), ('PSO', 'PSO', BLUE), ('SA', 'SA', GREEN),
              ('GWO', 'GWO', YELLOW), ('WOA', 'WOA', RED)]
    x = np.arange(len(order)); w = 0.16
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for i, (name, key, col) in enumerate(series):
        if key is None:
            vals = [base[s] for s in order]
        else:
            vals = [opt[(opt.smell == s) & (opt.optimizer == key)]['f1'].values[0]
                    for s in order]
        ax.bar(x + (i - 2) * w, vals, w, label=name, color=col)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel('F1 score (10-fold CV)'); ax.set_xlabel('Code smell')
    ax.set_ylim(0, 0.62); ax.grid(axis='x', visible=False)
    ax.legend(frameon=False, ncol=5, loc='upper center', bbox_to_anchor=(0.5, 1.10))
    save(fig, 'fig9_metaheuristic_fs.png')


# ------------------------------------------------------- IST2021 loading helper
IST2021_SMELLS = {
    'god_class': ('GodClass.csv', 'is_god_class'),
    'data_class': ('DataClass.csv', 'is_data_class'),
    'long_method': ('LongMethod.csv', 'is_long_method'),
    'feature_envy': ('FeatureEnvy.csv', 'is_feature_envy'),
    'long_parameter_list': ('LongParameterList.csv', 'is_long_parameters_list'),
    'switch_statements': ('SwitchStatements.csv', 'is_switch_statements'),
}


def _common_features():
    sets = []
    for _, (fn, tgt) in IST2021_SMELLS.items():
        df = pd.read_csv(IST_DIR / fn)
        sets.append({c for c in df.columns if c != tgt})
    return sorted(set.intersection(*sets))


def _clean_name(c):
    return c.replace('_type', '').replace('_package', ' (pkg)').replace('_project', ' (prj)')


# ---------------------------------------------------------- fig5 feature heatmap
def fig5_feature_heatmap():
    if not IST_DIR.exists():
        print("  skip fig5: IST2021 data not present"); return
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    feats = _common_features()
    order = ['god_class', 'data_class', 'long_method', 'feature_envy',
             'long_parameter_list', 'switch_statements']
    imp = {}
    for sm in order:
        fn, tgt = IST2021_SMELLS[sm]
        df = pd.read_csv(IST_DIR / fn)
        X = np.nan_to_num(df[feats].values, nan=0.0)
        y = df[tgt].apply(lambda v: 1 if v in [True, 'TRUE', 1, '1'] else 0).values
        Xs = StandardScaler().fit_transform(X)
        clf = RandomForestClassifier(n_estimators=100, class_weight='balanced',
                                     max_features='sqrt', random_state=42, n_jobs=-1)
        clf.fit(Xs, y)
        imp[sm] = clf.feature_importances_
    M = pd.DataFrame(imp, index=feats).T          # smells x features
    # keep features that are top-6 for at least one smell (readability)
    keep = set()
    for sm in order:
        keep |= set(M.loc[sm].sort_values(ascending=False).head(6).index)
    cols = [c for c in feats if c in keep]
    M = M[cols]
    fig, ax = plt.subplots(figsize=(min(1.0 + 0.55 * len(cols), 13), 4.6))
    im = ax.imshow(M.values, aspect='auto', cmap='YlGnBu', vmin=0,
                   vmax=float(M.values.max()))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([_clean_name(c) for c in cols], rotation=45, ha='right',
                       fontsize=9)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(['God Class', 'Data Class', 'Long Method', 'Feature Envy',
                        'Long Param. List', 'Switch Statements'])
    for i in range(len(order)):
        for j in range(len(cols)):
            v = M.values[i, j]
            if v >= 0.02:
                ax.text(j, i, f"{v:.2f}", ha='center', va='center', fontsize=8,
                        color='white' if v > M.values.max() * 0.55 else 'black')
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label('RF feature importance')
    ax.set_xlabel('Feature (CK metric)'); ax.set_ylabel('Code smell')
    save(fig, 'fig5_feature_heatmap.png')


# ----------------------------------------------------- fig6 RF hyperparam sens.
def fig6_hyperparam():
    if not IST_DIR.exists():
        print("  skip fig6: IST2021 data not present"); return
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score
    feats = _common_features()
    order = list(IST2021_SMELLS)
    data = {}
    for sm in order:
        fn, tgt = IST2021_SMELLS[sm]
        df = pd.read_csv(IST_DIR / fn)
        X = np.nan_to_num(df[feats].values, nan=0.0)
        y = df[tgt].apply(lambda v: 1 if v in [True, 'TRUE', 1, '1'] else 0).values
        data[sm] = (X, y)

    def mean_f1(n_estimators=100, max_features='sqrt'):
        per = []
        for sm in order:
            X, y = data[sm]
            skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
            fs = []
            for tr, te in skf.split(X, y):
                sc = StandardScaler(); Xtr = sc.fit_transform(X[tr]); Xte = sc.transform(X[te])
                clf = RandomForestClassifier(n_estimators=n_estimators,
                                             max_features=max_features,
                                             class_weight='balanced',
                                             random_state=42, n_jobs=-1)
                clf.fit(Xtr, y[tr]); fs.append(f1_score(y[te], clf.predict(Xte), zero_division=0))
            per.append(np.mean(fs))
        return np.mean(per), np.std(per)

    n_grid = [25, 50, 100, 200]
    mf_grid = [4, 8, 12, 18, 24]
    n_m = [mean_f1(n_estimators=n) for n in n_grid]
    mf_m = [mean_f1(max_features=m) for m in mf_grid]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    m, s = np.array([v[0] for v in n_m]), np.array([v[1] for v in n_m])
    a1.plot(n_grid, m, '-o', color=BLUE)
    a1.fill_between(n_grid, m - s, m + s, color=BLUE, alpha=0.18)
    a1.set_xlabel('n_estimators'); a1.set_ylabel('Mean F1 across smells (10-fold CV)')
    a1.set_title('(a)', loc='left', fontweight='bold')
    m, s = np.array([v[0] for v in mf_m]), np.array([v[1] for v in mf_m])
    a2.plot(mf_grid, m, '-s', color=RED)
    a2.fill_between(mf_grid, m - s, m + s, color=RED, alpha=0.18)
    a2.set_xlabel('max_features'); a2.set_ylabel('Mean F1 across smells (10-fold CV)')
    a2.set_title('(b)', loc='left', fontweight='bold')
    for a in (a1, a2):
        a.set_ylim(0.60, 0.90); a.grid(axis='x', visible=False)
    save(fig, 'fig6_hyperparameter_sensitivity.png')


# ------------------------------------------------------ fig8 dataset comparison
def fig8_dataset_comparison():
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    (a, b), (c, d) = axes
    dnames = ['IST2021', 'ImprovMLCQ', 'SmellyCode++']
    dcol = [BLUE, YELLOW, RED]
    # (a) dataset size
    sizes = [2520, 13489, 107554]
    a.bar(dnames, sizes, color=dcol)
    a.set_yscale('log'); a.set_ylabel('Number of samples (log scale)')
    a.set_title('(a)', loc='left', fontweight='bold'); a.grid(axis='x', visible=False)
    # (b) class balance ranges
    pos = {'IST2021': (30.7, 33.3), 'ImprovMLCQ': (5.1, 10.3), 'SmellyCode++': (1.5, 4.0)}
    xs = np.arange(len(dnames))
    for xi, dn in zip(xs, dnames):
        lo, hi = pos[dn]
        b.bar(xi, hi - lo, bottom=lo, color=dcol[xi], width=0.5)
        b.plot([xi, xi], [lo, hi], color='k', lw=1)
    b.axhline(10, color=GREY, ls='--', lw=1); b.axhline(5, color=GREY, ls=':', lw=1)
    b.set_xticks(xs); b.set_xticklabels(dnames)
    b.set_ylabel('Positive rate (%)'); b.set_title('(b)', loc='left', fontweight='bold')
    b.grid(axis='x', visible=False)
    # (c) feature space
    c.bar(dnames, [83, 33, 14], color=dcol)
    c.set_ylabel('Number of features'); c.set_title('(c)', loc='left', fontweight='bold')
    c.grid(axis='x', visible=False)
    # (d) performance summary (current numbers)
    groups = ['IST2021\n(specific vs\nmulti-class)', 'ImprovMLCQ\n(ind. vs joint)',
              'SmellyCode++\n(ind. vs joint)']
    left = [0.775, 0.425, 0.341]   # specific / independent
    right = [0.715, 0.461, 0.343]  # multi-class / joint
    xg = np.arange(len(groups)); w = 0.36
    d.bar(xg - w/2, left, w, color=BLUE, label='Specific / Independent')
    d.bar(xg + w/2, right, w, color=RED, label='Multi-class / Joint')
    d.set_xticks(xg); d.set_xticklabels(groups, fontsize=9)
    d.set_ylabel('Average F1 score'); d.set_ylim(0, 0.9)
    d.set_title('(d)', loc='left', fontweight='bold'); d.grid(axis='x', visible=False)
    d.legend(frameon=False, fontsize=9, loc='upper right')
    fig.tight_layout()
    save(fig, 'fig8_dataset_comparison.png')


# ------------------------------------------------- fig5 decision flowchart
def fig5_decision_flowchart():
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(10, 6.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis('off')

    def box(x, y, w, h, text, fc, ec='#333333', fs=11):
        ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                     boxstyle='round,pad=0.08,rounding_size=0.12',
                     linewidth=1.3, edgecolor=ec, facecolor=fc))
        ax.text(x, y, text, ha='center', va='center', fontsize=fs, wrap=True)

    def arrow(x1, y1, x2, y2, label=None):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     arrowstyle='-|>', mutation_scale=16, lw=1.3, color='#333333'))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.18, label, ha='center',
                    va='bottom', fontsize=10, color='#333333',
                    bbox=dict(boxstyle='round,pad=0.15', fc='white', ec='none'))

    box(5, 9.1, 4.6, 1.0, 'Assess class balance\n(positive rate per smell)', '#FFE9B0')
    arrow(5, 8.6, 5, 8.0)
    box(5, 7.6, 2.0, 0.7, 'Positive rate?', '#FFE9B0', fs=10)
    # three branches
    box(2.0, 5.3, 3.2, 1.3, 'Use smell-specific\nclassifiers\n(specialization helps)', '#9BD7A6')
    box(5.0, 5.3, 3.0, 1.3, 'Evaluate empirically\n(tentative boundary)', '#E0E0E0')
    box(8.0, 5.3, 3.2, 1.3, 'A single multi-class\nmodel is equally\naccurate (and simpler)', '#F4A7A1')
    arrow(4.2, 7.4, 2.4, 6.0, r'$>$10%')
    arrow(5.0, 7.25, 5.0, 6.0, '5–10%')
    arrow(5.8, 7.4, 7.6, 6.0, r'$<$5%')
    # secondary note
    box(5, 2.6, 8.6, 1.4,
        'Secondary (architecture-internal): among joint multi-label models,\n'
        'prefer ClassifierChain over native RandomForest fitting;\n'
        'linear models benefit from chaining slightly more than tree-based ones.',
        '#EDE7F6', fs=9.5)
    save(fig, 'fig5_decision_flowchart.png')


def main():
    # Figures used in the manuscript (fig6 sensitivity, fig8 dataset comparison,
    # the timeline, and the fig4 mechanism diagram were cut during revision).
    print(f"Writing figures to {FIG_DIR}")
    fig1_ist2021(); fig2_smellycode(); fig2b_improvmlcq(); fig3_boundary()
    fig9_metaheuristic(); fig5_feature_heatmap(); fig5_decision_flowchart()
    print("done.")


if __name__ == '__main__':
    main()
