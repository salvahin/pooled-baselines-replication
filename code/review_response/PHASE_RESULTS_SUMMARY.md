# Phase 1+2 results summary (auto-generated)

## Phase 1 — code-smell datasets (ΔPR-AUC = specific − pooled)

| dataset | classifier | Δ vs intercept-pooled | Δ vs interaction-pooled |
|---|---|---|---|
| IST2021 | LogReg | +0.166 ± 0.008 | -0.001 ± 0.005 |
| IST2021 | LinearSVC | +0.170 ± 0.008 | +0.001 ± 0.005 |
| IST2021 | RF | +0.036 ± 0.011 | +0.006 ± 0.010 |
| IST2021 | HistGB | +0.008 ± 0.012 | -0.004 ± 0.009 |
| ImprovMLCQ | LogReg | +0.065 ± 0.001 | -0.000 ± 0.001 |
| ImprovMLCQ | LinearSVC | +0.066 ± 0.001 | -0.001 ± 0.001 |
| ImprovMLCQ | RF | -0.001 ± 0.004 | -0.001 ± 0.004 |
| ImprovMLCQ | HistGB | +0.016 ± 0.005 | +0.005 ± 0.005 |
| Crowdsmelling | LogReg | +0.026 ± 0.002 | +0.000 ± 0.003 |
| Crowdsmelling | LinearSVC | +0.025 ± 0.004 | +0.003 ± 0.004 |
| Crowdsmelling | RF | +0.001 ± 0.003 | +0.001 ± 0.004 |
| Crowdsmelling | HistGB | +0.001 ± 0.002 | +0.001 ± 0.003 |
| SmellyCode++ | LogReg | +0.018 ± 0.006 | -0.002 ± 0.002 |
| SmellyCode++ | LinearSVC | +0.030 ± 0.004 | -0.001 ± 0.001 |
| SmellyCode++ | RF | -0.013 ± 0.005 | -0.003 ± 0.004 |
| SmellyCode++ | HistGB | -0.003 ± 0.006 | +0.003 ± 0.003 |
| ml-Codesmell | LogReg | +0.479 ± 0.007 | +0.000 ± 0.004 |
| ml-Codesmell | LinearSVC | +0.434 ± 0.014 | -0.004 ± 0.006 |
| ml-Codesmell | RF | +0.002 ± 0.001 | +0.000 ± 0.000 |
| ml-Codesmell | HistGB | +0.000 ± 0.000 | -0.000 ± 0.000 |

- **LINEAR**: mean Δ(intercept-pooled)=+0.148, mean Δ(interaction-pooled)=-0.000
- **TREE**: mean Δ(intercept-pooled)=+0.005, mean Δ(interaction-pooled)=+0.001

## Phase 2 — non-code-smell multi-class datasets (OvR vs pooled)

| dataset | classifier | Δ vs intercept-pooled | Δ vs interaction-pooled |
|---|---|---|---|
| digits | LogReg | +0.887 ± 0.003 | +0.002 ± 0.003 |
| digits | LinearSVC | +0.864 ± 0.005 | -0.013 ± 0.005 |
| digits | RF | +0.029 ± 0.003 | +0.001 ± 0.001 |
| digits | HistGB | +0.011 ± 0.002 | +0.003 ± 0.002 |
| segment | LogReg | +0.732 ± 0.005 | -0.000 ± 0.002 |
| segment | LinearSVC | +0.734 ± 0.004 | +0.001 ± 0.002 |
| segment | RF | +0.001 ± 0.000 | -0.000 ± 0.000 |
| segment | HistGB | +0.003 ± 0.001 | +0.000 ± 0.001 |
| vehicle | LogReg | +0.567 ± 0.009 | +0.003 ± 0.004 |
| vehicle | LinearSVC | +0.577 ± 0.011 | +0.003 ± 0.003 |
| vehicle | RF | +0.119 ± 0.009 | +0.008 ± 0.006 |
| vehicle | HistGB | +0.091 ± 0.011 | +0.002 ± 0.004 |
| letter | LogReg | +0.469 ± 0.012 | +0.011 ± 0.005 |
| letter | LinearSVC | +0.464 ± 0.012 | +0.010 ± 0.004 |
| letter | RF | +0.053 ± 0.004 | +0.006 ± 0.006 |
| letter | HistGB | +0.162 ± 0.010 | +0.102 ± 0.004 |

- **LINEAR**: mean Δ(intercept-pooled)=+0.662, mean Δ(interaction-pooled)=+0.002
- **TREE**: mean Δ(intercept-pooled)=+0.059, mean Δ(interaction-pooled)=+0.015

## Auto-verdict
Artifact CONFIRMED if: LINEAR Δ(intercept) large & Δ(interaction)≈0, while TREE both ≈0 — in BOTH phases.