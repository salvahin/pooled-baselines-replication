# Data Directory

This directory should contain the datasets needed to reproduce the experiments.

---

## Required Datasets

### 1. IST2021 Dataset (Balanced, ~33% positive rate)

**Source:** Jamaan, H. (2021). IST2021-CodeSmellStackingEnsemble.

**Repository:** https://github.com/hjamaan/IST2021-CodeSmellStackingEnsemble

**Download Instructions:**

```bash
# Option A: Clone and copy
# NOTE: the CSVs live under Datasets/Original/ in the upstream repo (not Data/).
git clone https://github.com/hjamaan/IST2021-CodeSmellStackingEnsemble.git
mkdir -p IST2021
cp IST2021-CodeSmellStackingEnsemble/Datasets/Original/*.csv IST2021/
rm -rf IST2021-CodeSmellStackingEnsemble

# Option B: Direct download (if git unavailable)
mkdir -p IST2021
curl -L "https://raw.githubusercontent.com/hjamaan/IST2021-CodeSmellStackingEnsemble/main/Datasets/Original/GodClass.csv" -o IST2021/GodClass.csv
curl -L "https://raw.githubusercontent.com/hjamaan/IST2021-CodeSmellStackingEnsemble/main/Datasets/Original/DataClass.csv" -o IST2021/DataClass.csv
curl -L "https://raw.githubusercontent.com/hjamaan/IST2021-CodeSmellStackingEnsemble/main/Datasets/Original/LongMethod.csv" -o IST2021/LongMethod.csv
curl -L "https://raw.githubusercontent.com/hjamaan/IST2021-CodeSmellStackingEnsemble/main/Datasets/Original/FeatureEnvy.csv" -o IST2021/FeatureEnvy.csv
curl -L "https://raw.githubusercontent.com/hjamaan/IST2021-CodeSmellStackingEnsemble/main/Datasets/Original/LongParameterList.csv" -o IST2021/LongParameterList.csv
curl -L "https://raw.githubusercontent.com/hjamaan/IST2021-CodeSmellStackingEnsemble/main/Datasets/Original/SwitchStatements.csv" -o IST2021/SwitchStatements.csv
```

**Verification:**

| File | Expected Rows | Positive Rate |
|------|---------------|---------------|
| GodClass.csv | 420 | ~33% |
| DataClass.csv | 420 | ~33% |
| LongMethod.csv | 420 | ~33% |
| FeatureEnvy.csv | 420 | ~33% |
| LongParameterList.csv | 420 | ~33% |
| SwitchStatements.csv | 420 | ~33% |

---

### 2. SmellyCode++ Dataset (Imbalanced, 1.5-4% positive rate)

**Source:** Alomari, N., Alazba, A., Aljamaan, H., Alshayeb, M. (2025). SmellyCode++: Multi-Label Dataset for Code Smell Detection. *Scientific Data* (Nature). DOI [10.1038/s41597-025-05465-z](https://doi.org/10.1038/s41597-025-05465-z).

**Data DOI (Figshare):** [10.6084/m9.figshare.28519385](https://doi.org/10.6084/m9.figshare.28519385) (file `multi-smell-dataset-v1_2.csv`)

**License:** CC BY 4.0

> **Note:** an earlier DOI (`10.6084/m9.figshare.28234218`) referenced in prior drafts is no longer resolvable; use the record above.

**Download Instructions:**

```bash
# Option A: Direct download from Figshare (≈590 MB; the file includes a raw source-code column)
curl -L "https://ndownloader.figshare.com/files/52714583" -o "SmellyCode++.csv"

# Option B: Manual download
# 1. Visit: https://figshare.com/articles/dataset/SmellyCode_csv/28519385
#    (or the dataset record: https://figshare.com/articles/dataset/_/28531922)
# 2. Download "multi-smell-dataset-v1_2.csv"
# 3. Rename to: SmellyCode++.csv
# 4. Place in this directory (data/)
```

**Verification:**

| Property | Expected Value |
|----------|----------------|
| Total rows | 107,554 |
| File size | ~590 MB (includes a `Code` source-text column) |
| Halstead feature columns | 14 |
| God Class positive rate | ~4.0% (4.03%) |
| Long Method positive rate | ~1.5% (1.46%) |
| Feature Envy positive rate | ~1.9% (1.86%) |
| Data Class positive rate | ~3.1% (3.05%) |

**Required Columns:**
- `Logical Lines`, `Distinct Operators`, `Distinct Operands`, `Total Operators`, `Total Operands`
- `Vocabulary`, `Length`, `Calculated Length`, `Volume`, `Difficulty`, `Effort`
- `Time Required`, `Bugs`, `Cyclomatic Complexity`
- `God class`, `Long method`, `Feature envy`, `Data class` (labels)

---

### 3. ImprovMLCQ Dataset (Intermediate, 5-10% positive rate)

**Source:** Silva, C. et al. (2025). ImprovMLCQ: A Feature-Enriched Dataset for Advancing Code Smell Detection. SBCARS 2025.

**DOI:** [10.5281/zenodo.14834187](https://doi.org/10.5281/zenodo.14834187)

**Paper:** https://sol.sbc.org.br/index.php/sbcars/article/view/36970

**Download Instructions:**

> **Note:** the Zenodo record exposes the file as `out_clean.csv` (3.7 MB), not `improvmlcq_dataset.csv`. The record also contains a much larger `out.csv` (~428 MB) and model artifacts — use `out_clean.csv`. This file is **not** bundled in this repository and must be downloaded.

```bash
# Option A: Direct download from Zenodo
curl -L "https://zenodo.org/api/records/14834187/files/out_clean.csv/content" -o ImprovMLCQ.csv

# Option B: Manual download
# 1. Visit: https://zenodo.org/records/14834187
# 2. Download "out_clean.csv"
# 3. Rename to: ImprovMLCQ.csv
# 4. Place in this directory (data/)
```

**Verification:**

| Property | Expected Value |
|----------|----------------|
| Total rows | 13,489 |
| Features | 33 CK metrics (columns starting with `ck_`) |
| Blob positive rate | ~10.3% |
| Data Class positive rate | ~9.8% |
| Feature Envy positive rate | ~5.6% |
| Long Method positive rate | ~5.1% |

**Required Columns:**
- `ck_*` (33 CK metric columns)
- `blob_label`, `dataclass_label`, `featureenvy_label`, `longmethod_label` (labels)

---

## Verification Script

Run this Python script to verify all datasets are correctly installed:

```python
import pandas as pd
import os

def verify_datasets():
    errors = []

    # IST2021
    ist_files = ['GodClass.csv', 'DataClass.csv', 'LongMethod.csv',
                 'FeatureEnvy.csv', 'LongParameterList.csv', 'SwitchStatements.csv']
    for f in ist_files:
        path = f'IST2021/{f}'
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"✓ {path}: {len(df)} rows")
        else:
            errors.append(f"✗ Missing: {path}")

    # SmellyCode++
    if os.path.exists('SmellyCode++.csv'):
        df = pd.read_csv('SmellyCode++.csv')
        print(f"✓ SmellyCode++.csv: {len(df):,} rows")
        if len(df) < 100000:
            errors.append(f"✗ SmellyCode++ has only {len(df)} rows (expected ~107,325)")
    else:
        errors.append("✗ Missing: SmellyCode++.csv")

    # ImprovMLCQ
    if os.path.exists('ImprovMLCQ.csv'):
        df = pd.read_csv('ImprovMLCQ.csv')
        print(f"✓ ImprovMLCQ.csv: {len(df):,} rows")
        ck_cols = [c for c in df.columns if c.startswith('ck_')]
        print(f"  Found {len(ck_cols)} CK metric columns")
    else:
        errors.append("✗ Missing: ImprovMLCQ.csv")

    if errors:
        print("\nErrors found:")
        for e in errors:
            print(f"  {e}")
        return False
    else:
        print("\n✓ All datasets verified successfully!")
        return True

if __name__ == "__main__":
    verify_datasets()
```

Save as `verify_data.py` and run:
```bash
cd data
python verify_data.py
```

---

## Pre-computed Results

The following result files are included for reference:

| File | Description |
|------|-------------|
| `ist2021_results_pr_*.csv` | IST2021 experiment results (RQ1) |
| `smellycode_results_pr_*.csv` | SmellyCode++ experiment results (RQ2a) |
| `improvmlcq_results_pr_*.csv` | ImprovMLCQ experiment results (RQ2b) |
| `boundary_conditions_pr_*.csv` | SMOTE boundary conditions results (RQ3) |
| `multilabel_*.csv` | Multi-label classification results |

These can be used to verify your reproduction matches our published results.

---

## Directory Structure After Setup

```
data/
├── README.md                    # This file
├── verify_data.py               # Verification script (create from above)
├── IST2021/                     # IST2021 dataset
│   ├── GodClass.csv
│   ├── DataClass.csv
│   ├── LongMethod.csv
│   ├── FeatureEnvy.csv
│   ├── LongParameterList.csv
│   └── SwitchStatements.csv
├── SmellyCode++.csv             # SmellyCode++ dataset
├── ImprovMLCQ.csv               # ImprovMLCQ dataset
└── *_results_*.csv              # Pre-computed results (included)
```

---

## Troubleshooting

**Q: Figshare/Zenodo download link doesn't work?**
A: DOI links are persistent. Visit the DOI URL directly and download manually.

**Q: CSV encoding issues?**
A: All datasets use UTF-8 encoding. If issues occur, try: `pd.read_csv(path, encoding='utf-8')`

**Q: Different row counts than expected?**
A: Dataset versions may be updated. Our experiments used the versions available as of May 2026. Results should be similar with minor variations.
