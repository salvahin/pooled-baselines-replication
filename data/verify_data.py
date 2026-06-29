#!/usr/bin/env python3
"""
Dataset Verification Script

Verifies that all required datasets are correctly installed before running experiments.

Usage:
    cd data
    python verify_data.py
"""

import pandas as pd
import os
import sys

def verify_datasets():
    """Verify all datasets are present and have expected structure."""
    errors = []
    warnings = []

    print("="*60)
    print("DATASET VERIFICATION")
    print("="*60)

    # =========================================================================
    # IST2021
    # =========================================================================
    print("\n--- IST2021 Dataset ---")
    ist_files = [
        ('IST2021/GodClass.csv', 'is_god_class'),
        ('IST2021/DataClass.csv', 'is_data_class'),
        ('IST2021/LongMethod.csv', 'is_long_method'),
        ('IST2021/FeatureEnvy.csv', 'is_feature_envy'),
        ('IST2021/LongParameterList.csv', 'is_long_parameters_list'),
        ('IST2021/SwitchStatements.csv', 'is_switch_statements'),
    ]

    for path, target_col in ist_files:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                if target_col in df.columns:
                    pos_rate = df[target_col].mean() if df[target_col].dtype in ['int64', 'float64', 'bool'] else 'N/A'
                    print(f"  [OK] {path}: {len(df)} rows, target='{target_col}'")
                else:
                    errors.append(f"{path}: missing target column '{target_col}'")
            except Exception as e:
                errors.append(f"{path}: read error - {e}")
        else:
            errors.append(f"MISSING: {path}")

    # =========================================================================
    # SmellyCode++
    # =========================================================================
    print("\n--- SmellyCode++ Dataset ---")
    smelly_path = 'SmellyCode++.csv'

    if os.path.exists(smelly_path):
        try:
            df = pd.read_csv(smelly_path)
            print(f"  [OK] {smelly_path}: {len(df):,} rows")

            # Check expected columns
            expected_labels = ['God class', 'Long method', 'Feature envy', 'Data class']
            missing_labels = [c for c in expected_labels if c not in df.columns]
            if missing_labels:
                errors.append(f"{smelly_path}: missing label columns {missing_labels}")
            else:
                for label in expected_labels:
                    pos_rate = (df[label] == 1).mean()
                    print(f"       {label}: {pos_rate:.1%} positive")

            # Check row count
            if len(df) < 100000:
                warnings.append(f"{smelly_path}: only {len(df):,} rows (expected ~107,554)")

        except Exception as e:
            errors.append(f"{smelly_path}: read error - {e}")
    else:
        errors.append(f"MISSING: {smelly_path}")
        print(f"  [MISSING] {smelly_path}")
        print(f"            Download from: https://doi.org/10.6084/m9.figshare.28234218")

    # =========================================================================
    # ImprovMLCQ
    # =========================================================================
    print("\n--- ImprovMLCQ Dataset ---")
    imlcq_path = 'ImprovMLCQ.csv'

    if os.path.exists(imlcq_path):
        try:
            df = pd.read_csv(imlcq_path)
            print(f"  [OK] {imlcq_path}: {len(df):,} rows")

            # Check CK metrics
            ck_cols = [c for c in df.columns if c.startswith('ck_')]
            print(f"       Found {len(ck_cols)} CK metric columns")

            if len(ck_cols) < 30:
                warnings.append(f"{imlcq_path}: only {len(ck_cols)} CK columns (expected ~33)")

            # Check labels
            expected_labels = ['blob_label', 'dataclass_label', 'featureenvy_label', 'longmethod_label']
            missing_labels = [c for c in expected_labels if c not in df.columns]
            if missing_labels:
                errors.append(f"{imlcq_path}: missing label columns {missing_labels}")
            else:
                for label in expected_labels:
                    pos_rate = df[label].mean()
                    print(f"       {label}: {pos_rate:.1%} positive")

        except Exception as e:
            errors.append(f"{imlcq_path}: read error - {e}")
    else:
        errors.append(f"MISSING: {imlcq_path}")
        print(f"  [MISSING] {imlcq_path}")
        print(f"            Download from: https://doi.org/10.5281/zenodo.14834187")

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*60)

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  - {w}")

    if errors:
        print("\nERRORS:")
        for e in errors:
            print(f"  - {e}")
        print("\n" + "="*60)
        print("VERIFICATION FAILED")
        print("Please download missing datasets before running experiments.")
        print("See data/README.md for download instructions.")
        print("="*60)
        return False
    else:
        print("\nAll datasets verified successfully!")
        print("="*60)
        return True


if __name__ == "__main__":
    success = verify_datasets()
    sys.exit(0 if success else 1)
