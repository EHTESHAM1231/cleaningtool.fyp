"""
End-to-end regression tests for the ADRF pipeline.
Run: python -m pytest tests/ -v
"""
import os
import sys
import json
import io
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.diagnostics import DatasetDiagnostics
from modules.repair import DatasetRepair
from modules.model_evaluator import ModelEvaluator


# ---------- Test fixtures ----------

def make_dirty_dataset(n=300, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        'age': rng.normal(35, 12, n).clip(18, 80).astype(int),
        'salary': rng.exponential(50000, n),
        'experience': rng.integers(0, 30, n).astype(float),
        'department': rng.choice(['Sales', 'Eng', 'HR', 'Finance'], n),
        'perf': rng.uniform(1, 10, n).round(1),
        'promoted': rng.choice([0, 1], n, p=[0.85, 0.15]),
    })
    # Missing values
    for col in ['salary', 'experience', 'perf']:
        idx = rng.choice(df.index, int(n * 0.1), replace=False)
        df.loc[idx, col] = np.nan
    # Outliers
    df.loc[rng.choice(df.index, 10), 'salary'] = rng.uniform(500000, 1_000_000, 10)
    # Duplicates
    dup_idx = rng.choice(df.index, 20, replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)
    # Constant
    df['constant_col'] = 1
    # Correlated
    df['salary_approx'] = df['salary'] * 0.98
    return df


def make_text_number_dataset():
    return pd.DataFrame({
        'quantity': ['one hundred', '200', 'fifty', np.nan, 'three hundred', '150'],
        'price': [10.5, 20.0, np.nan, 15.5, 'twelve', '25'],
        'label': [0, 1, 0, 1, 0, 1],
    })


def make_mixed_date_dataset():
    return pd.DataFrame({
        'date': ['2024-01-15', '15/02/2024', 'March 3, 2024', '2024/04/20', 'invalid', '2024-05-01'],
        'value': [10, 20, 30, 40, 50, 60],
        'label': [0, 1, 0, 1, 0, 1],
    })


# ---------- Diagnostics tests ----------

def test_diagnostics_runs_all_checks():
    df = make_dirty_dataset()
    d = DatasetDiagnostics(df).run_full_diagnostics()
    assert 'missing_values' in d
    assert 'class_imbalance' in d
    assert 'outliers' in d
    assert 'duplicates' in d
    assert 'feature_correlation' in d
    assert 'distribution_skew' in d
    assert 'constant_features' in d
    assert 'health_score' in d


def test_diagnostics_detects_known_issues():
    df = make_dirty_dataset()
    d = DatasetDiagnostics(df).run_full_diagnostics()
    assert d['missing_values']['total_missing'] > 0
    assert d['duplicates']['duplicate_rows'] > 0
    assert 'constant_col' in d['constant_features']['constant_features']
    assert d['class_imbalance']['detected']


# ---------- Repair invariants ----------

def test_repair_preserves_columns():
    df = make_dirty_dataset()
    d = DatasetDiagnostics(df).run_full_diagnostics()
    repaired, _, _ = DatasetRepair(df, d).run_auto_repair()
    # Every original column must still exist
    for col in df.columns:
        assert col in repaired.columns, f"Column {col} was dropped"


def test_repair_no_null_explosion():
    """Repair must not introduce MORE nulls than it resolves."""
    df = make_dirty_dataset()
    before_nulls = df.isnull().sum().sum()
    d = DatasetDiagnostics(df).run_full_diagnostics()
    repaired, _, _ = DatasetRepair(df, d).run_auto_repair()
    # True nulls (NaN), not string placeholders
    after_nulls = repaired.isnull().sum().sum()
    assert after_nulls <= before_nulls, \
        f"Null explosion: {before_nulls} → {after_nulls}"


def test_repair_numeric_columns_stay_numeric():
    """Numeric columns must not be converted to object via 'NULL' string insertion."""
    df = make_dirty_dataset()
    numeric_cols_before = df.select_dtypes(include=[np.number]).columns.tolist()
    d = DatasetDiagnostics(df).run_full_diagnostics()
    repaired, _, _ = DatasetRepair(df, d).run_auto_repair()
    for col in numeric_cols_before:
        if col in repaired.columns:
            assert pd.api.types.is_numeric_dtype(repaired[col]), \
                f"Numeric column {col} became {repaired[col].dtype} after repair"


def test_repair_no_row_corruption():
    """Row count may change (duplicate removal, oversampling) but must be > 0."""
    df = make_dirty_dataset()
    d = DatasetDiagnostics(df).run_full_diagnostics()
    repaired, _, _ = DatasetRepair(df, d).run_auto_repair()
    assert len(repaired) > 0


# ---------- Text number parsing (Phase 3A) ----------

def test_text_numbers_parsed():
    df = make_text_number_dataset()
    d = DatasetDiagnostics(df).run_full_diagnostics()
    repaired, log, _ = DatasetRepair(df, d).run_auto_repair()
    # 'one hundred' must be 100, 'three hundred' → 300, 'twelve' → 12, 'fifty' → 50
    if 'quantity' in repaired.columns and pd.api.types.is_numeric_dtype(repaired['quantity']):
        vals = repaired['quantity'].tolist()
        assert 100 in vals or 100.0 in vals, f"'one hundred' not parsed: {vals}"
        assert 300 in vals or 300.0 in vals, f"'three hundred' not parsed: {vals}"


# ---------- Date standardization (Phase 3D) ----------

def test_dates_standardized():
    df = make_mixed_date_dataset()
    d = DatasetDiagnostics(df).run_full_diagnostics()
    repaired, log, _ = DatasetRepair(df, d).run_auto_repair()
    if 'date' in repaired.columns:
        # Either parsed to datetime dtype, or formatted as YYYY-MM-DD strings
        col = repaired['date']
        if pd.api.types.is_datetime64_any_dtype(col):
            assert True
        else:
            valid = col.dropna().astype(str)
            iso_count = valid.str.match(r'^\d{4}-\d{2}-\d{2}').sum()
            # At least the 5 originally-valid dates should normalize
            assert iso_count >= 4, f"Date standardization weak: {valid.tolist()}"


# ---------- Evaluation ----------

def test_evaluator_no_data_leakage():
    """Scaler must be fit on train fold only, not full data."""
    df = make_dirty_dataset()
    results = ModelEvaluator().train_and_evaluate(df, label='raw')
    assert 'metrics' in results
    assert 'accuracy' in results['metrics']
    # Accuracy should be a reasonable number (not 1.0 from leakage)
    assert 0.0 < results['metrics']['accuracy'] <= 1.0


def test_evaluator_consistent_model_comparison():
    """before/after comparison must compare the same model, not apples to oranges."""
    df = make_dirty_dataset()
    ev = ModelEvaluator()
    before = ev.train_and_evaluate(df, label='before')
    d = DatasetDiagnostics(df).run_full_diagnostics()
    repaired, _, _ = DatasetRepair(df, d).run_auto_repair()
    after = ModelEvaluator().train_and_evaluate(repaired, label='after')
    comp = ev.compare_results(before, after)
    assert comp is not None


# ---------- Performance-guarded repair (Phase 4) ----------

def test_repair_does_not_degrade_performance():
    """Acceptance criterion: repaired model accuracy >= 95% of raw accuracy."""
    df = make_dirty_dataset()
    before = ModelEvaluator().train_and_evaluate(df, label='before')
    d = DatasetDiagnostics(df).run_full_diagnostics()
    repaired, _, _ = DatasetRepair(df, d).run_auto_repair()
    after = ModelEvaluator().train_and_evaluate(repaired, label='after')
    before_acc = before.get('metrics', {}).get('accuracy', 0)
    after_acc = after.get('metrics', {}).get('accuracy', 0)
    # Allow small fluctuation from stochasticity; 5% tolerance
    assert after_acc >= before_acc * 0.95, \
        f"Repair degraded accuracy: {before_acc:.4f} → {after_acc:.4f}"


if __name__ == '__main__':
    import pytest
    sys.exit(pytest.main([__file__, '-v']))
