"""
Generate sample datasets for testing the ADRF framework.
Run: python generate_sample_data.py
"""
import pandas as pd
import numpy as np

np.random.seed(42)
n = 500

# Create a realistic classification dataset with intentional issues
df = pd.DataFrame({
    'age': np.random.normal(35, 12, n).clip(18, 80).astype(int),
    'salary': np.random.exponential(50000, n),  # skewed
    'experience': np.random.randint(0, 30, n),
    'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n, p=[0.3, 0.4, 0.2, 0.1]),
    'department': np.random.choice(['Sales', 'Engineering', 'HR', 'Finance', 'Marketing'], n),
    'performance_score': np.random.uniform(1, 10, n).round(1),
    'tenure_months': np.random.randint(1, 120, n),
})

# Add target with class imbalance (80/20)
df['promoted'] = np.random.choice([0, 1], n, p=[0.80, 0.20])

# Introduce missing values
for col in ['salary', 'experience', 'performance_score']:
    idx = np.random.choice(df.index, size=int(n * 0.08), replace=False)
    df.loc[idx, col] = np.nan

# Add outliers
df.loc[np.random.choice(df.index, 15), 'salary'] = np.random.uniform(500000, 1000000, 15)

# Add some duplicates
dup_idx = np.random.choice(df.index, 20, replace=False)
df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)

# Add a constant column
df['constant_col'] = 1

# Add correlated feature
df['salary_approx'] = df['salary'] * 0.98 + np.random.normal(0, 100, len(df))

df.to_csv('sample_dataset.csv', index=False)
print(f"Generated sample_dataset.csv with {len(df)} rows and {len(df.columns)} columns")
print(f"Class distribution:\n{df['promoted'].value_counts()}")
print(f"\nMissing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
