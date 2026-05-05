"""
Dataset Repair Module
Applies targeted interventions based on diagnostics results.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.impute import SimpleImputer, KNNImputer
import warnings
warnings.filterwarnings('ignore')


class DatasetRepair:
    def __init__(self, df: pd.DataFrame, diagnostics_results: dict):
        self.original_df = df.copy()
        self.df = df.copy()
        self.diagnostics = diagnostics_results
        self.repair_log = []
        self.versions = []

    def run_auto_repair(self, preserve_columns=True, advanced_cleaning=True, column_specific_processing=None):
        """Automatically select and apply repairs based on diagnostics."""
        self._save_version("original")
        
        # Column consistency processing
        if column_specific_processing and column_specific_processing.get('enforce_column_consistency', False):
            self.enforce_column_consistency()
            self.standardize_column_formats()

        # 1. Handle missing values with column-aware imputation
        if self.diagnostics.get('missing_values', {}).get('total_missing', 0) > 0:
            self.impute_missing_values_column_aware()

        # 2. Remove duplicates (preserve columns)
        if self.diagnostics.get('duplicates', {}).get('duplicate_rows', 0) > 0:
            self.remove_duplicates()

        # 3. Handle outliers with column-specific detection
        if self.diagnostics.get('outliers', {}).get('severity') in ['medium', 'high']:
            self.cap_outliers_advanced()

        # 4. Handle skewed distributions
        if self.diagnostics.get('distribution_skew', {}).get('total_skewed', 0) > 0:
            self.fix_skewed_distributions()

        # 5. Balance classes
        if self.diagnostics.get('class_imbalance', {}).get('severity') in ['medium', 'high']:
            self.balance_classes()

        # Final consistency verification
        if column_specific_processing and column_specific_processing.get('standardize_column_formats', False):
            self.verify_column_consistency()

        self._save_version("repaired")
        return self.df, self.repair_log, self.versions

    def enforce_column_consistency(self):
        """Enforce same data type per column - no column dropping"""
        consistency_issues = 0
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Check for mixed data types in object columns
                sample_values = self.df[col].dropna().head(100)
                if len(sample_values) > 0:
                    # Detect if column should be numeric
                    numeric_count = sum(1 for v in sample_values if self._is_numeric_string(str(v)))
                    text_count = len(sample_values) - numeric_count
                    
                    if numeric_count > text_count * 0.8:  # 80% threshold for numeric
                        # Convert to numeric with NULL for non-numeric
                        self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna('NULL')
                        # Replace any remaining NaN with NULL
                        self.df[col] = self.df[col].replace([np.nan, 'nan', 'NaN'], 'NULL')
                        consistency_issues += 1
                    else:
                        # Ensure all are strings and replace NaN with NULL
                        self.df[col] = self.df[col].astype(str).replace(['nan', 'NaN'], 'NULL')
                        consistency_issues += 1
        
        if consistency_issues > 0:
            self.repair_log.append({
                'action': 'Column Consistency Enforcement',
                'details': f'Enforced consistent data types in {consistency_issues} column(s) - no columns dropped',
                'status': 'success'
            })

    def standardize_column_formats(self):
        """Standardize formats within each column"""
        format_standardized = 0
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Standardize text format
                self.df[col] = self.df[col].astype(str).str.strip()
                format_standardized += 1
            elif pd.api.types.is_numeric_dtype(self.df[col]):
                # Standardize numeric format
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                format_standardized += 1
        
        if format_standardized > 0:
            self.repair_log.append({
                'action': 'Column Format Standardization',
                'details': f'Standardized formats in {format_standardized} column(s)',
                'status': 'success'
            })

    def verify_column_consistency(self):
        """Verify column consistency after processing"""
        inconsistent_cols = []
        for col in self.df.columns:
            if self.df[col].dtype == 'object':
                # Check for remaining inconsistencies
                unique_types = set(type(v).__name__ for v in self.df[col].dropna().head(50))
                if len(unique_types) > 1:
                    inconsistent_cols.append(col)
        
        if inconsistent_cols:
            self.repair_log.append({
                'action': 'Column Consistency Verification',
                'details': f'Found {len(inconsistent_cols)} column(s) with remaining inconsistencies: {", ".join(inconsistent_cols)}',
                'status': 'warning'
            })
        else:
            self.repair_log.append({
                'action': 'Column Consistency Verification',
                'details': 'All columns maintain consistent data types and formats',
                'status': 'success'
            })

    def impute_missing_values_column_aware(self, imputation_strategy=None):
        """Intelligent dataset-aware imputation - only NULL in truly empty cells"""
        imputed = 0
        imputation_details = []
        
        for col in self.df.columns:
            # Only process truly empty cells - preserve "naan", "null", etc.
            truly_empty_mask = self._get_empty_cell_mask(col)
            missing_count = truly_empty_mask.sum()
            
            if missing_count == 0:
                continue
                
            # Deep analysis of column characteristics
            col_analysis = self._analyze_column_deeply(col)
            
            if pd.api.types.is_numeric_dtype(self.df[col]):
                # Numeric columns - intelligent imputation for truly empty cells only
                imputation_method, value = self._determine_numeric_imputation(col, col_analysis)
                # Ensure we never use NaN
                if pd.isna(value) or value == "nan" or value == "NaN":
                    value = "NULL"
                self.df.loc[truly_empty_mask, col] = value
                imputed += missing_count
                imputation_details.append(f"{col}: {imputation_method}")
                
            else:
                # Text columns - intelligent imputation for truly empty cells only
                imputation_method, value = self._determine_text_imputation(col, col_analysis)
                # Ensure we never use NaN
                if pd.isna(value) or value == "nan" or value == "NaN":
                    value = "NULL"
                self.df.loc[truly_empty_mask, col] = value
                imputed += missing_count
                imputation_details.append(f"{col}: {imputation_method}")
        
        if imputed > 0:
            self.repair_log.append({
                'action': 'Precise Empty Cell Imputation',
                'details': f'Filled {imputed} truly empty cells only (preserved "naan" and other values): {", ".join(imputation_details)}',
                'status': 'success'
            })

    def _analyze_column_deeply(self, col):
        """Deep analysis of column characteristics"""
        analysis = {
            'dtype': self.df[col].dtype,
            'missing_ratio': self.df[col].isnull().sum() / len(self.df),
            'unique_count': self.df[col].nunique(),
            'most_frequent': None,
            'data_patterns': [],
            'is_id_column': False,
            'is_categorical': False,
            'is_temporal': False
        }
        
        # Most frequent value
        if not self.df[col].mode().empty:
            analysis['most_frequent'] = self.df[col].mode()[0]
            
        # Detect patterns
        non_null_values = self.df[col].dropna()
        if len(non_null_values) > 0:
            # Check if it's an ID column
            if self._is_likely_id_column(col, non_null_values):
                analysis['is_id_column'] = True
                
            # Check if categorical
            if self._is_likely_categorical(non_null_values):
                analysis['is_categorical'] = True
                
            # Check if temporal
            if self._is_likely_temporal(non_null_values):
                analysis['is_temporal'] = True
                
        return analysis

    def _determine_numeric_imputation(self, col, analysis):
        """Determine best imputation for numeric columns using existing values"""
        missing_ratio = analysis['missing_ratio']
        
        # If too many missing values (>50%), use NULL to avoid false data
        if missing_ratio > 0.5:
            return "NULL (high missing ratio)", "NULL"
            
        # If ID column, use NULL to avoid false IDs
        if analysis['is_id_column']:
            return "NULL (ID column)", "NULL"
            
        # Get existing non-empty values from the column
        existing_values = self.df[col].dropna()
        if len(existing_values) == 0:
            return "NULL (no existing values)", "NULL"
            
        # Try to use most frequent existing value first
        most_frequent = existing_values.mode()
        if not most_frequent.empty:
            freq_value = most_frequent[0]
            # Ensure we don't use NaN as the frequent value
            if pd.isna(freq_value):
                freq_value = "NULL"
            freq_ratio = (existing_values == freq_value).sum() / len(existing_values)
            if freq_ratio > 0.1:  # At least 10% frequency
                return f"existing frequent ({freq_value})", freq_value
            
        # If no clear frequent value, use median (existing value)
        median_value = existing_values.median()
        # Ensure we don't use NaN as median
        if pd.isna(median_value):
            median_value = "NULL"
        return f"existing median ({median_value})", median_value

    def _determine_text_imputation(self, col, analysis):
        """Determine best imputation for text columns using existing values"""
        missing_ratio = analysis['missing_ratio']
        
        # If too many missing values (>50%), use NULL
        if missing_ratio > 0.5:
            return "NULL (high missing ratio)", "NULL"
            
        # If ID column, use NULL
        if analysis['is_id_column']:
            return "NULL (ID column)", "NULL"
            
        # Get existing non-empty values from the column
        existing_values = self.df[col].dropna()
        if len(existing_values) == 0:
            return "NULL (no existing values)", "NULL"
            
        # Try to use most frequent existing value first
        most_frequent = existing_values.mode()
        if not most_frequent.empty:
            freq_value = most_frequent[0]
            # Ensure we don't use NaN as the frequent value
            if pd.isna(freq_value):
                freq_value = "NULL"
            freq_ratio = (existing_values == freq_value).sum() / len(existing_values)
            if freq_ratio > 0.1:  # At least 10% frequency
                return f"existing frequent ('{freq_value}')", freq_value
                
        # If no clear frequent value, try to use any existing value
        if len(existing_values) > 0:
            first_valid = existing_values.iloc[0]
            # Ensure we don't use NaN as the value
            if pd.isna(first_valid):
                first_valid = "NULL"
            return f"existing value ('{first_valid}')", first_valid
            
        # Fallback to NULL only if no existing values found
        return "NULL (no existing values)", "NULL"

    def _is_likely_id_column(self, col_name, values):
        """Detect if column is likely an ID column"""
        # Check column name patterns
        id_patterns = ['id', 'identifier', 'code', 'key', 'num', 'number', 'ref', 'reference']
        if any(pattern in col_name.lower() for pattern in id_patterns):
            return True
            
        # Check if all values are unique integers
        if pd.api.types.is_numeric_dtype(values):
            if values.nunique() == len(values) and len(values) > 100:
                return True
                
        return False

    def _is_likely_categorical(self, values):
        """Detect if column is likely categorical"""
        # If few unique values relative to total
        if values.nunique() <= 20 and values.nunique() / len(values) < 0.1:
            return True
        return False

    def _is_likely_temporal(self, values):
        """Detect if column is likely temporal"""
        # Try to parse as dates
        try:
            pd.to_datetime(values, errors='coerce')
            return True
        except:
            return False

    def _is_truly_empty(self, value):
        """Check if value is truly empty (not just placeholder)"""
        if pd.isna(value):
            return True
        if value == '' or value == ' ':
            return True
        if isinstance(value, str) and value.strip() == '':
            return True
        return False

    def _get_empty_cell_mask(self, col):
        """Get mask for truly empty cells only (preserve "naan", "null", etc.)"""
        # Common placeholder values to preserve
        placeholders = {'naan', 'null', 'none', 'n/a', 'na', '-', '--', '...', 'missing', 'unknown'}
        
        # Check for truly empty cells
        truly_empty_mask = self.df[col].isnull() | (self.df[col] == '') | (self.df[col] == ' ')
        
        # For string columns, also check if string is empty after stripping
        if self.df[col].dtype == 'object':
            truly_empty_mask = truly_empty_mask | (self.df[col].astype(str).str.strip() == '')
        
        # Preserve any non-empty values including placeholders like "naan"
        # Only fill cells that are genuinely empty
        return truly_empty_mask

    def cap_outliers_advanced(self):
        """Advanced outlier detection with IQR + Z-score"""
        capped = 0
        for col in self.df.select_dtypes(include=[np.number]).columns:
            if col in self.df.columns:
                # IQR method
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_iqr, upper_iqr = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                
                # Z-score method
                mean = self.df[col].mean()
                std = self.df[col].std()
                lower_z, upper_z = mean - 3 * std, mean + 3 * std
                
                # Use both methods (more conservative)
                lower_bound = max(lower_iqr, lower_z)
                upper_bound = min(upper_iqr, upper_z)
                
                outliers = ((self.df[col] < lower_bound) | (self.df[col] > upper_bound)).sum()
                if outliers > 0:
                    self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
                    capped += outliers
        
        if capped > 0:
            self.repair_log.append({
                'action': 'Advanced Outlier Detection (IQR + Z-score)',
                'details': f'Capped {capped} outlier values using combined IQR and Z-score methods',
                'status': 'success'
            })

    def _is_numeric_string(self, value):
        """Check if string represents a numeric value"""
        try:
            float(value)
            return True
        except (ValueError, TypeError):
            return False

    def impute_missing_values(self):
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        imputed = 0
        if num_cols:
            num_missing = self.df[num_cols].isnull().sum().sum()
            if num_missing > 0:
                imp = SimpleImputer(strategy='median')
                self.df[num_cols] = imp.fit_transform(self.df[num_cols])
                imputed += num_missing
        if cat_cols:
            cat_missing = self.df[cat_cols].isnull().sum().sum()
            if cat_missing > 0:
                imp = SimpleImputer(strategy='most_frequent')
                self.df[cat_cols] = imp.fit_transform(self.df[cat_cols])
                imputed += cat_missing
        if imputed > 0:
            self.repair_log.append({
                'action': 'Missing Value Imputation',
                'details': f'Imputed {imputed} missing values (median for numeric, mode for categorical)',
                'status': 'success'
            })

    def remove_duplicates(self):
        before = len(self.df)
        self.df.drop_duplicates(inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        removed = before - len(self.df)
        self.repair_log.append({
            'action': 'Duplicate Removal',
            'details': f'Removed {removed} duplicate row(s)',
            'status': 'success'
        })

    def cap_outliers(self):
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        capped = 0
        for col in num_cols:
            Q1 = self.df[col].quantile(0.05)
            Q3 = self.df[col].quantile(0.95)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            before = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            self.df[col] = self.df[col].clip(lower=lower, upper=upper)
            capped += before
        self.repair_log.append({
            'action': 'Outlier Capping (Winsorization)',
            'details': f'Capped {capped} outlier values using 5th-95th percentile bounds',
            'status': 'success'
        })

    def fix_skewed_distributions(self):
        skewed_cols = list(self.diagnostics.get('distribution_skew', {}).get('skewed_columns', {}).keys())
        transformed = []
        for col in skewed_cols:
            if col not in self.df.columns:
                continue
            try:
                col_data = self.df[col]
                if (col_data > 0).all():
                    self.df[col] = np.log1p(col_data)
                    transformed.append(col)
            except Exception:
                continue
        if transformed:
            self.repair_log.append({
                'action': 'Skew Correction (Log Transform)',
                'details': f'Applied log1p transform to {len(transformed)} skewed column(s)',
                'status': 'success'
            })

    def remove_correlated_features(self):
        pairs = self.diagnostics.get('feature_correlation', {}).get('high_correlation_pairs', [])
        to_drop = list({p['feature2'] for p in pairs if p['feature2'] in self.df.columns})
        if to_drop:
            self.df.drop(columns=to_drop, inplace=True)
            self.repair_log.append({
                'action': 'Correlated Feature Removal',
                'details': f'Removed {len(to_drop)} highly correlated feature(s)',
                'status': 'success'
            })

    def balance_classes(self):
        target_col = self.diagnostics.get('class_imbalance', {}).get('target_column')
        if not target_col or target_col not in self.df.columns:
            return
        counts = self.df[target_col].value_counts()
        majority_class = counts.idxmax()
        minority_class = counts.idxmin()
        maj_df = self.df[self.df[target_col] == majority_class]
        min_df = self.df[self.df[target_col] == minority_class]
        # Oversample minority
        target_size = len(maj_df)
        if len(min_df) > 0:
            oversampled = min_df.sample(n=target_size, replace=True, random_state=42)
            self.df = pd.concat([maj_df, oversampled], ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
            self.repair_log.append({
                'action': 'Class Balancing (Oversampling)',
                'details': f'Oversampled minority class "{minority_class}" to match "{majority_class}"',
                'status': 'success'
            })

    def _save_version(self, name):
        self.versions.append({
            'name': name,
            'rows': len(self.df),
            'columns': len(self.df.columns),
            'snapshot': self.df.copy()
        })

    def get_repair_summary(self):
        return {
            'original_shape': self.original_df.shape,
            'repaired_shape': self.df.shape,
            'actions_taken': len(self.repair_log),
            'repair_log': self.repair_log
        }
