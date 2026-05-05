"""
Dataset Diagnostics Module
Automated Dataset Diagnostics and Repair Framework
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class DatasetDiagnostics:
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.results = {}
        self.issues = []
        self.severity_scores = {}

    def run_full_diagnostics(self):
        """Run all diagnostic checks and return a comprehensive report."""
        self.results['metadata'] = self._extract_metadata()
        self.results['missing_values'] = self._check_missing_values()
        self.results['class_imbalance'] = self._check_class_imbalance()
        self.results['outliers'] = self._check_outliers()
        self.results['duplicates'] = self._check_duplicates()
        self.results['feature_correlation'] = self._check_feature_correlation()
        self.results['data_leakage'] = self._check_data_leakage()
        self.results['distribution_skew'] = self._check_distribution_skew()
        self.results['constant_features'] = self._check_constant_features()
        self.results['label_noise'] = self._estimate_label_noise()
        self.results['issues_summary'] = self._summarise_issues()
        self.results['health_score'] = self._compute_health_score()
        return self.results

    def _extract_metadata(self):
        df = self.df
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        return {
            'rows': int(df.shape[0]),
            'columns': int(df.shape[1]),
            'numeric_features': len(num_cols),
            'categorical_features': len(cat_cols),
            'memory_usage_kb': round(df.memory_usage(deep=True).sum() / 1024, 2),
            'column_names': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'numeric_columns': num_cols,
            'categorical_columns': cat_cols,
        }

    def _check_missing_values(self):
        missing = self.df.isnull().sum()
        missing_pct = (missing / len(self.df) * 100).round(2)
        cols_with_missing = missing[missing > 0]
        severity = 'none'
        if len(cols_with_missing) > 0:
            max_pct = missing_pct.max()
            if max_pct > 30:
                severity = 'high'
            elif max_pct > 10:
                severity = 'medium'
            else:
                severity = 'low'
            self.issues.append({'type': 'Missing Values', 'severity': severity,
                                 'detail': f'{len(cols_with_missing)} columns with missing data'})
        self.severity_scores['missing_values'] = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}[severity]
        return {
            'total_missing': int(missing.sum()),
            'columns_affected': int(len(cols_with_missing)),
            'per_column': {col: {'count': int(cnt), 'percentage': float(missing_pct[col])}
                           for col, cnt in cols_with_missing.items()},
            'severity': severity
        }

    def _check_class_imbalance(self):
        target_col = self._detect_target_column()
        if target_col is None:
            return {'detected': False, 'message': 'No target column detected'}
        counts = self.df[target_col].value_counts()
        total = len(self.df)
        ratios = (counts / total * 100).round(2)
        min_ratio = float(ratios.min())
        imbalance_ratio = float(counts.max() / counts.min()) if counts.min() > 0 else float('inf')
        severity = 'none'
        if imbalance_ratio > 10:
            severity = 'high'
        elif imbalance_ratio > 3:
            severity = 'medium'
        elif imbalance_ratio > 1.5:
            severity = 'low'
        if severity != 'none':
            self.issues.append({'type': 'Class Imbalance', 'severity': severity,
                                 'detail': f'Imbalance ratio: {imbalance_ratio:.2f}x'})
        self.severity_scores['class_imbalance'] = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}[severity]
        return {
            'detected': True,
            'target_column': target_col,
            'class_distribution': {str(k): int(v) for k, v in counts.items()},
            'class_percentages': {str(k): float(v) for k, v in ratios.items()},
            'imbalance_ratio': round(imbalance_ratio, 2),
            'severity': severity
        }

    def _check_outliers(self):
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            return {'detected': False, 'message': 'No numeric columns'}
        outlier_info = {}
        total_outliers = 0
        for col in num_cols:
            col_data = self.df[col].dropna()
            if len(col_data) < 10:
                continue
            Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
            IQR = Q3 - Q1
            lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
            outliers = ((col_data < lower) | (col_data > upper)).sum()
            pct = round(outliers / len(col_data) * 100, 2)
            if outliers > 0:
                outlier_info[col] = {'count': int(outliers), 'percentage': float(pct),
                                     'lower_bound': round(float(lower), 4), 'upper_bound': round(float(upper), 4)}
                total_outliers += outliers
        outlier_pct = total_outliers / (len(self.df) * len(num_cols)) * 100 if num_cols else 0
        severity = 'high' if outlier_pct > 10 else 'medium' if outlier_pct > 3 else 'low' if outlier_pct > 0 else 'none'
        if severity != 'none':
            self.issues.append({'type': 'Outliers', 'severity': severity,
                                 'detail': f'{total_outliers} outlier data points detected'})
        self.severity_scores['outliers'] = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}[severity]
        return {'total_outliers': int(total_outliers), 'columns_affected': len(outlier_info),
                'per_column': outlier_info, 'severity': severity}

    def _check_duplicates(self):
        dup_count = int(self.df.duplicated().sum())
        dup_pct = round(dup_count / len(self.df) * 100, 2)
        severity = 'high' if dup_pct > 20 else 'medium' if dup_pct > 5 else 'low' if dup_pct > 0 else 'none'
        if severity != 'none':
            self.issues.append({'type': 'Duplicate Rows', 'severity': severity,
                                 'detail': f'{dup_count} duplicate rows ({dup_pct}%)'})
        self.severity_scores['duplicates'] = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}[severity]
        return {'duplicate_rows': dup_count, 'percentage': float(dup_pct), 'severity': severity}

    def _check_feature_correlation(self):
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return {'detected': False, 'message': 'Insufficient numeric columns'}
        corr_matrix = self.df[num_cols].corr().abs()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                val = corr_matrix.iloc[i, j]
                if val > 0.85:
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': round(float(val), 4)
                    })
        severity = 'high' if len(high_corr_pairs) > 5 else 'medium' if len(high_corr_pairs) > 2 else 'low' if high_corr_pairs else 'none'
        if severity != 'none':
            self.issues.append({'type': 'High Feature Correlation', 'severity': severity,
                                 'detail': f'{len(high_corr_pairs)} highly correlated feature pairs'})
        self.severity_scores['feature_correlation'] = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}[severity]
        # Return top 20 correlations for heatmap
        corr_dict = {}
        for col in num_cols[:15]:
            corr_dict[col] = {c: round(float(corr_matrix.loc[col, c]), 4) for c in num_cols[:15]}
        return {'high_correlation_pairs': high_corr_pairs, 'severity': severity,
                'correlation_matrix': corr_dict, 'columns': num_cols[:15]}

    def _check_data_leakage(self):
        target_col = self._detect_target_column()
        if target_col is None:
            return {'detected': False, 'message': 'No target column detected'}
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        leakage_suspects = []
        for col in num_cols:
            if col == target_col:
                continue
            try:
                if self.df[target_col].dtype in ['object', 'category']:
                    le = LabelEncoder()
                    target_enc = le.fit_transform(self.df[target_col].astype(str))
                else:
                    target_enc = self.df[target_col].values
                col_data = self.df[col].fillna(self.df[col].median())
                corr, pval = stats.pointbiserialr(target_enc, col_data) if len(set(target_enc)) == 2 else stats.pearsonr(target_enc, col_data)
                if abs(corr) > 0.95 and pval < 0.001:
                    leakage_suspects.append({'column': col, 'correlation': round(float(corr), 4), 'p_value': round(float(pval), 6)})
            except Exception:
                continue
        severity = 'high' if leakage_suspects else 'none'
        if severity != 'none':
            self.issues.append({'type': 'Data Leakage', 'severity': severity,
                                 'detail': f'{len(leakage_suspects)} potential leakage features'})
        self.severity_scores['data_leakage'] = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}[severity]
        return {'suspects': leakage_suspects, 'severity': severity}

    def _check_distribution_skew(self):
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        skewed = {}
        for col in num_cols:
            col_data = self.df[col].dropna()
            if len(col_data) < 10:
                continue
            skewness = float(stats.skew(col_data))
            if abs(skewness) > 1.0:
                skewed[col] = {'skewness': round(skewness, 4),
                               'direction': 'right-skewed' if skewness > 0 else 'left-skewed'}
        severity = 'high' if len(skewed) > len(num_cols) * 0.5 else 'medium' if skewed else 'none'
        if severity != 'none':
            self.issues.append({'type': 'Distribution Skew', 'severity': severity,
                                 'detail': f'{len(skewed)} highly skewed features'})
        self.severity_scores['distribution_skew'] = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}[severity]
        return {'skewed_columns': skewed, 'total_skewed': len(skewed), 'severity': severity}

    def _check_constant_features(self):
        constant = []
        near_constant = []
        for col in self.df.columns:
            unique_ratio = self.df[col].nunique() / len(self.df)
            if self.df[col].nunique() <= 1:
                constant.append(col)
            elif unique_ratio < 0.01:
                near_constant.append(col)
        severity = 'high' if constant else 'low' if near_constant else 'none'
        if severity != 'none':
            self.issues.append({'type': 'Constant/Near-Constant Features', 'severity': severity,
                                 'detail': f'{len(constant)} constant, {len(near_constant)} near-constant'})
        self.severity_scores['constant_features'] = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}[severity]
        return {'constant_features': constant, 'near_constant_features': near_constant, 'severity': severity}

    def _estimate_label_noise(self):
        target_col = self._detect_target_column()
        if target_col is None or self.df[target_col].dtype not in ['object', 'category', 'int64']:
            return {'detected': False, 'message': 'Cannot estimate label noise'}
        num_cols = self.df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) < 2:
            return {'detected': False, 'message': 'Insufficient features for label noise estimation'}
        try:
            from sklearn.neighbors import KNeighborsClassifier
            le = LabelEncoder()
            X = self.df[num_cols].fillna(self.df[num_cols].median())
            y = le.fit_transform(self.df[target_col].astype(str))
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X, y)
            pred = knn.predict(X)
            noise_indices = (pred != y).sum()
            noise_pct = round(noise_indices / len(y) * 100, 2)
            severity = 'high' if noise_pct > 15 else 'medium' if noise_pct > 5 else 'low' if noise_pct > 0 else 'none'
            if severity != 'none':
                self.issues.append({'type': 'Label Noise', 'severity': severity,
                                     'detail': f'~{noise_pct}% estimated noisy labels'})
            self.severity_scores['label_noise'] = {'none': 0, 'low': 1, 'medium': 2, 'high': 3}[severity]
            return {'estimated_noise_pct': float(noise_pct), 'noisy_samples': int(noise_indices), 'severity': severity}
        except Exception as e:
            return {'detected': False, 'message': str(e)}

    def _detect_target_column(self):
        candidates = [c for c in self.df.columns if c.lower() in
                      ['target', 'label', 'class', 'y', 'output', 'result', 'outcome',
                       'category', 'promoted', 'churn', 'survived', 'default', 'fraud',
                       'diagnosis', 'species', 'purchase', 'clicked', 'converted']]
        if candidates:
            return candidates[0]
        low_card = [c for c in self.df.columns if 1 < self.df[c].nunique() <= 20]
        if low_card:
            binary = [c for c in low_card if self.df[c].nunique() == 2]
            if binary:
                return binary[-1]
            return low_card[-1]
        last_col = self.df.columns[-1]
        if self.df[last_col].nunique() < 20:
            return last_col
        return None

    def _summarise_issues(self):
        high = [i for i in self.issues if i['severity'] == 'high']
        medium = [i for i in self.issues if i['severity'] == 'medium']
        low = [i for i in self.issues if i['severity'] == 'low']
        return {'high': high, 'medium': medium, 'low': low,
                'total_issues': len(self.issues), 'all_issues': self.issues}

    def _compute_health_score(self):
        total_weight = sum(self.severity_scores.values())
        max_possible = len(self.severity_scores) * 3
        penalty = (total_weight / max_possible) * 100 if max_possible > 0 else 0
        score = max(0, 100 - penalty)
        if score >= 80:
            grade = 'A'
        elif score >= 65:
            grade = 'B'
        elif score >= 50:
            grade = 'C'
        elif score >= 35:
            grade = 'D'
        else:
            grade = 'F'
        return {'score': round(score, 1), 'grade': grade,
                'severity_breakdown': self.severity_scores}
