"""
Model Training and Evaluation Module
Trains models before/after repair and compares performance.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              classification_report, mean_squared_error, r2_score)
import warnings
warnings.filterwarnings('ignore')


class ModelEvaluator:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def detect_task_type(self, df, target_col):
        if target_col not in df.columns:
            return None
        target = df[target_col]
        if target.dtype in ['object', 'category'] or target.nunique() <= 20:
            return 'classification'
        return 'regression'

    def detect_target_column(self, df):
        # Named match
        candidates = [c for c in df.columns if c.lower() in
                      ['target', 'label', 'class', 'y', 'output', 'result', 'outcome',
                       'category', 'promoted', 'churn', 'survived', 'default', 'fraud',
                       'diagnosis', 'species', 'purchase', 'clicked', 'converted']]
        if candidates:
            return candidates[0]
        # Low-cardinality columns (likely classification targets)
        low_card = [c for c in df.columns if 1 < df[c].nunique() <= 20]
        if low_card:
            # Prefer binary columns
            binary = [c for c in low_card if df[c].nunique() == 2]
            if binary:
                return binary[-1]
            return low_card[-1]
        # Last column fallback
        last_col = df.columns[-1]
        if df[last_col].nunique() < 50:
            return last_col
        return None

    def prepare_data(self, df):
        df = df.copy()
        target_col = self.detect_target_column(df)
        if target_col is None:
            return None, None, None

        # Encode categoricals
        for col in df.select_dtypes(include=['object', 'category']).columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        # Fill remaining NaN
        df = df.fillna(df.median(numeric_only=True))

        X = df.drop(columns=[target_col])
        y = df[target_col]
        task = self.detect_task_type(df, target_col)
        return X, y, task

    def train_and_evaluate(self, df, label='Dataset'):
        X, y, task = self.prepare_data(df)
        if X is None or len(X) < 20:
            return {'error': 'Insufficient data for model training', 'label': label}

        results = {'label': label, 'task': task, 'rows': len(X), 'features': X.shape[1]}

        try:
            X_scaled = self.scaler.fit_transform(X)

            if task == 'classification':
                models = {
                    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
                    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
                    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42),
                }
                cv = StratifiedKFold(n_splits=min(5, len(set(y))), shuffle=True, random_state=42)
                model_results = {}
                best_score = -1
                best_model_name = None

                for name, model in models.items():
                    try:
                        input_X = X_scaled if name == 'Logistic Regression' else X.values
                        scores = cross_val_score(model, input_X, y, cv=cv, scoring='accuracy')
                        f1_scores = cross_val_score(model, input_X, y, cv=cv, scoring='f1_weighted')
                        model_results[name] = {
                            'accuracy': round(float(scores.mean()), 4),
                            'accuracy_std': round(float(scores.std()), 4),
                            'f1_weighted': round(float(f1_scores.mean()), 4),
                        }
                        if scores.mean() > best_score:
                            best_score = scores.mean()
                            best_model_name = name
                    except Exception as e:
                        model_results[name] = {'error': str(e)}

                # Detailed metrics for best model
                best_model = models[best_model_name]
                input_X = X_scaled if best_model_name == 'Logistic Regression' else X.values
                X_train, X_test, y_train, y_test = train_test_split(input_X, y, test_size=0.2, random_state=42, stratify=y)
                best_model.fit(X_train, y_train)
                y_pred = best_model.predict(X_test)

                results['models'] = model_results
                results['best_model'] = best_model_name
                results['metrics'] = {
                    'accuracy': round(float(accuracy_score(y_test, y_pred)), 4),
                    'precision': round(float(precision_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
                    'recall': round(float(recall_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
                    'f1_score': round(float(f1_score(y_test, y_pred, average='weighted', zero_division=0)), 4),
                }

            else:  # regression
                models = {
                    'Random Forest': RandomForestRegressor(n_estimators=50, random_state=42),
                    'Linear Regression': LinearRegression(),
                }
                model_results = {}
                for name, model in models.items():
                    try:
                        input_X = X_scaled if name == 'Linear Regression' else X.values
                        scores = cross_val_score(model, input_X, y, cv=5, scoring='r2')
                        model_results[name] = {
                            'r2_score': round(float(scores.mean()), 4),
                            'r2_std': round(float(scores.std()), 4),
                        }
                    except Exception as e:
                        model_results[name] = {'error': str(e)}

                results['models'] = model_results
                results['metrics'] = {'r2_score': max(r['r2_score'] for r in model_results.values() if 'r2_score' in r)}

        except Exception as e:
            results['error'] = str(e)

        return results

    def compare_results(self, before_results, after_results):
        comparison = {'before': before_results, 'after': after_results, 'improvement': {}}
        if 'metrics' in before_results and 'metrics' in after_results:
            for metric in before_results['metrics']:
                if metric in after_results['metrics']:
                    before_val = before_results['metrics'][metric]
                    after_val = after_results['metrics'][metric]
                    diff = after_val - before_val
                    comparison['improvement'][metric] = {
                        'before': before_val,
                        'after': after_val,
                        'delta': round(diff, 4),
                        'improved': diff > 0
                    }
        return comparison
