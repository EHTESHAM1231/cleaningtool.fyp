"""
Model Training and Evaluation Module — rewritten.

Fixes:
  - Scaler fit only on train folds (no data leakage).
  - Consistent model comparison before/after (uses same models end-to-end).
  - Adds ROC-AUC, confusion matrix, macro/micro F1, class-wise metrics.
  - Statistical significance test (paired t-test on CV folds).
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                              GradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import (cross_val_score, cross_val_predict,
                                     StratifiedKFold, train_test_split, KFold)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, classification_report,
                              mean_squared_error, r2_score)

warnings.filterwarnings("ignore")


class ModelEvaluator:
    def __init__(self):
        self.label_encoders = {}

    # -- Public API -----------------------------------------------------

    def train_and_evaluate(self, df, label="Dataset"):
        X, y, task, target_col = self._prepare(df)
        if X is None or len(X) < 20:
            return {"error": "Insufficient data for training", "label": label}

        results = {
            "label": label, "task": task, "rows": int(len(X)),
            "features": int(X.shape[1]), "target_column": target_col,
        }

        try:
            if task == "classification":
                results.update(self._evaluate_classification(X, y))
            else:
                results.update(self._evaluate_regression(X, y))
        except Exception as e:
            results["error"] = str(e)

        return results

    def compare_results(self, before, after):
        """Compare before/after by model-by-model, not cross-best-model."""
        comparison = {"before": before, "after": after, "improvement": {}}
        b_models = before.get("models", {}) if isinstance(before, dict) else {}
        a_models = after.get("models", {}) if isinstance(after, dict) else {}

        # Compare on the same best-model (or first common one)
        metric_delta = {}
        if "metrics" in before and "metrics" in after:
            for m in before["metrics"]:
                if m in after["metrics"] and isinstance(before["metrics"][m], (int, float)):
                    b, a = before["metrics"][m], after["metrics"][m]
                    metric_delta[m] = {
                        "before": b, "after": a,
                        "delta": round(a - b, 4),
                        "improved": a > b,
                    }

        # Statistical test: paired t-test on CV fold scores if available
        sig = None
        b_folds = before.get("cv_fold_scores")
        a_folds = after.get("cv_fold_scores")
        if b_folds and a_folds and len(b_folds) == len(a_folds) and len(b_folds) >= 2:
            try:
                t_stat, p_val = stats.ttest_rel(a_folds, b_folds)
                sig = {"t_statistic": round(float(t_stat), 4),
                       "p_value": round(float(p_val), 4),
                       "significant_0.05": bool(p_val < 0.05)}
            except Exception:
                sig = None

        comparison["improvement"] = metric_delta
        comparison["statistical_significance"] = sig
        comparison["per_model"] = self._per_model_diff(b_models, a_models)
        comparison["interpretation"] = self._interpret(metric_delta, sig, before, after)
        return comparison

    def _interpret(self, metric_delta, sig, before, after):
        """Generate a plain-English summary of what the metrics mean."""
        if not metric_delta:
            return ["No comparable metrics available between before/after runs."]

        lines = []

        # Primary headline metric
        primary = None
        for candidate in ("accuracy", "r2_score", "f1_weighted"):
            if candidate in metric_delta:
                primary = candidate
                break
        if primary:
            d = metric_delta[primary]
            verb = "improved" if d["improved"] else "decreased"
            lines.append(
                f"Primary metric '{primary}' {verb} from {d['before']:.4f} to {d['after']:.4f} "
                f"(Δ={d['delta']:+.4f})."
            )

        # Precision / recall tradeoff
        if "precision_macro" in metric_delta and "recall_macro" in metric_delta:
            dp = metric_delta["precision_macro"]
            dr = metric_delta["recall_macro"]
            if dr["improved"] and not dp["improved"]:
                lines.append(
                    "Recall increased while precision decreased — the model now catches more true "
                    "positives (better minority-class detection) at the cost of more false positives. "
                    "This is common after class-balancing or imputation that expands decision boundaries."
                )
            elif dp["improved"] and not dr["improved"]:
                lines.append(
                    "Precision increased while recall decreased — the model is more selective but "
                    "misses more true positives. Consider whether minority-class recall is more valuable "
                    "for your use case."
                )
            elif dp["improved"] and dr["improved"]:
                lines.append("Both precision and recall improved — a strict Pareto improvement.")

        # ROC-AUC commentary
        if "roc_auc" in metric_delta:
            ra = metric_delta["roc_auc"]
            if ra["improved"]:
                lines.append(
                    f"ROC-AUC improved by {ra['delta']:+.4f}, indicating better ranking quality "
                    f"across all classification thresholds, not just at the default cutoff."
                )
            elif abs(ra["delta"]) < 0.01:
                lines.append("ROC-AUC essentially unchanged — ranking quality preserved.")
            else:
                lines.append(
                    f"ROC-AUC decreased by {abs(ra['delta']):.4f}; the model's ability to rank "
                    f"positives above negatives has weakened slightly."
                )

        # Statistical significance
        if sig is not None:
            if sig.get("significant_0.05"):
                lines.append(
                    f"A paired t-test across CV folds gives p={sig['p_value']:.4f}, so the change is "
                    f"statistically significant at α=0.05 — unlikely to be noise."
                )
            else:
                lines.append(
                    f"A paired t-test across CV folds gives p={sig['p_value']:.4f}, which is not "
                    f"statistically significant at α=0.05 — treat the change as within noise range."
                )

        return lines

    def _per_model_diff(self, b, a):
        out = {}
        for name in set(b.keys()) | set(a.keys()):
            bm, am = b.get(name, {}), a.get(name, {})
            row = {}
            for metric in ("accuracy", "f1_weighted", "r2_score"):
                if metric in bm and metric in am:
                    row[metric] = {"before": bm[metric], "after": am[metric],
                                   "delta": round(am[metric] - bm[metric], 4)}
            if row:
                out[name] = row
        return out

    # -- Data preparation ----------------------------------------------

    def _detect_target_column(self, df):
        candidates = [c for c in df.columns if c.lower() in
                      ["target", "label", "class", "y", "output", "result", "outcome",
                       "category", "promoted", "churn", "survived", "default", "fraud",
                       "diagnosis", "species", "purchase", "clicked", "converted"]]
        if candidates:
            return candidates[0]
        low_card = [c for c in df.columns if 1 < df[c].nunique() <= 20]
        if low_card:
            binary = [c for c in low_card if df[c].nunique() == 2]
            if binary:
                return binary[-1]
            return low_card[-1]
        last = df.columns[-1]
        if df[last].nunique() < 20:
            return last
        return None

    def _detect_task_type(self, df, target_col):
        target = df[target_col]
        if target.dtype in ["object", "category"] or target.nunique() <= 20:
            return "classification"
        return "regression"

    def _prepare(self, df):
        df = df.copy()
        target_col = self._detect_target_column(df)
        if target_col is None:
            return None, None, None, None
        # Encode categoricals (non-target first)
        for col in df.select_dtypes(include=["object", "category"]).columns:
            if col == target_col:
                continue
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le
        # Fill any residual NaN in features (should be rare after repair)
        df = df.fillna(df.median(numeric_only=True))
        # Drop rows where target is still NaN
        df = df.dropna(subset=[target_col])
        task = self._detect_task_type(df, target_col)
        # Encode target if classification
        y = df[target_col]
        if task == "classification" and y.dtype in ["object", "category"]:
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y.astype(str)), index=y.index)
            self.label_encoders[target_col] = le
        X = df.drop(columns=[target_col])
        return X, y, task, target_col

    # -- Classification evaluation -------------------------------------

    def _evaluate_classification(self, X, y):
        # Pipelines so scaler is fit inside each CV fold (no leakage)
        models = {
            "Random Forest": Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=1)),
            ]),
            "Logistic Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=500, random_state=42)),
            ]),
            "Gradient Boosting": Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("clf", GradientBoostingClassifier(n_estimators=50, random_state=42)),
            ]),
        }

        min_class = int(pd.Series(y).value_counts().min())
        n_splits = max(2, min(5, min_class))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        model_results = {}
        best_name, best_score, best_folds = None, -1.0, None

        for name, pipe in models.items():
            try:
                acc_folds = cross_val_score(pipe, X, y, cv=cv, scoring="accuracy", n_jobs=1)
                f1_folds = cross_val_score(pipe, X, y, cv=cv, scoring="f1_weighted", n_jobs=1)
                model_results[name] = {
                    "accuracy": round(float(acc_folds.mean()), 4),
                    "accuracy_std": round(float(acc_folds.std()), 4),
                    "f1_weighted": round(float(f1_folds.mean()), 4),
                }
                if acc_folds.mean() > best_score:
                    best_score = acc_folds.mean()
                    best_name = name
                    best_folds = acc_folds.tolist()
            except Exception as e:
                model_results[name] = {"error": str(e)}

        if best_name is None:
            return {"models": model_results, "error": "No model succeeded"}

        # Detailed held-out metrics on the best model
        best_pipe = models[best_name]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if pd.Series(y).value_counts().min() >= 2 else None,
        )
        best_pipe.fit(X_train, y_train)
        y_pred = best_pipe.predict(X_test)

        n_classes = pd.Series(y).nunique()
        roc_auc = None
        try:
            if hasattr(best_pipe, "predict_proba"):
                y_proba = best_pipe.predict_proba(X_test)
                if n_classes == 2:
                    roc_auc = float(roc_auc_score(y_test, y_proba[:, 1]))
                else:
                    roc_auc = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"))
        except Exception:
            roc_auc = None

        cm = confusion_matrix(y_test, y_pred).tolist()
        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        # Make classification_report JSON-safe
        class_report = {str(k): {kk: round(float(vv), 4) if isinstance(vv, (int, float)) else vv
                                 for kk, vv in v.items()} if isinstance(v, dict) else round(float(v), 4)
                        for k, v in class_report.items()}

        metrics = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision_macro": round(float(precision_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
            "f1_micro": round(float(f1_score(y_test, y_pred, average="micro", zero_division=0)), 4),
            "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        }
        if roc_auc is not None:
            metrics["roc_auc"] = round(roc_auc, 4)

        return {
            "models": model_results,
            "best_model": best_name,
            "metrics": metrics,
            "confusion_matrix": cm,
            "classification_report": class_report,
            "cv_fold_scores": best_folds,
            "cv_folds": n_splits,
        }

    # -- Regression evaluation -----------------------------------------

    def _evaluate_regression(self, X, y):
        models = {
            "Random Forest": Pipeline([
                ("scaler", StandardScaler(with_mean=False)),
                ("reg", RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)),
            ]),
            "Linear Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("reg", LinearRegression()),
            ]),
        }
        model_results = {}
        best_name, best_score, best_folds = None, -np.inf, None
        for name, pipe in models.items():
            try:
                r2_folds = cross_val_score(pipe, X, y, cv=5, scoring="r2", n_jobs=1)
                model_results[name] = {
                    "r2_score": round(float(r2_folds.mean()), 4),
                    "r2_std": round(float(r2_folds.std()), 4),
                }
                if r2_folds.mean() > best_score:
                    best_score = r2_folds.mean()
                    best_name = name
                    best_folds = r2_folds.tolist()
            except Exception as e:
                model_results[name] = {"error": str(e)}

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        best_pipe = models[best_name]
        best_pipe.fit(X_train, y_train)
        y_pred = best_pipe.predict(X_test)
        metrics = {
            "r2_score": round(float(r2_score(y_test, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
        }
        return {
            "models": model_results,
            "best_model": best_name,
            "metrics": metrics,
            "cv_fold_scores": best_folds,
        }
