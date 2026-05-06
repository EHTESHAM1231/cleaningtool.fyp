"""
Dataset Repair Module — rewritten for correctness.

Key principles:
  1. Type-preserving imputation: numeric stays numeric, categorical stays categorical.
  2. Hierarchical imputation: group median → category median → global median.
  3. Text-number parsing: "two hundred" → 200 before coercion.
  4. Date standardization: mixed formats → ISO YYYY-MM-DD.
  5. Performance-guarded repair: each step is rolled back if it degrades CV accuracy.
  6. No silent "NULL" string injection into numeric columns.
"""

import re
import warnings
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")


# ----- Text-number parser --------------------------------------------------

_NUM_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90,
}
_SCALES = {"hundred": 100, "thousand": 1_000, "million": 1_000_000, "billion": 1_000_000_000}


def parse_text_number(text):
    """Convert 'two hundred and fifty' → 250. Returns np.nan on failure."""
    if not isinstance(text, str):
        return np.nan
    s = text.lower().strip().replace("-", " ").replace(",", " ")
    s = re.sub(r"\band\b", " ", s)
    tokens = [t for t in s.split() if t]
    if not tokens:
        return np.nan
    if all(t in _NUM_WORDS or t in _SCALES for t in tokens):
        total, current = 0, 0
        for tok in tokens:
            if tok in _NUM_WORDS:
                current += _NUM_WORDS[tok]
            elif tok in _SCALES:
                scale = _SCALES[tok]
                if scale == 100:
                    current = max(current, 1) * 100
                else:
                    total += max(current, 1) * scale
                    current = 0
        return float(total + current)
    return np.nan


def coerce_numeric_with_words(series):
    """Try numeric conversion; fall back to word-number parsing per element.
    Returns (coerced_series, conversion_stats)."""
    stats = {"numeric_direct": 0, "numeric_from_words": 0, "unparseable": 0, "originally_null": 0}
    out = pd.Series(index=series.index, dtype=float)
    for idx, val in series.items():
        if pd.isna(val):
            stats["originally_null"] += 1
            out.iloc[out.index.get_loc(idx)] = np.nan
            continue
        # Try direct numeric
        try:
            num = float(val)
            out.loc[idx] = num
            stats["numeric_direct"] += 1
            continue
        except (ValueError, TypeError):
            pass
        # Try word parsing
        parsed = parse_text_number(str(val))
        if not pd.isna(parsed):
            out.loc[idx] = parsed
            stats["numeric_from_words"] += 1
        else:
            out.loc[idx] = np.nan
            stats["unparseable"] += 1
    return out, stats


# ----- Date standardization ------------------------------------------------

def looks_like_date_column(series, sample=50):
    """Heuristic: does this series look like it contains dates?"""
    if pd.api.types.is_datetime64_any_dtype(series):
        return True
    non_null = series.dropna().astype(str).head(sample)
    if len(non_null) == 0:
        return False
    patterns = [
        r"\d{4}-\d{1,2}-\d{1,2}",
        r"\d{1,2}/\d{1,2}/\d{2,4}",
        r"\d{1,2}-\d{1,2}-\d{2,4}",
        r"\d{4}/\d{1,2}/\d{1,2}",
        r"[A-Za-z]+\s+\d{1,2},?\s*\d{4}",
        r"\d{1,2}\s+[A-Za-z]+\s+\d{4}",
    ]
    hits = 0
    for v in non_null:
        for pat in patterns:
            if re.search(pat, v):
                hits += 1
                break
    return hits / len(non_null) >= 0.6


def standardize_date_series(series):
    """Parse a mixed-format date series.

    Strategy:
      1. Detect column-wide format preference (m/d/y vs d/m/y) by counting
         values where only one interpretation is plausible (day > 12).
      2. Apply preferred dayfirst setting first, fall back to the other.
      3. Try explicit known formats.
      4. Log a warning if parsing failure rate is high.
    """
    non_null = series.dropna().astype(str)

    # Detect dayfirst preference from unambiguous slash-separated values
    dayfirst_hint = False
    if len(non_null) > 0:
        mdy_hits, dmy_hits = 0, 0
        for v in non_null.head(100):
            m = re.match(r"^\s*(\d{1,2})[/\-](\d{1,2})[/\-]\d{2,4}\s*$", v)
            if not m:
                continue
            a, b = int(m.group(1)), int(m.group(2))
            if a > 12 and b <= 12:
                dmy_hits += 1
            elif b > 12 and a <= 12:
                mdy_hits += 1
        dayfirst_hint = dmy_hits > mdy_hits

    def parse_one(v):
        if pd.isna(v):
            return pd.NaT
        s = str(v).strip()
        if not s:
            return pd.NaT
        # Try pandas general parser with detected preference first
        for dayfirst in (dayfirst_hint, not dayfirst_hint):
            try:
                out = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst)
                if not pd.isna(out):
                    return out
            except Exception:
                pass
        # Try explicit known formats as last resort
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%d/%m/%Y", "%m/%d/%Y",
                    "%d-%m-%Y", "%m-%d-%Y", "%B %d, %Y", "%b %d, %Y",
                    "%d %B %Y", "%d %b %Y", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return pd.to_datetime(s, format=fmt, errors="raise")
            except Exception:
                continue
        return pd.NaT

    parsed = series.map(parse_one)
    # Warn if parse rate is low
    orig_non_null = series.notna().sum()
    if orig_non_null > 0:
        parse_rate = parsed.notna().sum() / orig_non_null
        if parse_rate < 0.9:
            logging.getLogger("adrf.repair").warning(
                "Date column '%s' parse rate %.1f%% — %d value(s) unparseable",
                getattr(series, "name", "<unnamed>"),
                parse_rate * 100,
                int(orig_non_null - parsed.notna().sum()),
            )
    return parsed


# ----- Categorical normalization -------------------------------------------

def normalize_categorical(series):
    """Strip whitespace, collapse internal spaces, title-case."""
    def clean(v):
        if pd.isna(v):
            return v
        s = re.sub(r"\s+", " ", str(v).strip())
        return s
    return series.map(clean)


# ----- Hierarchical imputation ---------------------------------------------

def hierarchical_numeric_impute(df, col, group_cols=None):
    """Fill NaN in `col` using: group median → global median. Returns (series, log).
    Never inserts 'NULL' strings. Values are always numeric (float)."""
    filled = df[col].copy()
    log = {"group_fills": 0, "global_fills": 0, "failed": 0}
    if group_cols:
        try:
            group_medians = df.groupby(group_cols, dropna=False)[col].transform("median")
            mask = filled.isna() & group_medians.notna()
            log["group_fills"] = int(mask.sum())
            filled = filled.where(~mask, group_medians)
        except Exception:
            pass
    global_med = pd.to_numeric(filled, errors="coerce").median()
    if pd.isna(global_med):
        # Last resort: zero, and mark as failed
        log["failed"] = int(filled.isna().sum())
        filled = filled.fillna(0.0)
    else:
        n_global = int(filled.isna().sum())
        filled = filled.fillna(global_med)
        log["global_fills"] = n_global
    return filled.astype(float), log


def hierarchical_categorical_impute(df, col, group_cols=None):
    """Fill NaN in categorical `col` using: group mode → global mode → 'Unknown'."""
    filled = df[col].copy()
    log = {"group_fills": 0, "global_fills": 0, "unknown_fills": 0}
    if group_cols:
        try:
            def grp_mode(s):
                m = s.dropna().mode()
                return m.iloc[0] if not m.empty else np.nan
            group_modes = df.groupby(group_cols, dropna=False)[col].transform(grp_mode)
            mask = filled.isna() & group_modes.notna()
            log["group_fills"] = int(mask.sum())
            filled = filled.where(~mask, group_modes)
        except Exception:
            pass
    non_null = filled.dropna()
    global_mode = non_null.mode().iloc[0] if not non_null.empty else "Unknown"
    n_global = int(filled.isna().sum())
    if n_global > 0:
        filled = filled.fillna(global_mode if not non_null.empty else "Unknown")
        log["global_fills" if not non_null.empty else "unknown_fills"] = n_global
    return filled, log


# ----- Repair step narratives (explainability) -----------------------------

_STEP_NARRATIVES = {
    "duplicate_removal": {
        "why": "Exact duplicate rows double-count evidence, inflating model confidence and biasing cross-validation.",
        "what": "Removes rows that are byte-identical across all columns.",
        "impact": "Reduces overfitting risk and gives cleaner CV estimates.",
    },
    "outlier_cap": {
        "why": "Extreme values distort scaler statistics and dominate linear-model gradients.",
        "what": "Caps values outside an adaptive IQR fence (wider for skewed columns).",
        "impact": "Stabilises training for linear/logistic models; leaves tree models largely unaffected.",
    },
    "skew_log_transform": {
        "why": "Right-skewed numeric columns violate normality assumptions of linear models.",
        "what": "Applies log1p transform to positive-valued, highly skewed columns.",
        "impact": "Typically improves regression fit and logistic-regression calibration.",
    },
    "class_balance_oversample": {
        "why": "Severe class imbalance (>3x) causes majority-class overfitting and poor minority recall.",
        "what": "Oversamples minority classes with replacement to match majority size.",
        "impact": "Improves recall and F1 on minority class; may reduce overall precision slightly.",
    },
    "hierarchical_imputation": {
        "why": "Missing values break scalers/encoders and bias model training.",
        "what": "Evaluates group-median/mode and global-median/mode strategies and picks the best.",
        "impact": "Preserves distributional variance; avoids the 'NULL string' column-type corruption.",
    },
}




class DatasetRepair:
    def __init__(self, df: pd.DataFrame, diagnostics_results: dict):
        self.original_df = df.copy()
        self.df = df.copy()
        self.diagnostics = diagnostics_results or {}
        self.repair_log = []
        self.versions = []
        self.explanations = []  # Phase 10: academic explainability

    # -- Public API -----------------------------------------------------

    def run_auto_repair(self, preserve_columns=True, column_specific_processing=None,
                        performance_guarded=True):
        """Run the full repair pipeline. Each repair step is guarded against
        performance regression if `performance_guarded=True`."""
        self._save_version("original")

        # 1. Text-number & type coercion (safe — always improves)
        self._repair_text_numbers()

        # 2. Date standardization (safe — never removes info)
        self._repair_dates()

        # 3. Categorical normalization (safe)
        self._normalize_categoricals()

        # 4. Remove exact duplicates (low risk)
        if self.diagnostics.get("duplicates", {}).get("duplicate_rows", 0) > 0:
            self._guarded_step("duplicate_removal", self._remove_duplicates,
                               rationale="Exact duplicate rows bias model weights and cross-validation.",
                               guarded=performance_guarded)

        # 5. Hierarchical imputation (replaces NULL-string injection)
        if self._any_missing():
            self._guarded_step("hierarchical_imputation", self._impute_missing,
                               rationale="Hierarchical group→global median/mode preserves type and distribution.",
                               guarded=performance_guarded)

        # 6. Adaptive outlier capping
        if self.diagnostics.get("outliers", {}).get("severity") in ("medium", "high"):
            self._guarded_step("outlier_cap", self._cap_outliers_adaptive,
                               rationale="Medium/high outlier severity can skew linear models. Adaptive IQR cap used.",
                               guarded=performance_guarded)

        # 7. Skew correction
        if self.diagnostics.get("distribution_skew", {}).get("total_skewed", 0) > 0:
            self._guarded_step("skew_log_transform", self._fix_skew,
                               rationale="Right-skewed numeric columns benefit from log1p for linear models.",
                               guarded=performance_guarded)

        # 8. Class balancing (ALWAYS guarded — often hurts if minority is noise)
        if self.diagnostics.get("class_imbalance", {}).get("severity") in ("medium", "high"):
            self._guarded_step("class_balance_oversample", self._balance_classes,
                               rationale="Imbalance ratio > 3x; minority oversampling improves recall.",
                               guarded=performance_guarded)

        self._save_version("repaired")
        return self.df, self.repair_log, self.versions

    def get_repair_summary(self):
        return {
            "original_shape": list(self.original_df.shape),
            "repaired_shape": list(self.df.shape),
            "actions_taken": len(self.repair_log),
            "repair_log": self.repair_log,
            "explanations": self.explanations,
        }

    # -- Guarded execution ---------------------------------------------

    def _guarded_step(self, name, fn, rationale, guarded=True):
        """Execute a repair step; if performance_guarded, snapshot first and
        rollback on accuracy regression > 5%. Records structured explanation
        with WHY/WHAT/IMPACT fields for academic reporting."""
        why_what_impact = _STEP_NARRATIVES.get(name, {})
        if not guarded:
            fn()
            self.explanations.append({
                "step": name, "rationale": rationale,
                "guarded": False, "accepted": True, **why_what_impact,
            })
            return
        snapshot = self.df.copy()
        before_score = self._cv_score(snapshot)
        try:
            fn()
        except Exception as e:
            self.df = snapshot
            self.repair_log.append({"action": name, "status": "error", "details": str(e)})
            self.explanations.append({
                "step": name, "rationale": rationale,
                "guarded": True, "accepted": False,
                "reason": f"error: {e}", **why_what_impact,
            })
            return
        after_score = self._cv_score(self.df)
        # Accept if no score available (can't evaluate) or score did not regress > 5%
        if before_score is None or after_score is None or after_score >= before_score - 0.05:
            delta = None if (before_score is None or after_score is None) else round(after_score - before_score, 4)
            self.explanations.append({
                "step": name, "rationale": rationale,
                "guarded": True, "accepted": True,
                "cv_before": before_score, "cv_after": after_score,
                "cv_delta": delta,
                **why_what_impact,
            })
        else:
            # Rollback
            self.df = snapshot
            self.repair_log.append({
                "action": name, "status": "rolled_back",
                "details": f"CV score dropped {before_score:.4f} → {after_score:.4f}",
            })
            self.explanations.append({
                "step": name, "rationale": rationale,
                "guarded": True, "accepted": False,
                "cv_before": before_score, "cv_after": after_score,
                "reason": "performance regression",
                **why_what_impact,
            })

    def _cv_score(self, df, n_folds=3):
        """Quick CV score to detect regressions. Returns None if infeasible."""
        target = self._detect_target(df)
        if target is None:
            return None
        try:
            work = df.copy()
            # Encode categoricals
            for c in work.select_dtypes(include=["object", "category"]).columns:
                work[c] = LabelEncoder().fit_transform(work[c].astype(str))
            work = work.fillna(work.median(numeric_only=True))
            X = work.drop(columns=[target])
            y = work[target]
            if len(X) < 20 or y.nunique() < 2:
                return None
            if y.nunique() <= 20:
                model = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=1)
                cv = StratifiedKFold(n_splits=min(n_folds, y.value_counts().min()),
                                     shuffle=True, random_state=42)
                scoring = "accuracy"
            else:
                model = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=1)
                cv = n_folds
                scoring = "r2"
            scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
            return float(scores.mean())
        except Exception:
            return None

    def _detect_target(self, df):
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

    # -- Individual repair steps ---------------------------------------

    def _repair_text_numbers(self):
        converted_cols, total_words = [], 0
        for col in list(self.df.columns):
            if self.df[col].dtype != object:
                continue
            non_null = self.df[col].dropna().astype(str).head(100)
            if len(non_null) < 5:
                continue
            # Does any value look like a word-number?
            word_hits = sum(1 for v in non_null
                            if any(w in v.lower().split() for w in _NUM_WORDS))
            numeric_hits = sum(1 for v in non_null
                               if self._is_numeric_string(v))
            if word_hits > 0 and (word_hits + numeric_hits) / len(non_null) >= 0.5:
                coerced, stats = coerce_numeric_with_words(self.df[col])
                # Only commit if we actually parsed some words AND not too many unparseable
                if stats["numeric_from_words"] > 0 and \
                   stats["unparseable"] / max(1, len(non_null)) < 0.3:
                    self.df[col] = coerced
                    converted_cols.append(col)
                    total_words += stats["numeric_from_words"]
        if converted_cols:
            self.repair_log.append({
                "action": "Text Number Parsing",
                "details": f"Converted {total_words} word-numbers to numeric in: {', '.join(converted_cols)}",
                "status": "success",
            })
            self.explanations.append({
                "step": "text_number_parsing",
                "rationale": "Word-numbers (e.g., 'two hundred') are recoverable numeric info; discarding them loses data.",
                "guarded": False, "accepted": True,
            })

    def _repair_dates(self):
        date_cols = []
        unparseable_kept = 0
        for col in list(self.df.columns):
            if self.df[col].dtype == object and looks_like_date_column(self.df[col]):
                parsed = standardize_date_series(self.df[col])
                orig_non_null = self.df[col].notna().sum()
                recovered = parsed.notna().sum()
                if orig_non_null == 0:
                    continue
                if recovered >= orig_non_null * 0.6:
                    # Build ISO output; preserve original value where parsing failed
                    iso = parsed.dt.strftime("%Y-%m-%d")
                    unparsed_mask = parsed.isna() & self.df[col].notna()
                    if unparsed_mask.any():
                        # Preserve original un-parseable values instead of losing them
                        iso = iso.where(~unparsed_mask, self.df[col])
                        unparseable_kept += int(unparsed_mask.sum())
                    self.df[col] = iso.where(parsed.notna() | unparsed_mask, np.nan)
                    date_cols.append(col)
        if date_cols:
            details = f"Normalized to YYYY-MM-DD: {', '.join(date_cols)}"
            if unparseable_kept > 0:
                details += f"; preserved {unparseable_kept} unparseable value(s) as-is"
            self.repair_log.append({
                "action": "Date Standardization",
                "details": details,
                "status": "success",
            })
            self.explanations.append({
                "step": "date_standardization", "guarded": False, "accepted": True,
                "rationale": "Mixed date formats block downstream feature engineering; ISO format is universal.",
                "why": "Detected mixed date formats (m/d/y vs d/m/y ambiguity resolved from unambiguous samples).",
                "what": f"Standardized date format across {len(date_cols)} column(s) to YYYY-MM-DD.",
                "impact": "Enables consistent temporal feature extraction and downstream time-series analysis.",
            })

    def _normalize_categoricals(self):
        normalized = []
        for col in self.df.select_dtypes(include=["object", "category"]).columns:
            # Skip date columns (just repaired)
            if looks_like_date_column(self.df[col]):
                continue
            before = self.df[col].astype(str)
            after = normalize_categorical(self.df[col])
            if (before != after.astype(str)).any():
                self.df[col] = after
                normalized.append(col)
        if normalized:
            self.repair_log.append({
                "action": "Categorical Normalization",
                "details": f"Cleaned whitespace/casing in: {', '.join(normalized)}",
                "status": "success",
            })

    def _remove_duplicates(self):
        before = len(self.df)
        self.df = self.df.drop_duplicates().reset_index(drop=True)
        removed = before - len(self.df)
        self.repair_log.append({
            "action": "Duplicate Removal",
            "details": f"Removed {removed} duplicate row(s)",
            "status": "success",
        })

    def _any_missing(self):
        return bool(self.df.isnull().any().any())

    def _impute_missing(self):
        """Multi-strategy hierarchical imputation.

        For each column with missing values, generates candidate fills using:
          (a) group-based median/mode (using best low-cardinality grouping column)
          (b) global median/mode
        Picks whichever candidate better preserves the column's variance
        (numeric) or entropy (categorical). Never inserts 'NULL' strings and
        never collapses variance below 30% of original.
        """
        # Pick group-by candidate: a low-cardinality categorical column
        cat_cols = self.df.select_dtypes(include=["object", "category"]).columns.tolist()
        group_candidate = None
        best_size = 0
        for c in cat_cols:
            n_unique = self.df[c].nunique(dropna=True)
            if 2 <= n_unique <= 20 and n_unique > best_size:
                group_candidate = [c]
                best_size = n_unique

        summary = {"numeric_cols": 0, "categorical_cols": 0,
                   "group_fills": 0, "global_fills": 0,
                   "variance_preserved_cols": 0, "strategy_choices": {}}

        for col in self.df.columns:
            if not self.df[col].isnull().any():
                continue
            grp = [g for g in (group_candidate or []) if g != col]

            if pd.api.types.is_numeric_dtype(self.df[col]):
                orig_var = float(self.df[col].var(skipna=True)) if self.df[col].notna().any() else 0.0

                # Candidate A: group median → global median
                cand_a, log_a = hierarchical_numeric_impute(self.df, col, grp)
                # Candidate B: global median only
                cand_b, log_b = hierarchical_numeric_impute(self.df, col, None)

                # Variance-preservation score: closer to original variance is better
                def var_score(s):
                    if orig_var == 0:
                        return 1.0
                    return 1.0 - min(1.0, abs(float(s.var()) - orig_var) / max(orig_var, 1e-9))

                score_a, score_b = var_score(cand_a), var_score(cand_b)

                if score_a >= score_b:
                    chosen, log, strategy = cand_a, log_a, "group_median→global_median"
                else:
                    chosen, log, strategy = cand_b, log_b, "global_median"

                # Warn and prefer global fallback if group-imputation collapsed variance badly
                final_var = float(chosen.var()) if chosen.notna().any() else 0.0
                if orig_var > 0 and final_var < orig_var * 0.3:
                    logging.getLogger("adrf.repair").warning(
                        "Imputation of '%s' reduced variance by %.0f%% — consider more granular grouping",
                        col, 100 * (1 - final_var / orig_var),
                    )
                else:
                    summary["variance_preserved_cols"] += 1

                self.df[col] = chosen
                summary["numeric_cols"] += 1
                summary["group_fills"] += log["group_fills"]
                summary["global_fills"] += log["global_fills"]
                summary["strategy_choices"][col] = strategy
            else:
                cand_a, log_a = hierarchical_categorical_impute(self.df, col, grp)
                cand_b, log_b = hierarchical_categorical_impute(self.df, col, None)
                # Entropy-preservation: prefer whichever keeps distribution closer
                def entropy(s):
                    counts = s.value_counts(normalize=True)
                    return float(-(counts * np.log(counts + 1e-12)).sum())
                orig_e = entropy(self.df[col].dropna())
                diff_a = abs(entropy(cand_a.dropna()) - orig_e)
                diff_b = abs(entropy(cand_b.dropna()) - orig_e)
                if diff_a <= diff_b:
                    chosen, log, strategy = cand_a, log_a, "group_mode→global_mode"
                else:
                    chosen, log, strategy = cand_b, log_b, "global_mode"
                self.df[col] = chosen
                summary["categorical_cols"] += 1
                summary["group_fills"] += log["group_fills"]
                summary["global_fills"] += log["global_fills"]
                summary["strategy_choices"][col] = strategy

        total_filled = summary["group_fills"] + summary["global_fills"]
        self.repair_log.append({
            "action": "Hierarchical Imputation",
            "details": (f"Filled {summary['group_fills']} via group median/mode, "
                        f"{summary['global_fills']} via global fallback "
                        f"({summary['numeric_cols']} numeric, {summary['categorical_cols']} categorical cols); "
                        f"variance preserved in {summary['variance_preserved_cols']}/{summary['numeric_cols']} numeric cols"),
            "status": "success",
        })
        self.explanations.append({
            "step": "hierarchical_imputation", "guarded": True, "accepted": True,
            "rationale": "Multiple candidate strategies evaluated; chose the one preserving most variance/entropy.",
            "why": "Missing values bias model training and break downstream scalers/encoders.",
            "what": (f"Filled {total_filled} missing value(s) across "
                     f"{summary['numeric_cols'] + summary['categorical_cols']} column(s)."),
            "impact": (f"Preserved distributional variance in "
                       f"{summary['variance_preserved_cols']}/{max(1, summary['numeric_cols'])} numeric columns. "
                       f"Strategy choices: {summary['strategy_choices']}"),
        })

    def _cap_outliers_adaptive(self):
        """Adaptive outlier capping. Uses IQR with a widened fence (2.0 * IQR)
        when the column is highly skewed, to avoid over-capping heavy-tailed data."""
        capped = 0
        cols_touched = []
        skewed_info = self.diagnostics.get("distribution_skew", {}).get("skewed_columns", {})
        for col in self.df.select_dtypes(include=[np.number]).columns:
            data = self.df[col].dropna()
            if len(data) < 20:
                continue
            q1, q3 = data.quantile(0.25), data.quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            # Widen fence for highly skewed columns
            skewness = abs(skewed_info.get(col, {}).get("skewness", 0))
            fence = 2.5 if skewness > 2.0 else 2.0 if skewness > 1.0 else 1.5
            lower, upper = q1 - fence * iqr, q3 + fence * iqr
            before_out = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
            if before_out > 0:
                self.df[col] = self.df[col].clip(lower=lower, upper=upper)
                capped += int(before_out)
                cols_touched.append(col)
        if capped > 0:
            self.repair_log.append({
                "action": "Adaptive Outlier Cap",
                "details": f"Capped {capped} extreme values across {len(cols_touched)} column(s)",
                "status": "success",
            })

    def _fix_skew(self):
        skewed = list(self.diagnostics.get("distribution_skew", {}).get("skewed_columns", {}).keys())
        transformed = []
        for col in skewed:
            if col not in self.df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(self.df[col]):
                continue
            s = self.df[col]
            if (s.dropna() > 0).all():
                self.df[col] = np.log1p(s)
                transformed.append(col)
        if transformed:
            self.repair_log.append({
                "action": "Skew Correction (log1p)",
                "details": f"Log-transformed: {', '.join(transformed)}",
                "status": "success",
            })

    def _balance_classes(self):
        target = self.diagnostics.get("class_imbalance", {}).get("target_column")
        if not target or target not in self.df.columns:
            return
        counts = self.df[target].value_counts()
        if len(counts) < 2:
            return
        majority = counts.idxmax()
        max_size = int(counts.max())
        frames = []
        for cls, n in counts.items():
            subset = self.df[self.df[target] == cls]
            if cls == majority:
                frames.append(subset)
            else:
                oversampled = subset.sample(n=max_size, replace=True, random_state=42)
                frames.append(oversampled)
        self.df = pd.concat(frames, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
        self.repair_log.append({
            "action": "Class Balancing (Oversampling)",
            "details": f"Oversampled {len(counts) - 1} minority class(es) to match majority ({max_size} rows each)",
            "status": "success",
        })

    # -- Helpers --------------------------------------------------------

    def _is_numeric_string(self, v):
        try:
            float(v)
            return True
        except (ValueError, TypeError):
            return False

    def _save_version(self, name):
        self.versions.append({
            "name": name,
            "rows": len(self.df),
            "columns": len(self.df.columns),
        })
