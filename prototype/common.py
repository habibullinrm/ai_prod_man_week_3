"""Shared config for the churn prototype: paths, features, models, the top-k policy."""
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ.get("OULAD_DIR", ROOT.parent / "data" / "oulad"))
REPORTS = ROOT / "reports"
SNAPSHOTS_PATH = ROOT / "snapshots.parquet"
SCORED_PATH = ROOT / "test_scored.parquet"
RESULTS_PATH = REPORTS / "results.json"
BOOTSTRAP_PATH = REPORTS / "bootstrap.json"

SNAPSHOTS = list(range(30, 211, 30))   # snapshot day t inside a presentation
HORIZON = 30                           # label: withdrawal in (t, t + 30]
PRICE = 8000                           # rub per month, D-005/D-014
TOP = 0.20                             # retention team treats top-20% of each snapshot
RULE_DAYS = 7                          # baseline: no activity for 7+ days
TRAIN_YEAR, TEST_YEAR = 2013, 2014
KEY = ["code_module", "code_presentation", "id_student"]

ACT = ["t", "module_presentation_length", "date_registration", "clicks_total", "active_days_total",
       "clicks_7", "active_days_7", "clicks_14", "active_days_14", "clicks_30", "active_days_30",
       "clicks_prev14", "days_since_click", "click_trend", "clicks_per_day"]
KNOW = ["n_due", "n_sub", "n_missed", "n_late", "score_mean", "score_last", "score_min", "score_drop"]
SCORE_COLS = ["score_mean", "score_last", "score_min", "score_drop"]


class TabularPrep(BaseEstimator, TransformerMixin):
    """For linear models: impute missing grades (+ indicator), log-compress heavy-tailed counts."""

    def fit(self, X, y=None):
        self.medians_ = X[[c for c in SCORE_COLS if c in X]].median()
        return self

    def transform(self, X):
        X = X.copy()
        for c in self.medians_.index:
            X[c + "_na"] = X[c].isna().astype(int)
            X[c] = X[c].fillna(self.medians_[c])
        for c in X.columns:
            if c.startswith(("clicks_", "active_days_", "n_")) and not c.endswith("_na"):
                X[c] = np.log1p(X[c].clip(lower=0))
        if "click_trend" in X:
            X["click_trend"] = np.log(X.click_trend)
        X["days_since_click"] = np.log1p(X.days_since_click)
        return X


def lgbm():
    return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              verbose=-1, random_state=42)


def logreg_spline():
    # GAM-like: each feature gets a smooth non-linear shape, effects add up in the logit
    return make_pipeline(TabularPrep(), StandardScaler(), SplineTransformer(n_knots=5, degree=3),
                         LogisticRegression(C=0.3, max_iter=3000))


# name -> (label for charts, feature list, factory). Baseline is a rule, not a fitted model.
BASELINE = "baseline_7days"
MODELS = {
    "lgbm_activity": ("LightGBM: активность", ACT, lgbm),
    "lgbm_activity_grades": ("LightGBM: активность + оценки", ACT + KNOW, lgbm),
    "logreg_activity_grades": ("LogReg (сплайны): активность + оценки", ACT + KNOW, logreg_spline),
}
LABELS = {BASELINE: f"Правило: {RULE_DAYS} дней без активности", **{k: v[0] for k, v in MODELS.items()}}
# validated categorical palette (light surface), fixed order
COLORS = dict(zip(LABELS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))


def baseline_score(df):
    """Rank by days since the last click; ties broken by remaining contract value."""
    return df.days_since_click.values + df.value_rub.values / 1e9


def ranking_score(df, name):
    """What the retention list is sorted by: expected loss for models, inactivity for the rule."""
    return baseline_score(df) if name == BASELINE else df[f"p_{name}"].values * df.value_rub.values


def topk_flags(df, score, share=TOP):
    """Flag top `share` of each snapshot population (each renewal date we pick whom to treat).
    `share` may be a scalar or a per-snapshot Series indexed by t."""
    r = df.assign(_s=score).groupby("t")._s.rank(ascending=False, method="first")
    n = df.groupby("t").t.transform("size")
    s = df.t.map(share) if isinstance(share, pd.Series) else share
    return (r <= np.ceil(s * n)).values
