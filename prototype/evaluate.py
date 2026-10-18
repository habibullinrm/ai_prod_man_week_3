"""Step 2. Train on 2013 presentations, score 2014, compute metrics (Gate 2 D3).

Main metric: share of money at risk caught when the team treats the top-20% of each snapshot.
Also reported at the rule's own workload (same number of flagged students as '7 days')."""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from common import (BASELINE, MODELS, REPORTS, RESULTS_PATH, RULE_DAYS, SCORED_PATH, SNAPSHOTS_PATH,
                    TEST_YEAR, TOP, TRAIN_YEAR, ranking_score, topk_flags)


def budget_metrics(df, flag):
    loss = df.y * df.value_rub
    return {"loss_caught": round((loss * flag).sum() / loss.sum(), 4),
            "recall": round((df.y * flag).sum() / df.y.sum(), 4),
            "precision": round(df.y[flag].mean(), 4),
            "flagged": int(flag.sum())}


def report(name, df, rule_share):
    score = ranking_score(df, name)
    prob = df.days_since_click.values if name == BASELINE else df[f"p_{name}"].values
    flag = topk_flags(df, score)
    loss = df.y * df.value_rub
    top = budget_metrics(df, flag)
    out = {"model": name,
           "loss_caught_top20": top["loss_caught"], "recall_top20": top["recall"],
           "precision_top20": top["precision"], "lift_top20": round(top["precision"] / df.y.mean(), 2),
           "pr_auc": round(average_precision_score(df.y, prob), 4),
           "roc_auc": round(roc_auc_score(df.y, prob), 4),
           "median_lead_days": float(df.loc[flag & (df.y == 1), "days_to_leave"].median()),
           "loss_caught_by_t": (pd.Series(flag * loss).groupby(df.t.values).sum()
                                / loss.groupby(df.t.values).sum()).round(3).to_dict(),
           "at_rule_budget": budget_metrics(df, topk_flags(df, score, rule_share))}
    if name != BASELINE:
        q = pd.qcut(prob, 10, labels=False, duplicates="drop")
        cal = pd.DataFrame({"p": prob, "y": df.y.values, "q": q}).groupby("q").agg(p=("p", "mean"), y=("y", "mean"))
        out["calib_max_gap_pp"] = round(float((cal.p - cal.y).abs().max() * 100), 2)
        out["brier"] = round(brier_score_loss(df.y, prob), 5)
    return out


def main():
    ds = pd.read_parquet(SNAPSHOTS_PATH)
    tr, te = ds[ds.year == TRAIN_YEAR], ds[ds.year == TEST_YEAR].copy()
    print(f"train {TRAIN_YEAR}: rows {len(tr)}, pos {int(tr.y.sum())} | test {TEST_YEAR}: rows {len(te)}, pos {int(te.y.sum())}")

    extra = {}
    for name, (label, cols, factory) in MODELS.items():
        m = factory().fit(tr[cols], tr.y)
        t0 = time.perf_counter()
        te[f"p_{name}"] = m.predict_proba(te[cols])[:, 1]
        extra[name] = {"predict_ms_per_1000": round((time.perf_counter() - t0) / len(te) * 1e6, 3)}
        if hasattr(m, "booster_"):
            imp = pd.Series(m.booster_.feature_importance("gain"), index=cols)
            extra[name]["feature_importance_gain"] = (imp / imp.sum()).round(4).sort_values(ascending=False).to_dict()
        print(f"trained {name}")

    rule_share = (te.days_since_click >= RULE_DAYS).groupby(te.t).mean()
    print(f"rule '{RULE_DAYS} days' flags {(te.days_since_click >= RULE_DAYS).mean():.1%} of test rows "
          f"(vs fixed top-{TOP:.0%} budget)")

    res = [report(BASELINE, te, rule_share)] + [report(n, te, rule_share) | extra[n] for n in MODELS]
    cols = ["model", "loss_caught_top20", "recall_top20", "precision_top20", "lift_top20", "pr_auc", "roc_auc",
            "median_lead_days", "calib_max_gap_pp"]
    table = pd.DataFrame(res)[cols].set_index("model")
    table["loss_caught_at_rule_budget"] = [r["at_rule_budget"]["loss_caught"] for r in res]
    table["gain_vs_baseline"] = (table.loss_caught_top20 / table.loss_caught_top20[BASELINE] - 1).round(3)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table)

    REPORTS.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    table.to_csv(REPORTS / "metrics.csv")
    keep = ["enr", "code_module", "code_presentation", "t", "y", "days_to_leave", "value_rub", "days_since_click"]
    te[keep + [f"p_{n}" for n in MODELS]].to_parquet(SCORED_PATH)
    print("->", RESULTS_PATH.relative_to(RESULTS_PATH.parents[1]), "and", SCORED_PATH.name)


if __name__ == "__main__":
    main()
