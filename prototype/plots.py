"""Step 4. Charts with the main metrics -> reports/*.png"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

from common import (BASELINE, BOOTSTRAP_PATH, COLORS, LABELS, MODELS, REPORTS, RESULTS_PATH, SCORED_PATH,
                    TOP, ranking_score)

INK, MUTED, GRID = "#2b2b29", "#6b6b66", "#e4e3df"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.bbox": "tight", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "lines.linewidth": 2, "legend.frameon": False,
})
NAMES = list(LABELS)
PCT = matplotlib.ticker.PercentFormatter(1, decimals=0)
STYLE = {k: ("--" if k == BASELINE else "-") for k in NAMES}   # rule is dashed: identity not by color alone


def save(fig, name):
    fig.savefig(REPORTS / name)
    plt.close(fig)
    print("  ", name)


def hbars(ax, values, fmt="{:.1%}", err=None):
    y = np.arange(len(NAMES))[::-1]
    v = [values[k] for k in NAMES]
    known = [x for x in v if not np.isnan(x)]
    v = [0 if np.isnan(x) else x for x in v]
    xerr = None if err is None else np.array([[values[k] - err[k][0], err[k][2] - values[k]] for k in NAMES]).T
    ax.barh(y, v, height=0.6, color=[COLORS[k] for k in NAMES], edgecolor="#fcfcfb", linewidth=2,
            xerr=xerr, error_kw={"ecolor": INK, "elinewidth": 1, "capsize": 3})
    right = [err[k][2] for k in NAMES] if err else v
    for yi, vi, ri, k in zip(y, v, right, NAMES):
        label = "н/д" if np.isnan(values[k]) else fmt.format(vi)
        ax.text(ri + max(right) * 0.02, yi, label, va="center", fontsize=9, color=INK if known else MUTED)
    if "%" in fmt:
        ax.xaxis.set_major_formatter(PCT)
    ax.set_yticks(y, [LABELS[k] for k in NAMES])
    ax.set_xlim(0, max(right) * 1.25)
    ax.grid(axis="y", visible=False)


def summary(res, bs):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"wspace": 0.9})
    hbars(a, {k: res[k]["loss_caught_top20"] for k in NAMES}, err=bs["loss_caught"])
    target = res[BASELINE]["loss_caught_top20"] * (1 + bs["target_gain"])
    a.axvline(target, color=MUTED, ls=":", lw=1.5)
    a.text(target, 1.0, f"цель: правило +{bs['target_gain']:.0%}", color=MUTED, fontsize=8.5, va="bottom",
           ha="center", transform=a.get_xaxis_transform())
    a.set_title(f"Доля пойманных потерь, топ-{TOP:.0%} (95% CI)", pad=18)

    models = [k for k in NAMES if k != BASELINE]
    y = np.arange(len(models))[::-1]
    g = bs["gain_vs_baseline"]
    for yi, k in zip(y, models):
        lo, mid, hi = g[k]
        b.plot([lo, hi], [yi, yi], color=COLORS[k], lw=3, solid_capstyle="round")
        b.plot(mid, yi, "o", ms=9, color=COLORS[k], mec="#fcfcfb", mew=2)
        b.text(hi + 0.02, yi, f"{mid:+.0%}  [{lo:+.0%}; {hi:+.0%}]  P(≥цели)={bs['p_gain_ge_target'][k]:.0%}",
               va="center", fontsize=8.5)
    b.axvline(0, color=INK, lw=1)
    b.axvline(bs["target_gain"], color=MUTED, ls=":", lw=1.5)
    b.set_yticks(y, [LABELS[k] for k in models])
    b.set_ylim(-0.6, len(models) - 0.4)
    b.set_xlim(min(-0.1, min(v[0] for v in g.values()) - 0.05), max(v[2] for v in g.values()) + 0.6)
    b.grid(axis="y", visible=False)
    b.xaxis.set_major_formatter(PCT)
    b.set_title("Прирост к правилу «7 дней» (бутстрэп по студентам)", pad=18)
    fig.suptitle(f"Тест {bs['n_students']} студентов, {bs['n_churn_events']} событий оттока", x=0.01, ha="left",
                 y=1.04, fontsize=10, color=MUTED)
    save(fig, "01_summary.png")


def metric_grid(res):
    specs = [("recall_top20", f"Recall, топ-{TOP:.0%}", "{:.1%}"),
             ("precision_top20", f"Precision, топ-{TOP:.0%}", "{:.1%}"),
             ("lift_top20", "Lift к базовой частоте", "{:.2f}×"),
             ("pr_auc", "PR-AUC", "{:.3f}"),
             ("roc_auc", "ROC-AUC", "{:.3f}"),
             ("rule_budget", "Пойманные потери при нагрузке правила", "{:.1%}"),
             ("median_lead_days", "Медиана дней до ухода (пойманные)", "{:.0f} дн."),
             ("calib_max_gap_pp", "Макс. ошибка калибровки, п.п.", "{:.2f}")]
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), gridspec_kw={"hspace": 0.55, "wspace": 0.9})
    for ax, (key, title, fmt) in zip(axes.flat, specs):
        if key == "rule_budget":
            vals = {k: res[k]["at_rule_budget"]["loss_caught"] for k in NAMES}
        else:
            vals = {k: res[k].get(key, np.nan) for k in NAMES}
        hbars(ax, vals, fmt=fmt)
        if key == "calib_max_gap_pp":
            ax.text(0.98, 0.98, "у правила нет вероятностей", transform=ax.transAxes, ha="right",
                    va="top", fontsize=8, color=MUTED)
        ax.set_title(title)
    save(fig, "02_metrics_grid.png")


def gain_curve(te):
    fig, ax = plt.subplots(figsize=(7, 5))
    loss = (te.y * te.value_rub).values
    for k in NAMES:
        s = ranking_score(te, k)
        # rank inside each snapshot, then pool: x = share flagged per snapshot
        pct = te.assign(_s=s).groupby("t")._s.rank(ascending=False, pct=True).values
        order = np.argsort(pct)
        ax.plot(pct[order], np.cumsum(loss[order]) / loss.sum(), color=COLORS[k], ls=STYLE[k], label=LABELS[k])
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":", label="Случайный выбор")
    ax.axvline(TOP, color=MUTED, lw=1)
    ax.text(TOP + 0.01, 0.03, f"бюджет {TOP:.0%}", color=MUTED, fontsize=8.5)
    ax.set_xlabel("Доля студентов в списке на удержание (в каждом срезе)")
    ax.set_ylabel("Доля пойманных потерь, ₽")
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    ax.set_title("Кривая пойманных потерь (gain curve)")
    ax.legend(loc="lower right", fontsize=8.5)
    save(fig, "03_gain_curve.png")


def pr_roc(te):
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5))
    for k in NAMES:
        p = te.days_since_click.values if k == BASELINE else te[f"p_{k}"].values
        pr, rc, _ = precision_recall_curve(te.y, p)
        a.plot(rc, pr, color=COLORS[k], ls=STYLE[k], label=LABELS[k])
        fpr, tpr, _ = roc_curve(te.y, p)
        b.plot(fpr, tpr, color=COLORS[k], ls=STYLE[k], label=LABELS[k])
    a.axhline(te.y.mean(), color=MUTED, lw=1, ls=":")
    a.text(0.5, te.y.mean() + 0.003, f"базовая частота {te.y.mean():.1%}", ha="center", va="bottom", color=MUTED,
           fontsize=8.5)
    a.set(xlabel="Recall", ylabel="Precision", ylim=(0, 0.3), title="Precision–Recall (ось Y обрезана до 30%)")
    b.plot([0, 1], [0, 1], color=MUTED, lw=1, ls=":")
    b.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC")
    b.legend(loc="lower right", fontsize=8.5)
    save(fig, "04_pr_roc.png")


def by_snapshot(res):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for k in NAMES:
        s = pd.Series(res[k]["loss_caught_by_t"])
        x = s.index.astype(int)
        ax.plot(x, s.values, color=COLORS[k], ls=STYLE[k], marker="o", ms=6, mec="#fcfcfb", mew=1.5, label=LABELS[k])
    ax.set_xticks(x)
    ax.set_xlabel("День курса на момент среза (t)")
    ax.yaxis.set_major_formatter(PCT)
    ax.set_title(f"Доля пойманных потерь в топ-{TOP:.0%} по срезам")
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    save(fig, "05_by_snapshot.png")


def calibration(te):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    hi = 0
    for k in MODELS:
        p = te[f"p_{k}"]
        cal = pd.DataFrame({"p": p, "y": te.y}).groupby(pd.qcut(p, 10, labels=False, duplicates="drop")).mean()
        ax.plot(cal.p, cal.y, color=COLORS[k], marker="o", ms=6, mec="#fcfcfb", mew=1.5, label=LABELS[k])
        hi = max(hi, cal.p.max(), cal.y.max())
    ax.plot([0, hi], [0, hi], color=MUTED, lw=1, ls=":", label="Идеальная калибровка")
    ax.set(xlabel="Предсказанная вероятность (среднее по децилю)", ylabel="Фактическая доля ушедших",
           title="Калибровка по децилям")
    ax.xaxis.set_major_formatter(PCT)
    ax.yaxis.set_major_formatter(PCT)
    ax.legend(fontsize=8.5, loc="upper left")
    save(fig, "06_calibration.png")


def importance(res):
    k = "lgbm_activity_grades"
    imp = pd.Series(res[k]["feature_importance_gain"]).head(12)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.barh(imp.index, imp.values, height=0.6, color=COLORS[k], edgecolor="#fcfcfb", linewidth=2)
    for i, v in enumerate(imp.values):
        ax.text(v + imp.max() * 0.01, i, f"{v:.1%}", va="center", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(PCT)
    ax.set_title(f"Важность признаков (gain), {LABELS[k]}")
    save(fig, "07_feature_importance.png")


def main():
    res = {r["model"]: r for r in json.loads(RESULTS_PATH.read_text())}
    bs = json.loads(BOOTSTRAP_PATH.read_text())
    te = pd.read_parquet(SCORED_PATH)
    print("charts ->", REPORTS)
    summary(res, bs)
    metric_grid(res)
    gain_curve(te)
    pr_roc(te)
    by_snapshot(res)
    calibration(te)
    importance(res)


if __name__ == "__main__":
    main()
