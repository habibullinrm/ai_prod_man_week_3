"""Step 5. Everything the landing page shows -> landing/data.json.

Reads pipeline artifacts (results.json, bootstrap.json, snapshots, test_scored), refits the product
model (common.MODELS[PRODUCT], same 2013 data) only to get exact per-feature logit contributions.
Does not change any model, feature, label or metric. Demographics never leave this script."""
import json
from datetime import date

import numpy as np
import pandas as pd

from common import (ACT, BASELINE, BOOTSTRAP_PATH, COLORS, DATA, KEY, KNOW, LABELS, MODELS, RESULTS_PATH,
                    ROOT, RULE_DAYS, SCORED_PATH, SNAPSHOTS_PATH, TEST_YEAR, TOP, TRAIN_YEAR, ranking_score,
                    topk_flags)

PRODUCT = "logreg_activity_grades"
N_STREAMS = 3
OUT = ROOT.parent / "landing" / "data.json"
DEMOGRAPHICS = {"gender", "region", "highest_education", "imd_band", "age_band", "disability"}
SERVICE = {"t", "module_presentation_length"}   # stage of course: never shown as a reason
REASON_FEATURES = [f for f in ACT + KNOW if f not in SERVICE]
CURVE_POINTS = 101
LIST_CAP = 0.50   # max budget in the UI; rows outside top-50% of both rankings are not needed


# ---------- product model contributions ----------

def fit_product(tr, te):
    label, cols, factory = MODELS[PRODUCT]
    pipe = factory().fit(tr[cols], tr.y)
    prep, scaler, spline, lr = (pipe.steps[i][1] for i in range(4))
    zcols = list(prep.transform(tr[cols].head(1)).columns)
    names = spline.get_feature_names_out(zcols)
    owner = [n.rsplit("_sp_", 1)[0].removesuffix("_na") for n in names]   # spline column -> raw feature
    B_tr = spline.transform(scaler.transform(prep.transform(tr[cols])))
    B_te = spline.transform(scaler.transform(prep.transform(te[cols])))
    mean_tr = B_tr.mean(axis=0)
    coef = lr.coef_[0]
    per_col = (B_te - mean_tr) * coef
    contrib = pd.DataFrame(per_col, columns=names).T.groupby(owner).sum().T
    contrib.index = te.index
    base = float(lr.intercept_[0] + coef @ mean_tr)
    logit = lr.decision_function(B_te)
    assert np.allclose(contrib.sum(axis=1).values + base, logit, atol=1e-6), "contributions != logit"
    p = pipe.predict_proba(te[cols])[:, 1]
    return p, contrib, base


# ---------- reasons ----------

def fmt_num(x, digits=0):
    s = f"{x:,.{digits}f}".replace(",", " ").replace(".", ",")
    return s


def reason_text(f, r):
    v = r[f]
    if f in ("score_mean", "score_min", "score_last"):
        word = {"score_mean": "средний", "score_min": "минимальный", "score_last": "последний"}[f]
        return f"низкий {word} балл (нет оценок)" if pd.isna(v) else f"низкий {word} балл ({fmt_num(v)})"
    if f == "score_drop":
        return "нет оценок для сравнения" if pd.isna(v) else f"последний балл ниже среднего (на {fmt_num(-v, 1)})"
    if f == "n_missed":
        return f"пропущено заданий ({fmt_num(v)})"
    if f == "n_late":
        return f"сдано с опозданием ({fmt_num(v)})"
    if f in ("n_sub", "n_due"):
        return f"сдано заданий ({fmt_num(r.n_sub)} из {fmt_num(r.n_due)})"
    # activity: neutral wording — a non-linear model may raise risk for a value that is not "low"
    if f == "days_since_click":
        return f"дней без заходов: {fmt_num(v)}"
    if f.startswith("clicks_") and f[7:] in ("7", "14", "30"):
        return f"кликов за {f[7:]} дней: {fmt_num(v)}"
    if f == "active_days_total":
        return f"дней с заходами с начала курса: {fmt_num(v)}"
    if f.startswith("active_days_"):
        return f"дней с заходами за {f[12:]} дней: {fmt_num(v)}"
    if f == "click_trend":
        return f"активность к прошлым 2 неделям: ×{fmt_num(v, 2)}"
    if f == "clicks_total":
        return f"кликов с начала курса: {fmt_num(v)}"
    if f == "clicks_per_day":
        return f"кликов в день с начала курса: {fmt_num(v, 1)}"
    if f == "date_registration":
        return f"поздняя запись на курс (день {fmt_num(v)})"
    if f == "clicks_prev14":
        return f"кликов 2–4 недели назад: {fmt_num(v)}"
    raise KeyError(f)


def top_reasons(contrib_row, row, k=3):
    c = contrib_row[REASON_FEATURES]
    c = c[c > 0].sort_values(ascending=False)
    out, seen = [], set()
    for f, val in c.items():
        text = reason_text(f, row)
        if text in seen:          # n_sub / n_due render the same text
            continue
        seen.add(text)
        out.append({"f": f, "text": text, "logit": round(float(val), 3)})
        if len(out) == k:
            break
    return out


# ---------- per-stream rankings ----------

def stream_ranks(df):
    """Rank inside each (stream, t): the curator of one stream picks from his own students."""
    df = df.copy()
    g = df.groupby(["code_module", "code_presentation", "t"])
    df["n_pop"] = g.t.transform("size")
    df["rank_model"] = df.assign(_s=ranking_score(df, PRODUCT)).groupby(
        ["code_module", "code_presentation", "t"])._s.rank(ascending=False, method="first").astype(int)
    df["rank_rule"] = df.assign(_s=ranking_score(df, BASELINE)).groupby(
        ["code_module", "code_presentation", "t"])._s.rank(ascending=False, method="first").astype(int)
    return df


def in_top(rank, n, share):
    return rank <= np.ceil(share * n)


# ---------- case selection ----------

CASE_RULES = {
    "quiet": "y=1; вне топ-20% правила; в топ-5% модели; n_missed ≥ 1 или score_drop < 0; days_to_leave ≥ 14",
    "obvious": "y=1; в топ-20% и правила, и модели",
    "false_alarm": "y=0; в топ-5% модели; final_result ∈ {Pass, Distinction}",
    "miss": "y=1; вне топ-20% и правила, и модели; clicks_30 и score_mean в межквартильном "
            "диапазоне [25%; 75%] своего потока на тот же день t; n_missed = 0; из подходящих берётся ученик с наименьшим риском модели",
}


def case_masks(df):
    top20_m, top20_r = in_top(df.rank_model, df.n_pop, .2), in_top(df.rank_rule, df.n_pop, .2)
    top5_m = in_top(df.rank_model, df.n_pop, .05)
    g = df.groupby(["code_module", "code_presentation", "t"])
    def iqr(col):
        lo, hi = g[col].transform(lambda s: s.quantile(.25)), g[col].transform(lambda s: s.quantile(.75))
        return df[col].between(lo, hi)
    return {
        "quiet": (df.y == 1) & ~top20_r & top5_m & ((df.n_missed >= 1) | (df.score_drop < 0)) & (df.days_to_leave >= 14),
        "obvious": (df.y == 1) & top20_r & top20_m,
        "false_alarm": (df.y == 0) & top5_m & df.final_result.isin(["Pass", "Distinction"]),
        "miss": (df.y == 1) & ~top20_r & ~top20_m & iqr("clicks_30") & iqr("score_mean") & (df.n_missed == 0),
    }


def pick_cases(df):
    masks = case_masks(df)
    chosen, used = {}, set()
    for case, m in masks.items():
        # highest product-model risk among matches; for "miss" the lowest (a truly signal-free exit)
        cand = df[m & ~df.enr.isin(used)].sort_values(["p", "enr", "t"], ascending=[case == "miss", True, True])
        assert len(cand), f"no candidate for case {case}"
        row = cand.iloc[0]
        chosen[case] = row
        used.add(row.enr)
    return chosen, masks


def card_payload(case, row, vle_s, asm_all, sub_all, contrib):
    """Weekly clicks for the whole course (after t is drawn pale), assessments, verdicts."""
    t = int(row.t)
    mod, pres, sid = row.code_module, row.code_presentation, int(row.id_student)
    v = vle_s[(vle_s.code_module == mod) & (vle_s.code_presentation == pres) & (vle_s.id_student == sid)]
    # recheck: features shown were computed from days < t only
    before = v[v.date < t]
    dsc = (t - before.date.max()) if len(before) else t + 60
    assert dsc == row.days_since_click, (case, dsc, row.days_since_click)
    assert before.sum_click.sum() == row.clicks_total, case
    length = int(row.module_presentation_length)
    weeks = np.arange(int(np.floor(min(v.date.min(), 0) / 7)) if len(v) else 0, int(np.ceil(length / 7)))
    wk = v.assign(w=np.floor(v.date / 7).astype(int)).groupby("w").sum_click.sum().reindex(weeks, fill_value=0)
    a = asm_all[(asm_all.code_module == mod) & (asm_all.code_presentation == pres)].sort_values("date")
    s = sub_all[sub_all.id_student == sid].set_index("id_assessment")
    grades = []
    for _, q in a.iterrows():
        got = q.id_assessment in s.index
        sub = s.loc[q.id_assessment] if got else None
        grades.append({"due": int(q.date), "type": q.assessment_type,
                       "submitted": int(sub.date_submitted) if got else None,
                       "score": None if (not got or pd.isna(sub.score)) else float(sub.score)})
    leave = None if pd.isna(row.date_unregistration) else int(row.date_unregistration)
    outcome = {1: "ушёл в ближайшие 30 дней", 0: "остался"}[int(row.y)]
    if row.y == 0:
        outcome += {"Pass": " и сдал курс", "Distinction": " и сдал курс с отличием",
                    "Fail": ", но не сдал курс", "Withdrawn": ", ушёл позже"}[row.final_result]
    return {
        "case": case, "rule": CASE_RULES[case], "id": sid, "stream": f"{mod} {pres}", "t": t,
        "length": length, "y": int(row.y), "outcome": outcome, "leave_day": leave,
        "days_to_leave": None if pd.isna(row.days_to_leave) else int(row.days_to_leave),
        "weeks": [int(w) for w in wk.index], "clicks": [int(c) for c in wk.values],
        "grades": grades,
        "rule_verdict": {"in_list": bool(in_top(row.rank_rule, row.n_pop, TOP)), "days_since_click": int(row.days_since_click),
                         "rank": int(row.rank_rule), "n": int(row.n_pop), "flag_7days": bool(row.days_since_click >= RULE_DAYS)},
        "model_verdict": {"risk": round(float(row.p), 4), "rank": int(row.rank_model), "n": int(row.n_pop),
                          "in_list": bool(in_top(row.rank_model, row.n_pop, TOP)),
                          "expected_loss": round(float(row.p * row.value_rub))},
        "reasons": top_reasons(contrib.loc[row.name], row),
        "features": {k: (None if pd.isna(row[k]) else round(float(row[k]), 2))
                     for k in ["days_since_click", "clicks_7", "clicks_30", "n_due", "n_sub", "n_missed",
                               "n_late", "score_mean", "score_last"]},
    }


# ---------- curves ----------

def gain_curve(te, name):
    loss = (te.y * te.value_rub).values
    pct = te.assign(_s=ranking_score(te, name)).groupby("t")._s.rank(ascending=False, pct=True).values
    o = np.argsort(pct, kind="stable")
    x, y = pct[o], np.cumsum(loss[o]) / loss.sum()
    grid = np.linspace(0, 1, CURVE_POINTS)
    idx = np.searchsorted(x, grid, side="right") - 1
    return [0.0] + [round(float(y[i]), 4) if i >= 0 else 0.0 for i in idx[1:]]


# ---------- main ----------

def r4(x):
    return round(float(x), 4)


def main():
    res = {r["model"]: r for r in json.loads(RESULTS_PATH.read_text())}
    bs = json.loads(BOOTSTRAP_PATH.read_text())
    ds = pd.read_parquet(SNAPSHOTS_PATH)
    tr, te = ds[ds.year == TRAIN_YEAR], ds[ds.year == TEST_YEAR].copy()
    scored = pd.read_parquet(SCORED_PATH)
    assert (scored.enr.values == te.enr.values).all() and (scored.t.values == te.t.values).all()

    p, contrib, base = fit_product(tr, te)
    assert np.allclose(p, scored[f"p_{PRODUCT}"].values, atol=1e-9), "refit differs from pipeline scores"
    for k in MODELS:
        te[f"p_{k}"] = scored[f"p_{k}"].values
    te["p"] = te[f"p_{PRODUCT}"]

    # --- recheck headline numbers against the pipeline (no silent fixes) ---
    for k in LABELS:
        flag = topk_flags(te, ranking_score(te, k))
        loss = te.y * te.value_rub
        assert round((loss * flag).sum() / loss.sum(), 4) == res[k]["loss_caught_top20"], k

    te = stream_ranks(te)
    streams = (te[te.y == 1].groupby(["code_module", "code_presentation"]).size()
               .sort_values(ascending=False).reset_index(name="churn"))
    streams = streams.drop_duplicates("code_module").head(N_STREAMS)
    keys = list(zip(streams.code_module, streams.code_presentation))
    pool = te[[tuple(x) in set(keys) for x in te[["code_module", "code_presentation"]].values]]

    # --- cases ---
    chosen, masks = pick_cases(pool)
    for case, row in chosen.items():
        assert masks[case].loc[row.name], case
    sids = {int(r.id_student) for r in chosen.values()}
    cat = {"code_module": "category", "code_presentation": "category"}
    vle = pd.read_csv(DATA / "studentVle.csv", usecols=KEY + ["date", "sum_click"], dtype=cat)
    vle_s = vle[vle.id_student.isin(sids)].groupby(KEY + ["date"], observed=True, as_index=False).sum_click.sum()
    vle_s[["code_module", "code_presentation"]] = vle_s[["code_module", "code_presentation"]].astype(str)
    asm_all = pd.read_csv(DATA / "assessments.csv")
    asm_all = asm_all[asm_all.assessment_type != "Exam"].dropna(subset=["date"])
    sub_all = pd.read_csv(DATA / "studentAssessment.csv")
    sub_all = sub_all[sub_all.id_student.isin(sids)]
    cards = [card_payload(c, r, vle_s, asm_all, sub_all, contrib) for c, r in chosen.items()]

    # --- curator list: compact columnar per stream ---
    stream_data, why_dict, why_idx = [], [], {}   # reason texts repeat a lot: store each once
    def wid(text):
        if text not in why_idx:
            why_idx[text] = len(why_dict)
            why_dict.append(text)
        return why_idx[text]
    for mod, pres in keys:
        s = pool[(pool.code_module == mod) & (pool.code_presentation == pres)]
        per_t = []
        for t, g in s.groupby("t"):
            keep = g[in_top(g.rank_model, g.n_pop, LIST_CAP) | in_top(g.rank_rule, g.n_pop, LIST_CAP)]
            keep = keep.sort_values("rank_model")
            reasons = [top_reasons(contrib.loc[i], r) for i, r in keep.iterrows()]
            per_t.append({
                "t": int(t), "n": int(len(g)), "churn": int(g.y.sum()),
                "loss_total": round(float((g.y * g.value_rub).sum())),
                "rows": {
                    "id": keep.id_student.astype(int).tolist(),
                    "p": [round(float(x), 4) for x in keep.p],
                    "loss": [round(float(x)) for x in keep.p * keep.value_rub],
                    "value": [round(float(x)) for x in keep.value_rub],
                    "rm": keep.rank_model.astype(int).tolist(),
                    "rr": keep.rank_rule.astype(int).tolist(),
                    "dsc": keep.days_since_click.astype(int).tolist(),
                    "y": keep.y.astype(int).tolist(),
                    "dtl": [None if pd.isna(x) else int(x) for x in keep.days_to_leave],
                    "why": [[wid(x["text"]) for x in rs] for rs in reasons],
                },
            })
        stream_data.append({"key": f"{mod} {pres}", "module": mod, "presentation": pres,
                            "students": int(s.enr.nunique()), "churn": int(s.y.sum()), "snapshots": per_t})

    # --- evidence ---
    natural = (te.days_since_click >= RULE_DAYS).groupby(te.t).mean()
    def stability(k):
        by_t = pd.Series(res[k]["loss_caught_by_t"]).astype(float)
        worst = float((res[k]["loss_caught_top20"] - by_t).max())
        return {"worst_gap_pp": round(worst * 100, 1), "worst_t": int(by_t.idxmin())}
    si = pd.read_csv(DATA / "studentInfo.csv")
    sr = pd.read_csv(DATA / "studentRegistration.csv")
    w = si.merge(sr, on=KEY)
    w = w[w.final_result == "Withdrawn"]
    pre_start_share = float((w.date_unregistration <= 0).sum() / len(w))

    models = {}
    for k in LABELS:
        r = res[k]
        models[k] = {
            "label": LABELS[k], "color": COLORS[k],
            "loss_caught_top20": r["loss_caught_top20"], "recall_top20": r["recall_top20"],
            "precision_top20": r["precision_top20"], "pr_auc": r["pr_auc"], "roc_auc": r["roc_auc"],
            "median_lead_days": r["median_lead_days"], "calib_max_gap_pp": r.get("calib_max_gap_pp"),
            "predict_ms_per_1000": r.get("predict_ms_per_1000"),
            "loss_caught_by_t": {int(t): v for t, v in r["loss_caught_by_t"].items()},
            "at_rule_budget": r["at_rule_budget"]["loss_caught"],
            "loss_ci": bs["loss_caught"][k],
            "gain_ci": bs["gain_vs_baseline"].get(k), "p_gain_ge_target": bs["p_gain_ge_target"].get(k),
            "stability": stability(k),
            "curve": gain_curve(te, k),
        }
    data = {
        "built": date.today().isoformat(),
        "product": PRODUCT, "baseline": BASELINE, "order": list(LABELS),
        "top": TOP, "rule_days": RULE_DAYS, "price_rub_month": 8000, "horizon_days": 30,
        "snapshots": sorted(int(t) for t in te.t.unique()), "train_year": TRAIN_YEAR, "test_year": TEST_YEAR,
        "test": {"students": bs["n_students"], "churn_events": bs["n_churn_events"], "rows": int(len(te))},
        "train": {"rows": int(len(tr)), "churn_events": int(tr.y.sum())},
        "pre_start_share_of_withdrawals": round(pre_start_share, 3),
        "rule_natural_load": round(float((te.days_since_click >= RULE_DAYS).mean()), 3),
        "rule_natural_load_by_t": {int(t): round(float(v), 3) for t, v in natural.items()},
        "bootstrap": {"B": bs["B"], "target_gain": bs["target_gain"],
                      "logreg_vs_lgbm": bs["logreg_vs_lgbm_grades"]},
        "models": models,
        "streams": stream_data,
        "default_stream": stream_data[0]["key"],
        "cases": cards,
        "case_rules": CASE_RULES,
        "why_dict": why_dict,
    }

    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    for field in DEMOGRAPHICS | {"final_result"}:
        assert f'"{field}"' not in text, field
    # page numbers == pipeline numbers
    for k in LABELS:
        assert data["models"][k]["loss_caught_top20"] == res[k]["loss_caught_top20"]
        assert data["models"][k]["pr_auc"] == res[k]["pr_auc"]
        assert data["models"][k]["loss_ci"] == bs["loss_caught"][k]
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(text)
    print(f"-> {OUT.relative_to(ROOT.parent)}  {len(text) / 1e6:.2f} MB")
    print("streams:", [(s["key"], s["students"], s["churn"]) for s in stream_data])
    for c in cards:
        print(f"[{c['case']}] id {c['id']} {c['stream']} t={c['t']} | {c['outcome']} | risk "
              f"{c['model_verdict']['risk']:.1%} rank {c['model_verdict']['rank']}/{c['model_verdict']['n']} | "
              f"rule rank {c['rule_verdict']['rank']}, {c['rule_verdict']['days_since_click']} дн. без заходов | "
              + "; ".join(r["text"] for r in c["reasons"]))


if __name__ == "__main__":
    main()
