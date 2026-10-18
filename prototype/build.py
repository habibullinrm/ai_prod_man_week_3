"""Step 1. Build training/test snapshots from OULAD.

Every 30 days (t = 30..210) we take a snapshot of students who are still enrolled.
Label y = the student withdraws within the next 30 days. Features use only days < t.
Excluded: withdrawals before the course start (no activity to learn from) and
withdrawals with an unknown date."""
import numpy as np
import pandas as pd

from common import DATA, HORIZON, KEY, PRICE, SNAPSHOTS, SNAPSHOTS_PATH


def load_enrollments():
    si = pd.read_csv(DATA / "studentInfo.csv")
    sr = pd.read_csv(DATA / "studentRegistration.csv")
    co = pd.read_csv(DATA / "courses.csv")
    enr = si.merge(sr, on=KEY).merge(co, on=["code_module", "code_presentation"])
    n0 = len(enr)
    unknown = (enr.final_result == "Withdrawn") & enr.date_unregistration.isna()
    pre_start = enr.date_unregistration <= 0   # D-043: left before day 0
    print(f"enrollments {n0}: drop withdrawn-without-date {unknown.sum()}, pre-start withdrawals {pre_start.sum()}")
    enr = enr[~unknown & ~pre_start].reset_index(drop=True)
    enr["enr"] = np.arange(len(enr))
    enr["year"] = enr.code_presentation.str[:4].astype(int)
    return enr


def load_vle(enr):
    """Daily clicks per enrollment."""
    cat = {"code_module": "category", "code_presentation": "category"}
    vle = pd.read_csv(DATA / "studentVle.csv", usecols=KEY + ["date", "sum_click"], dtype=cat)
    vle = vle.groupby(KEY + ["date"], observed=True, as_index=False).sum_click.sum()
    return vle.merge(enr[KEY + ["enr"]].astype(cat), on=KEY)[["enr", "date", "sum_click"]]


def load_assessments(enr):
    """Submissions and due dates of non-exam assessments."""
    asm = pd.read_csv(DATA / "assessments.csv")
    sas = pd.read_csv(DATA / "studentAssessment.csv")
    asm = asm[asm.assessment_type != "Exam"].dropna(subset=["date"])
    sub = sas.merge(asm, on="id_assessment").merge(enr[KEY + ["enr"]], on=KEY)
    due = asm.merge(enr[["code_module", "code_presentation", "enr"]], on=["code_module", "code_presentation"])
    return sub, due


def activity_features(vle, pop, t):
    v = vle[(vle.date < t) & vle.enr.isin(pop.enr)]
    g = v.groupby("enr")
    f = pd.DataFrame({"clicks_total": g.sum_click.sum(), "active_days_total": g.size(), "last_click_day": g.date.max()})
    for w in (7, 14, 30):
        vv = v[v.date >= t - w].groupby("enr")
        f[f"clicks_{w}"] = vv.sum_click.sum()
        f[f"active_days_{w}"] = vv.size()
    f["clicks_prev14"] = v[(v.date >= t - 28) & (v.date < t - 14)].groupby("enr").sum_click.sum()
    pop = pop.merge(f, left_on="enr", right_index=True, how="left")
    counts = [c for c in f.columns if c != "last_click_day"]
    pop[counts] = pop[counts].fillna(0)
    pop["days_since_click"] = (t - pop.last_click_day).fillna(t + 60)   # never clicked: far beyond any window
    pop["click_trend"] = (pop.clicks_14 + 1) / (pop.clicks_prev14 + 1)
    pop["clicks_per_day"] = pop.clicks_total / (t - pop.date_registration.clip(upper=0) + 1)
    return pop


def grade_features(sub, due, pop, t):
    d = due[(due.date < t) & due.enr.isin(pop.enr)].groupby("enr").size().rename("n_due")
    s = sub[(sub.date_submitted < t) & sub.enr.isin(pop.enr)].copy()
    s["late"] = (s.date_submitted > s.date).astype(int)
    gs = s.sort_values("date_submitted").groupby("enr")
    k = pd.DataFrame({"n_sub": gs.size(), "score_mean": gs.score.mean(), "score_last": gs.score.last(),
                      "score_min": gs.score.min(), "n_late": gs.late.sum()})
    pop = pop.merge(d, left_on="enr", right_index=True, how="left").merge(k, left_on="enr", right_index=True, how="left")
    pop[["n_due", "n_sub", "n_late"]] = pop[["n_due", "n_sub", "n_late"]].fillna(0)
    pop["n_missed"] = (pop.n_due - pop.n_sub).clip(lower=0)
    pop["score_drop"] = pop.score_last - pop.score_mean   # grades stay NaN if nothing submitted yet
    return pop


def snapshot(enr, vle, sub, due, t):
    active = (enr.date_registration <= t) & ~(enr.date_unregistration <= t) & (enr.module_presentation_length > t)
    pop = enr[active].copy()
    pop["t"] = t
    leaves = (pop.final_result == "Withdrawn") & (pop.date_unregistration > t) & (pop.date_unregistration <= t + HORIZON)
    pop["y"] = leaves.astype(int)
    pop["days_to_leave"] = np.where(leaves, pop.date_unregistration - t, np.nan)
    pop["value_rub"] = (pop.module_presentation_length - t) / 30 * PRICE   # remaining payments at risk
    pop = activity_features(vle, pop, t)
    return grade_features(sub, due, pop, t)


def main():
    enr = load_enrollments()
    vle = load_vle(enr)
    sub, due = load_assessments(enr)
    ds = pd.concat([snapshot(enr, vle, sub, due, t) for t in SNAPSHOTS], ignore_index=True)
    ds.to_parquet(SNAPSHOTS_PATH)
    print(ds.groupby(["year", "t"]).agg(n=("y", "size"), churn=("y", "sum"), rate=("y", "mean")).round(4))
    print("total rows", len(ds), "positives", int(ds.y.sum()), "->", SNAPSHOTS_PATH.name)


if __name__ == "__main__":
    main()
