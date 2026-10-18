"""Step 3. Bootstrap CI over students (enrollments) for loss caught in the top-20%.

Resampling is by enrollment, not by row: one student appears in several snapshots,
so rows are not independent."""
import json

import numpy as np
import pandas as pd

from common import BASELINE, BOOTSTRAP_PATH, LABELS, SCORED_PATH, ranking_score, topk_flags

B = 1000
TARGET_GAIN = 0.20   # Gate 2 target: +20% loss caught vs the rule


def ci(x):
    return [round(float(v), 4) for v in np.percentile(x, [2.5, 50, 97.5])]


def main():
    te = pd.read_parquet(SCORED_PATH)
    loss = (te.y * te.value_rub).values
    uniq, inv = np.unique(te.enr.values, return_inverse=True)
    L = np.bincount(inv, loss)
    caught = {k: np.bincount(inv, loss * topk_flags(te, ranking_score(te, k))) for k in LABELS}

    rng = np.random.default_rng(0)
    w = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=B)   # B x n_students weights
    share = {k: (w @ c) / (w @ L) for k, c in caught.items()}

    out = {"n_students": int(len(uniq)), "n_churn_events": int(te.y.sum()), "B": B, "target_gain": TARGET_GAIN,
           "loss_caught": {}, "gain_vs_baseline": {}, "p_gain_ge_target": {}}
    for k, s in share.items():
        out["loss_caught"][k] = ci(s)
        if k != BASELINE:
            gain = s / share[BASELINE] - 1
            out["gain_vs_baseline"][k] = ci(gain)
            out["p_gain_ge_target"][k] = round(float((gain >= TARGET_GAIN).mean()), 3)
    names = [k for k in LABELS if k != BASELINE]
    # pairwise: does the new model beat LightGBM with the same features?
    out["logreg_vs_lgbm_grades"] = ci(share["logreg_activity_grades"] / share["lgbm_activity_grades"] - 1)

    print(f"students {out['n_students']}, churn events {out['n_churn_events']}, B={B}")
    for k in LABELS:
        line = f"{k:24s} loss caught 95% CI {out['loss_caught'][k]}"
        if k in names:
            line += f" | gain vs rule {out['gain_vs_baseline'][k]} | P(gain>={TARGET_GAIN:.0%}) {out['p_gain_ge_target'][k]}"
        print(line)
    print("logreg vs lgbm (same features), rel. diff CI:", out["logreg_vs_lgbm_grades"])
    BOOTSTRAP_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("->", BOOTSTRAP_PATH.name)


if __name__ == "__main__":
    main()
