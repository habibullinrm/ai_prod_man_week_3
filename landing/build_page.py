"""template.html + data.json -> index.html (full document) and fragment.html (for a claude.ai artifact).

The template is written as a fragment: everything above <!--/head--> goes into <head> of index.html,
the rest into <body>. fragment.html keeps it all as is (the artifact host adds its own skeleton)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MAX_BYTES = 5 * 1024 * 1024
MARK = "<!--/head-->"


def main():
    tpl = (HERE / "template.html").read_text(encoding="utf-8")
    data = json.loads((HERE / "data.json").read_text(encoding="utf-8"))
    # numbers on the page come from pipeline files: recheck before embedding
    res = {r["model"]: r for r in json.loads((ROOT / "prototype/reports/results.json").read_text())}
    bs = json.loads((ROOT / "prototype/reports/bootstrap.json").read_text())
    for k, m in data["models"].items():
        for f in ("loss_caught_top20", "pr_auc", "median_lead_days", "calib_max_gap_pp", "predict_ms_per_1000"):
            assert m[f] == res[k].get(f), (k, f)
        assert m["loss_ci"] == bs["loss_caught"][k] and m["gain_ci"] == bs["gain_vs_baseline"].get(k), k
    assert data["bootstrap"]["logreg_vs_lgbm"] == bs["logreg_vs_lgbm_grades"]

    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    assert tpl.count("__DATA__") == 1 and tpl.count(MARK) == 1
    fragment = tpl.replace("__DATA__", payload)
    head, body = fragment.split(MARK)
    full = ('<!doctype html>\n<html lang="ru">\n<head>\n<meta charset="utf-8">\n'
            '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
            f"{head.strip()}\n</head>\n<body>\n{body.strip()}\n</body>\n</html>\n")
    (HERE / "index.html").write_text(full, encoding="utf-8")
    (HERE / "fragment.html").write_text(fragment.replace(MARK, ""), encoding="utf-8")
    size = (HERE / "index.html").stat().st_size
    assert size <= MAX_BYTES, f"index.html is {size / 1e6:.2f} MB > 5 MB"
    print(f"-> landing/index.html {size / 1e6:.2f} MB, landing/fragment.html")


if __name__ == "__main__":
    main()
