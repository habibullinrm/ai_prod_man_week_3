"""Section 8 checks on the built page: screenshots (1280/390, light/dark), no horizontal scroll at 390,
only allowed network hosts, no console errors, numbers on the page == pipeline numbers."""
import json
from pathlib import Path
from urllib.parse import urlparse

from playwright.sync_api import sync_playwright

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ALLOWED = {"fonts.googleapis.com", "fonts.gstatic.com", "cdnjs.cloudflare.com", "cdn.jsdelivr.net", "unpkg.com"}


def ru_pct(x, d=1):
    return f"{x * 100:.{d}f}".replace(".", ",") + "%"


def signed(x):
    return ("+" if x >= 0 else "−") + f"{abs(x) * 100:.0f}%"


def main():
    res = {r["model"]: r for r in json.loads((ROOT / "prototype/reports/results.json").read_text())}
    bs = json.loads((ROOT / "prototype/reports/bootstrap.json").read_text())
    prod = "logreg_activity_grades"
    expect = {"prod-loss": ru_pct(res[prod]["loss_caught_top20"]),
              "prod-gain": signed(bs["gain_vs_baseline"][prod][1]),
              "prod-lead": f"{res[prod]['median_lead_days']:.0f}"}
    (HERE / "screens").mkdir(exist_ok=True)
    url = (HERE / "index.html").as_uri()
    problems = []
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        for scheme in ("light", "dark"):
            for width in (1280, 390):
                ctx = browser.new_context(viewport={"width": width, "height": 900}, color_scheme=scheme,
                                          device_scale_factor=1)
                page = ctx.new_page()
                errors, hosts = [], set()
                page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
                page.on("pageerror", lambda e: errors.append(str(e)))
                page.on("request", lambda r: hosts.add(urlparse(r.url).hostname) if r.url.startswith("http") else None)
                page.goto(url, wait_until="networkidle")
                tag = f"{width}_{scheme}"
                page.screenshot(path=HERE / "screens" / f"page_{tag}.png", full_page=True)
                sw = page.evaluate("document.documentElement.scrollWidth")
                if width == 390 and sw > 390:
                    problems.append(f"{tag}: scrollWidth {sw} > 390")
                if errors:
                    problems.append(f"{tag}: console errors {errors}")
                bad = hosts - ALLOWED
                if bad:
                    problems.append(f"{tag}: requests to {bad}")
                for key, val in expect.items():
                    got = page.inner_text(f'[data-check="{key}"]')
                    if not got.replace(" ", " ").startswith(val):
                        problems.append(f"{tag}: {key} shows {got!r}, expected {val!r}")
                ev = page.inner_text("#ev-table").replace(" ", " ").replace(" ", " ")
                for val in (f"{res[prod]['pr_auc']:.3f}".replace(".", ","), f"{res[prod]['calib_max_gap_pp']:.2f}".replace(".", ",")):
                    if val not in ev:
                        problems.append(f"{tag}: evidence table lacks {val}")
                if page.locator(".case").count() != 4:
                    problems.append(f"{tag}: expected 4 case cards")
                if page.locator("#list-body tr").count() == 0:
                    problems.append(f"{tag}: curator list is empty")
                if width == 1280 and scheme == "light":
                    page.click("#reveal")
                    page.locator("#list").screenshot(path=HERE / "screens" / "list_revealed_1280_light.png")
                    page.locator("#cases").screenshot(path=HERE / "screens" / "cases_1280_light.png")
                    page.locator("#evidence").screenshot(path=HERE / "screens" / "evidence_1280_light.png")
                print(f"{tag}: scrollWidth {sw}, hosts {sorted(h for h in hosts if h)}, console errors {len(errors)}")
                ctx.close()
        browser.close()
    size = (HERE / "index.html").stat().st_size
    if size > 5 * 1024 * 1024:
        problems.append(f"index.html {size} bytes > 5 MB")
    if problems:
        raise SystemExit("FAILED:\n" + "\n".join(problems))
    print(f"all checks passed; index.html {size / 1e6:.2f} MB; screenshots in landing/screens/")


if __name__ == "__main__":
    main()
