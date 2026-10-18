"""Run the whole pipeline: build snapshots -> train & evaluate -> bootstrap CI -> charts -> landing data.
Usage: python run_all.py [--rebuild]   (snapshots are reused if present, --rebuild forces step 1)"""
import sys

import bootstrap
import build
import evaluate
import export_landing
import plots
from common import SNAPSHOTS_PATH

if __name__ == "__main__":
    steps = [("evaluate", evaluate.main), ("bootstrap", bootstrap.main), ("plots", plots.main),
             ("export", export_landing.main)]
    if "--rebuild" in sys.argv or not SNAPSHOTS_PATH.exists():
        steps.insert(0, ("build", build.main))
    for name, step in steps:
        print(f"\n=== {name} ===")
        step()
