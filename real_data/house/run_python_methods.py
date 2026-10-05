"""
run_python_methods.py
Fits one Python method (SAMBoost, GPBoost or XGBoost) in one variant on the
house-price test years, with the sale year as a covariate. The fitting code is
in methods.py; INLA is fitted by run_inla.R.

Variants: ex = coordinates only in the spatial part (SAMBoost, GPBoost);
geo = coordinates also tree features (for XGBoost, the only way to use them).

By default each test year is fitted once with the selected hyperparameters
(methods.OPTIMAL_PARAMS). With --grid the grid search is run again (36 combinations
for SAMBoost, 27 for GPBoost, 108 for XGBoost); this takes about 1 to 2 hours
per test year for the two spatial methods. SAMBoost and XGBoost are
deterministic; GPBoost varies very slightly between runs (random Vecchia ordering).

Output: outputs/results/house_price/final_syear/fold_{t}/{method}_{variant}/

Usage (from the repository root):
    python real_data/house/run_python_methods.py --method samboost --variant ex
    python real_data/house/run_python_methods.py --method gpboost   --variant ex  --threads 6
    python real_data/house/run_python_methods.py --method xgboost   --variant geo --fold 1998
    python real_data/house/run_python_methods.py --method samboost --variant ex  --dry-run
"""
import os
import sys
import argparse
from pathlib import Path

_p = argparse.ArgumentParser()
_p.add_argument("--method",  choices=["gpboost", "xgboost", "samboost"], required=True)
_p.add_argument("--variant", choices=["ex", "geo"], required=True)
_p.add_argument("--fold",    type=str, default="")   # comma-sep years; blank = all
_p.add_argument("--threads", type=int, default=4)
_p.add_argument("--dry-run", action="store_true",
                help="print the plan (method/variant/folds/grid size/out dirs) and exit, no fitting")
_p.add_argument("--out-suffix", default="",
                help="output folder name instead of {method}_{variant} (e.g. for a test run)")
_p.add_argument("--tree-backend", choices=["sklearn", "xgboost"], default="sklearn",
                help="samboost trees: sklearn (paper) or xgboost (faster, model_extended)")
_p.add_argument("--xgb-threads", type=int, default=0,
                help="threads for xgboost tree building (0 = all cores), separate from --threads "
                     "(sparse solver). Only used with --tree-backend xgboost")
_p.add_argument("--grid", action="store_true",
                help="run the grid search again; by default the selected hyperparameters "
                     "(OPTIMAL_PARAMS) are fitted directly, one fit per test year")
ARGS = _p.parse_args()

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(ARGS.threads)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import itertools
import real_data.house.methods as M
from real_data.house.data_folds import (
    load_house_sigrist, get_fold, TEST_YEARS, COORD_COLS, BOOSTING_FEAT_COLS,
)

FEAT  = BOOSTING_FEAT_COLS + ["year"]          # sale year as a covariate
FINAL = M.OUT_BASE / "final_syear"
GEO   = (ARGS.variant == "geo")

# Output folder per method and variant (read by tables_figures.py)
OUT_SUFFIX = ARGS.out_suffix or {
    "gpboost":   f"gpboost_{ARGS.variant}",
    "xgboost":   f"xgboost_{ARGS.variant}",
    "samboost": f"samboost_{ARGS.variant}",
}[ARGS.method]

GRID_SIZE = {
    "gpboost":   len(list(itertools.product(*M.GPBOOST_GRID_SIGRIST.values()))),
    "xgboost":   len(list(itertools.product(*M.XGBOOST_GRID.values()))),
    "samboost": len(list(itertools.product(*M.SAMBOOST_GRID.values()))),
}[ARGS.method]


def main():
    folds = ([int(x) for x in ARGS.fold.split(",")] if ARGS.fold.strip()
             else list(TEST_YEARS))
    mode = f"GRID SEARCH ({GRID_SIZE} combos)" if ARGS.grid else "fixed optimal params (no grid, default)"
    print(f"# FINAL | {ARGS.method} | variant={ARGS.variant} | folds={folds} "
          f"| {mode} | feat={len(FEAT)} (incl year) | threads={ARGS.threads}", flush=True)
    print(f"# writes -> final_syear/fold_{{t}}/{OUT_SUFFIX}/", flush=True)

    if ARGS.dry_run:
        print("# --dry-run: no fitting performed.", flush=True)
        return

    df = load_house_sigrist()

    # Grid search only with --grid; otherwise OPTIMAL_PARAMS are fitted
    if ARGS.method == "gpboost":
        M.GRID_SEARCH_GPBOOST = ARGS.grid
    elif ARGS.method == "xgboost":
        M.GRID_SEARCH_XGBOOST = ARGS.grid
    else:
        M.GRID_SEARCH_SAMBOOST = ARGS.grid
        all_coords = df[COORD_COLS].values          # mesh domain: all sales

    for t in folds:
        fo  = get_fold(df, t)
        out = FINAL / f"fold_{t}" / OUT_SUFFIX
        fp  = None if ARGS.grid else M.OPTIMAL_PARAMS[ARGS.method][ARGS.variant][t]
        print(f"\n--- {OUT_SUFFIX}  fold {t}  (train={len(fo['train'])}) ---", flush=True)
        if ARGS.method == "gpboost":
            M.run_gpboost(fo, FEAT, out, include_coords_in_F=GEO, fixed_params=fp)
        elif ARGS.method == "xgboost":
            M.run_xgboost(fo, FEAT, out, include_coords_in_trees=GEO, fixed_params=fp)
        else:
            M.run_samboost_final(fo, FEAT, out, all_coords, include_coords_in_trees=GEO,
                                  tree_backend=ARGS.tree_backend, xgb_nthread=ARGS.xgb_threads,
                                  fixed_params=fp)
        print(f"  -> {out}", flush=True)


if __name__ == "__main__":
    main()
    # Exit directly: with the xgboost backend the normal interpreter shutdown can
    # crash after all outputs are written. Errors in main() still exit non-zero.
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
