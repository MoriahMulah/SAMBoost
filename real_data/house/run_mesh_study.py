"""
run_mesh_study.py
Mesh-resolution study of SAMBoost on the house-price data (ex variant, sale
year as a covariate, hyperparameters from methods.OPTIMAL_PARAMS). Two curves:

  reselect : at each of 7 meshes (about 500 to 17,000 nodes), kappa is chosen
             again among {14, 30, 50} on val, refitted on train and scored on
             test. One JSON per mesh and test year in final_syear/mesh_study/,
             aggregated into plots/mesh_curve_ex_data.csv (the mesh figure).
  fixed    : kappa kept at its selected value, 12 meshes. Results in
             final_syear/mesh_study/curve_fold_{t}.csv, aggregated into
             plots/mesh_curve_v2_data.csv.

Fit time in mesh_curve_ex_data.csv is the time of the final fit (refit_secs).
For JSONs that do not record it, the aggregation keeps the fit time already in
the CSV. The RMSE is always recomputed from the JSONs.

Usage (from the repository root):
    python real_data/house/run_mesh_study.py                    # aggregate the stored results
    python real_data/house/run_mesh_study.py --mode reselect --fold 1996
    python real_data/house/run_mesh_study.py --mode reselect --fold 1996 --cutoff 0.002 --label 9k
    python real_data/house/run_mesh_study.py --mode fixed --shard 0 --nshards 6 --threads 2
The reselect and fixed modes refit the models and are slow.
"""
import os, sys, csv, io, json, time, glob, warnings, contextlib, argparse
from pathlib import Path

_p = argparse.ArgumentParser()
_p.add_argument("--mode", choices=["aggregate", "reselect", "fixed"], default="aggregate")
_p.add_argument("--fold", type=int)                 # reselect
_p.add_argument("--cutoff", type=float)             # reselect single point (optional)
_p.add_argument("--label", type=str)                # reselect single point (optional)
_p.add_argument("--shard", type=int, default=0)     # fixed
_p.add_argument("--nshards", type=int, default=1)   # fixed
_p.add_argument("--threads", type=int, default=1)
ARGS = _p.parse_args()
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = str(ARGS.threads)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import numpy as np, pandas as pd
from sklearn.metrics import mean_squared_error
from real_data.house.data_folds import (
    load_house_sigrist, get_fold, TEST_YEARS, COORD_COLS, TARGET_COL,
    BOOSTING_FEAT_COLS, CATEGORICAL_COLS)
from real_data.house.methods import OUT_BASE, OPTIMAL_PARAMS, _samboost_design
from samboost.model_continuous import SAMBoostContinuous
from utils.mesh_building import build_inla_mesh, assemble_spde_matrices, build_A_matrix

FEAT   = BOOSTING_FEAT_COLS + ["year"]                  # sale year as a covariate
FINAL  = OUT_BASE / "final_syear"
STUDY  = FINAL / "mesh_study"     # raw results of both curves
PLOTS  = FINAL / "plots"
# reselect: mesh label -> mesh cutoff (on coordinates scaled to [0,1]^2)
RESELECT_MESHES = [("0.5k", 0.0270), ("1.5k", 0.0165), ("5k", 0.0062),
                   ("6k", 0.0050), ("9k", 0.0020), ("13k", 0.0014), ("17k", 0.0010)]
RESELECT_KAPPAS = [14.0, 30.0, 50.0]
# fixed: the 12 mesh cutoffs
FIXED_CUTOFFS = [0.0270, 0.0165, 0.0130, 0.0100, 0.0078, 0.0062,
                 0.0050, 0.0035, 0.0029, 0.0020, 0.0017, 0.0015]


def _global_norm(df):
    c = df[COORD_COLS].values
    lo = c.min(0); span = c.max(0) - lo; span = np.where(span == 0, 1.0, span)
    return lambda x: (x - lo) / span


def _mesh(cn_all, cut):
    m = build_inla_mesh(obs_locs=cn_all, cutoff=cut, max_edge_inner=2.5 * cut,
                        max_edge_outer=10.0 * cut, random_state=42)
    nodes = m["mesh_nodes"]
    G, C = assemble_spde_matrices(nodes)
    return nodes, G, C


def _fit(params, X, y, cg, A, G, C):
    m = SAMBoostContinuous(A=A, G=G, C=C, **params)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        m.fit(X, y, coords=cg)
    return m


# >> Aggregation
def aggregate():
    PLOTS.mkdir(parents=True, exist_ok=True)

    # Reselect JSONs -> mesh_curve_ex_data.csv (means over the test years; fit time
    # from refit_secs if present, else the value already in the CSV)
    prev_fit = {}
    ex_csv = PLOTS / "mesh_curve_ex_data.csv"
    if ex_csv.exists():
        try:
            prev = pd.read_csv(ex_csv)
            prev_fit = dict(zip(prev["nodes"].astype(int), prev["fit"]))
        except Exception:
            pass
    groups = {}
    for f in glob.glob(str(STUDY / "exmeshk_*.json")):
        d = json.load(open(f))
        groups.setdefault(d["label"], []).append(d)
    rows = []
    for label, ds in groups.items():
        nodes = int(ds[0]["nodes"])
        rmse = float(np.mean([d["test_rmse"] for d in ds]))
        if all("refit_secs" in d for d in ds):
            fit = float(np.mean([d["refit_secs"] for d in ds]))
        elif nodes in prev_fit:
            fit = float(prev_fit[nodes])           # keep the stored value
        else:
            fit = float(np.mean([d["fit_secs"] for d in ds]))   # otherwise: time of the whole kappa search
        rows.append((nodes, round(rmse, 4), round(fit, 1)))
    if rows:
        pd.DataFrame(sorted(rows), columns=["nodes", "rmse", "fit"]).to_csv(ex_csv, index=False)
        print(f"wrote {ex_csv.name} ({len(rows)} mesh points, RMSE from JSONs, fit preserved)")

    # Fixed-kappa curves -> mesh_curve_v2_data.csv (means over the test years)
    parts = [pd.read_csv(p) for p in glob.glob(str(STUDY / "curve_fold_*.csv"))]
    if parts:
        allc = pd.concat(parts, ignore_index=True)
        agg = (allc.groupby("nodes")
                   .agg(rmse=("test_rmse", "mean"), fit=("fit_secs", "mean"))
                   .reset_index().sort_values("nodes"))
        agg["rmse"] = agg["rmse"].round(4); agg["fit"] = agg["fit"].round(1)
        agg.to_csv(PLOTS / "mesh_curve_v2_data.csv", index=False)
        print(f"wrote mesh_curve_v2_data.csv ({len(agg)} mesh points, mean over folds)")
    else:
        print("no curve_fold_*.csv found; skipped mesh_curve_v2_data.csv")


# >> Reselect mode
def _reselect_point(df, g, cn_all, fold, cut, label):
    out = STUDY / f"exmeshk_{label}_fold{fold}.json"
    if out.exists():
        print(f"  {label} fold{fold}: exists, skip"); return
    base = dict(OPTIMAL_PARAMS["samboost"]["ex"][fold])
    numeric = [c for c in FEAT if c not in CATEGORICAL_COLS]   # ex: no coordinates
    fo = get_fold(df, fold); inn, val, tr, te = fo["inner_train"], fo["val"], fo["train"], fo["test"]
    Xin, yin = _samboost_design(inn, numeric), inn[TARGET_COL].values
    Xva, yva = _samboost_design(val, numeric), val[TARGET_COL].values
    Xtr, ytr = _samboost_design(tr, numeric), tr[TARGET_COL].values
    Xte, yte = _samboost_design(te, numeric), te[TARGET_COL].values
    cin, cva = g(inn[COORD_COLS].values), g(val[COORD_COLS].values)
    ctr, cte = g(tr[COORD_COLS].values), g(te[COORD_COLS].values)
    nodes, G, C = _mesh(cn_all, cut)
    A_in, A_va = build_A_matrix(cin, nodes), build_A_matrix(cva, nodes)
    A_tr, A_te = build_A_matrix(ctr, nodes), build_A_matrix(cte, nodes)
    t0 = time.perf_counter()
    best = None
    for k in RESELECT_KAPPAS:
        m = _fit({**base, "kappa": k}, Xin, yin, cin, A_in, G, C)
        vm = float(mean_squared_error(yva, m.predict(Xva, A_va)))
        if best is None or vm < best[1]:
            best = (k, vm)
    kbest = best[0]
    t_refit = time.perf_counter()
    m = _fit({**base, "kappa": kbest}, Xtr, ytr, ctr, A_tr, G, C)
    refit_secs = time.perf_counter() - t_refit
    rmse = float(np.sqrt(np.mean((m.predict(Xte, A_te) - yte) ** 2)))
    sweep_secs = time.perf_counter() - t0
    STUDY.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"label": label, "fold": fold, "nodes": int(len(nodes)),
                               "kappa": kbest, "test_rmse": rmse,
                               "fit_secs": round(sweep_secs, 1),
                               "refit_secs": round(refit_secs, 1)}))
    print(f"  {label} fold{fold}: nodes={len(nodes)} kbest={kbest} test={rmse:.4f} "
          f"refit={refit_secs:.0f}s (sweep {sweep_secs:.0f}s)")


def reselect():
    if ARGS.fold is None:
        sys.exit("--mode reselect needs --fold")
    df = load_house_sigrist(); g = _global_norm(df); cn_all = g(df[COORD_COLS].values)
    meshes = ([(ARGS.label, ARGS.cutoff)] if ARGS.cutoff is not None and ARGS.label
              else RESELECT_MESHES)
    print(f"# reselect | fold {ARGS.fold} | {len(meshes)} mesh points")
    for label, cut in meshes:
        _reselect_point(df, g, cn_all, ARGS.fold, cut, label)


# >> Fixed mode
def fixed():
    STUDY.mkdir(parents=True, exist_ok=True)
    df = load_house_sigrist(); g = _global_norm(df); cn_all = g(df[COORD_COLS].values)
    numeric = [c for c in FEAT if c not in CATEGORICAL_COLS]
    cache = {}
    for t in TEST_YEARS:
        fo = get_fold(df, t); tr, te = fo["train"], fo["test"]
        cache[t] = dict(cfg=dict(OPTIMAL_PARAMS["samboost"]["ex"][t]),
                        Xtr=_samboost_design(tr, numeric), ytr=tr[TARGET_COL].values,
                        Xte=_samboost_design(te, numeric), yte=te[TARGET_COL].values,
                        ctr=g(tr[COORD_COLS].values), cte=g(te[COORD_COLS].values))
    tasks = [(t, cut) for t in TEST_YEARS for cut in FIXED_CUTOFFS]
    mine = [tc for i, tc in enumerate(tasks) if i % ARGS.nshards == ARGS.shard]
    CSV_COLS = ["fold", "cutoff", "edge_inner", "edge_outer", "nodes", "version",
                "kappa", "test_rmse", "fit_secs"]
    print(f"# fixed | shard {ARGS.shard}/{ARGS.nshards} | {len(mine)} (fold,cutoff) fits")
    for t, cut in mine:
        path = STUDY / f"curve_fold_{t}.csv"
        done = set()
        if path.exists() and path.stat().st_size:
            try: done = set(pd.read_csv(path)["cutoff"].round(4).tolist())
            except Exception: pass
        if round(cut, 4) in done:
            continue
        nodes, G, C = _mesh(cn_all, cut)
        c = cache[t]
        A_tr, A_te = build_A_matrix(c["ctr"], nodes), build_A_matrix(c["cte"], nodes)
        t0 = time.perf_counter()
        m = _fit(c["cfg"], c["Xtr"], c["ytr"], c["ctr"], A_tr, G, C)
        rmse = float(np.sqrt(mean_squared_error(c["yte"], m.predict(c["Xte"], A_te))))
        secs = time.perf_counter() - t0
        new = not path.exists() or path.stat().st_size == 0
        with open(path, "a", newline="", buffering=1) as fh:
            w = csv.DictWriter(fh, fieldnames=CSV_COLS)
            if new: w.writeheader()
            w.writerow({"fold": t, "cutoff": round(cut, 4), "edge_inner": round(2.5 * cut, 4),
                        "edge_outer": round(10.0 * cut, 4), "nodes": len(nodes),
                        "version": "A_fixed", "kappa": c["cfg"]["kappa"],
                        "test_rmse": round(rmse, 4), "fit_secs": round(secs, 1)})
            fh.flush(); os.fsync(fh.fileno())
        print(f"  fold {t} cut={cut:.4f} nodes={len(nodes)} test={rmse:.4f} [{secs:.0f}s]")


if __name__ == "__main__":
    {"aggregate": aggregate, "reselect": reselect, "fixed": fixed}[ARGS.mode]()
