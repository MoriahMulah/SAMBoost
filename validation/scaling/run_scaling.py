"""
run_scaling.py
==============
Fit-time scaling study for the Python methods: fit time as a function of the
number of training points N, everything else fixed.

  N       : 500 to 64,000, seeds 1-3; data generated as in Simulation 2 and saved
            in validation/scaling/data/ (reused by bench_inla.R)
  Mesh    : one mesh built once on a 40x40 reference grid and reused for every N
            (only the projection matrix A changes with N)
  Models  : 100 trees and fixed settings for every method, one fit at a time:
            SAMBoost, GPBoost (Vecchia, and exact GP up to N = 2,000),
            XGBoost with the coordinates as features
  Output  : results_python.csv (one row per method, N and seed)

Usage (from the repository root):
    python validation/scaling/run_scaling.py           # full sweep
    python validation/scaling/run_scaling.py --smoke   # N = 500, 1000, one seed
"""
import sys, time, warnings, argparse, gc
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
import numpy as np, pandas as pd
warnings.simplefilter("ignore")
from samboost.model_continuous import SAMBoostContinuous
from utils import build_inla_mesh, build_A_matrix, assemble_spde_matrices
from simulations.generate_hotspot_river import hotspot_river_effect
from simulations.generate_sim2_hotspot_river import river_indicator_wide, F_raw, F_VAR_TARGET
import gpboost as gpb
from xgboost import XGBRegressor

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"; DATA.mkdir(parents=True, exist_ok=True)
RESCSV = HERE / "results_python.csv"

NS = [500, 1000, 2000, 4000, 8000, 16000, 32000, 64000]
SEEDS = [1, 2, 3]
CAP_EXACT = 2000
N_TREES = 100
VECCHIA_NEIGHBORS = 20
COV_COLS = ["x1","x2","x3","x4","x5","x_noise1","x_noise2"]
SB_CFG = dict(alpha=1.0, kappa=5.0, n_estimators=N_TREES, learning_rate=0.1, max_depth_stage1=3,
              n_freeze=10, min_samples_leaf=20, w_mode="no_geo", ratio=2.0, n_levels=1,
              alpha_start_scale=1.0, gamma=0)

# One mesh, built once on a fixed 40x40 reference grid (independent of N)
_g = np.linspace(0, 1, 40)
_gx, _gy = np.meshgrid(_g, _g)
FIXED_REF = np.column_stack([_gx.ravel(), _gy.ravel()])   # fixed 40x40 grid -> fixed mesh
_MESH = build_inla_mesh(obs_locs=FIXED_REF, cutoff=0.01, max_edge_inner=0.05,
                        max_edge_outer=0.15, random_state=42)
MESH_NODES = _MESH["mesh_nodes"]
G_FIXED, C_FIXED = assemble_spde_matrices(MESH_NODES)
print(f"FIXED mesh: {len(MESH_NODES)} nodes (shared across all N)")


def gen(n, seed):
    d = DATA / f"N{n}_s{seed}"; d.mkdir(exist_ok=True); fp = d / "train_df.csv"
    if fp.exists(): return pd.read_csv(fp)
    rng = np.random.default_rng(seed); rc = np.random.default_rng(seed + 1000)
    Lc = rc.uniform(0,1,(50000,2)); Xc = rc.standard_normal((50000,7))
    fc = F_raw(Xc, river_indicator_wide(Lc)); C1, C2 = fc.mean(), np.sqrt(F_VAR_TARGET)/fc.std()
    L = rng.uniform(0,1,(n,2)); X = rng.standard_normal((n,7)); r = river_indicator_wide(L)
    F = C2*(F_raw(X,r)-C1); y = F + hotspot_river_effect(L) + rng.normal(0,0.5,n)
    df = pd.DataFrame(L, columns=["s1","s2"])
    for i,c in enumerate(COV_COLS): df[c] = X[:,i]
    df["x_hotspot"]=r; df["F_effect"]=F; df["spatial_effect"]=hotspot_river_effect(L); df["m"]=F+df["spatial_effect"]; df["y"]=y
    df.to_csv(fp, index=False); return df


def append(row):
    pd.DataFrame([row]).to_csv(RESCSV, mode="a", header=not RESCSV.exists(), index=False)
    print("   ", {k:(round(v,2) if isinstance(v,float) else v) for k,v in row.items()})


def load(n, seed):
    df = gen(n, seed)
    feats = [c for c in df.columns if c not in ("s1","s2","F_effect","spatial_effect","m","y")]
    return df[feats].values, df["y"].values, df[["s1","s2"]].values


def fit_sb(n, seed):
    X, y, coords = load(n, seed)
    A = build_A_matrix(coords, MESH_NODES)          # only A changes with N; mesh/G/C fixed
    m = SAMBoostContinuous(A=A, G=G_FIXED, C=C_FIXED, objective="reg:squarederror", **SB_CFG)
    t=time.perf_counter(); m.fit(X, y, coords=coords); dt=time.perf_counter()-t
    del m; gc.collect(); return dt

def fit_gpb(n, seed, approx):
    X, y, coords = load(n, seed)
    kw = dict(gp_coords=coords, cov_function="exponential", likelihood="gaussian", gp_approx=approx)
    if approx == "vecchia": kw["num_neighbors"] = VECCHIA_NEIGHBORS
    gm = gpb.GPModel(**kw); ds = gpb.Dataset(data=X, label=y)
    t=time.perf_counter()
    gpb.train(params={"objective":"regression","learning_rate":0.1,"max_depth":3,"verbose":-1},
              train_set=ds, gp_model=gm, num_boost_round=N_TREES); dt=time.perf_counter()-t
    del gm, ds; gc.collect(); return dt

def fit_xgb(n, seed):
    X, y, coords = load(n, seed)
    Xc = np.hstack([X, coords])                     # coordinates as features
    m = XGBRegressor(n_estimators=N_TREES, max_depth=3, learning_rate=0.1, n_jobs=-1, verbosity=0)
    t=time.perf_counter(); m.fit(Xc, y); dt=time.perf_counter()-t
    del m; gc.collect(); return dt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--smoke", action="store_true"); a = ap.parse_args()
    ns = [500,1000] if a.smoke else NS; seeds = [1] if a.smoke else SEEDS
    if not a.smoke and RESCSV.exists(): RESCSV.unlink()
    print(f"Scaling: N={ns} seeds={seeds} trees={N_TREES} exact_cap={CAP_EXACT} mesh_nodes={len(MESH_NODES)}")
    # method-ordered: SAMBoost -> GPBoost(vecchia,exact) -> XGBoost
    for n in ns:
        for s in seeds:
            print(f"[SAMBoost] N={n} s={s}"); append(dict(N=n,seed=s,mesh_nodes=len(MESH_NODES),method="SAMBoost",fit_s=fit_sb(n,s)))
    for n in ns:
        for s in seeds:
            print(f"[GPBoost_vecchia] N={n} s={s}"); append(dict(N=n,seed=s,mesh_nodes=len(MESH_NODES),method="GPBoost_vecchia",fit_s=fit_gpb(n,s,"vecchia")))
    for n in ns:
        if n > CAP_EXACT: continue
        for s in seeds:
            print(f"[GPBoost_exact] N={n} s={s}"); append(dict(N=n,seed=s,mesh_nodes=len(MESH_NODES),method="GPBoost_exact",fit_s=fit_gpb(n,s,"none")))
    for n in ns:
        for s in seeds:
            print(f"[XGBoost] N={n} s={s}"); append(dict(N=n,seed=s,mesh_nodes=len(MESH_NODES),method="XGBoost",fit_s=fit_xgb(n,s)))
    print("\nDONE ->", RESCSV)


if __name__ == "__main__":
    main()
