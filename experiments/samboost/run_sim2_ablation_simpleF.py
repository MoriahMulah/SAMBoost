"""
run_sim2_ablation_simpleF.py  (SAMBoost)
===========================================
SAMBoost on the ablation (Simulation 2 with a nearly additive covariate effect), squared-error objective (model_continuous).

  Mesh      : training locations plus the domain corners and edge midpoints
              (cutoff 0.01, boundary spacing 0.05 / 0.15), shared by all folds.
  Tuning    : grid search over PARAM_GRID, 3-fold CV (KFold), mean RMSE.
  Final fit : selected values on the full training set; prediction and
              dissociation metrics, predictions, spatial field on a 50x50 grid,
              mesh files and plots in out_dir.

Called by run_pipeline/run_multi_seed.py as summary = run(out_dir); the driver
tunes on seed 1 and reuses the selected values for seeds 2-30. Run as a script,
it reads outputs/data/sim2_ablation_simpleF/.
"""

import sys
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import time
import json
import warnings
import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from samboost.model_continuous import SAMBoostContinuous
from utils import build_inla_mesh, build_A_matrix, assemble_spde_matrices
from simulations.generate_hotspot_river import hotspot_river_effect


# >> Simulation settings

SIM_NAME = "sim2_ablation_simpleF"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SIM_DIR   = REPO_ROOT / "outputs" / "data" / "sim2_ablation_simpleF"

COORD_COLS    = ["s1", "s2"]
TARGET_COL    = "y"
NON_FEAT_COLS = ["s1", "s2", "F_effect", "spatial_effect", "m", "y"]

# Mesh settings
MESH_CUTOFF     = 0.01
MESH_EDGE_INNER = 0.05
MESH_EDGE_OUTER = 0.15


# >> Hyperparameter grid (16 combinations)

PARAM_GRID = {
    "alpha"            : [1.0, 2.0],
    "n_estimators"     : [200],
    "learning_rate"    : [0.1],
    "max_depth_stage1" : [3, 5],
    "min_samples_leaf" : [20],
    "gamma"            : [0],
    "n_freeze"         : [10, 30],
    "w_mode"           : ["no_geo"],
    "kappa"            : [5.0, 7.0],
    "ratio"            : [2.0],
    "n_levels"         : [1],
    "alpha_start_scale": [1.0],
}
OBJECTIVE = "reg:squarederror"
N_CV_FOLDS = 3


# >> Helpers

def _arr(df_or_arr):
    return df_or_arr.values if hasattr(df_or_arr, "values") else df_or_arr


def score_split(name, y_true, y_pred):
    return {
        "split" : name,
        "n"     : int(len(y_true)),
        "rmse"  : round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "r2"    : round(float(r2_score(y_true, y_pred)), 4),
        "mae"   : round(float(mean_absolute_error(y_true, y_pred)), 4),
    }


def dissociation_metrics(split_name, b_true, b_hat, F_true, F_hat):
    return {
        "split"        : split_name,
        "r_spatial"    : round(float(np.corrcoef(b_true, b_hat)[0, 1]), 4),
        "rmse_spatial" : round(float(np.sqrt(np.mean((b_true - b_hat) ** 2))), 4),
        "r_cov"        : round(float(np.corrcoef(F_true, F_hat)[0, 1]), 4),
        "rmse_cov"     : round(float(np.sqrt(np.mean((F_true - F_hat) ** 2))), 4),
    }


def load_data(sim_dir):
    train_df = pd.read_csv(sim_dir / "train_df.csv")
    test_df  = pd.read_csv(sim_dir / "test_df.csv")
    print(f"  Train : {len(train_df)} rows")
    print(f"  Test  : {len(test_df)} rows")
    return train_df, test_df


# >> Spatial components (mesh, A, G, C)

def build_spatial_components(train_df, test_df):
    # Add the domain corners and edge midpoints so that the mesh covers the whole
    # unit square (otherwise points near the corners get a zero spatial effect).
    _frame = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.],
                       [0.5, 0.], [0.5, 1.], [0., 0.5], [1., 0.5]])
    obs_locs = np.vstack([train_df[COORD_COLS].values, _frame])
    mesh_result = build_inla_mesh(
        obs_locs       = obs_locs,
        cutoff         = MESH_CUTOFF,
        max_edge_inner = MESH_EDGE_INNER,
        max_edge_outer = MESH_EDGE_OUTER,
        random_state   = 42,
    )
    mesh_nodes = mesh_result["mesh_nodes"]
    print(f"  Mesh   nodes={len(mesh_nodes)}  quality={mesh_result['quality']}")
    G, C = assemble_spde_matrices(mesh_nodes)
    all_locs = np.vstack([train_df[COORD_COLS].values, test_df[COORD_COLS].values])
    A_global = build_A_matrix(all_locs, mesh_nodes)
    n_tr = len(train_df)
    return mesh_nodes, mesh_result, G, C, A_global[:n_tr], A_global[n_tr:]


# >> Hyperparameter tuning (3-fold CV, RMSE)

def run_grid_search(X_train, y_train, coords_train, A_train, G, C, out_dir):
    """KFold regression CV over PARAM_GRID. For each fold the mesh nodes are
    shared (built on the full training cloud); A_train rows are subset per fold,
    mirroring SAMBoostCV's classification protocol. Selects min mean RMSE."""
    from sklearn.model_selection import KFold
    keys   = list(PARAM_GRID.keys())
    combos = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    kf = KFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=42)
    Xa, ya = _arr(X_train), _arr(y_train)
    print(f"  Grid size: {len(combos)} combos x {N_CV_FOLDS} CV folds (regression RMSE)")

    rows = []
    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        fold_rmses = []
        try:
            for tr, va in kf.split(Xa):
                model = SAMBoostContinuous(A=A_train[tr], G=G, C=C,
                                              objective=OBJECTIVE, **params)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(Xa[tr], ya[tr], coords=coords_train[tr])
                pred = model.predict(Xa[va], A_new=A_train[va])
                fold_rmses.append(np.sqrt(mean_squared_error(ya[va], pred)))
            row = {**params, "mean_rmse": float(np.mean(fold_rmses)),
                   "std_rmse": float(np.std(fold_rmses))}
        except Exception as e:
            print(f"  [{i+1:>3}/{len(combos)}] FAILED: {e}")
            row = {**params, "mean_rmse": np.nan, "std_rmse": np.nan}
        rows.append(row)
        if (i + 1) % 12 == 0:
            print(f"  [{i+1:>3}/{len(combos)}] scanned")

    results_df = (pd.DataFrame(rows).dropna(subset=["mean_rmse"])
                  .sort_values("mean_rmse", ascending=True).reset_index(drop=True))
    results_df.insert(0, "rank", range(1, len(results_df) + 1))
    results_df.to_csv(out_dir / "tuning_grid.csv", index=False)

    best_row    = results_df.iloc[0]
    best_params = {k: best_row[k] for k in keys}
    for k, v in best_params.items():
        if isinstance(v, np.integer):
            best_params[k] = int(v)
        elif isinstance(v, np.floating):
            best_params[k] = float(v)
    rmse_cv = float(best_row["mean_rmse"])
    with open(out_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"  Best CV RMSE: {rmse_cv:.4f} +/- {best_row['std_rmse']:.4f}")
    print(f"  Best params : {best_params}")
    return best_params, rmse_cv


# >> Evaluate the selected model

def evaluate_model(best_params, X_train, y_train, coords_train,
                   X_test, y_test, train_df, test_df,
                   A_train, A_test, G, C, mesh_nodes, mesh_result, out_dir):
    print(f"\n  Fitting model on full training set ...")
    model = SAMBoostContinuous(A=A_train, G=G, C=C, objective=OBJECTIVE, **best_params)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(_arr(X_train), _arr(y_train), coords=coords_train)
    fit_time = time.perf_counter() - t0
    print(f"  Fit time: {fit_time:.1f}s")

    y_pred_train = model.predict(_arr(X_train), A_new=A_train)
    y_pred_test  = model.predict(_arr(X_test),  A_new=A_test)

    scores_df = pd.DataFrame([score_split("train", _arr(y_train), y_pred_train),
                              score_split("test",  _arr(y_test),  y_pred_test)])
    scores_df.to_csv(out_dir / "scores.csv", index=False)
    print("\n  Prediction scores (regression):")
    print(scores_df.to_string(index=False))

    b_train_hat = model.spatial_effect(A_train)
    b_test_hat  = model.spatial_effect(A_test)
    F_train_hat = model.covariate_effect(_arr(X_train))
    F_test_hat  = model.covariate_effect(_arr(X_test))

    dissoc_df = pd.DataFrame([
        dissociation_metrics("train", train_df["spatial_effect"].values, b_train_hat,
                             train_df["F_effect"].values, F_train_hat),
        dissociation_metrics("test", test_df["spatial_effect"].values, b_test_hat,
                             test_df["F_effect"].values, F_test_hat)])
    dissoc_df["fit_time"] = round(fit_time, 2)
    dissoc_df.to_csv(out_dir / "dissociation.csv", index=False)
    print("\n  Dissociation metrics:")
    print(dissoc_df[["split", "r_spatial", "rmse_spatial", "r_cov", "rmse_cov"]].to_string(index=False))

    pd.concat([
        train_df[COORD_COLS + ["y"]].assign(
            F_true=train_df["F_effect"].values, b_true=train_df["spatial_effect"].values,
            y_hat=y_pred_train, b_hat=b_train_hat, F_hat=F_train_hat, split="train"),
        test_df[COORD_COLS + ["y"]].assign(
            F_true=test_df["F_effect"].values, b_true=test_df["spatial_effect"].values,
            y_hat=y_pred_test, b_hat=b_test_hat, F_hat=F_test_hat, split="test"),
    ], ignore_index=True).to_csv(out_dir / "predictions.csv", index=False)

    _plot_dissociation("train", train_df, b_train_hat, F_train_hat, out_dir)
    _plot_dissociation("test",  test_df,  b_test_hat,  F_test_hat,  out_dir)

    pd.DataFrame(mesh_nodes, columns=["s1", "s2"]).to_csv(out_dir / "mesh_nodes.csv", index=False)
    pd.DataFrame(mesh_result["tri"].simplices, columns=["v1", "v2", "v3"]).to_csv(
        out_dir / "mesh_triangles.csv", index=False)

    grid_1d = np.linspace(0, 1, 50)
    gs1, gs2 = np.meshgrid(grid_1d, grid_1d)
    grid_locs = np.column_stack([gs1.ravel(), gs2.ravel()])
    A_grid = build_A_matrix(grid_locs, mesh_nodes)
    grid_df = pd.DataFrame(grid_locs, columns=["s1", "s2"])
    grid_df["b_hat"]  = model.spatial_effect(A_grid)
    grid_df["b_true"] = hotspot_river_effect(grid_locs)
    grid_df.to_csv(out_dir / "grid_spatial_field.csv", index=False)

    return scores_df, dissoc_df, fit_time


# >> Plots

def _plot_dissociation(split_name, df, b_hat, F_hat, out_dir):
    s1, s2 = df["s1"].values, df["s2"].values
    b_true = df["spatial_effect"].values
    F_true = df["F_effect"].values
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))

    def _map(ax, vals, title, cmap):
        sc = ax.scatter(s1, s2, c=vals, cmap=cmap, s=14, alpha=0.85,
                        vmin=np.percentile(vals, 2), vmax=np.percentile(vals, 98))
        plt.colorbar(sc, ax=ax, shrink=0.8)
        ax.set_title(title, fontsize=10); ax.set_xlabel("s1"); ax.set_ylabel("s2")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

    _map(axes[0, 0], b_true, "True  b(s)", "RdBu_r")
    _map(axes[0, 1], b_hat, "Recovered  b_hat(s)", "RdBu_r")
    _map(axes[0, 2], b_true - b_hat, "Residual  b - b_hat", "bwr")
    _map(axes[1, 0], F_true, "True  F(X)", "PiYG")
    _map(axes[1, 1], F_hat, "Recovered  F_hat(X)", "PiYG")
    ax = axes[1, 2]
    ax.scatter(F_true, F_hat, s=8, alpha=0.4, color="#1f77b4")
    lim = [min(F_true.min(), F_hat.min()) - 0.2, max(F_true.max(), F_hat.max()) + 0.2]
    ax.plot(lim, lim, "r--", lw=1.2); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True F(X)"); ax.set_ylabel("Recovered F_hat(X)")
    ax.set_title("True vs Recovered  F(X)", fontsize=10)
    r_s = np.corrcoef(b_true, b_hat)[0, 1]; r_f = np.corrcoef(F_true, F_hat)[0, 1]
    fig.suptitle(f"{SIM_NAME}  [{split_name}]  -  r_spatial={r_s:.3f}   r_cov={r_f:.3f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / f"dissociation_{split_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# >> Main entry point

def run(out_dir=None):
    if out_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir   = REPO_ROOT / "outputs" / "results" / "samboost" / f"run_{timestamp}" / SIM_NAME
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n  {SIM_NAME}  [regression]\n{'='*60}")
    print("\n[1] Loading simulation data ...")
    train_df, test_df = load_data(SIM_DIR)

    feat_cols    = [c for c in train_df.columns if c not in NON_FEAT_COLS]
    X_train, y_train = train_df[feat_cols], train_df[TARGET_COL]
    X_test,  y_test  = test_df[feat_cols],  test_df[TARGET_COL]
    coords_train = train_df[COORD_COLS].values
    print(f"  Feature cols ({len(feat_cols)}): {feat_cols}")

    print("\n[2] Building spatial components ...")
    mesh_nodes, mesh_result, G, C, A_train, A_test = build_spatial_components(train_df, test_df)

    print("\n[3] Hyperparameter tuning (KFold regression CV on training set) ...")
    best_params, rmse_cv = run_grid_search(
        X_train, y_train, coords_train, A_train, G, C, out_dir)

    print("\n[4] Evaluating model on all splits ...")
    scores_df, dissoc_df, fit_time = evaluate_model(
        best_params, X_train, y_train, coords_train, X_test, y_test,
        train_df, test_df, A_train, A_test, G, C, mesh_nodes, mesh_result, out_dir)

    def _d(split, col): return dissoc_df.loc[dissoc_df.split == split, col].values[0]
    def _s(split, col): return scores_df.loc[scores_df.split == split, col].values[0]
    summary = {
        "sim": SIM_NAME,
        "rmse_train": _s("train", "rmse"), "rmse_test": _s("test", "rmse"),
        "r2_test": _s("test", "r2"), "mae_test": _s("test", "mae"),
        "r_spatial_train": _d("train", "r_spatial"), "rmse_spatial_train": _d("train", "rmse_spatial"),
        "r_cov_train": _d("train", "r_cov"), "rmse_cov_train": _d("train", "rmse_cov"),
        "r_spatial_test": _d("test", "r_spatial"), "rmse_spatial_test": _d("test", "rmse_spatial"),
        "r_cov_test": _d("test", "r_cov"), "rmse_cov_test": _d("test", "rmse_cov"),
        "fit_time": round(fit_time, 2),
        **{f"best_{k}": v for k, v in best_params.items()},
    }
    print(f"\n  Done. All outputs in {out_dir}")
    return summary


if __name__ == "__main__":
    run()
