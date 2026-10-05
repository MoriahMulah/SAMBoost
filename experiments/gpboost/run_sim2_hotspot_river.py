"""
run_sim2_hotspot_river.py  (GPBoost)
====================================
GPBoost on Simulation 2 (regression, confounded river covariate): tree ensemble plus a Gaussian process with an
exponential covariance (Gaussian likelihood, exact GP at N = 1000).

  Tuning    : gpb.cv, 3 folds, early stopping (10 rounds, at most 500), RMSE.
  Effects   : covariate effect = tree ensemble; spatial effect = kriging
              prediction of the GP random effect.
  Outputs   : metrics, predictions, spatial field on a 50x50 grid and plots in out_dir.

Called by run_pipeline/run_multi_seed.py as summary = run(out_dir); the driver
tunes on seed 1 and reuses the selected values for seeds 2-30. Run as a script,
it reads outputs/data/sim2_hotspot_river/.
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
import gpboost as gpb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from simulations.generate_hotspot_river import hotspot_river_effect


# >> Simulation settings

SIM_NAME = "sim2_hotspot_river"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SIM_DIR   = REPO_ROOT / "outputs" / "data" / "sim2_hotspot_river"

COORD_COLS    = ["s1", "s2"]
TARGET_COL    = "y"
NON_FEAT_COLS = ["s1", "s2", "F_effect", "spatial_effect", "m", "y"]

COV_FUNCTION = "exponential"
LIKELIHOOD   = "gaussian"


# >> Hyperparameter grid

# Exact GP only (gp_approx="none"): at N = 1000 the Vecchia approximation is not needed.
GRID_FULL = {
    "learning_rate"    : [0.05, 0.1],
    "max_depth"        : [3, 5, 7],
    "min_data_in_leaf" : [10, 20],
    "num_leaves"       : [15, 31],
    "gp_approx"        : ["none"],
}
GRID_VECCHIA = {}   # Vecchia sweep disabled (exact GP at N = 1000)

N_CV_FOLDS       = 3
CV_METRIC        = "rmse"       # regression -> lower is better
EARLY_STOPPING   = 10
MAX_BOOST_ROUNDS = 500


# >> Helpers

def _arr(x):
    return x.values if hasattr(x, "values") else x


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


def _make_gp_model(coords, gp_approx, num_neighbors=None):
    kwargs = dict(gp_coords=coords, cov_function=COV_FUNCTION,
                  likelihood=LIKELIHOOD, gp_approx=gp_approx)
    if gp_approx == "vecchia" and num_neighbors is not None:
        kwargs["num_neighbors"] = int(num_neighbors)
    return gpb.GPModel(**kwargs)


def _make_dataset(X, y):
    return gpb.Dataset(data=_arr(X), label=_arr(y))


def _build_tree_params(combo):
    gp_keys = {"gp_approx", "num_neighbors"}
    return {k: v for k, v in combo.items() if k not in gp_keys}


def _build_combos():
    combos = []
    kf = list(GRID_FULL.keys())
    for vals in itertools.product(*[GRID_FULL[k] for k in kf]):
        combos.append(dict(zip(kf, vals)))
    if GRID_VECCHIA:
        kv = list(GRID_VECCHIA.keys())
        for vals in itertools.product(*[GRID_VECCHIA[k] for k in kv]):
            combos.append(dict(zip(kv, vals)))
    return combos


def load_data(sim_dir):
    train_df = pd.read_csv(sim_dir / "train_df.csv")
    test_df  = pd.read_csv(sim_dir / "test_df.csv")
    print(f"  Train : {len(train_df)} rows")
    print(f"  Test  : {len(test_df)} rows")
    return train_df, test_df


# >> Tuning (gpb.cv, 3 folds, RMSE)

def run_grid_search(X_train, y_train, coords_train, out_dir):
    combos  = _build_combos()
    n_total = len(combos)
    print(f"  Grid size  : {n_total} combinations  CV folds: {N_CV_FOLDS}  metric: {CV_METRIC}")

    rows = []
    for i, combo in enumerate(combos):
        gp_approx     = combo.get("gp_approx", "full")
        num_neighbors = combo.get("num_neighbors", None)
        tree_params   = _build_tree_params(combo); tree_params["verbose"] = -1
        print(f"  [{i+1:>4}/{n_total}]  {combo}", end="  ")
        try:
            gp_cv   = _make_gp_model(coords_train, gp_approx, num_neighbors)
            data_cv = _make_dataset(X_train, y_train)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                cv_result = gpb.cv(
                    params=tree_params, train_set=data_cv, gp_model=gp_cv,
                    num_boost_round=MAX_BOOST_ROUNDS, early_stopping_rounds=EARLY_STOPPING,
                    nfold=N_CV_FOLDS, metric=CV_METRIC, use_gp_model_for_validation=True,
                    verbose_eval=False, seed=42)
            metric_mean = cv_result[f"{CV_METRIC}-mean"]
            best_round  = int(np.argmin(metric_mean)) + 1       # lower RMSE = better
            best_metric = float(min(metric_mean))
            best_std    = float(cv_result[f"{CV_METRIC}-stdv"][best_round - 1])
            row = {**combo, "best_num_rounds": best_round, "mean_rmse": best_metric, "std_rmse": best_std}
            print(f"RMSE={best_metric:.4f}  rounds={best_round}")
        except Exception as e:
            print(f"FAILED: {e}")
            row = {**combo, "best_num_rounds": np.nan, "mean_rmse": np.nan, "std_rmse": np.nan}
        rows.append(row)

    results_df = (pd.DataFrame(rows).dropna(subset=["mean_rmse"])
                  .sort_values("mean_rmse", ascending=True).reset_index(drop=True))
    results_df.insert(0, "rank", range(1, len(results_df) + 1))
    results_df.to_csv(out_dir / "tuning_grid.csv", index=False)

    best_row        = results_df.iloc[0]
    rmse_cv         = float(best_row["mean_rmse"])
    best_num_rounds = int(best_row["best_num_rounds"])
    all_keys = list(GRID_FULL.keys()) + [k for k in GRID_VECCHIA if k not in GRID_FULL]
    best_combo = {k: best_row[k] for k in all_keys if k in best_row.index and not pd.isna(best_row[k])}
    for k in ["max_depth", "min_data_in_leaf", "num_leaves", "num_neighbors"]:
        if k in best_combo:
            best_combo[k] = int(best_combo[k])
    with open(out_dir / "best_params.json", "w") as f:
        json.dump({**best_combo, "num_boost_round": best_num_rounds}, f, indent=2)
    print(f"  Best CV RMSE: {rmse_cv:.4f}  rounds={best_num_rounds}")
    return results_df, best_combo, best_num_rounds, rmse_cv


# >> Evaluate

def evaluate_best_model(best_combo, best_num_rounds, X_train, y_train, X_test, y_test,
                        train_df, test_df, coords_train, coords_test, out_dir):
    gp_approx     = str(best_combo.get("gp_approx", "full"))
    num_neighbors = best_combo.get("num_neighbors", None)
    tree_params   = _build_tree_params(best_combo); tree_params["verbose"] = -1

    print(f"\n  Fitting best model  gp_approx={gp_approx}  rounds={best_num_rounds}")
    gp_model   = _make_gp_model(coords_train, gp_approx, num_neighbors)
    data_train = _make_dataset(X_train, y_train)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bst = gpb.train(params=tree_params, train_set=data_train,
                        gp_model=gp_model, num_boost_round=best_num_rounds)
    fit_time = time.perf_counter() - t0
    print(f"  Fit time: {fit_time:.1f}s")

    def _pred(X, coords):
        return bst.predict(data=_arr(X), gp_coords_pred=coords, pred_latent=False)["response_mean"]
    def _latent(X, coords):
        return bst.predict(data=_arr(X), gp_coords_pred=coords, pred_latent=True)

    y_pred_train = _pred(X_train, coords_train)
    y_pred_test  = _pred(X_test,  coords_test)
    pred_train_lat = _latent(X_train, coords_train)
    pred_test_lat  = _latent(X_test,  coords_test)
    b_train_hat = pred_train_lat["random_effect_mean"]
    b_test_hat  = pred_test_lat["random_effect_mean"]
    F_train_hat = pred_train_lat["fixed_effect"]
    F_test_hat  = pred_test_lat["fixed_effect"]

    scores_df = pd.DataFrame([score_split("train", _arr(y_train), y_pred_train),
                              score_split("test",  _arr(y_test),  y_pred_test)])
    scores_df.to_csv(out_dir / "scores.csv", index=False)
    print("\n  Prediction scores (regression):")
    print(scores_df.to_string(index=False))

    dissoc_df = pd.DataFrame([
        dissociation_metrics("train", train_df["spatial_effect"].values, b_train_hat,
                             train_df["F_effect"].values, F_train_hat),
        dissociation_metrics("test", test_df["spatial_effect"].values, b_test_hat,
                             test_df["F_effect"].values, F_test_hat)])
    dissoc_df["fit_time"] = round(fit_time, 2)
    dissoc_df.to_csv(out_dir / "dissociation.csv", index=False)
    print("\n  Dissociation:")
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

    grid_1d = np.linspace(0, 1, 50)
    gs1, gs2 = np.meshgrid(grid_1d, grid_1d)
    grid_locs = np.column_stack([gs1.ravel(), gs2.ravel()])
    X_dummy = np.zeros((len(grid_locs), _arr(X_train).shape[1]), dtype=np.float64)
    pred_grid = bst.predict(data=X_dummy, gp_coords_pred=grid_locs, pred_latent=True)
    grid_df = pd.DataFrame(grid_locs, columns=["s1", "s2"])
    grid_df["b_hat"]  = pred_grid["random_effect_mean"]
    grid_df["b_true"] = hotspot_river_effect(grid_locs)
    grid_df.to_csv(out_dir / "grid_spatial_field.csv", index=False)

    return scores_df, dissoc_df, fit_time


def _plot_dissociation(split_name, df, b_hat, F_hat, out_dir):
    s1, s2 = df["s1"].values, df["s2"].values
    b_true = df["spatial_effect"].values
    F_true = df["F_effect"].values
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    def _m(ax, vals, title, cmap):
        sc = ax.scatter(s1, s2, c=vals, cmap=cmap, s=14, alpha=0.85,
                        vmin=np.percentile(vals, 2), vmax=np.percentile(vals, 98))
        plt.colorbar(sc, ax=ax, shrink=0.8); ax.set_title(title, fontsize=10)
        ax.set_xlabel("s1"); ax.set_ylabel("s2"); ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
    _m(axes[0, 0], b_true, "True  b(s)", "RdBu_r")
    _m(axes[0, 1], b_hat, "GPBoost  b_hat(s)", "RdBu_r")
    _m(axes[0, 2], b_true - b_hat, "Residual", "bwr")
    _m(axes[1, 0], F_true, "True  F(X)", "PiYG")
    _m(axes[1, 1], F_hat, "GPBoost  F_hat(X)", "PiYG")
    ax = axes[1, 2]; ax.scatter(F_true, F_hat, s=8, alpha=0.4, color="#d62728")
    lim = [min(F_true.min(), F_hat.min()) - 0.2, max(F_true.max(), F_hat.max()) + 0.2]
    ax.plot(lim, lim, "k--", lw=1.2); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True F(X)"); ax.set_ylabel("GPBoost F_hat(X)"); ax.set_title("True vs Recovered F(X)", fontsize=10)
    r_s = np.corrcoef(b_true, b_hat)[0, 1]; r_f = np.corrcoef(F_true, F_hat)[0, 1]
    fig.suptitle(f"GPBoost  [{split_name}]  [{SIM_NAME}]\nr_spatial={r_s:.3f}   r_cov={r_f:.3f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / f"dissociation_{split_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(out_dir=None):
    if out_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir   = REPO_ROOT / "outputs" / "results" / "gpboost" / f"run_{timestamp}" / SIM_NAME
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n  GPBoost  [{SIM_NAME}]  likelihood={LIKELIHOOD}\n{'='*60}")
    print("\n[1] Loading simulation data ...")
    train_df, test_df = load_data(SIM_DIR)
    feat_cols    = [c for c in train_df.columns if c not in NON_FEAT_COLS]
    X_train, y_train = train_df[feat_cols], train_df[TARGET_COL]
    X_test,  y_test  = test_df[feat_cols],  test_df[TARGET_COL]
    coords_train = train_df[COORD_COLS].values
    coords_test  = test_df[COORD_COLS].values
    print(f"  Feature cols ({len(feat_cols)}): {feat_cols}")

    print("\n[2] Hyperparameter tuning (gpb.cv, RMSE) ...")
    results_df, best_combo, best_num_rounds, rmse_cv = run_grid_search(X_train, y_train, coords_train, out_dir)

    print("\n[3] Evaluating best model ...")
    scores_df, dissoc_df, fit_time = evaluate_best_model(
        best_combo, best_num_rounds, X_train, y_train, X_test, y_test,
        train_df, test_df, coords_train, coords_test, out_dir)

    def _d(split, col):
        row = dissoc_df.loc[dissoc_df.split == split, col]
        return float(row.values[0]) if len(row) else float("nan")
    def _s(split, col):
        row = scores_df.loc[scores_df.split == split, col]
        return float(row.values[0]) if len(row) else float("nan")
    summary = {
        "sim": SIM_NAME, "model": "gpboost", "rmse_cv": round(rmse_cv, 4),
        "rmse_train": _s("train", "rmse"), "rmse_test": _s("test", "rmse"),
        "r2_test": _s("test", "r2"), "mae_test": _s("test", "mae"),
        "r_spatial_train": _d("train", "r_spatial"), "rmse_spatial_train": _d("train", "rmse_spatial"),
        "r_cov_train": _d("train", "r_cov"), "rmse_cov_train": _d("train", "rmse_cov"),
        "r_spatial_test": _d("test", "r_spatial"), "rmse_spatial_test": _d("test", "rmse_spatial"),
        "r_cov_test": _d("test", "r_cov"), "rmse_cov_test": _d("test", "rmse_cov"),
        "fit_time": round(fit_time, 2), "best_num_rounds": best_num_rounds,
        **{f"best_{k}": v for k, v in best_combo.items()},
    }
    print(f"\n  Done. All outputs in {out_dir}")
    return summary


if __name__ == "__main__":
    run()
