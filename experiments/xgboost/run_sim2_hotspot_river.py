"""
run_sim2_hotspot_river.py  (XGBoost)
====================================
XGBoost on Simulation 2 (regression, confounded river covariate), with the coordinates s1, s2 as extra features.

  Tuning    : grid search over PARAM_GRID, 5-fold CV, RMSE.
  Effects   : post-hoc SHAP decomposition; spatial effect = SHAP(s1) + SHAP(s2),
              covariate effect = sum of the other SHAP values.
  Outputs   : metrics, predictions, feature importance and plots in out_dir.

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
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import xgboost as xgb


# >> Simulation settings

SIM_NAME = "sim2_hotspot_river"

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SIM_DIR   = REPO_ROOT / "outputs" / "data" / "sim2_hotspot_river"

COORD_COLS    = ["s1", "s2"]
TARGET_COL    = "y"
NON_FEAT_COLS = ["s1", "s2", "F_effect", "spatial_effect", "m", "y"]

INCLUDE_COORDS = True

PARAM_GRID = {
    "n_estimators"     : [50],
    "learning_rate"    : [0.1],
    "max_depth"        : [3, 5, 7],
    "subsample"        : [0.7, 1.0],
    "colsample_bytree" : [0.7, 1.0],
    "min_child_weight" : [1, 5, 10],
}


# >> Helpers

def _arr(x):
    return x.values if hasattr(x, "values") else x


def _build_feature_matrix(df, base_feat_cols):
    X = df[base_feat_cols].copy()
    if INCLUDE_COORDS:
        X["s1"] = df["s1"].values
        X["s2"] = df["s2"].values
    return X


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


def _nan_dissoc(split_name):
    return {"split": split_name, "r_spatial": float("nan"), "rmse_spatial": float("nan"),
            "r_cov": float("nan"), "rmse_cov": float("nan")}


def load_data(sim_dir):
    train_df = pd.read_csv(sim_dir / "train_df.csv")
    test_df  = pd.read_csv(sim_dir / "test_df.csv")
    print(f"  Train : {len(train_df)} rows")
    print(f"  Test  : {len(test_df)} rows")
    return train_df, test_df


# >> Cross-validation (5-fold, RMSE)

def run_grid_search(X_train, y_train, out_dir, n_splits=5, random_state=42):
    kf      = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    keys    = list(PARAM_GRID.keys())
    combos  = list(itertools.product(*[PARAM_GRID[k] for k in keys]))
    n_total = len(combos)
    print(f"  Grid size : {n_total} combinations x {n_splits} CV folds (RMSE)")
    X_arr, y_arr = _arr(X_train), _arr(y_train)

    rows = []
    for i, values in enumerate(combos):
        params = dict(zip(keys, values))
        print(f"  [{i+1:>4}/{n_total}]  {params}", end="  ")
        fold_rmses = []
        try:
            for tr_idx, val_idx in kf.split(X_arr):
                model = xgb.XGBRegressor(**params, objective="reg:squarederror",
                                         random_state=random_state, verbosity=0)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_arr[tr_idx], y_arr[tr_idx])
                pred = model.predict(X_arr[val_idx])
                fold_rmses.append(np.sqrt(mean_squared_error(y_arr[val_idx], pred)))
            row = {**params, "mean_rmse": float(np.mean(fold_rmses)), "std_rmse": float(np.std(fold_rmses))}
            print(f"RMSE={row['mean_rmse']:.4f}")
        except Exception as e:
            print(f"FAILED: {e}")
            row = {**params, "mean_rmse": np.nan, "std_rmse": np.nan}
        rows.append(row)

    results_df = (pd.DataFrame(rows).dropna(subset=["mean_rmse"])
                  .sort_values("mean_rmse", ascending=True).reset_index(drop=True))
    results_df.insert(0, "rank", range(1, len(results_df) + 1))
    results_df.to_csv(out_dir / "tuning_grid.csv", index=False)

    best_row    = results_df.iloc[0]
    best_params = {k: best_row[k] for k in keys}
    for k in ["n_estimators", "max_depth", "min_child_weight"]:
        if k in best_params:
            best_params[k] = int(best_params[k])
    rmse_cv = float(best_row["mean_rmse"])
    with open(out_dir / "best_params.json", "w") as f:
        json.dump(best_params, f, indent=2)
    print(f"  Best CV RMSE: {rmse_cv:.4f} +/- {best_row['std_rmse']:.4f}")
    return results_df, best_params, rmse_cv


# >> SHAP decomposition

def compute_shap_effects(model, X):
    try:
        contribs  = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        shap_vals = contribs[:, :-1]  # drop bias
        return shap_vals[:, -2] + shap_vals[:, -1], shap_vals[:, :-2].sum(axis=1)
    except Exception as e:
        print(f"  [warn] SHAP failed: {e}")
        return None, None


# >> Evaluate

def evaluate_best_model(best_params, X_train, y_train, X_test, y_test,
                        train_df, test_df, out_dir, random_state=42):
    print(f"\n  Fitting best model  INCLUDE_COORDS={INCLUDE_COORDS}  params={best_params}")
    model = xgb.XGBRegressor(**best_params, objective="reg:squarederror",
                             random_state=random_state, verbosity=0)
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(_arr(X_train), _arr(y_train))
    fit_time = time.perf_counter() - t0
    print(f"  Fit time: {fit_time:.1f}s")

    y_pred_train = model.predict(_arr(X_train))
    y_pred_test  = model.predict(_arr(X_test))
    scores_df = pd.DataFrame([score_split("train", _arr(y_train), y_pred_train),
                              score_split("test",  _arr(y_test),  y_pred_test)])
    scores_df.to_csv(out_dir / "scores.csv", index=False)
    print("\n  Prediction scores (regression):")
    print(scores_df.to_string(index=False))

    if INCLUDE_COORDS:
        b_train_hat, F_train_hat = compute_shap_effects(model, X_train)
        b_test_hat,  F_test_hat  = compute_shap_effects(model, X_test)
    else:
        b_train_hat = b_test_hat = F_train_hat = F_test_hat = None

    def _row(split, df, b_hat, F_hat):
        if b_hat is None or F_hat is None:
            return _nan_dissoc(split)
        return dissociation_metrics(split, df["spatial_effect"].values, b_hat,
                                    df["F_effect"].values, F_hat)
    dissoc_df = pd.DataFrame([_row("train", train_df, b_train_hat, F_train_hat),
                              _row("test",  test_df,  b_test_hat,  F_test_hat)])
    dissoc_df["fit_time"] = round(fit_time, 2)
    dissoc_df.to_csv(out_dir / "dissociation.csv", index=False)
    print("\n  Dissociation (SHAP proxy):")
    print(dissoc_df[["split", "r_spatial", "rmse_spatial", "r_cov", "rmse_cov"]].to_string(index=False))

    def _pred_row(df, y_hat, b_hat, F_hat, split):
        out = df[COORD_COLS + ["y"]].copy()
        out["F_true"] = df["F_effect"].values
        out["b_true"] = df["spatial_effect"].values
        out["y_hat"]  = y_hat
        out["b_hat"]  = b_hat if b_hat is not None else np.nan
        out["F_hat"]  = F_hat if F_hat is not None else np.nan
        out["split"]  = split
        return out
    pd.concat([_pred_row(train_df, y_pred_train, b_train_hat, F_train_hat, "train"),
               _pred_row(test_df,  y_pred_test,  b_test_hat,  F_test_hat,  "test")],
              ignore_index=True).to_csv(out_dir / "predictions.csv", index=False)

    if INCLUDE_COORDS and b_train_hat is not None:
        _plot_shap_spatial("test", test_df, b_test_hat, F_test_hat, out_dir)
    _plot_feature_importance(model, X_train, out_dir)
    return scores_df, dissoc_df, fit_time


def _plot_shap_spatial(split_name, df, b_hat, F_hat, out_dir):
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
    _m(axes[0, 1], b_hat, "SHAP spatial  s1+s2", "RdBu_r")
    _m(axes[0, 2], b_true - b_hat, "Residual", "bwr")
    _m(axes[1, 0], F_true, "True  F(X)", "PiYG")
    _m(axes[1, 1], F_hat, "SHAP covariate", "PiYG")
    ax = axes[1, 2]; ax.scatter(F_true, F_hat, s=8, alpha=0.4, color="#d62728")
    lim = [min(F_true.min(), F_hat.min()) - 0.2, max(F_true.max(), F_hat.max()) + 0.2]
    ax.plot(lim, lim, "k--", lw=1.2); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("True F(X)"); ax.set_ylabel("SHAP covariate"); ax.set_title("True vs SHAP F(X)", fontsize=10)
    r_s = np.corrcoef(b_true, b_hat)[0, 1]; r_f = np.corrcoef(F_true, F_hat)[0, 1]
    fig.suptitle(f"XGBoost  [{split_name}]  SHAP dissociation  [{SIM_NAME}]\nr_spatial={r_s:.3f}   r_cov={r_f:.3f}", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / f"dissociation_{split_name}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_feature_importance(model, X_train, out_dir):
    importance = model.get_booster().get_score(importance_type="gain")
    if not importance:
        return
    feat_names = list(X_train.columns) if hasattr(X_train, "columns") else [f"f{i}" for i in range(X_train.shape[1])]
    scores = [importance.get(f, 0.0) for f in feat_names]
    fig, ax = plt.subplots(figsize=(8, max(3, len(feat_names) * 0.35)))
    colors = ["#d62728" if n in ("s1", "s2") else "#1f77b4" for n in feat_names]
    ax.barh(feat_names, scores, color=colors); ax.set_xlabel("Gain")
    ax.set_title(f"XGBoost feature importance (gain)  [{SIM_NAME}]")
    fig.tight_layout()
    fig.savefig(out_dir / "feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(out_dir=None):
    coord_label = "coords" if INCLUDE_COORDS else "no_coords"
    if out_dir is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = REPO_ROOT / "outputs" / "results" / "xgboost" / f"run_{timestamp}" / f"{SIM_NAME}_{coord_label}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}\n  XGBoost  [{SIM_NAME}]  [{coord_label}]  [regression]\n{'='*60}")
    print("\n[1] Loading simulation data ...")
    train_df, test_df = load_data(SIM_DIR)
    base_feat_cols = [c for c in train_df.columns if c not in NON_FEAT_COLS]
    X_train = _build_feature_matrix(train_df, base_feat_cols)
    X_test  = _build_feature_matrix(test_df,  base_feat_cols)
    y_train, y_test = train_df[TARGET_COL], test_df[TARGET_COL]
    print(f"  Feature cols ({len(X_train.columns)}): {list(X_train.columns)}")

    print("\n[2] Hyperparameter tuning (KFold CV, RMSE) ...")
    results_df, best_params, rmse_cv = run_grid_search(X_train, y_train, out_dir)

    print("\n[3] Evaluating best model ...")
    scores_df, dissoc_df, fit_time = evaluate_best_model(
        best_params, X_train, y_train, X_test, y_test, train_df, test_df, out_dir)

    def _d(split, col):
        row = dissoc_df.loc[dissoc_df.split == split, col]
        return float(row.values[0]) if len(row) else float("nan")
    def _s(split, col):
        row = scores_df.loc[scores_df.split == split, col]
        return float(row.values[0]) if len(row) else float("nan")
    summary = {
        "sim": SIM_NAME, "model": f"xgboost_{coord_label}", "rmse_cv": round(rmse_cv, 4),
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
