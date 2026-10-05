"""
methods.py
Fit and scoring functions for the three Python methods of the house-price
application (SAMBoost, GPBoost, XGBoost), called by run_python_methods.py.

For each test year t (folds in data_folds.py):
    tuning    : fit each grid combination on inner_train (years < t-1) and
                select the one with the lowest MSE on val (year t-1)
    final fit : refit the selected configuration on train (years < t) and
                score it on test (year t)

Each fit writes to its output folder: scores.csv (RMSE, MAE, R2 and fit time on
train and test), predictions.csv (prediction y_hat, covariate part F_hat and
spatial part b_hat for every sale), best_params.json and, after a grid search,
tuning_grid.csv.

OPTIMAL_PARAMS holds the configuration selected for each method, variant and
test year, so the results can be reproduced without the grid search.
"""

import io
import sys
import time
import json
import itertools
import warnings
import contextlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from real_data.house.data_folds import (
    load_house_sigrist, get_fold, TEST_YEARS, COORD_COLS, TARGET_COL,
    BOOSTING_FEAT_COLS, CATEGORICAL_COLS,
)
from samboost.model_continuous import SAMBoostContinuous
from utils.mesh_building import (
    build_inla_mesh, build_A_matrix, assemble_spde_matrices,
)


# >> Settings

# Grid search switches (run_python_methods.py turns them off unless --grid is given)
GRID_SEARCH_GPBOOST   = True
GRID_SEARCH_XGBOOST   = True
GRID_SEARCH_SAMBOOST = True

OUT_BASE = REPO_ROOT / "outputs" / "results" / "house_price"

# XGBoost grid search: up to MAX_BOOST_ROUNDS trees, early stopping after
# EARLY_STOPPING rounds without improvement on val
MAX_BOOST_ROUNDS = 2000
EARLY_STOPPING    = 10

# >> GPBoost
# The GPBoost settings start from Sigrist (2022). With the help of Fabio Sigrist,
# some changes were made to them that improve on the results of the paper.

# Default parameters, used only without grid search and without fixed parameters
GPBOOST_PARAMS = dict(
    learning_rate=0.1, max_depth=5, min_data_in_leaf=20, num_boost_round=100,
)

# Grid: 27 combinations. For each one the number of rounds is chosen by early
# stopping (up to GPBOOST_NROUNDS_MAX), with the GP model used for validation;
# the combination with the lowest val MSE is selected.
GPBOOST_GRID_SIGRIST = dict(
    learning_rate=[0.01, 0.05, 0.1],
    min_data_in_leaf=[1, 10, 100],
    max_depth=[1, 5, 10],
)
GPBOOST_NUM_LEAVES   = 1024   # 2^10
GPBOOST_NROUNDS_MAX  = 1000   # early-stopping ceiling
GPBOOST_EARLY_STOP   = 10     # early_stopping_rounds

GPBOOST_COV_FUNCTION = "exponential"

# Vecchia approximation (an exact GP is O(N^3), too slow for 11k-21k training
# sales): 20 neighbours for training and 40 for prediction, instead of 50 / 500
# in the original setup, which reduces time and memory without losing accuracy.
# Prediction conditions on the observed data only.
GPBOOST_GP_APPROX          = "vecchia"
GPBOOST_NUM_NEIGHBORS      = 20
GPBOOST_VECCHIA_ORDERING   = "random"
GPBOOST_VECCHIA_PRED_TYPE  = "order_obs_first_cond_obs_only"
GPBOOST_NUM_NEIGHBORS_PRED = 40

# >> XGBoost
# Default parameters, used only without grid search and without fixed parameters
XGBOOST_PARAMS = dict(
    n_estimators=200, learning_rate=0.05, max_depth=5,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
)
# Grid: 108 combinations; the number of trees is chosen by early stopping on val
XGBOOST_GRID = dict(
    learning_rate=[0.01, 0.03, 0.05],
    max_depth=[5, 7, 9],
    subsample=[0.7, 1.0],
    colsample_bytree=[0.7, 1.0],
    min_child_weight=[1, 5, 10],
)

# >> SAMBoost
# Grid: 36 combinations (kappa x w_mode x min_samples_leaf x n_estimators), the
# other arguments fixed. Used with the mesh of run_samboost_final (about 9,250 nodes).
SAMBOOST_GRID = dict(
    kappa=[30.0, 40.0, 50.0], w_mode=["uniform", "no_geo"],
    min_samples_leaf=[1, 5, 10], n_estimators=[400, 800],
    learning_rate=[0.05], max_depth_stage1=[3], alpha=[0.2], n_freeze=[10],
    gamma=[0], ratio=[2.0], n_levels=[1], alpha_start_scale=[1.0],
)

# >> Selected hyperparameters
# Configuration selected by the grid search for each method, variant and test
# year. run_python_methods.py fits these directly unless --grid is given.
#   samboost: every SAMBoost argument
#   gpboost  : includes num_boost_round (chosen by early stopping)
#   xgboost  : includes n_estimators (chosen by early stopping)
OPTIMAL_PARAMS = {
    "samboost": {
        "ex": {
            1996: dict(alpha=0.2, kappa=30.0, max_depth_stage1=3, min_samples_leaf=5,  w_mode="uniform", learning_rate=0.05, n_estimators=400, n_freeze=10, gamma=0, ratio=2.0, n_levels=1, alpha_start_scale=1.0),
            1997: dict(alpha=0.2, kappa=30.0, max_depth_stage1=3, min_samples_leaf=10, w_mode="no_geo",  learning_rate=0.05, n_estimators=800, n_freeze=10, gamma=0, ratio=2.0, n_levels=1, alpha_start_scale=1.0),
            1998: dict(alpha=0.2, kappa=30.0, max_depth_stage1=3, min_samples_leaf=5,  w_mode="no_geo",  learning_rate=0.05, n_estimators=800, n_freeze=10, gamma=0, ratio=2.0, n_levels=1, alpha_start_scale=1.0),
        },
        "geo": {
            1996: dict(alpha=0.2, kappa=40.0, max_depth_stage1=3, min_samples_leaf=1,  w_mode="uniform", learning_rate=0.05, n_estimators=800, n_freeze=10, gamma=0, ratio=2.0, n_levels=1, alpha_start_scale=1.0),
            1997: dict(alpha=0.2, kappa=30.0, max_depth_stage1=3, min_samples_leaf=10, w_mode="uniform", learning_rate=0.05, n_estimators=800, n_freeze=10, gamma=0, ratio=2.0, n_levels=1, alpha_start_scale=1.0),
            1998: dict(alpha=0.2, kappa=30.0, max_depth_stage1=3, min_samples_leaf=5,  w_mode="no_geo",  learning_rate=0.05, n_estimators=800, n_freeze=10, gamma=0, ratio=2.0, n_levels=1, alpha_start_scale=1.0),
        },
    },
    "gpboost": {
        "ex": {
            1996: dict(learning_rate=0.01, min_data_in_leaf=1,   max_depth=1, num_leaves=1024, verbose=-1, num_boost_round=243),
            1997: dict(learning_rate=0.01, min_data_in_leaf=10,  max_depth=1, num_leaves=1024, verbose=-1, num_boost_round=358),
            1998: dict(learning_rate=0.01, min_data_in_leaf=1,   max_depth=1, num_leaves=1024, verbose=-1, num_boost_round=412),
        },
        "geo": {
            1996: dict(learning_rate=0.01, min_data_in_leaf=1,   max_depth=1, num_leaves=1024, verbose=-1, num_boost_round=243),
            1997: dict(learning_rate=0.01, min_data_in_leaf=10,  max_depth=1, num_leaves=1024, verbose=-1, num_boost_round=358),
            1998: dict(learning_rate=0.01, min_data_in_leaf=100, max_depth=5, num_leaves=1024, verbose=-1, num_boost_round=67),
        },
    },
    "xgboost": {
        "ex": {
            1996: dict(learning_rate=0.01, max_depth=5, subsample=1.0, colsample_bytree=0.7, min_child_weight=1,  n_estimators=787),
            1997: dict(learning_rate=0.05, max_depth=5, subsample=0.7, colsample_bytree=1.0, min_child_weight=10, n_estimators=390),
            1998: dict(learning_rate=0.01, max_depth=7, subsample=1.0, colsample_bytree=0.7, min_child_weight=10, n_estimators=1031),
        },
        "geo": {
            1996: dict(learning_rate=0.05, max_depth=7, subsample=1.0, colsample_bytree=0.7, min_child_weight=5,  n_estimators=314),
            1997: dict(learning_rate=0.01, max_depth=7, subsample=0.7, colsample_bytree=0.7, min_child_weight=10, n_estimators=1372),
            1998: dict(learning_rate=0.01, max_depth=9, subsample=0.7, colsample_bytree=0.7, min_child_weight=5,  n_estimators=1006),
        },
    },
}


# >> Helpers

def score_split_reg(name, y_true, y_pred):
    return {
        "split" : name,
        "n"     : int(len(y_true)),
        "rmse"  : round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "mae"   : round(float(mean_absolute_error(y_true, y_pred)), 4),
        "r2"    : round(float(r2_score(y_true, y_pred)), 4),
    }


def save_results(out_dir, scores_df, best_params=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(out_dir / "scores.csv", index=False)
    if best_params:
        with open(out_dir / "best_params.json", "w") as f:
            json.dump(best_params, f, indent=2)


def _samboost_design(df, numeric_cols):
    """
    Numeric design matrix for SAMBoost (sklearn trees need numeric input).
    Numeric covariates kept as-is; stories/wall/garage one-hot encoded.
    Category levels are fixed on the full df at load time, so get_dummies
    yields identical columns for any train/test subset.
    """
    X_num = df[numeric_cols].to_numpy(dtype=np.float32)
    X_cat = pd.get_dummies(df[CATEGORICAL_COLS]).to_numpy(dtype=np.float32)
    return np.hstack([X_num, X_cat])


# >> SAMBoost

def run_samboost_final(fold, feat_cols, out_dir, all_coords,
                        include_coords_in_trees=False, grid=None,
                        mesh_params=(0.002, 0.005, 0.02), tree_backend="sklearn",
                        xgb_nthread=0, fixed_params=None):
    """
    SAMBoost for one test year: tune on inner_train / val (unless
    fixed_params is given), refit on train and score on test.

    The coordinates are scaled to [0,1]^2 on the bounding box of the full
    dataset (all_coords), and one SPDE mesh is built over that domain and used
    for every test year. mesh_params = (cutoff, max_edge_inner, max_edge_outer)
    on the scaled coordinates; the default gives about 9,250 nodes.

    include_coords_in_trees: False = ex variant (coordinates only in the spatial
    field), True = geo variant (coordinates also tree features).
    tree_backend: "sklearn" (default, used for the paper) fits
    samboost.model_continuous; "xgboost" fits samboost.model_extended with XGBoost
    trees, which is faster.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = grid if grid is not None else SAMBOOST_GRID

    # Scale the coordinates on the full dataset and build one mesh over the whole domain
    lo   = all_coords.min(axis=0)
    span = all_coords.max(axis=0) - lo
    span = np.where(span == 0, 1.0, span)
    g = lambda c: (c - lo) / span
    cut, edge_inner, edge_outer = mesh_params
    mesh = build_inla_mesh(
        obs_locs=g(all_coords), cutoff=cut,
        max_edge_inner=edge_inner, max_edge_outer=edge_outer, random_state=42,
    )
    nodes = mesh["mesh_nodes"]
    G, C = assemble_spde_matrices(nodes)

    if tree_backend not in ("sklearn", "xgboost"):
        raise ValueError(f"tree_backend must be 'sklearn' or 'xgboost', got {tree_backend!r}")
    if tree_backend == "xgboost":
        from samboost.model_extended import SAMBoost as _ExtendedModel

    def _make_model(A_mat, params):
        if tree_backend == "xgboost":
            # min_samples_leaf is passed as xgboost's min_child_weight (the same
            # leaf-size limit for squared error, where every hessian is 1)
            q = {k: v for k, v in params.items() if k != "min_samples_leaf"}
            # xgb_nthread: threads for xgboost tree building, separate from the
            # threads of the sparse solver (0 = all cores)
            return _ExtendedModel(A=A_mat, G=G, C=C, likelihood="gaussian",
                              tree_backend="xgboost", xgb_nthread=xgb_nthread,
                              xgb_min_child_weight=float(params["min_samples_leaf"]), **q)
        return SAMBoostContinuous(A=A_mat, G=G, C=C, **params)

    numeric_cols = [c for c in feat_cols if c not in CATEGORICAL_COLS]
    if include_coords_in_trees:
        numeric_cols = numeric_cols + [c for c in COORD_COLS if c not in numeric_cols]

    train, test         = fold["train"], fold["test"]
    inner_train, val    = fold["inner_train"], fold["val"]

    if fixed_params is not None:
        # No grid search: fit the given parameters directly
        best_params = dict(fixed_params)
        print(f"    SAMBoost FIXED (no grid) | mesh {len(nodes)} nodes | "
              f"{'geo' if include_coords_in_trees else 'ex'} | params={best_params}")
    else:
        # Tuning: fit each combination on inner_train, select on val MSE
        X_inner = _samboost_design(inner_train, numeric_cols); y_inner = inner_train[TARGET_COL].values
        X_val   = _samboost_design(val, numeric_cols);          y_val  = val[TARGET_COL].values
        c_inner = g(inner_train[COORD_COLS].values); c_val = g(val[COORD_COLS].values)
        A_inner = build_A_matrix(c_inner, nodes);    A_val = build_A_matrix(c_val, nodes)

        keys   = list(grid.keys())
        combos = list(itertools.product(*[grid[k] for k in keys]))
        print(f"    SAMBoost FINAL: {len(combos)} combos | mesh {len(nodes)} nodes | "
              f"{'geo' if include_coords_in_trees else 'ex'} (fit inner_train, MSE on val)")

        rows, best_mse, best_params = [], np.inf, None
        for i, vals in enumerate(combos):
            # Do not delete the model or call gc.collect() between combinations:
            # with the xgboost backend this crashes the process on the next fit.
            p = dict(zip(keys, vals))
            try:
                m = _make_model(A_inner, p)
                with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
                    warnings.simplefilter("ignore")
                    m.fit(X_inner, y_inner, coords=c_inner)
                mse = float(mean_squared_error(y_val, m.predict(X_val, A_val)))
                rows.append({**p, "val_mse": mse})
                if mse < best_mse:
                    best_mse, best_params = mse, p
                print(f"    [{i+1:>2}/{len(combos)}]  {p}  val_mse={mse:.5f}")
            except Exception as e:
                print(f"    [{i+1:>2}/{len(combos)}]  {p}  FAILED: {e}")
        pd.DataFrame(rows).to_csv(out_dir / "tuning_grid.csv", index=False)
        print(f"    best val MSE {best_mse:.5f}  params={best_params}")

    # Refit the selected configuration on train, score on test
    X_train = _samboost_design(train, numeric_cols); y_train = train[TARGET_COL].values
    X_test  = _samboost_design(test,  numeric_cols); y_test  = test[TARGET_COL].values
    c_train = g(train[COORD_COLS].values); c_test = g(test[COORD_COLS].values)
    A_train = build_A_matrix(c_train, nodes); A_test = build_A_matrix(c_test, nodes)

    t0 = time.perf_counter()
    model = _make_model(A_train, best_params)
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train, coords=c_train)
    fit_time = time.perf_counter() - t0

    p_train = model.predict(X_train, A_train)
    p_test  = model.predict(X_test,  A_test)
    scores_df = pd.DataFrame([
        score_split_reg("train", y_train, p_train),
        score_split_reg("test",  y_test,  p_test),
    ])
    scores_df["fit_time"] = round(fit_time, 2)
    save_results(out_dir, scores_df, best_params)

    b_train = model.spatial_effect(A_train);  b_test  = model.spatial_effect(A_test)
    F_train = model.covariate_effect(X_train); F_test = model.covariate_effect(X_test)
    pred_df = pd.concat([
        train[COORD_COLS + [TARGET_COL]].assign(
            y_hat=p_train, F_hat=F_train, b_hat=b_train, split="train"),
        test[COORD_COLS + [TARGET_COL]].assign(
            y_hat=p_test, F_hat=F_test, b_hat=b_test, split="test"),
    ], ignore_index=True)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    print(f"    test RMSE={scores_df.iloc[1]['rmse']:.4f}  [{fit_time:.0f}s]  -> {out_dir}")
    return scores_df


# >> GPBoost

def run_gpboost(fold, feat_cols, out_dir, include_coords_in_F=True, fixed_params=None):
    import gpboost as gpb

    out_dir.mkdir(parents=True, exist_ok=True)
    train, test = fold["train"], fold["test"]
    inner_train, val = fold["inner_train"], fold["val"]

    # include_coords_in_F: True = geo variant (coordinates also tree features),
    # False = ex variant (coordinates only in the GP). Kept as a DataFrame so
    # stories / wall / garage stay categorical and GPBoost splits on them directly.
    tree_cols = feat_cols + COORD_COLS if include_coords_in_F else feat_cols
    X_train, y_train = train[tree_cols], train[TARGET_COL].values
    X_test,  y_test  = test[tree_cols],  test[TARGET_COL].values
    coords_train = train[COORD_COLS].values
    coords_test  = test[COORD_COLS].values

    def _make_gp(coords):
        return gpb.GPModel(
            gp_coords=coords, cov_function=GPBOOST_COV_FUNCTION,
            likelihood="gaussian",
            gp_approx=GPBOOST_GP_APPROX,
            num_neighbors=GPBOOST_NUM_NEIGHBORS,
            vecchia_ordering=GPBOOST_VECCHIA_ORDERING,
        )

    def _predict(bst, gp_model, X, coords, **kwargs):
        # The Vecchia prediction settings must be set on the GP model
        # (Booster.predict no longer accepts them)
        gp_model.set_prediction_data(
            vecchia_pred_type=GPBOOST_VECCHIA_PRED_TYPE,
            num_neighbors_pred=GPBOOST_NUM_NEIGHBORS_PRED,
        )
        return bst.predict(data=X, gp_coords_pred=coords, **kwargs)

    if GRID_SEARCH_GPBOOST:
        # For each combination, early stopping on val (GP model used for
        # validation) sets the number of rounds; selection on val MSE.
        X_inner, y_inner = inner_train[tree_cols], inner_train[TARGET_COL].values
        X_val,   y_val   = val[tree_cols],         val[TARGET_COL].values
        coords_inner = inner_train[COORD_COLS].values
        coords_val   = val[COORD_COLS].values

        keys   = list(GPBOOST_GRID_SIGRIST.keys())
        combos = list(itertools.product(*[GPBOOST_GRID_SIGRIST[k] for k in keys]))
        print(f"    Grid (sigrist): {len(combos)} combos "
              f"(early stop to {GPBOOST_NROUNDS_MAX}, num_leaves={GPBOOST_NUM_LEAVES}, "
              f"GP-aware val)")

        rows = []
        best_mse, best_combo, best_rounds = np.inf, None, 100
        for i, vals in enumerate(combos):
            combo = dict(zip(keys, vals))
            params = {**combo, "objective": "regression",
                      "num_leaves": GPBOOST_NUM_LEAVES, "verbose": -1}
            try:
                gp_cv   = _make_gp(coords_inner)
                gp_cv.set_prediction_data(gp_coords_pred=coords_val)
                data_cv = gpb.Dataset(data=X_inner, label=y_inner)
                val_cv  = gpb.Dataset(data=X_val, label=y_val, reference=data_cv)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    bst_cv = gpb.train(
                        params=params, train_set=data_cv, gp_model=gp_cv,
                        num_boost_round=GPBOOST_NROUNDS_MAX, valid_sets=[val_cv],
                        use_gp_model_for_validation=True,
                        early_stopping_rounds=GPBOOST_EARLY_STOP,
                    )
                n_best   = bst_cv.best_iteration or GPBOOST_NROUNDS_MAX
                pred_val = _predict(bst_cv, gp_cv, X_val, coords_val,
                                    pred_latent=False, num_iteration=n_best)["response_mean"]
                mse      = float(mean_squared_error(y_val, pred_val))
                rows.append({**combo, "best_iteration": n_best, "val_mse": mse})
                if mse < best_mse:
                    best_mse    = mse
                    best_combo  = {**combo, "num_leaves": GPBOOST_NUM_LEAVES}
                    best_rounds = n_best
                print(f"    [{i+1:>3}/{len(combos)}]  {combo}  best_it={n_best}  val_mse={mse:.4f}")
            except Exception as e:
                print(f"    [{i+1:>3}/{len(combos)}]  {combo}  FAILED: {e}")

        out_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out_dir / "tuning_grid.csv", index=False)
        if best_combo is None:
            raise RuntimeError(
                f"All {len(combos)} GPBoost (sigrist) combos failed (see tuning_grid.csv). "
                f"Common cause: xgboost imported in the same process before gpboost."
            )
        tree_params, num_rounds = best_combo, best_rounds
        print(f"    Best val MSE: {best_mse:.4f}  rounds={num_rounds}  params={tree_params}")

    else:
        # No grid search: fit fixed_params if given, else GPBOOST_PARAMS
        src = fixed_params if fixed_params is not None else GPBOOST_PARAMS
        tree_params = {k: v for k, v in src.items() if k != "num_boost_round"}
        num_rounds  = src.get("num_boost_round", MAX_BOOST_ROUNDS)
        if fixed_params is not None:
            print(f"    GPBoost FIXED (no grid): rounds={num_rounds}  params={tree_params}")

    tree_params = {**tree_params, "verbose": -1}
    gp_model = _make_gp(coords_train)
    data     = gpb.Dataset(data=X_train, label=y_train)

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bst = gpb.train(params=tree_params, train_set=data, gp_model=gp_model,
                        num_boost_round=num_rounds)
    fit_time = time.perf_counter() - t0

    # Scores use the response prediction, as in the grid search; the latent
    # prediction only gives the covariate part F_hat and the spatial part b_hat
    p_train = _predict(bst, gp_model, X_train, coords_train,
                       pred_latent=False)["response_mean"]
    p_test  = _predict(bst, gp_model, X_test,  coords_test,
                       pred_latent=False)["response_mean"]
    lat_train = _predict(bst, gp_model, X_train, coords_train, pred_latent=True)
    lat_test  = _predict(bst, gp_model, X_test,  coords_test,  pred_latent=True)

    scores_df = pd.DataFrame([
        score_split_reg("train", y_train, p_train),
        score_split_reg("test",  y_test,  p_test),
    ])
    scores_df["fit_time"] = round(fit_time, 2)
    save_results(out_dir, scores_df, {**tree_params, "num_boost_round": num_rounds})

    pred_df = pd.concat([
        train[COORD_COLS + [TARGET_COL]].assign(
            y_hat=p_train, F_hat=lat_train["fixed_effect"],
            b_hat=lat_train["random_effect_mean"], split="train"),
        test[COORD_COLS + [TARGET_COL]].assign(
            y_hat=p_test, F_hat=lat_test["fixed_effect"],
            b_hat=lat_test["random_effect_mean"], split="test"),
    ], ignore_index=True)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    return scores_df


# >> XGBoost

def run_xgboost(fold, feat_cols, out_dir, include_coords_in_trees=True, fixed_params=None):
    import xgboost as xgb

    out_dir.mkdir(parents=True, exist_ok=True)
    train, test = fold["train"], fold["test"]
    inner_train, val = fold["inner_train"], fold["val"]

    # include_coords_in_trees: True = geo variant (coordinates as features),
    # False = ex variant (no coordinates). Kept as a DataFrame so stories / wall /
    # garage stay categorical and XGBoost splits on them directly.
    all_cols = feat_cols + COORD_COLS if include_coords_in_trees else list(feat_cols)
    X_train = train[all_cols]
    y_train = train[TARGET_COL].values
    X_test  = test[all_cols]
    y_test  = test[TARGET_COL].values

    if GRID_SEARCH_XGBOOST:
        X_inner = inner_train[all_cols]
        y_inner = inner_train[TARGET_COL].values
        X_val   = val[all_cols]
        y_val   = val[TARGET_COL].values

        keys    = list(XGBOOST_GRID.keys())
        combos  = list(itertools.product(*[XGBOOST_GRID[k] for k in keys]))
        print(f"    Grid: {len(combos)} combos (tune on inner_train, score on val)")

        rows = []
        best_mse, best_params, best_rounds = np.inf, None, 200
        for i, vals in enumerate(combos):
            p = dict(zip(keys, vals))
            try:
                model = xgb.XGBRegressor(
                    **p, n_estimators=MAX_BOOST_ROUNDS,
                    objective="reg:squarederror", eval_metric="rmse",
                    early_stopping_rounds=EARLY_STOPPING,
                    tree_method="hist", enable_categorical=True,
                    random_state=42, verbosity=0,
                )
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model.fit(X_inner, y_inner, eval_set=[(X_val, y_val)], verbose=False)
                pred_val = model.predict(X_val)
                mse      = float(mean_squared_error(y_val, pred_val))
                round_n  = model.best_iteration + 1 if model.best_iteration is not None else MAX_BOOST_ROUNDS
                rows.append({**p, "val_mse": mse, "num_rounds": round_n})
                if mse < best_mse:
                    best_mse, best_params, best_rounds = mse, p, round_n
                print(f"    [{i+1:>3}/{len(combos)}]  {p}  val_mse={mse:.4f}")
            except Exception as e:
                print(f"    [{i+1:>3}/{len(combos)}]  {p}  FAILED: {e}")

        pd.DataFrame(rows).to_csv(out_dir / "tuning_grid.csv", index=False)
        xgb_params, n_estimators = best_params, best_rounds
        print(f"    Best val MSE: {best_mse:.4f}  rounds={n_estimators}  params={xgb_params}")
    else:
        # No grid search: fit fixed_params if given, else XGBOOST_PARAMS
        src = fixed_params if fixed_params is not None else XGBOOST_PARAMS
        xgb_params  = {k: v for k, v in src.items() if k != "n_estimators"}
        n_estimators = src["n_estimators"]
        if fixed_params is not None:
            print(f"    XGBoost FIXED (no grid): n_estimators={n_estimators}  params={xgb_params}")

    model = xgb.XGBRegressor(
        **xgb_params, n_estimators=n_estimators,
        objective="reg:squarederror", eval_metric="rmse",
        tree_method="hist", enable_categorical=True,
        random_state=42, verbosity=0,
    )
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    p_train = model.predict(X_train)
    p_test  = model.predict(X_test)

    scores_df = pd.DataFrame([
        score_split_reg("train", y_train, p_train),
        score_split_reg("test",  y_test,  p_test),
    ])
    scores_df["fit_time"] = round(fit_time, 2)
    save_results(out_dir, scores_df, {**xgb_params, "n_estimators": n_estimators})

    # SHAP split: b_hat = contribution of the last two features, F_hat = the rest.
    # Meaningful for the geo variant only, where the last two features are the
    # coordinates.
    b_train = F_train = b_test = F_test = None
    try:
        c_train = model.get_booster().predict(
            xgb.DMatrix(X_train, enable_categorical=True), pred_contribs=True)
        s_train = c_train[:, :-1]
        b_train = s_train[:, -2] + s_train[:, -1]
        F_train = s_train[:, :-2].sum(axis=1)
        c_test  = model.get_booster().predict(
            xgb.DMatrix(X_test, enable_categorical=True), pred_contribs=True)
        s_test  = c_test[:, :-1]
        b_test  = s_test[:, -2] + s_test[:, -1]
        F_test  = s_test[:, :-2].sum(axis=1)
    except Exception as e:
        print(f"    [warn] SHAP failed: {e}")

    pred_df = pd.concat([
        train[COORD_COLS + [TARGET_COL]].assign(
            y_hat=p_train, F_hat=F_train, b_hat=b_train, split="train"),
        test[COORD_COLS + [TARGET_COL]].assign(
            y_hat=p_test, F_hat=F_test, b_hat=b_test, split="test"),
    ], ignore_index=True)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)

    return scores_df
