"""
run_multi_seed.py
=================
Runs the simulation study for the Python methods (SAMBoost, GPBoost, XGBoost)
over several seeds. For each seed: generate the data with the simulation
generator, run each method's experiment runner (experiments/<method>/), and
collect the test metrics.

Tuning modes (one per method):
  "first_seed" : grid search on seed 1, selected values reused for seeds 2-30
                 (used for the paper)
  "per_seed"   : grid search on every seed
  "fixed"      : the values in *_FIXED_PARAMS for every seed

Output: outputs/results/multi_seed/<sim>/run_<timestamp>/ with one folder per
seed and method, all_seeds_raw.csv and summary.csv.

Usage (from the repository root, after setting the switches below):
    python run_pipeline/run_multi_seed.py
"""

import sys
import json
import time
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd


# >> Settings

# Which simulations to run (set False to skip)
RUN_SIM1                = False   # Simulation 1 (classification, no confounding)
RUN_SIM2        = False   # Simulation 2 (regression, confounded river covariate)
RUN_SIM2_ABLATION = False   # ablation (Simulation 2 with a nearly additive covariate effect)

# Seeds
SEEDS = list(range(1, 31))  # 30 seeds

# Which methods to run
RUN_SAMBOOST = True
RUN_GPBOOST   = True
RUN_XGBOOST   = True

# SAMBoost tuning mode (see module docstring). "first_seed" for the paper,
# as for GPBoost and XGBoost below.
TUNING_MODE = "first_seed"  # "fixed", "first_seed", "per_seed"

# Fixed params (used when TUNING_MODE = "fixed")
SAMBOOST_FIXED_PARAMS = {
    "alpha": 0.9, "n_estimators": 50, "learning_rate": 0.1,
    "max_depth_stage1": 3, "min_samples_leaf": 20, "gamma": 0,
    "n_freeze": 5, "w_mode": "full", "kappa": 10.0, "ratio": 2.0,
    "n_levels": 1, "alpha_start_scale": 1.0,
}

# Output
MULTI_SEED_DIR = REPO_ROOT / "outputs" / "results" / "multi_seed"


# >> Simulations and runners

SIM_CONFIG = {
    "sim1_wave": {
        "gen_module":  "simulations.generate_sim1_wave",
        "data_subdir": "sim1_wave",
        "sim_name":    "sim1_wave",
        "runners": {
            "SAMBoost": "experiments.samboost.run_sim1_wave",
            "GPBoost":     "experiments.gpboost.run_sim1_wave",
            "XGBoost":     "experiments.xgboost.run_sim1_wave",
        },
    },
    "sim2_hotspot_river": {
        "gen_module":  "simulations.generate_sim2_hotspot_river",
        "data_subdir": "sim2_hotspot_river",
        "sim_name":    "sim2_hotspot_river",
        "runners": {
            "SAMBoost": "experiments.samboost.run_sim2_hotspot_river",
            "GPBoost":     "experiments.gpboost.run_sim2_hotspot_river",
            "XGBoost":     "experiments.xgboost.run_sim2_hotspot_river",
        },
    },
    "sim2_ablation_simpleF": {
        "gen_module":  "simulations.generate_sim2_ablation_simpleF",
        "data_subdir": "sim2_ablation_simpleF",
        "sim_name":    "sim2_ablation_simpleF",
        "runners": {
            "SAMBoost": "experiments.samboost.run_sim2_ablation_simpleF",
            "GPBoost":     "experiments.gpboost.run_sim2_ablation_simpleF",
            "XGBoost":     "experiments.xgboost.run_sim2_ablation_simpleF",
        },
    },
}


# >> Helpers

def load_metrics_from_folder(folder):
    """Read scores.csv and dissociation.csv, return test-set metrics dict."""
    folder = Path(folder)
    out = {}

    scores_path = folder / "scores.csv"
    if scores_path.exists():
        scores = pd.read_csv(scores_path)
        test_s = scores[scores["split"] == "test"]
        if len(test_s) > 0:
            # classification metrics + regression metrics (whichever the runner
            # wrote, so both task types work).
            for col in ["auc", "log_loss", "brier", "rmse", "r2", "mae"]:
                if col in test_s.columns:
                    out[col] = float(test_s[col].values[0])

    dissoc_path = folder / "dissociation.csv"
    if dissoc_path.exists():
        dissoc = pd.read_csv(dissoc_path)
        test_d = dissoc[dissoc["split"] == "test"]
        if len(test_d) > 0:
            for col in ["r_spatial", "rmse_spatial", "r_cov", "rmse_cov"]:
                if col in test_d.columns:
                    out[col] = float(test_d[col].values[0])
        train_d = dissoc[dissoc["split"] == "train"]
        if len(train_d) > 0 and "fit_time" in train_d.columns:
            out["fit_time"] = float(train_d["fit_time"].values[0])

    return out


def print_summary(df, sim_name, out_dir):
    """Print and save mean +/- std summary."""
    metric_cols = ["auc", "log_loss", "brier", "rmse", "r2", "mae",
                   "r_spatial", "rmse_spatial",
                   "r_cov", "rmse_cov", "fit_time", "total_run_time"]
    metric_cols = [c for c in metric_cols if c in df.columns]

    summary_rows = []
    for method in ["SAMBoost", "GPBoost", "XGBoost"]:
        sub = df[df["method"] == method]
        if len(sub) == 0:
            continue
        row = {"Method": method, "n_seeds": len(sub)}
        for col in metric_cols:
            vals = sub[col].dropna()
            if len(vals) > 0:
                row[f"{col}_mean"] = round(float(vals.mean()), 4)
                row[f"{col}_std"]  = round(float(vals.std()), 4)
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    w = 20
    header = f"  {'Method':<15}  {'n':>3}"
    for col in metric_cols:
        header += f"  {col:>{w}}"
    print(header)
    print(f"  {'-'*15}  {'-'*3}" + f"  {'-'*w}" * len(metric_cols))

    for _, row in summary_df.iterrows():
        line = f"  {row['Method']:<15}  {int(row['n_seeds']):>3}"
        for col in metric_cols:
            mk = f"{col}_mean"
            sk = f"{col}_std"
            if mk in row and not pd.isna(row[mk]):
                cell = f"{row[mk]:.4f}+/-{row[sk]:.4f}"
            else:
                cell = "-"
            line += f"  {cell:>{w}}"
        print(line)

    return summary_df


# >> Tuning mode per method

# Each method can have its own tuning mode: "fixed", "first_seed", "per_seed"
# and fixed params (used when mode is "fixed")

GPBOOST_TUNING_MODE = "first_seed"  # "fixed", "first_seed", "per_seed"
GPBOOST_FIXED_PARAMS = {
    "learning_rate": 0.1, "max_depth": 5, "min_data_in_leaf": 10,
    "num_leaves": 15, "gp_approx": "none",
}

XGBOOST_TUNING_MODE = "first_seed"  # "fixed", "first_seed", "per_seed"
XGBOOST_FIXED_PARAMS = {
    "n_estimators": 50, "learning_rate": 0.1, "max_depth": 5,
    "subsample": 0.7, "colsample_bytree": 0.7, "min_child_weight": 5,
}


# >> Grid patching

def _get_grid_attr(runner, method_name):
    """Get the name of the param grid attribute for each method."""
    if method_name == "SAMBoost":
        return "PARAM_GRID"
    elif method_name == "GPBoost":
        # GPBoost has GRID_FULL and GRID_VECCHIA
        return "GRID_FULL"
    elif method_name == "XGBoost":
        return "PARAM_GRID"
    return None


def _save_original_grids(runners):
    """Save original grid dicts to restore later."""
    originals = {}
    for method_name, runner in runners.items():
        attr = _get_grid_attr(runner, method_name)
        if attr and hasattr(runner, attr):
            originals[method_name] = {attr: getattr(runner, attr).copy()}
            # GPBoost also has GRID_VECCHIA
            if method_name == "GPBoost" and hasattr(runner, "GRID_VECCHIA"):
                originals[method_name]["GRID_VECCHIA"] = runner.GRID_VECCHIA.copy()
    return originals


def _restore_original_grids(runners, originals):
    """Restore original grid dicts."""
    for method_name, grids in originals.items():
        if method_name in runners:
            for attr, grid in grids.items():
                setattr(runners[method_name], attr, grid)


def _patch_grid_from_best_params(runner, method_name, best_params_path, original_grids):
    """Read best_params.json and patch the runner's grid to a single point."""
    if not best_params_path.exists():
        return False

    with open(best_params_path) as fp:
        best = json.load(fp)

    if method_name == "GPBoost":
        # Patch GRID_FULL, disable GRID_VECCHIA
        orig_keys = set(original_grids.get(method_name, {}).get("GRID_FULL", {}).keys())
        patched = {k: [v] for k, v in best.items() if k in orig_keys}
        if patched:
            runner.GRID_FULL = patched
            # Empty vecchia grid so it produces no combos
            runner.GRID_VECCHIA = {"__dummy__": []}
    elif method_name == "XGBoost":
        orig_keys = set(original_grids.get(method_name, {}).get("PARAM_GRID", {}).keys())
        patched = {k: [v] for k, v in best.items() if k in orig_keys}
        if patched:
            runner.PARAM_GRID = patched
    elif method_name == "SAMBoost":
        orig_keys = set(original_grids.get(method_name, {}).get("PARAM_GRID", {}).keys())
        patched = {k: [v] for k, v in best.items() if k in orig_keys}
        if patched:
            runner.PARAM_GRID = patched

    return True


def _apply_fixed_params(runner, method_name, fixed_params, original_grids):
    """Patch grid to use fixed params."""
    if method_name == "GPBoost":
        orig_keys = set(original_grids.get(method_name, {}).get("GRID_FULL", {}).keys())
        runner.GRID_FULL = {k: [v] for k, v in fixed_params.items() if k in orig_keys}
        runner.GRID_VECCHIA = {"__dummy__": []}
    elif method_name in ("XGBoost", "SAMBoost"):
        attr = _get_grid_attr(runner, method_name)
        orig_keys = set(original_grids.get(method_name, {}).get(attr, {}).keys())
        setattr(runner, attr, {k: [v] for k, v in fixed_params.items() if k in orig_keys})


def _get_tuning_mode(method_name):
    if method_name == "SAMBoost": return TUNING_MODE
    if method_name == "GPBoost":     return GPBOOST_TUNING_MODE
    if method_name == "XGBoost":     return XGBOOST_TUNING_MODE
    return "per_seed"


def _get_fixed_params(method_name):
    if method_name == "SAMBoost": return SAMBOOST_FIXED_PARAMS
    if method_name == "GPBoost":     return GPBOOST_FIXED_PARAMS
    if method_name == "XGBoost":     return XGBOOST_FIXED_PARAMS
    return {}


# >> One simulation over all seeds

def run_simulation_multi_seed(sim_key):
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    config   = SIM_CONFIG[sim_key]
    data_dir = REPO_ROOT / "outputs" / "data" / config["data_subdir"]
    out_dir  = MULTI_SEED_DIR / sim_key / f"run_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Multi-Seed: {sim_key} ({config['sim_name']})")
    print(f"  Seeds: {len(SEEDS)}")
    print(f"  Tuning modes:")
    print(f"    SAMBoost : {TUNING_MODE}")
    print(f"    GPBoost     : {GPBOOST_TUNING_MODE}")
    print(f"    XGBoost     : {XGBOOST_TUNING_MODE}")
    print(f"{'='*70}")

    # Import simulation module
    gen_module = __import__(config["gen_module"], fromlist=["main"])

    # Import runner modules
    runners = {}
    method_flags = {"SAMBoost": RUN_SAMBOOST, "GPBoost": RUN_GPBOOST, "XGBoost": RUN_XGBOOST}
    for method_name, module_path in config["runners"].items():
        if method_flags.get(method_name, False):
            try:
                runners[method_name] = __import__(module_path, fromlist=["run", "PARAM_GRID"])
            except ImportError as e:
                print(f"  [warn] Cannot import {method_name}: {e}")

    # Save original grids
    original_grids = _save_original_grids(runners)

    # Apply fixed params where needed
    for method_name, runner in runners.items():
        mode = _get_tuning_mode(method_name)
        if mode == "fixed":
            _apply_fixed_params(runner, method_name, _get_fixed_params(method_name), original_grids)
            print(f"  [tuning] {method_name}: fixed params applied")

    all_rows = []
    import io, contextlib

    for i, seed in enumerate(SEEDS):
        print(f"\n{'-'*70}")
        print(f"  Seed {seed}  ({i+1}/{len(SEEDS)})")
        print(f"{'-'*70}")

        # 1. Regenerate data
        print(f"  [data] Generating with seed={seed} ...", end="", flush=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f = io.StringIO()
            with contextlib.redirect_stdout(f):
                gen_module.main(output_dir=str(data_dir), seed=seed)
        print(" done")

        # 2. Run each method with timing
        for method_name, runner in runners.items():
            if method_name == "XGBoost":
                runner.INCLUDE_COORDS = True

            seed_out = out_dir / f"seed_{seed:04d}" / method_name.lower()

            print(f"  [{method_name:<15}] ", end="", flush=True)
            method_t0 = time.perf_counter()
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    f = io.StringIO()
                    with contextlib.redirect_stdout(f):
                        summary = runner.run(out_dir=seed_out)

                method_time = time.perf_counter() - method_t0
                m = load_metrics_from_folder(seed_out)
                m["method"] = method_name
                m["seed"] = seed
                m["total_run_time"] = round(method_time, 2)
                all_rows.append(m)
                # task-aware progress line: AUC for classification, RMSE for regression
                _pm = (f"AUC={m['auc']:.4f}" if "auc" in m
                       else f"RMSE={m['rmse']:.4f}" if "rmse" in m else "score=?")
                _rs = f"{m['r_spatial']:.4f}" if "r_spatial" in m else "?"
                print(f"{_pm}  r_sp={_rs}  time={method_time:.1f}s")

            except Exception as e:
                method_time = time.perf_counter() - method_t0
                print(f"FAILED ({method_time:.1f}s): {e}")

        # 3. After first seed: apply first_seed tuning for all methods
        if i == 0:
            for method_name, runner in runners.items():
                mode = _get_tuning_mode(method_name)
                if mode == "first_seed":
                    bp_path = out_dir / f"seed_{seed:04d}" / method_name.lower() / "best_params.json"
                    if _patch_grid_from_best_params(runner, method_name, bp_path, original_grids):
                        with open(bp_path) as fp:
                            bp = json.load(fp)
                        print(f"  [tuning] {method_name}: first-seed params applied -> {bp}")
                    else:
                        print(f"  [tuning] {method_name}: best_params.json not found at {bp_path}")

    # Restore original grids
    _restore_original_grids(runners, original_grids)

    # Restore default seed data
    print(f"\n  Restoring default seed=42 data ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            gen_module.main(output_dir=str(data_dir), seed=42)
    print("  Done.")

    if not all_rows:
        print("  No results collected.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    df["simulation"] = sim_key
    df.to_csv(out_dir / "all_seeds_raw.csv", index=False)

    print(f"\n  {'='*70}")
    print(f"  SUMMARY - {sim_key} ({len(SEEDS)} seeds)")
    print(f"  {'='*70}")
    print_summary(df, sim_key, out_dir)
    print(f"\n  Raw -> {out_dir / 'all_seeds_raw.csv'}")
    print(f"  Summary -> {out_dir / 'summary.csv'}")

    return df


# >> Main

def main():
    total_t0 = time.perf_counter()

    print(f"{'='*70}")
    print(f"  Multi-Seed Experiment Runner")
    print(f"  Seeds       : {len(SEEDS)}")
    print(f"  Tuning      : {TUNING_MODE}")
    sims = []
    if RUN_SIM1:                sims.append("sim1_wave")
    if RUN_SIM2:        sims.append("sim2_hotspot_river")
    if RUN_SIM2_ABLATION: sims.append("sim2_ablation_simpleF")
    print(f"  Simulations : {', '.join(sims)}")
    methods = []
    if RUN_SAMBOOST: methods.append("SAMBoost")
    if RUN_GPBOOST:   methods.append("GPBoost")
    if RUN_XGBOOST:   methods.append("XGBoost")
    print(f"  Methods     : {', '.join(methods)}")
    print(f"{'='*70}")

    all_dfs = []
    for sim_key in sims:
        df = run_simulation_multi_seed(sim_key)
        if len(df) > 0:
            all_dfs.append(df)

    # Combined summary across simulations
    if len(all_dfs) > 1:
        combined = pd.concat(all_dfs, ignore_index=True)
        from datetime import datetime
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        combined.to_csv(MULTI_SEED_DIR / f"all_simulations_raw_{ts}.csv", index=False)
        print(f"\n  Combined raw -> {MULTI_SEED_DIR / f'all_simulations_raw_{ts}.csv'}")

    total_time = time.perf_counter() - total_t0
    print(f"\n  Total time: {total_time:.0f}s ({total_time/60:.1f}min)")


if __name__ == "__main__":
    main()