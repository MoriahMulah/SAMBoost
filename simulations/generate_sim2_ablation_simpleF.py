"""
generate_sim2_ablation_simpleF.py
=================================
Ablation of the paper (role of covariate complexity): Simulation 2 with a
simpler covariate effect. Everything else is identical to
generate_sim2_hotspot_river.py (hotspot, river of half-width 0.15, noise,
N = 1000 train and 1000 test).

Covariate effect (nearly additive)
  F_raw = 1.5*x1 + x2^2 + 3*1{x3 > 0} + x4*x5 + 4.5*x_river
  F     = C2 * (F_raw - C1), mean 0 and variance 0.5.

The river covariate is stored in the column "x_hotspot", as in Simulation 2.
Output: outputs/data/sim2_ablation_simpleF/{train_df.csv, test_df.csv, summary}
"""
import argparse
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root, so the file also runs as a script
from simulations.generate_hotspot_river import hotspot_river_effect, RIVER_P1, RIVER_P2

SEED = 42
N_TRAIN = 1000
N_TEST  = 1000
N_CALIB = 50_000
F_VAR_TARGET      = 0.5
DEFAULT_NOISE_STD = 0.5
RIVER_COEFF       = 4.5
RIVER_HALFWIDTH   = 0.15                      # river corridor half-width, as in Sim 2
COV_COLS = ["x1", "x2", "x3", "x4", "x5", "x_noise1", "x_noise2"]


def river_indicator_wide(locs, halfwidth=RIVER_HALFWIDTH):
    lv = RIVER_P2 - RIVER_P1
    d = np.abs((locs - RIVER_P1)[:, 0]*lv[1] - (locs - RIVER_P1)[:, 1]*lv[0]) / np.linalg.norm(lv)
    return (d < halfwidth).astype(float)


def F_raw(X, river):
    """Nearly additive F: 1.5*x1 + x2^2 + 3*1{x3>0} + x4*x5 + 4.5*river."""
    return (1.5*X[:, 0] + X[:, 1]**2 + 3.0*(X[:, 2] > 0)
            + X[:, 3]*X[:, 4] + RIVER_COEFF*river)


def main(output_dir=None, task="regression", noise_std=DEFAULT_NOISE_STD, seed=SEED):
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "outputs" / "data" / "sim2_ablation_simpleF"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 58)
    print(f"  Sim 2 ablation: hotspot+river, simple F  [{task}, N={N_TRAIN}, halfwidth={RIVER_HALFWIDTH}]")
    print("=" * 58)

    rng = np.random.default_rng(seed); rc = np.random.default_rng(seed + 1000)
    Lc = rc.uniform(0, 1, (N_CALIB, 2)); Xc = rc.standard_normal((N_CALIB, 7))
    fc = F_raw(Xc, river_indicator_wide(Lc)); C1, C2 = fc.mean(), np.sqrt(F_VAR_TARGET)/fc.std()

    def make(n):
        L = rng.uniform(0, 1, (n, 2)); X = rng.standard_normal((n, 7))
        r = river_indicator_wide(L)
        F = C2*(F_raw(X, r) - C1); b = hotspot_river_effect(L)
        if task == "classification":
            from scipy.special import expit
            y = rng.binomial(1, expit(F + b)).astype(int)
        else:
            y = F + b + rng.normal(0, noise_std, n)
        df = pd.DataFrame(L, columns=["s1", "s2"])
        for i, c in enumerate(COV_COLS):
            df[c] = X[:, i]
        df["x_hotspot"] = r                 # the river covariate (see module docstring)
        df["F_effect"] = F; df["spatial_effect"] = b; df["m"] = F + b; df["y"] = y
        return df

    train, test = make(N_TRAIN), make(N_TEST)
    train.to_csv(output_dir / "train_df.csv", index=False)
    test.to_csv(output_dir / "test_df.csv", index=False)
    print(f"  b_train std={train.spatial_effect.std():.3f}  F_train std={train.F_effect.std():.3f}")
    print(f"  river prevalence train={train.x_hotspot.mean():.3f}")
    (output_dir / "sim2_ablation_simpleF_simulation_summary.txt").write_text(
        f"Sim 2 ablation: hotspot+river, simple F ({task})\nSeed {seed}  N {N_TRAIN}/{N_TEST}\n"
        f"River halfwidth={RIVER_HALFWIDTH}, coeff={RIVER_COEFF}\n"
        f"F_raw = 1.5*x1 + x2^2 + 3*1{{x3>0}} + x4*x5 + {RIVER_COEFF}*x_river;  F_var_target={F_VAR_TARGET}\n"
        f"C1={C1:.6f} C2={C2:.6f}\n", encoding="utf-8")
    print("Done.\n" + "=" * 58)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate Sim 2 ablation (simple F) data")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--task", type=str, default="regression", choices=["regression", "classification"])
    p.add_argument("--noise_std", type=float, default=DEFAULT_NOISE_STD)
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args()
    main(output_dir=a.output_dir, task=a.task, noise_std=a.noise_std, seed=a.seed)
