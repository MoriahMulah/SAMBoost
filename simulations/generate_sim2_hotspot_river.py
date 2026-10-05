"""
generate_sim2_hotspot_river.py
==============================
Simulation 2 of the paper: regression with a spatially confounded covariate.

Spatial effect
  b(s) = Mexican-hat hotspot centred at (0.5, 0.2), see
         generate_hotspot_river.hotspot_river_effect

River covariate (fully determined by location, hence confounded)
  x_river = 1 if the perpendicular distance to the line (0, 0.70) -> (1, 0.90)
            is below 0.15, else 0
  It is stored in the column "x_hotspot" (column name shared with the other
  generators and expected by the experiment scripts).

Covariate effect (with interactions)
  F_raw = 1.5*x1 + 3*1{x3 > 0} + 2.5*x1*x2 + 2*x2*x3 + 1.5*x4*x5 + 4.5*x_river
  F     = C2 * (F_raw - C1), with C1, C2 set on a draw of 50,000 points so
          that F has mean 0 and variance 0.5.
  x1..x5 and the noise covariates x_noise1, x_noise2 are i.i.d. N(0, 1).

Response
  y = F(x) + b(s) + eps,  eps ~ N(0, 0.5^2)   (regression, default)

Locations are uniform on [0, 1]^2, N = 1000 train and 1000 test.
Output: outputs/data/sim2_hotspot_river/{train_df.csv, test_df.csv, summary}
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
RIVER_HALFWIDTH   = 0.15                      # river corridor half-width
COV_COLS = ["x1", "x2", "x3", "x4", "x5", "x_noise1", "x_noise2"]


def river_indicator_wide(locs, halfwidth=RIVER_HALFWIDTH):
    lv = RIVER_P2 - RIVER_P1
    d = np.abs((locs - RIVER_P1)[:, 0]*lv[1] - (locs - RIVER_P1)[:, 1]*lv[0]) / np.linalg.norm(lv)
    return (d < halfwidth).astype(float)


def F_raw(X, river):
    return (1.5*X[:, 0] + 3.0*(X[:, 2] > 0) + 2.5*X[:, 0]*X[:, 1]
            + 2.0*X[:, 1]*X[:, 2] + 1.5*X[:, 3]*X[:, 4] + RIVER_COEFF*river)


def main(output_dir=None, task="regression", noise_std=DEFAULT_NOISE_STD, seed=SEED):
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "outputs" / "data" / "sim2_hotspot_river"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 58)
    print(f"  Sim 2: hotspot+river  [{task}, N={N_TRAIN}, halfwidth={RIVER_HALFWIDTH}]")
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
    (output_dir / "sim2_hotspot_river_simulation_summary.txt").write_text(
        f"Sim 2: hotspot+river ({task})\nSeed {seed}  N {N_TRAIN}/{N_TEST}\n"
        f"River halfwidth={RIVER_HALFWIDTH}, coeff={RIVER_COEFF}\n"
        f"F_raw = 1.5*x1 + 3*1{{x3>0}} + 2.5*x1*x2 + 2*x2*x3 + 1.5*x4*x5 + {RIVER_COEFF}*x_river;  F_var_target={F_VAR_TARGET}\n"
        f"C1={C1:.6f} C2={C2:.6f}\n", encoding="utf-8")
    print("Done.\n" + "=" * 58)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Generate Sim 2 hotspot+river data")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--task", type=str, default="regression", choices=["regression", "classification"])
    p.add_argument("--noise_std", type=float, default=DEFAULT_NOISE_STD)
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args()
    main(output_dir=a.output_dir, task=a.task, noise_std=a.noise_std, seed=a.seed)
