"""
generate_sim1_wave.py
=====================
Simulation 1 of the paper: classification, no spatial confounding.

Spatial effect (smooth chequerboard, one sign change per axis)
  base(s) = 1.4 * cos(pi s1) * cos(pi s2)
  b(s)    = base          where base >= 0
          = 0.65 * base   where base <  0   (negative lobes softened)

Covariate effect (nearly additive)
  F_raw = 1.5*x1 + x2^2 + 3*1{x3 > 0} + x4*x5
  F     = C2 * (F_raw - C1), with C1, C2 set on a draw of 50,000 points so
          that F has mean 0 and variance 0.5.
  x1..x5 and the noise covariates x_noise1, x_noise2 are i.i.d. N(0, 1);
  the noise covariates are not used in F.

Response
  m = F(x) + b(s)
  classification (default): y ~ Bernoulli(sigmoid(m))
  regression:               y = m + eps, eps ~ N(0, noise_std^2)

Locations are uniform on [0, 1]^2, N = 1000 train and 1000 test.
Output: outputs/data/sim1_wave/{train_df.csv, test_df.csv, summary}
"""

import argparse
import numpy as np
import pandas as pd
from scipy.special import expit
from pathlib import Path

# Reproducibility and sizes
SEED    = 42
N_TRAIN = 1000
N_TEST  = 1000
N_CALIB = 50_000

F_VAR_TARGET      = 0.5
DEFAULT_NOISE_STD = 0.5

COV_COLS = ["x1", "x2", "x3", "x4", "x5", "x_noise1", "x_noise2"]


# >> 1. Spatial effect

def sim1_wave_effect(locs):
    """b(s) = 1.4 cos(pi s1) cos(pi s2), negative half softened (x0.65)."""
    s1, s2 = locs[:, 0], locs[:, 1]
    base = 1.4 * np.cos(np.pi * s1) * np.cos(np.pi * s2)
    return np.where(base < 0, 0.65 * base, base)


# >> 2. Covariates and F

def build_X(rng, n):
    """Covariate matrix (n, 7): x1..x5, x_noise1, x_noise2, i.i.d. N(0,1)."""
    X = {c: rng.standard_normal(n) for c in COV_COLS}
    return pd.DataFrame(X)


def F_raw(X_df):
    """Simple F: 1.5*x1 + x2^2 + 3*1{x3>0} + x4*x5."""
    return (1.5 * X_df["x1"]
            + X_df["x2"] ** 2
            + 3.0 * (X_df["x3"] > 0).astype(float)
            + X_df["x4"] * X_df["x5"]
            ).values


def calibrate_F(rng_cal):
    """C1, C2 on a large draw so F = C2*(F_raw - C1) has mean 0, Var=F_VAR_TARGET.
    The 50,000 locations drawn first are not used, but do not remove that line:
    it would change the random numbers drawn next, hence C1, C2 and all the
    generated data."""
    _ = rng_cal.uniform(0, 1, (N_CALIB, 2))   # not used, see docstring
    X_cal = build_X(rng_cal, N_CALIB)
    f = F_raw(X_cal)
    c1 = float(f.mean())
    c2 = float(np.sqrt(F_VAR_TARGET) / f.std())
    return c1, c2


# >> 3. Response and data frame

def make_response(rng, m, task="classification", noise_std=DEFAULT_NOISE_STD):
    if task == "classification":
        return rng.binomial(1, expit(m)).astype(int)
    return m + rng.normal(0, noise_std, size=len(m))


def build_df(rng, locs, X_df, F_eff, b, task, noise_std):
    m = F_eff + b
    y = make_response(rng, m, task=task, noise_std=noise_std)
    df = pd.DataFrame(locs, columns=["s1", "s2"])
    for c in COV_COLS:
        df[c] = X_df[c].values
    df["F_effect"]       = F_eff
    df["spatial_effect"] = b
    df["m"]              = m
    df["y"]              = y
    return df


# >> 4. Main

def main(output_dir=None, task="classification", noise_std=DEFAULT_NOISE_STD, seed=SEED):
    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "outputs" / "data" / "sim1_wave"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 58)
    print(f"  Sim 1: classification, no confounding  [{task}, N={N_TRAIN}]")
    print("=" * 58)

    rng     = np.random.default_rng(seed)
    rng_cal = np.random.default_rng(seed + 1000)

    C1, C2 = calibrate_F(rng_cal)

    train_locs = rng.uniform(0, 1, (N_TRAIN, 2))
    test_locs  = rng.uniform(0, 1, (N_TEST, 2))
    b_train = sim1_wave_effect(train_locs)
    b_test  = sim1_wave_effect(test_locs)
    X_train = build_X(rng, N_TRAIN)
    X_test  = build_X(rng, N_TEST)
    F_train = C2 * (F_raw(X_train) - C1)
    F_test  = C2 * (F_raw(X_test)  - C1)

    train_df = build_df(rng, train_locs, X_train, F_train, b_train, task, noise_std)
    test_df  = build_df(rng, test_locs,  X_test,  F_test,  b_test,  task, noise_std)

    train_df.to_csv(output_dir / "train_df.csv", index=False)
    test_df.to_csv(output_dir / "test_df.csv",   index=False)

    print(f"  b_train std={b_train.std():.3f}  range=[{b_train.min():.3f}, {b_train.max():.3f}]")
    print(f"  F_train std={F_train.std():.3f}")
    if task == "classification":
        print(f"  y=1 rate  train={train_df.y.mean():.4f}  test={test_df.y.mean():.4f}")

    task_label = "y ~ Bernoulli(sigma(m))" if task == "classification" else f"y = m + N(0,{noise_std}^2)"
    summary = (
        f"\nSim 1: classification, no confounding ({task})\n"
        f"Seed : {seed}\nN : {N_TRAIN}/{N_TEST}\nResponse : {task_label}\n"
        f"Spatial: b(s) = 1.4 cos(pi s1) cos(pi s2), neg half x0.65  (smooth chequerboard)\n"
        f"F_raw = 1.5*x1 + x2^2 + 3*1{{x3>0}} + x4*x5  (simple)\n"
        f"C1={C1:.6f}, C2={C2:.6f}\n"
    )
    (output_dir / "sim1_wave_simulation_summary.txt").write_text(summary, encoding="utf-8")
    print(summary)
    print("Done.\n" + "=" * 58)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Simulation 1 data (chequerboard field, classification)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--task", type=str, default="classification",
                        choices=["classification", "regression"])
    parser.add_argument("--noise_std", type=float, default=DEFAULT_NOISE_STD)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    main(output_dir=args.output_dir, task=args.task,
         noise_std=args.noise_std, seed=args.seed)
