"""
plot_scaling_linear.py
Same fit times as plot_scaling.py on linear axes: fit_time_scaling_linear.png,
the scaling figure of the paper. Run after both benchmarks.
"""
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
frames = [pd.read_csv(HERE / f) for f in ["results_python.csv", "results_inla.csv"] if (HERE / f).exists()]
df = pd.concat(frames, ignore_index=True).dropna(subset=["fit_s"])
LABEL = {"SAMBoost":"SAMBoost", "GPBoost_vecchia":"GPBoost (Vecchia)",
         "GPBoost_exact":"GPBoost (exact GP)", "INLA_spde":"INLA (SPDE)", "XGBoost":"XGBoost"}
COLOR = {"SAMBoost":"#1f77b4","GPBoost_vecchia":"#2ca02c","GPBoost_exact":"#d62728",
         "INLA_spde":"#9467bd","XGBoost":"#ff7f0e"}
agg = df.groupby(["method","N"]).fit_s.agg(["mean","std"]).reset_index()

fig, ax = plt.subplots(figsize=(7.6, 5.6))
for m in ["SAMBoost","GPBoost_vecchia","GPBoost_exact","INLA_spde","XGBoost"]:
    d = agg[agg.method==m].sort_values("N")
    if not len(d): continue
    ax.errorbar(d.N, d["mean"], yerr=d["std"].fillna(0), marker="o", capsize=3,
                color=COLOR[m], label=LABEL[m])
ax.set_xlabel("N (training points)"); ax.set_ylabel("fit time (s)")
ax.set_title("Fit-time scaling with N (LINEAR axes)")
ax.legend(); ax.grid(True, alpha=0.3)
out = HERE / "fit_time_scaling_linear.png"; fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved ->", out)
