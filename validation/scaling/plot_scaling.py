"""
plot_scaling.py
Combines results_python.csv and results_inla.csv, fits a power law t = c N^b for
each method (the exponent b quoted in the paper, about 0.37 for SAMBoost), and
writes the log-log figure fit_time_scaling.png and fit_time_table.csv.
Run after both benchmarks.
"""
import sys, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent
frames = []
for f in ["results_python.csv", "results_inla.csv"]:
    p = HERE / f
    if p.exists(): frames.append(pd.read_csv(p))
df = pd.concat(frames, ignore_index=True)
df = df.dropna(subset=["fit_s"])
LABEL = {"SAMBoost":"SAMBoost", "GPBoost_vecchia":"GPBoost (Vecchia)",
         "GPBoost_exact":"GPBoost (exact GP)", "INLA_spde":"INLA (SPDE)"}
COLOR = {"SAMBoost":"#1f77b4","GPBoost_vecchia":"#2ca02c","GPBoost_exact":"#d62728","INLA_spde":"#9467bd"}
agg = df.groupby(["method","N"]).fit_s.agg(["mean","std","count"]).reset_index()

fig, ax = plt.subplots(figsize=(7.6, 5.6))
summary = []
for m in ["SAMBoost","GPBoost_vecchia","GPBoost_exact","INLA_spde"]:
    d = agg[agg.method==m].sort_values("N")
    if not len(d): continue
    ax.errorbar(d.N, d["mean"], yerr=d["std"].fillna(0), marker="o", capsize=3,
                color=COLOR[m], label=LABEL[m])
    # power-law fit: log t = a + b log N  -> slope b = scaling exponent
    if len(d) >= 3:
        b, a = np.polyfit(np.log(d.N), np.log(d["mean"]), 1)
        ax.plot(d.N, np.exp(a)*d.N**b, ls="--", color=COLOR[m], alpha=0.4)
        summary.append((LABEL[m], b, d.N.min(), d.N.max(), d["mean"].max()))
        ax.annotate(f"~N^{b:.2f}", (d.N.iloc[-1], d["mean"].iloc[-1]),
                    color=COLOR[m], fontsize=9, xytext=(5,0), textcoords="offset points")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("N (training points)"); ax.set_ylabel("fit time (s)")
ax.set_title("Fit-time scaling with N (fixed 100 trees, single fit, sequential)")
ax.legend(); ax.grid(True, which="both", alpha=0.3)
out = HERE / "fit_time_scaling.png"; fig.savefig(out, dpi=150, bbox_inches="tight")
print("saved ->", out, "\n")

print("=== scaling exponents (log-log slope) ===")
for lab, b, nmin, nmax, tmax in summary:
    print(f"  {lab:24s}  ~ N^{b:.2f}   (N {int(nmin)}-{int(nmax)}, max {tmax:.1f}s)")
piv = agg.pivot(index="N", columns="method", values="mean").round(2)
piv.to_csv(HERE / "fit_time_table.csv")
print("\n=== mean fit time (s) by N ===\n", piv.to_string())
print("\nsaved -> fit_time_table.csv")
