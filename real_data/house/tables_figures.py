"""
tables_figures.py
Builds the house-price tables and figures from the saved results in
outputs/results/house_price/final_syear/ (no refitting). Methods compared:
SAMBoost ex, GPBoost ex, XGBoost geo and INLA.

Tables:
  plots/consolidated_metrics_ex.csv   test RMSE and fit time per method and test year
  plots/final_grid_ex_bootstrap.json  paired bootstrap of the test errors pooled over the
                                      three years: SAMBoost ex vs GPBoost ex and vs
                                      XGBoost geo (confidence intervals)
  plots/final_grid_bootstrap.json     the same for the geo variants
Figures (plots/):
  mesh_resolution_v3.png   test RMSE and fit time of SAMBoost against the number
                           of mesh nodes (data from run_mesh_study.py)
  maps_ex.png              observed and predicted log price, and the spatial effect
                           of each method
  accuracy_vs_cost_ex.png  mean test RMSE against mean fit time
  rmse_by_fold_ex.png      test RMSE per test year

Usage (from the repository root):
    python real_data/house/tables_figures.py            # tables and figures
    python real_data/house/tables_figures.py --tables   # tables only
    python real_data/house/tables_figures.py --figures  # figures only
"""
import os, sys, json, argparse
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "4")
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from real_data.house.data_folds import TEST_YEARS

BASE = REPO_ROOT / "outputs" / "results" / "house_price" / "final_syear"
OUT  = BASE / "plots"; OUT.mkdir(parents=True, exist_ok=True)

TITLE_FS, LABEL_FS, TICK_FS, LEG_FS = 20, 18, 14, 14
COL = {"SAMBoost": "#0072B2", "GPBoost": "#E69F00", "XGBoost": "#009E73", "INLA": "#CC79A7"}
# Result folder, colour and marker per method
METHODS = {
    "XGBoost":   ("xgboost_geo",   COL["XGBoost"],   "s"),
    "GPBoost":   ("gpboost_ex",    COL["GPBoost"],   "o"),
    "SAMBoost": ("samboost_ex",  COL["SAMBoost"], "D"),
    "INLA":      ("inla_spline",   COL["INLA"],      "^"),
}
MESH_SB_COL, MESH_FIT_COL, MESH_GPB_COL = "#0072B2", "#FDB863", "#8c8c8c"
GPB_MEAN, GPB_CI = 0.2461, 0.002   # GPBoost ex mean RMSE and half-width of its bootstrap CI
GPB_FIT_MEAN, GPB_FIT_COL = 850, "#D55E00"   # GPBoost ex mean fit time (s), drawn on the mesh figure


# >> Reading results
def load_scores():
    rows = []
    for label, (d, c, mk) in METHODS.items():
        for t in TEST_YEARS:
            p = BASE / f"fold_{t}" / d / "scores.csv"
            if not p.exists():
                continue
            r = pd.read_csv(p); r = r[r.split == "test"].iloc[0]
            rows.append(dict(method=label, fold=t, rmse=r["rmse"], fit=r["fit_time"]))
    return pd.DataFrame(rows)


def pooled(d):
    parts = []
    for t in TEST_YEARS:
        p = BASE / f"fold_{t}" / d / "predictions.csv"
        if p.exists():
            q = pd.read_csv(p); parts.append(q[q.split == "test"])
    return pd.concat(parts, ignore_index=True) if parts else None


def _test_err(fold, subdir):
    q = pd.read_csv(BASE / f"fold_{fold}" / subdir / "predictions.csv")
    q = q[q.split == "test"].reset_index(drop=True)
    return q["y_hat"].values, q["log_price"].values


# >> Tables
def table_metrics():
    s = load_scores()
    s.to_csv(OUT / "consolidated_metrics_ex.csv", index=False)
    print(s.groupby("method").agg(rmse=("rmse", "mean"), fit=("fit", "mean")).round(4).to_string())
    print("wrote: consolidated_metrics_ex.csv")
    return s


def table_bootstrap(variant, nboot):
    """Paired bootstrap: SAMBoost vs GPBoost (same variant) and vs XGBoost geo."""
    sb_dir, gp_dir, xg_dir = f"samboost_{variant}", f"gpboost_{variant}", "xgboost_geo"
    sb_err, gp_err, xg_err = [], [], []
    for fold in TEST_YEARS:
        sb_hat, y = _test_err(fold, sb_dir)
        gp_hat, y_gp = _test_err(fold, gp_dir)
        xg_hat, y_xg = _test_err(fold, xg_dir)
        assert np.allclose(y, y_gp) and np.allclose(y, y_xg), f"row misalign fold {fold}"
        sb_err.append(sb_hat - y); gp_err.append(gp_hat - y); xg_err.append(xg_hat - y)

    sb, gp, xg = map(np.concatenate, (sb_err, gp_err, xg_err))
    n = len(sb)
    rmse = lambda e: float(np.sqrt(np.mean(e ** 2)))
    rs = np.random.RandomState(42)
    d_gp, d_xg = np.empty(nboot), np.empty(nboot)
    for b in range(nboot):
        idx = rs.randint(0, n, n)
        d_gp[b] = rmse(sb[idx]) - rmse(gp[idx])
        d_xg[b] = rmse(sb[idx]) - rmse(xg[idx])

    def summarize(d):
        lo, hi = np.percentile(d, [2.5, 97.5])
        return {"mean": float(d.mean()), "ci": [float(lo), float(hi)],
                "p_samboost_better": float(np.mean(d < 0)),
                "verdict": "SAMBoost better" if hi < 0 else ("TIE" if lo < 0 < hi else "other better")}

    res_gp, res_xg = summarize(d_gp), summarize(d_xg)
    print(f"# bootstrap {variant} | n={n} | SAMBoost={rmse(sb):.4f} GPB={rmse(gp):.4f} XGB={rmse(xg):.4f}")
    print(f"  SAMBoost vs GPBoost: {res_gp}\n  SAMBoost vs XGBoost: {res_xg}")

    if variant == "ex":
        out = {"variant": "ex (main-table)",
               "per_fold_SAMBoost_ex": {str(t): rmse(e) for t, e in zip(TEST_YEARS, sb_err)},
               "pooled_rmse": {"SAMBoost_ex": rmse(sb), "GPBoost_ex": rmse(gp), "XGBoost_geo": rmse(xg)},
               "SAMBoost_ex_vs_GPBoost_ex": res_gp, "SAMBoost_ex_vs_XGBoost_geo": res_xg}
        fname = "final_grid_ex_bootstrap.json"
    else:
        out = {"n_freeze": "10 (fixed, all folds)",
               "per_fold_SAMBoost": {str(t): rmse(e) for t, e in zip(TEST_YEARS, sb_err)},
               "pooled_rmse": {"SAMBoost": rmse(sb), "GPBoost": rmse(gp), "XGBoost": rmse(xg)},
               "SAMBoost_vs_GPBoost": res_gp, "SAMBoost_vs_XGBoost": res_xg}
        fname = "final_grid_bootstrap.json"
    (OUT / fname).write_text(json.dumps(out, indent=2))
    print(f"wrote: {fname}")


# >> Figures
def fig1_mesh():
    g = pd.read_csv(OUT / "mesh_curve_ex_data.csv").sort_values("nodes")
    fig, ax = plt.subplots(figsize=(8.2, 5.4), constrained_layout=True)
    ax.axhspan(GPB_MEAN - GPB_CI, GPB_MEAN + GPB_CI, color=MESH_GPB_COL, alpha=0.20)
    ax.axhline(GPB_MEAN, ls="--", color=MESH_GPB_COL, lw=1.4, label="GPBoost RMSE")
    ax.plot(g.nodes, g.rmse, "-o", color=MESH_SB_COL, label="SAMBoost RMSE")
    ax.set_xlabel("number of mesh nodes", fontsize=LABEL_FS)
    ax.set_ylabel("mean test RMSE (log price)", fontsize=LABEL_FS)
    ax.set_ylim(0.235, 0.285); ax.tick_params(labelsize=TICK_FS)
    ax2 = ax.twinx()
    ax2.plot(g.nodes, g.fit, "-o", color=MESH_FIT_COL, label="SAMBoost fit time")
    ax2.set_ylabel("mean fit time (s)", fontsize=LABEL_FS); ax2.tick_params(labelsize=TICK_FS)
    # GPBoost fit time as a reference line (labelled on the plot, not in the legend)
    ax2.axhline(GPB_FIT_MEAN, ls=":", color=GPB_FIT_COL, lw=1.6)
    ax2.set_ylim(0, 950)
    ax2.text(g.nodes.values[-1], GPB_FIT_MEAN + 20, f"GPBoost ({GPB_FIT_MEAN}s)",
             fontsize=12, color=GPB_FIT_COL, fontweight="bold", va="bottom", ha="right")
    lines = [l for l in ax.get_lines() + ax2.get_lines() if not l.get_label().startswith("_")]
    handles = [Patch(facecolor=MESH_GPB_COL, alpha=0.20)] + lines
    labs = ["GPBoost 95% CI"] + [l.get_label() for l in lines]
    ax.legend(handles, labs, loc="lower center", bbox_to_anchor=(0.5, 1.005),
              ncol=2, fontsize=LEG_FS, frameon=False)
    fig.savefig(OUT / "mesh_resolution_v3.png", dpi=150); plt.close(fig)
    print("wrote: mesh_resolution_v3.png")


def fig2_accuracy_cost(scores):
    mean = scores.groupby("method").agg(rmse=("rmse", "mean"), fit=("fit", "mean")).reset_index()
    try:
        mesh = pd.read_csv(OUT / "mesh_curve_v2_data.csv").sort_values("nodes")
    except Exception:
        mesh = None
    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    if mesh is not None:
        ax.plot(mesh.fit, mesh.rmse, "-", color=COL["SAMBoost"], alpha=0.35, lw=1.5,
                label="SAMBoost mesh sweep (diagnostic, geo)")
        ax.scatter(mesh.fit, mesh.rmse, s=14, color=COL["SAMBoost"], alpha=0.35)
    for _, r in mean.iterrows():
        mk = METHODS[r["method"]][2]; c = METHODS[r["method"]][1]
        ax.scatter(r["fit"], r["rmse"], s=100, color=c, marker=mk, edgecolor="k", linewidth=0.6, zorder=5)
        ax.annotate(r["method"], (r["fit"], r["rmse"]), fontsize=11, xytext=(6, 6), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("mean fit time per fold (s, log scale)", fontsize=LABEL_FS)
    ax.set_ylabel("mean test RMSE (log price)", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS); ax.legend(fontsize=LEG_FS, frameon=False, loc="upper right")
    fig.savefig(OUT / "accuracy_vs_cost_ex.png", dpi=150); plt.close(fig)
    print("wrote: accuracy_vs_cost_ex.png")


def fig3_rmse_by_fold(scores):
    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    for label in ["XGBoost", "GPBoost", "SAMBoost", "INLA"]:
        s = scores[scores.method == label].sort_values("fold")
        if s.empty:
            continue
        ax.plot(s.fold, s.rmse, "-", color=METHODS[label][1], marker=METHODS[label][2], markersize=8, label=label)
    ax.set_xticks(TEST_YEARS); ax.set_xlabel("test year", fontsize=LABEL_FS)
    ax.set_ylabel("test RMSE (log price)", fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS); ax.legend(fontsize=LEG_FS, frameon=False)
    fig.savefig(OUT / "rmse_by_fold_ex.png", dpi=150); plt.close(fig)
    print("wrote: rmse_by_fold_ex.png")


def _spanel(ax, q, title):
    vmax = np.nanpercentile(np.abs(q.b_hat), 98)
    s = ax.scatter(q.s1, q.s2, c=q.b_hat, s=4, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    cb = ax.figure.colorbar(s, ax=ax, shrink=0.8); cb.ax.tick_params(labelsize=TICK_FS)
    ax.set_title(title, fontsize=TITLE_FS); ax.set_xticks([]); ax.set_yticks([])


def fig4_maps():
    sb = pooled("samboost_ex"); gp = pooled("gpboost_ex")
    inla = pooled("inla_spline"); xgb = pooled("xgboost_geo")
    fig, ax = plt.subplots(2, 3, figsize=(12.5, 8.2), constrained_layout=True)
    vlo, vhi = np.nanpercentile(sb["log_price"], [1, 99])
    ax[0, 0].scatter(sb.s1, sb.s2, c=sb.log_price, s=4, cmap="viridis", vmin=vlo, vmax=vhi)
    ax[0, 0].set_title("Observed log-price", fontsize=TITLE_FS)
    sc = ax[0, 1].scatter(sb.s1, sb.s2, c=sb.y_hat, s=4, cmap="viridis", vmin=vlo, vmax=vhi)
    ax[0, 1].set_title("SAMBoost predicted log-price", fontsize=TITLE_FS)
    for a in (ax[0, 0], ax[0, 1]):
        a.set_xticks([]); a.set_yticks([])
    cb = fig.colorbar(sc, ax=[ax[0, 0], ax[0, 1]], shrink=0.7)
    cb.set_label("log price", fontsize=LABEL_FS); cb.ax.tick_params(labelsize=TICK_FS)
    _spanel(ax[0, 2], sb, "SAMBoost: spatial effect")
    _spanel(ax[1, 0], gp, "GPBoost: spatial effect")
    _spanel(ax[1, 1], inla, "INLA: spatial effect")
    _spanel(ax[1, 2], xgb, "XGBoost: spatial effect")
    fig.savefig(OUT / "maps_ex.png", dpi=150); plt.close(fig)
    print("wrote: maps_ex.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", action="store_true", help="tables only")
    ap.add_argument("--figures", action="store_true", help="figures only")
    ap.add_argument("--nboot", type=int, default=10000)
    args = ap.parse_args()
    do_tables = args.tables or not args.figures
    do_figs   = args.figures or not args.tables

    scores = load_scores()
    if do_tables:
        print("== TABLES ==")
        table_metrics()
        table_bootstrap("ex", args.nboot)
        table_bootstrap("geo", args.nboot)
    if do_figs:
        print("== FIGURES ==")
        fig1_mesh(); fig2_accuracy_cost(scores); fig3_rmse_by_fold(scores); fig4_maps()


if __name__ == "__main__":
    main()
