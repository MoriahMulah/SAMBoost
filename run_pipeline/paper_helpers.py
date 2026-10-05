r"""
paper_helpers.py

Shared configuration and helper functions for building the paper's figures and
tables. Import this from build_paper.py; nothing runs on import.

  Config   : SIM1 / SIM2 / ABLATION (single source of truth for run paths,
             method keys, task, geometry).
  Figures  : domain, field recovery (rep-seed + 30-seed mean),
             ablation (F-recovery, simple vs complex F).
  Tables   : 30-seed aggregate (mean +/- sd), augmented (adds leak / peak),
             ablation (SAMBoost vs INLA under simple vs complex F).
  Metrics  : leak and peak diagnostics (see leak_peak), not reported in the paper.
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
from simulations.generate_hotspot_river import river_indicator, HOTSPOT_CENTER

MS   = _REPO / "outputs" / "results" / "multi_seed"
DATA = _REPO / "outputs" / "data"
OUT  = _REPO / "outputs" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)

# Font sizes
FS_FIELD, FS_DOMAIN, FS_DISSOC, FS_CBAR, MARK = 24, 18, 20, 18, 12
# fixed title height (in axes units): titles with subscripts would otherwise be
# placed higher than plain ones by matplotlib's automatic positioning
TITLE_Y = 1.04
RIVER_P1, RIVER_P2, HOTSPOT_C = (0.0, 0.70), (1.0, 0.90), (0.5, 0.2)


# >> Config: run folders used for the paper
# Each sim: the run folder of each method (sb_run / gpb_run for the per-seed
# grids and predictions, raw_from for the 30-seed metrics of the tables), the
# INLA run and variant, and whether the river confounder is present.

_S1 = MS / "sim1_wave/run_20260717_212540"
SIM1 = dict(
    name="sim1_wave", subdir="sim1_wave", task="classification", river=False,
    sb_run=_S1,
    gpb_run=_S1,
    raw_from={"SAMBoost": _S1, "GPBoost": _S1,
              "XGBoost": MS / "sim1_wave/run_20261002_234315"},   # learning rate 0.1
    # XGBoost fit time from the run where all methods were timed together
    fit_time_from={"XGBoost": _S1},
    inla_run=MS / "sim1_wave/run_20260718_002155",
    inla_name="sim1_wave", inla_variant="spline",
    rep_methods=[("SAMBoost", "samboost", False), ("GPBoost", "gpboost", False),
                 ("INLA", "spline", True)],
)
# Sim 2: river confounding, covariate effect with interactions.
_S2_SB = MS / "sim2_hotspot_river/run_20260720_184054"
_S2_PY = MS / "sim2_hotspot_river/run_20260720_013119"
SIM2 = dict(
    name="sim2_hotspot_river", subdir="sim2_hotspot_river", task="regression", river=True,
    sb_run=_S2_SB,
    gpb_run=_S2_PY,
    raw_from={"SAMBoost": _S2_SB, "GPBoost": _S2_PY,
              "XGBoost": MS / "sim2_hotspot_river/run_20261002_234429"},   # learning rate 0.1
    fit_time_from={"XGBoost": _S2_PY},   # see SIM1
    inla_run=MS / "sim2_hotspot_river/run_20260720_013834",
    inla_name="sim2_hotspot_river", inla_variant="hybrid",
    rep_methods=[("SAMBoost", "samboost", False), ("GPBoost", "gpboost", False),
                 ("INLA", "hybrid", True)],
)
# Ablation: Sim 2 with a nearly additive covariate effect. In the ablation table
# GPBoost comes from gpb_fixed_run (hyperparameters of the Sim 2 run), so it is
# not listed in raw_from.
_AB_SB = MS / "sim2_ablation_simpleF/run_20260720_230225"
ABLATION = dict(
    name="sim2_ablation_simpleF", subdir="sim2_ablation_simpleF", task="regression", river=True,
    sb_run=_AB_SB,
    gpb_run=MS / "sim2_ablation_simpleF/run_20260720_124207",
    gpb_fixed_run=MS / "sim2_ablation_simpleF/gpb_fixed_params_check",
    raw_from={"SAMBoost": _AB_SB},
    inla_run=MS / "sim2_ablation_simpleF/run_20260720_124529",
    inla_name="sim2_ablation_simpleF", inla_variant="hybrid",
    rep_methods=[("SAMBoost", "samboost", False), ("GPBoost", "gpboost", False),
                 ("INLA", "hybrid", True)],
)


# >> Table formatting
_AGG_METHOD_ORDER = ["SAMBoost", "GPBoost", "INLA_linear", "INLA_spline",
                     "INLA_hybrid", "XGBoost"]
_AGG_METHOD_LABEL = {"SAMBoost": "SAMBoost", "GPBoost": "GPBoost",
                     "INLA_linear": "INLA linear", "INLA_spline": "INLA spline",
                     "INLA_hybrid": "INLA hybrid", "XGBoost": "XGBoost"}
_AGG_COLS = {
    "regression":     ["rmse", "r2", "mae", "r_spatial", "rmse_spatial", "r_cov", "rmse_cov", "fit_time"],
    "classification": ["auc", "log_loss", "brier", "r_spatial", "rmse_spatial", "r_cov", "rmse_cov", "fit_time"],
}
_HIGHER_BETTER = {"auc", "r2", "r_spatial", "r_cov"}
_COL_LABEL = {"rmse": "RMSE", "r2": "R2", "mae": "MAE", "auc": "AUC",
              "log_loss": "LogLoss", "brier": "Brier",
              "r_spatial": "r_spatial", "rmse_spatial": "RMSE_spatial",
              "r_cov": "r_cov", "rmse_cov": "RMSE_cov", "fit_time": "Time(s)"}
_COL_LABEL_TEX = {"rmse": "RMSE", "r2": r"$R^2$", "mae": "MAE", "auc": "AUC",
                  "log_loss": "LogLoss", "brier": "Brier",
                  "r_spatial": r"$r_{\mathrm{sp}}$", "rmse_spatial": r"RMSE$_{\mathrm{sp}}$",
                  "r_cov": r"$r_{\mathrm{cov}}$", "rmse_cov": r"RMSE$_{\mathrm{cov}}$",
                  "fit_time": "Time (s)"}


# >> Paths and data
def grid_path(cfg, key, is_inla, seed):
    """Per-seed recovered-field grid CSV for one method."""
    s = f"seed_{seed:04d}"
    if is_inla:
        return Path(cfg["inla_run"]) / s / cfg["inla_name"] / key / "grid_spatial_field.csv"
    run = cfg["sb_run"] if key == "samboost" else cfg["gpb_run"]
    return Path(run) / s / key / "grid_spatial_field.csv"


def pred_path(cfg, key, is_inla, seed):
    """Per-seed test predictions CSV for one method."""
    s = f"seed_{seed:04d}"
    if is_inla:
        return Path(cfg["inla_run"]) / s / cfg["inla_name"] / key / "predictions.csv"
    run = cfg["sb_run"] if key == "samboost" else cfg["gpb_run"]
    return Path(run) / s / key / "predictions.csv"


def rep_seed(cfg):
    """Representative seed = the one whose SAMBoost spatial RMSE is the median."""
    sb = pd.read_csv(Path(cfg["sb_run"]) / "all_seeds_raw.csv")
    sb = sb[sb.method == "SAMBoost"][["seed", "rmse_spatial"]]
    mv = sb.rmse_spatial.median()
    return int(sb.iloc[(sb.rmse_spatial - mv).abs().argmin()].seed)


def _load_raw(cfg):
    """30-seed metrics for one sim: each Python method from its run folder in
    cfg["raw_from"], plus all INLA variants from cfg["inla_run"]."""
    sources = [(Path(run) / "all_seeds_raw.csv", method) for method, run in cfg["raw_from"].items()]
    sources.append((Path(cfg["inla_run"]) / "all_seeds_raw_inla.csv", None))
    frames = []
    for path, method in sources:
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if "simulation" in df.columns:
            df = df[df["simulation"] == cfg["subdir"]]
        if method is not None:
            df = df[df["method"] == method]
            if method in cfg.get("fit_time_from", {}):
                t = pd.read_csv(Path(cfg["fit_time_from"][method]) / "all_seeds_raw.csv")
                t = t[t["method"] == method].set_index("seed")["fit_time"]
                df = df.assign(fit_time=df["seed"].map(t).values)
        if len(df):
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# >> Leak and peak diagnostics
def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def leak_peak(cfg, seeds=range(1, 31)):
    """Leakage and hotspot diagnostics per method (mean, sd over seeds), not
    reported in the paper. They need the per-seed field grids (seed_*/ folders,
    not published); without them the columns show "--".
      leak      : RMSE of the spatial field on grid points within 0.10 of the river
                  line (the central part of the 0.15-wide river of Simulation 2)
      peak_rmse : RMSE of the spatial field within 0.2 of the hotspot centre
      peak_h    : maximum of the estimated field within 0.2 of the hotspot centre
    Also returns the true peak height."""
    grid_methods = [("SAMBoost", "samboost", False),
                    ("GPBoost", "gpboost", False),
                    ("INLA_" + cfg["inla_variant"], cfg["inla_variant"], True)]
    res = {}
    for mlabel, key, is_inla in grid_methods:
        per = []
        for s in seeds:
            p = grid_path(cfg, key, is_inla, s)
            if not p.exists():
                continue
            g = pd.read_csv(p)
            crd = g[["s1", "s2"]].values
            riv = river_indicator(crd) == 1
            disk = np.sqrt(((crd - HOTSPOT_CENTER) ** 2).sum(1)) < 0.2
            per.append((_rmse(g.b_hat.values[riv], g.b_true.values[riv]),
                        _rmse(g.b_hat.values[disk], g.b_true.values[disk]),
                        float(g.b_hat.values[disk].max())))
        if per:
            a = np.array(per)
            res[mlabel] = dict(leak=(a[:, 0].mean(), a[:, 0].std()),
                               peakrmse=(a[:, 1].mean(), a[:, 1].std()),
                               peakh=(a[:, 2].mean(), a[:, 2].std()))
    # true peak height, from the SAMBoost grid of seed 1 (None if not available)
    p0 = grid_path(cfg, "samboost", False, 1)
    if not p0.exists():
        return res, None
    g0 = pd.read_csv(p0)
    d0 = np.sqrt(((g0[["s1", "s2"]].values - HOTSPOT_CENTER) ** 2).sum(1)) < 0.2
    true_peak = float(g0.b_true.values[d0].max())
    return res, true_peak


# >> Figures
def _hm(ax, s1, s2, v, title, vmax, river, title_fs=FS_FIELD):
    sc = ax.scatter(s1, s2, c=v, cmap="RdBu_r", s=MARK, vmin=-vmax, vmax=vmax)
    if river:
        ax.plot([RIVER_P1[0], RIVER_P2[0]], [RIVER_P1[1], RIVER_P2[1]],
                color="navy", lw=1.2, alpha=0.5)
        ax.plot(*HOTSPOT_C, "kx", ms=9, mew=2.5)
    ax.set_title(title, fontsize=title_fs, y=TITLE_Y)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    return sc


def field_figure(cfg, methods, seed, out_name):
    """Recovered spatial field for one seed: true field + each method."""
    ref = pd.read_csv(grid_path(cfg, "samboost", False, seed))
    s1, s2, bt = ref.s1.values, ref.s2.values, ref.b_true.values
    vmax = np.percentile(np.abs(bt), 99)
    n = 1 + len(methods)
    fig, ax = plt.subplots(1, n, figsize=(4.0 * n, 4.3))
    sc = _hm(ax[0], s1, s2, bt, r"True $F^{*}_{\mathrm{spa}}(s)$", vmax, cfg["river"])
    for a, (lab, key, is_inla) in zip(ax[1:], methods):
        g = pd.read_csv(grid_path(cfg, key, is_inla, seed))
        _hm(a, g.s1.values, g.s2.values, g.b_hat.values, lab, vmax, cfg["river"])
    cb = fig.colorbar(sc, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("Spatial effect", fontsize=FS_CBAR); cb.ax.tick_params(labelsize=FS_CBAR - 4)
    fig.savefig(OUT / out_name, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out_name)


def field_figure_mean(cfg, methods, out_name):
    """Recovered spatial field averaged over all seeds: true field + each method."""
    seeds = sorted(int(os.path.basename(p).split("_")[1])
                   for p in glob.glob(str(Path(cfg["sb_run"]) / "seed_*")))
    ref = pd.read_csv(grid_path(cfg, "samboost", False, seeds[0]))
    s1, s2, bt = ref.s1.values, ref.s2.values, ref.b_true.values
    vmax = np.percentile(np.abs(bt), 99)

    def meanf(key, is_inla):
        arr = np.vstack([pd.read_csv(grid_path(cfg, key, is_inla, sd)).b_hat.values
                         for sd in seeds if grid_path(cfg, key, is_inla, sd).exists()])
        return arr.mean(0)

    n = 1 + len(methods)
    fig, ax = plt.subplots(1, n, figsize=(4.0 * n, 4.3))
    sc = _hm(ax[0], s1, s2, bt, r"True $F^{*}_{\mathrm{spa}}(s)$", vmax, cfg["river"])
    for a, (lab, key, is_inla) in zip(ax[1:], methods):
        _hm(a, s1, s2, meanf(key, is_inla), lab, vmax, cfg["river"])
    cb = fig.colorbar(sc, ax=ax, fraction=0.02, pad=0.01)
    cb.set_label("Spatial effect", fontsize=FS_CBAR); cb.ax.tick_params(labelsize=FS_CBAR - 4)
    fig.savefig(OUT / out_name, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out_name)


def domain_figure(cfg, out_name):
    """Domain layout: true spatial field, true covariate effect, response y."""
    df = pd.read_csv(DATA / cfg["subdir"] / "train_df.csv")
    panels = [("spatial_effect", r"True spatial field $F^{*}_{\mathrm{spa}}(s)$", "RdBu_r", True),
              ("F_effect", r"Covariate effect $F^{*}_{\mathrm{cov}}(x)$", "RdBu_r", True)]
    n = len(panels) + 1
    fig, ax = plt.subplots(1, n, figsize=(4.4 * n, 4.3))
    for a, (col, title, cmap, sym) in zip(ax[:-1], panels):
        v = df[col].values
        m = np.percentile(np.abs(v), 98)
        sc = a.scatter(df.s1, df.s2, c=v, cmap=cmap, s=MARK, vmin=-m, vmax=m) if sym \
            else a.scatter(df.s1, df.s2, c=v, cmap=cmap, s=MARK)
        if cfg["river"]:
            a.plot([RIVER_P1[0], RIVER_P2[0]], [RIVER_P1[1], RIVER_P2[1]],
                   color="navy", lw=1.2, alpha=0.4)
        a.set_title(title, fontsize=FS_DOMAIN, y=TITLE_Y)
        a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
        cb = fig.colorbar(sc, ax=a, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=FS_CBAR - 4)
    a = ax[-1]
    sc = a.scatter(df.s1, df.s2, c=df.y, cmap="viridis", s=MARK)
    a.set_title(r"Response $y$", fontsize=FS_DOMAIN, y=TITLE_Y)
    a.set_xticks([]); a.set_yticks([]); a.set_aspect("equal")
    cb = fig.colorbar(sc, ax=a, fraction=0.046, pad=0.04); cb.ax.tick_params(labelsize=FS_CBAR - 4)
    fig.savefig(OUT / out_name, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out_name)


def ablation_figure(complex_cfg, simple_cfg, out_name):
    """2x2 covariate-effect recovery: rows = simple/complex F, cols = SAMBoost/INLA.
    Shows INLA's F collapse appears only under complex F."""
    ms = rep_seed(simple_cfg)
    mc = rep_seed(complex_cfg)
    fig, ax = plt.subplots(2, 2, figsize=(8.4, 8.4))
    panels = [(0, 0, "SAMBoost, simple F", simple_cfg, "samboost", False, ms),
              (0, 1, "INLA, simple F", simple_cfg, simple_cfg["inla_variant"], True, ms),
              (1, 0, "SAMBoost, complex F", complex_cfg, "samboost", False, mc),
              (1, 1, "INLA, complex F", complex_cfg, complex_cfg["inla_variant"], True, mc)]
    for i, j, title, cfg, key, is_inla, seed in panels:
        a = ax[i, j]
        te = pd.read_csv(pred_path(cfg, key, is_inla, seed)); te = te[te.split == "test"]
        a.scatter(te.F_true, te.F_hat, s=10, alpha=0.4, color="#d62728")
        lo, hi = te.F_true.min(), te.F_true.max()
        a.plot([lo, hi], [lo, hi], "k--", lw=1.2)
        r = float(np.corrcoef(te.F_true, te.F_hat)[0, 1])
        a.set_title(f"{title}   ($r_{{\\mathrm{{cov}}}}$={r:.3f})", fontsize=11)
        if j == 0:
            a.set_ylabel("predicted $F(x)$", fontsize=10)
        if i == 1:
            a.set_xlabel("true $F(x)$", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / out_name, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", out_name)


# >> Tables
def aggregate_table(cfg):
    """30-seed mean +/- sd table across all methods -> table_agg_<sim>.{csv,tex}."""
    sim, task = cfg["name"], cfg["task"]
    print(f"\n  Aggregate table: {sim} [{task}]")
    raw = _load_raw(cfg)
    if not len(raw):
        print("    [skip] no raw files found; run the multi-seed drivers first.")
        return
    cols = [c for c in _AGG_COLS[task] if c in raw.columns]
    methods = [m for m in _AGG_METHOD_ORDER if m in set(raw["method"])]
    if not methods:
        print("    [skip] no known methods present.")
        return
    agg_mean = raw.groupby("method")[cols].mean()
    agg_std = raw.groupby("method")[cols].std()
    n_seeds = raw.groupby("method").size()
    best = {c: (agg_mean.loc[methods, c].max() if c in _HIGHER_BETTER
                else agg_mean.loc[methods, c].min())
            for c in cols if c != "fit_time"}

    rows = []
    for m in methods:
        row = {"Method": _AGG_METHOD_LABEL.get(m, m), "n": int(n_seeds[m])}
        for c in cols:
            dec = 1 if c == "fit_time" else 3
            row[_COL_LABEL[c]] = f"{agg_mean.loc[m, c]:.{dec}f} +/- {agg_std.loc[m, c]:.{dec}f}"
        rows.append(row)
    disp = pd.DataFrame(rows)
    disp.to_csv(OUT / f"table_agg_{sim}.csv", index=False, encoding="utf-8")

    lines = ["% Auto-generated by build_paper.py", f"% {sim} -- 30-seed mean +/- sd; best per column bold.",
             "\\begin{tabular}{l" + "r" * len(cols) + "}", "\\toprule",
             "Method & " + " & ".join(_COL_LABEL_TEX[c] for c in cols) + r" \\", "\\midrule"]
    for m in methods:
        cells = []
        for c in cols:
            dec = 1 if c == "fit_time" else 3
            mu, sd = agg_mean.loc[m, c], agg_std.loc[m, c]
            is_best = (c != "fit_time") and np.isclose(mu, best.get(c, np.nan))
            mu_s = f"\\mathbf{{{mu:.{dec}f}}}" if is_best else f"{mu:.{dec}f}"
            cells.append(f"${mu_s} \\pm {sd:.{dec}f}$")
        lines.append(f"{_AGG_METHOD_LABEL.get(m, m)} & " + " & ".join(cells) + r" \\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    (OUT / f"table_agg_{sim}.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"    saved -> table_agg_{sim}.{{csv,tex}}")


def augmented_table(cfg, out_name):
    """Main dissociation metrics + leak / peak columns -> <out_name>.{csv,tex}."""
    print(f"\n  Augmented table: {cfg['name']} -> {out_name}")
    raw = _load_raw(cfg)
    if not len(raw):
        print("    [skip] no raw files found.")
        return
    MAIN = ["rmse", "r_spatial", "rmse_spatial", "r_cov", "rmse_cov"]
    main = raw.groupby("method")[MAIN].agg(["mean", "std"])
    leak, true_peak = leak_peak(cfg)

    present = [m for m in _AGG_METHOD_ORDER if m in main.index]
    cols = ["RMSE", "r_sp", "RMSE_sp", "r_cov", "RMSE_cov", "leak", "peak_rmse", "peak_h"]

    def cell(m, met):
        return f"{main.loc[m, (met, 'mean')]:.3f}+/-{main.loc[m, (met, 'std')]:.3f}"

    def lc(m, k):
        return f"{leak[m][k][0]:.3f}+/-{leak[m][k][1]:.3f}" if m in leak else "--"

    tbl = [[m, cell(m, "rmse"), cell(m, "r_spatial"), cell(m, "rmse_spatial"),
            cell(m, "r_cov"), cell(m, "rmse_cov"), lc(m, "leak"), lc(m, "peakrmse"), lc(m, "peakh")]
           for m in present]
    pd.DataFrame(tbl, columns=["Method"] + cols).to_csv(OUT / f"{out_name}.csv", index=False)

    peak_txt = f"{true_peak:.2f}" if true_peak is not None else "--"
    lines = [f"% Auto-generated by build_paper.py; true peak height ~ {peak_txt}",
             r"\begin{tabular}{l" + "c" * len(cols) + "}", r"\toprule",
             "Method & " + " & ".join(cols).replace("_", r"\_") + r" \\", r"\midrule"]
    for r_ in tbl:
        lines.append(r_[0].replace("_", r"\_") + " & "
                     + " & ".join(f"${x}$".replace("+/-", r"\pm") for x in r_[1:]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (OUT / f"{out_name}.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"    true peak height ~ {peak_txt}; saved -> {out_name}.{{csv,tex}}")


def ablation_table(complex_cfg, simple_cfg, out_name):
    """SAMBoost vs GPBoost vs INLA under simple vs complex F -> <out_name>.{csv,tex}.
    Shows INLA's r_cov collapses only under complex F while SAMBoost holds.
    GPBoost uses fixed rich-F params for the simple-F row (controlled comparison)."""
    print(f"\n  Ablation table -> {out_name}")
    MAIN = ["rmse", "r_spatial", "rmse_spatial", "r_cov", "rmse_cov"]
    inla_lbl = "INLA_" + complex_cfg["inla_variant"]
    rows = []
    for F_label, cfg in [("simple", simple_cfg), ("complex", complex_cfg)]:
        raw = _load_raw(cfg)
        if not len(raw):
            print(f"    [skip] no raw for {cfg['name']}")
            return
        # For simple F, load GPBoost fixed-params results (rich-F hyperparameters)
        if F_label == "simple" and "gpb_fixed_run" in cfg:
            gpb_csv = Path(cfg["gpb_fixed_run"]) / "all_seeds_raw.csv"
            if gpb_csv.exists():
                gpb_df = pd.read_csv(gpb_csv)
                gpb_df["method"] = "GPBoost"
                raw = pd.concat([raw, gpb_df], ignore_index=True)
        mean = raw.groupby("method")[MAIN].mean()
        leak, _ = leak_peak(cfg)
        for m, disp in [("SAMBoost", "SAMBoost"), ("GPBoost", "GPBoost"), (inla_lbl, "INLA " + complex_cfg["inla_variant"])]:
            if m not in mean.index:
                continue
            lk = leak[m]["leak"][0] if m in leak else np.nan
            rows.append({"Method": disp, "F": F_label,
                         "RMSE": mean.loc[m, "rmse"], "r_sp": mean.loc[m, "r_spatial"],
                         "RMSE_sp": mean.loc[m, "rmse_spatial"], "r_cov": mean.loc[m, "r_cov"],
                         "RMSE_cov": mean.loc[m, "rmse_cov"], "leak": lk})
    # order: method blocks, simple then complex
    df = pd.DataFrame(rows).sort_values(["Method", "F"], ascending=[True, True]).reset_index(drop=True)
    num = ["RMSE", "r_sp", "RMSE_sp", "r_cov", "RMSE_cov", "leak"]
    df_disp = df.copy()
    for c in num:
        df_disp[c] = df_disp[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "--")
    df_disp.to_csv(OUT / f"{out_name}.csv", index=False)

    lines = ["% Auto-generated by build_paper.py",
             r"\begin{tabular}{ll" + "c" * len(num) + "}", r"\toprule",
             "Method & F & " + " & ".join(c.replace("_", r"\_") for c in num) + r" \\", r"\midrule"]
    for _, r_ in df.iterrows():
        cells = " & ".join(f"${r_[c]:.3f}$" if pd.notna(r_[c]) else "--" for c in num)
        lines.append(f"{r_['Method']} & {r_['F']} & {cells}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    (OUT / f"{out_name}.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"    saved -> {out_name}.{{csv,tex}}")
