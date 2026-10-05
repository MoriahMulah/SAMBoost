"""
variogram.py
============
Empirical variogram and estimation of the spatial range.

The models call fit_variogram themselves when kappa=None: at the end of the
freeze rounds, on the residuals y - link(F_cov). No paper run uses this,
since kappa is always given (chosen by cross-validation).

    fit_variogram(coords, residuals)   empirical variogram + parametric fit,
                                       returns a VariogramResult
    plot_variogram(result)             four-panel diagnostic figure
    make_kappa_schedule(range, ...)    kappa per level for n_levels > 1

Note on the range: range_optimal is where the fitted variogram reaches 95% of
the sill. With the default Matern 2.5 model this is not exactly the practical
range of the paper (correlation about 0.1 for Matern nu = 1); kappa_optimal
= sqrt(8) / range_optimal is a starting value, not an exact equivalent.
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist, squareform
import warnings


# >> Parametric variogram models

def _variogram_exponential(h, nugget, sill, range_):
    """
    Exponential variogram: gamma(h) = nugget + (sill-nugget)*(1 - exp(-h/range_))
    Corresponds to Matern nu=0.5.
    """
    return nugget + (sill - nugget) * (1.0 - np.exp(-h / (range_ + 1e-12)))


def _variogram_matern15(h, nugget, sill, range_):
    """
    Matern nu=1.5 variogram (smooth, 1 mean-square differentiable).
    gamma(h) = nugget + (sill-nugget)*(1 - (1 + sqrt(3)*h/range_)*exp(-sqrt(3)*h/range_))
    """
    u = np.sqrt(3.0) * h / (range_ + 1e-12)
    return nugget + (sill - nugget) * (1.0 - (1.0 + u) * np.exp(-u))


def _variogram_matern25(h, nugget, sill, range_):
    """
    Matern nu=2.5 variogram (smooth, 2 mean-square differentiable).
    Used as default because it balances smoothness and flexibility.
    """
    u = np.sqrt(5.0) * h / (range_ + 1e-12)
    return nugget + (sill - nugget) * (1.0 - (1.0 + u + u**2 / 3.0) * np.exp(-u))


MODELS = {
    "exponential": _variogram_exponential,
    "matern15":    _variogram_matern15,
    "matern25":    _variogram_matern25,
}


# >> Result container

class VariogramResult:
    """
    Container for empirical variogram and fitted model parameters.

    Attributes
    ----------
    bin_centers   : (n_bins,)  midpoint distance of each bin (coord units)
    gamma_emp     : (n_bins,)  empirical semivariance at each bin
    bin_counts    : (n_bins,)  number of pairs in each bin
    nugget        : float      fitted nugget (variance at distance 0)
    sill          : float      fitted sill (total variance at large distances)
    range_param   : float      fitted range parameter in coordinate units
    model_name    : str        fitted variogram model name
    range_optimal : float      distance where the fitted variogram reaches 95%
                               of the sill (found numerically)
    kappa_optimal : float      SPDE kappa = sqrt(8) / range_optimal  (use in compute_Q)
    r2_fit        : float      R^2 of the variogram model fit
    n_sub         : int        number of points used in the subsample
    n_pairs       : int        total number of pairs used
    fit_success   : bool       whether curve_fit converged
    coords_sub    : (n_sub,2)  subsampled coordinates (for diagnostics)
    residuals_sub : (n_sub,)   subsampled residuals
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def summary(self) -> str:
        lines = [
            "",
            "=" * 55,
            "  Variogram Fit Summary",
            "=" * 55,
            f"  Model              : {self.model_name}",
            f"  Subsampled points  : {self.n_sub:,}",
            f"  Pairs used         : {self.n_pairs:,}",
            f"  Fit converged      : {self.fit_success}",
            f"  R^2 of fit         : {self.r2_fit:.4f}",
            "-" * 55,
            f"  Nugget             : {self.nugget:.6f}",
            f"  Sill               : {self.sill:.6f}",
            f"  Range parameter    : {self.range_param:.4f}  (coord units)",
            f"  Practical range    : {self.range_optimal:.4f}  (95% sill distance)",
            f"  SPDE kappa         : {self.kappa_optimal:.4f}  (sqrt(8) / practical range)",
            "=" * 55,
            "",
            "  HOW TO INTERPRET",
            "  Nugget  : variance at distance 0 - measurement noise or",
            "            micro-scale variation below the mesh resolution.",
            "            High nugget = noisy labels or very local effects.",
            "  Sill    : total spatial variance. (Sill - Nugget) is the",
            "            variance explained by spatial structure.",
            "  Range   : distance at which observations become ~independent.",
            "            Observations closer than the range are spatially",
            "            correlated; further apart they are independent.",
            f"  kappa={self.kappa_optimal:.4f} : use this as kappa in compute_Q().",
            "=" * 55,
            "",
        ]
        return "\n".join(lines)


# >> Core computation

def fit_variogram(
    coords: np.ndarray,
    residuals: np.ndarray,
    n_sub: int = 20_000,
    n_bins: int = 25,
    max_dist: float = None,
    model: str = "matern25",
    random_state: int = 42,
) -> VariogramResult:
    """
    Compute the empirical variogram of residuals and fit a parametric model.

    For large datasets a random subsample of n_sub points is used, since the
    number of pairs grows as n_sub^2.

    Algorithm
    ---------
    1. Draw n_sub random points from (coords, residuals).
    2. Compute all n_sub*(n_sub-1)/2 pairwise distances and semivariances:
           gamma_hat(h_ij) = 1/2 (residuals_i - residuals_j)^2
    3. Bin by distance into n_bins equally-spaced bins up to max_dist.
    4. Compute mean semivariance per bin (the empirical variogram).
    5. Fit the chosen parametric model by nonlinear least squares
       (scipy.optimize.curve_fit, weighted by bin pair counts).
    6. Derive the practical range and SPDE kappa from the fitted parameters.

    Parameters
    ----------
    coords : (N, 2)
        Observation coordinates. Units must be consistent - if lon/lat in
        degrees, the range output will be in degrees (1 deg ~ 111km).
        If projected (metres), the range will be in metres.
    residuals : (N,)
        Residuals to analyse. The models pass y - link(F_cov) after the
        freeze rounds, so that part of the covariate signal is removed.
    n_sub : int, default 20_000
        Subsample size. Runtime and memory grow as n_sub^2.
    n_bins : int, default 25
        Number of distance bins. More bins = finer resolution but noisier
        per-bin estimates at large distances.
    max_dist : float or None
        Maximum distance to include. None -> half of the domain diagonal
        (bins at larger distances have few pairs).
    model : {"exponential", "matern15", "matern25"}, default "matern25"
        Parametric variogram model to fit.
    random_state : int
        Seed for the subsample draw.

    Returns
    -------
    VariogramResult
        See class docstring for all attributes.
    """
    rng = np.random.default_rng(random_state)
    N   = len(residuals)

    # Subsample
    n_sub = min(n_sub, N)
    idx   = rng.choice(N, size=n_sub, replace=False)
    c_sub = coords[idx]        # (n_sub, 2)
    r_sub = residuals[idx]     # (n_sub,)

    # Pairwise distances and semivariances
    # pdist returns the upper triangle, shape n_sub*(n_sub-1)/2
    dists = pdist(c_sub, metric="euclidean")      # (n_pairs,)
    # Semivariance: gamma(h) = 1/2 E[(Z(s) - Z(s+h))^2]
    # For a pair (i,j): gamma_ij = 1/2 (r_i - r_j)^2
    r_pairs = pdist(r_sub.reshape(-1, 1), metric="euclidean")
    gamma   = 0.5 * r_pairs ** 2                  # (n_pairs,)

    # Automatic max_dist
    if max_dist is None:
        domain_diagonal = np.sqrt(
            (c_sub[:, 0].max() - c_sub[:, 0].min()) ** 2 +
            (c_sub[:, 1].max() - c_sub[:, 1].min()) ** 2
        )
        max_dist = 0.5 * domain_diagonal

    # Bin by distance
    bin_edges  = np.linspace(0, max_dist, n_bins + 1)
    bin_ids    = np.digitize(dists, bin_edges) - 1   # 0-indexed bin

    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    gamma_emp   = np.full(n_bins, np.nan)
    bin_counts  = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = (bin_ids == b)
        cnt  = mask.sum()
        bin_counts[b] = cnt
        if cnt >= 2:
            gamma_emp[b] = gamma[mask].mean()

    # Drop bins with too few pairs (unreliable estimates)
    valid = (bin_counts >= 10) & ~np.isnan(gamma_emp)
    h_fit = bin_centers[valid]
    g_fit = gamma_emp[valid]
    w_fit = np.sqrt(bin_counts[valid].astype(float))   # weight by sqrt(n_pairs)

    # Fit parametric model
    vfunc      = MODELS[model]
    sill_init  = float(np.nanpercentile(gamma_emp, 90))
    range_init = float(max_dist / 4.0)

    p0     = [0.0, sill_init, range_init]
    bounds = ([0, 1e-8, 1e-6], [sill_init, sill_init * 2, max_dist])

    fit_success = True
    try:
        popt, _ = curve_fit(
            vfunc, h_fit, g_fit,
            p0     = p0,
            bounds = bounds,
            sigma  = 1.0 / (w_fit + 1e-6),
            maxfev = 5000,
        )
        nugget, sill, range_param = popt
    except (RuntimeError, ValueError) as e:
        warnings.warn(f"Variogram fit did not converge: {e}. "
                      "Using moment estimates.", RuntimeWarning)
        fit_success = False
        nugget      = 0.0
        sill        = float(np.nanpercentile(gamma_emp, 95))
        range_param = range_init

    # R^2 of the fit
    g_pred = vfunc(h_fit, nugget, sill, range_param)
    ss_res = np.sum((g_fit - g_pred) ** 2)
    ss_tot = np.sum((g_fit - g_fit.mean()) ** 2)
    r2_fit = float(1.0 - ss_res / (ss_tot + 1e-12))

    # Practical range: first h with gamma(h) >= nugget + 0.95 (sill - nugget)
    target = nugget + 0.95 * (sill - nugget + 1e-12)
    h_test = np.linspace(0, max_dist * 2, 2000)
    g_test = vfunc(h_test, nugget, sill, range_param)
    # Find first crossing of target
    cross  = np.where(g_test >= target)[0]
    range_optimal = float(h_test[cross[0]]) if len(cross) > 0 else float(3 * range_param)

    # SPDE kappa
    kappa_optimal = float(np.sqrt(8.0) / (range_optimal + 1e-12))

    return VariogramResult(
        bin_centers   = bin_centers,
        gamma_emp     = gamma_emp,
        bin_counts    = bin_counts,
        nugget        = float(nugget),
        sill          = float(sill),
        range_param   = float(range_param),
        range_optimal = range_optimal,
        kappa_optimal = kappa_optimal,
        model_name    = model,
        r2_fit        = r2_fit,
        n_sub         = n_sub,
        n_pairs       = len(dists),
        fit_success   = fit_success,
        coords_sub    = c_sub,
        residuals_sub = r_sub,
        max_dist_used = max_dist,
    )


# >> Plotting

def plot_variogram(
    result: VariogramResult,
    title: str = None,
    save_path: str = None,
):
    """
    Four-panel diagnostic figure for the variogram fit.

    Panel 1 (top-left):   Empirical variogram + fitted model curve.
                          The most important panel. Shows how well the
                          parametric model captures the data.

    Panel 2 (top-right):  Pair count per bin.
                          Bins with few pairs have unreliable gamma estimates.
                          The fitted model ignores bins with < 10 pairs.

    Panel 3 (bottom-left): Spatial map of residuals on the subsample.
                           Shows the spatial pattern of what is being
                           analysed. Useful for spotting outliers or
                           non-stationarity.

    Panel 4 (bottom-right): Residual autocorrelation vs distance.
                            Equivalent to 1 - gamma(h)/sill, shows the
                            correlation decay. Reaches 0 at the practical
                            range.
    """
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.colors import TwoSlopeNorm

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32,
                            left=0.08, right=0.96, top=0.90, bottom=0.07)

    h_curve = np.linspace(0, result.max_dist_used * 1.05, 300)
    g_curve = MODELS[result.model_name](
        h_curve, result.nugget, result.sill, result.range_param
    )

    BLUE   = "#2563EB"
    RED    = "#DC2626"
    GREEN  = "#16A34A"
    ORANGE = "#D97706"

    # Panel 1: empirical variogram + fitted model
    ax1 = fig.add_subplot(gs[0, 0])

    valid = ~np.isnan(result.gamma_emp)
    ax1.scatter(
        result.bin_centers[valid], result.gamma_emp[valid],
        s      = np.sqrt(result.bin_counts[valid]) * 4,
        color  = BLUE, alpha=0.75, zorder=3,
        label  = "Empirical gamma (size prop. to sqrt(pairs))"
    )
    ax1.plot(h_curve, g_curve, color=RED, linewidth=2.2,
             label=f"Fitted {result.model_name}")

    # Mark nugget, sill, practical range
    ax1.axhline(result.nugget, color="gray", linestyle=":", linewidth=1,
                alpha=0.7, label=f"Nugget={result.nugget:.4f}")
    ax1.axhline(result.sill, color=ORANGE, linestyle="--", linewidth=1.2,
                alpha=0.7, label=f"Sill={result.sill:.4f}")
    ax1.axvline(result.range_optimal, color=GREEN, linestyle="--", linewidth=1.5,
                label=f"Practical range={result.range_optimal:.4f}")

    ax1.set_xlabel("Distance h (coord units)", fontsize=9)
    ax1.set_ylabel("Semivariance gamma(h)", fontsize=9)
    ax1.set_title(f"Empirical Variogram + Fit\n"
                  f"R^2={result.r2_fit:.3f}  kappa={result.kappa_optimal:.4f}",
                  fontweight="bold", fontsize=9)
    ax1.legend(fontsize=7, loc="lower right")
    ax1.grid(True, alpha=0.3, linewidth=0.6)
    ax1.set_xlim(0, result.max_dist_used * 1.05)
    ax1.set_ylim(bottom=0)

    # Panel 2: pair counts per bin
    ax2 = fig.add_subplot(gs[0, 1])
    colors_bar = [BLUE if c >= 10 else "lightgray" for c in result.bin_counts]
    ax2.bar(result.bin_centers, result.bin_counts,
            width=result.bin_centers[1] - result.bin_centers[0] if len(result.bin_centers) > 1 else 1,
            color=colors_bar, alpha=0.8, edgecolor="white")
    ax2.axhline(10, color=RED, linestyle="--", linewidth=1,
                label="Min threshold (10 pairs)")
    ax2.set_xlabel("Distance h (coord units)", fontsize=9)
    ax2.set_ylabel("Number of pairs", fontsize=9)
    ax2.set_title("Pairs per distance bin\n(gray = excluded from fit)",
                  fontweight="bold", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3, axis="y", linewidth=0.6)

    # Panel 3: spatial map of residuals
    ax3 = fig.add_subplot(gs[1, 0])
    vmax = np.percentile(np.abs(result.residuals_sub), 97)
    vmax = max(vmax, 1e-6)
    norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)
    sc   = ax3.scatter(
        result.coords_sub[:, 0], result.coords_sub[:, 1],
        c=result.residuals_sub, cmap="RdBu_r", norm=norm,
        s=6, alpha=0.6, linewidths=0,
    )
    plt.colorbar(sc, ax=ax3, shrink=0.85, label="Residual")
    ax3.set_xlabel("Coordinate 1", fontsize=9)
    ax3.set_ylabel("Coordinate 2", fontsize=9)
    ax3.set_title(f"Spatial pattern of residuals\n(n={result.n_sub:,} subsampled points)",
                  fontweight="bold", fontsize=9)
    ax3.set_aspect("equal")

    # Panel 4: correlation decay
    ax4 = fig.add_subplot(gs[1, 1])
    sill_nug = result.sill - result.nugget + 1e-12
    corr_emp = np.where(
        ~np.isnan(result.gamma_emp),
        1.0 - (result.gamma_emp - result.nugget) / sill_nug,
        np.nan,
    )
    corr_curve = 1.0 - (g_curve - result.nugget) / sill_nug

    valid2 = ~np.isnan(corr_emp)
    ax4.scatter(result.bin_centers[valid2], corr_emp[valid2],
                s=8, color=BLUE, alpha=0.75, zorder=3, label="Empirical correlation")
    ax4.plot(h_curve, corr_curve, color=RED, linewidth=2.2,
             label="Fitted model")
    ax4.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax4.axhline(0.05, color=GREEN, linewidth=1.2, linestyle="--",
                label="5% correlation (practical range)")
    ax4.axvline(result.range_optimal, color=GREEN, linewidth=1.5, linestyle="--")
    ax4.set_xlabel("Distance h (coord units)", fontsize=9)
    ax4.set_ylabel("Spatial correlation", fontsize=9)
    ax4.set_title("Correlation decay with distance",
                  fontweight="bold", fontsize=9)
    ax4.set_xlim(0, result.max_dist_used * 1.05)
    ax4.set_ylim(-0.1, 1.05)
    ax4.legend(fontsize=7)
    ax4.grid(True, alpha=0.3, linewidth=0.6)

    # Title
    main_title = title or "Variogram Analysis"
    fig.suptitle(
        f"{main_title}\n"
        f"Model: {result.model_name}  |  "
        f"Practical range: {result.range_optimal:.4f}  |  "
        f"kappa: {result.kappa_optimal:.4f}  |  "
        f"R^2: {result.r2_fit:.3f}",
        fontsize=10, fontweight="bold", y=0.97,
    )

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


# >> Range schedule for n_levels > 1

def make_kappa_schedule(
    range_base: float,
    n_levels: int = 3,
    ratio: float = 2.0,
) -> list:
    """
    Build a geometric sequence of kappa values for models with n_levels > 1.

    Given the base range rho_base (from variogram fitting), creates L levels
    with ranges rho_1 = rho_base, rho_2 = rho_base/ratio, ..., rho_L = rho_base/ratio^(L-1).

    Level 1 (largest range):  captures broad geographic trends.
    Level L (smallest range): captures local micro-variation.

    Parameters
    ----------
    range_base : float  practical range from fit_variogram (coordinate units)
    n_levels   : int    number of resolution levels (2 or 3 recommended)
    ratio      : float  range ratio between levels (2.0 = octave, like wavelets)

    Returns
    -------
    list of dicts:  [{"level": l, "rho": rho_l, "kappa": kappa_l}, ...]
    """
    schedule = []
    for l in range(n_levels):
        rho_l   = range_base / (ratio ** l)
        kappa_l = np.sqrt(8.0) / rho_l
        schedule.append({
            "level" : l + 1,
            "rho"   : round(rho_l, 6),
            "kappa" : round(kappa_l, 6),
        })
    return schedule