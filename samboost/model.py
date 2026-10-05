"""
model.py
========
SAMBoost: spatial gradient boosting for binary outcomes (Bernoulli
likelihood, logit link). This is the classification model of the paper.
model_continuous.py is the regression version and model_extended.py an extended
version; all three share the spatial part described here.

Each round has two stages (Algorithm 1 of the paper):
  Stage 1  a regression tree is grown on the negative gradient using the
           covariates only; its leaves get the Newton values
           -G_k / (H_k + reg_lambda).
  Stage 2  the spatial field z on the mesh nodes is updated by solving
           (A'HA + alpha Q) dz = -(A'g + alpha Q z).

Correspondence with the paper
-----------------------------
  alpha          lambda_spa, spatial penalty
  reg_lambda     lambda_tree, L2 penalty on the leaf values
  learning_rate  eta_tree (and eta_spa when w_mode="uniform")
  kappa          SPDE range parameter, practical range rho = sqrt(8) / kappa
  Q              tau^2 (kappa^4 C + 2 kappa^2 G + G C^-1 G), with
                 tau = 1 / (sqrt(4 pi) kappa) so that the marginal variance is 1

Spatial step size (w_mode)
--------------------------
The committed update is eta * w * dz.
  "uniform"  eta = learning_rate, w = 1. This is the update of Algorithm 1.
  "no_geo"   eta = Newton step size along dz, w = 1 (used in the simulations):
                 eta* = (b + alpha q) / (b + 2 alpha q), in (0.5, 1],
                 b = dz' A'HA dz,  q = dz' Q dz.
  "full"     Newton eta and a per-node weight w in [w_slow_factor, 1] that is
             larger where the covariate loss has stopped improving.
  "no_eta"   eta = 1 with the per-node weight.
  "unit"     eta = 1, w = 1 (full WLS step).
  "binary"   eta = 1, w in {learning_rate * w_slow_factor, learning_rate}.

Other options
-------------
  n_freeze        the first n_freeze rounds fit trees only.
  center_spatial  after each spatial update the mean of A dz over the training
                  points is moved from z to the intercept F0_, so that the
                  spatial effect has mean zero on the training data.
  penalize_z      True: the penalty acts on the accumulated field, as in the
                  system above. False: it acts on the increment dz only.
  kappa           a float, a list (one value per level), or None to estimate it
                  from a variogram of the residuals at the end of the freeze.
  n_levels        several fields z_1..z_L on the same mesh, from coarse to fine
                  (kappa_l = kappa * ratio^(l-1)), each active in turn for an
                  equal share of the rounds after the freeze. Not used in the
                  paper (n_levels=1). It is not the coarse-to-fine mesh
                  schedule mentioned as future work in the paper: all levels
                  share one mesh.

Attributes set by fit()
-----------------------
    alpha_schedule_         (T,)      alpha_t per round
    w_nodes_schedule_       (T, M)    per-node weight per round (NaN = freeze)
    eta_schedule_           (T,)      Newton step size (NaN when eta = 1)
    all_znodes_             (T, L, M) cumulative z_l per round and level
    loss_nodes_schedule_    (T, M)    covariate loss at the nodes per round
    kappa_per_level_        (L,)      kappa of each level
    rho_per_level_          (L,)      practical range of each level
    tau_per_level_          (L,)      tau of each level
    active_level_per_round_ (T,)      active level index (-1 = freeze)
    kappa_source_           str       "user_float", "user_list" or "variogram"
    variogram_result_       VariogramResult or None
"""

import warnings
import numpy as np
from collections import deque

from scipy.special import expit, xlogy
from scipy.sparse import issparse, diags
from scipy.sparse.linalg import spsolve, cg as sparse_cg
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from utils.mesh_building import compute_Q, assemble_spde_matrices
from utils.variogram import fit_variogram


class SAMBoost(BaseEstimator, ClassifierMixin):
    """
    Spatial gradient boosting for binary outcomes. See the module docstring.

    Parameters
    ----------
    A : sparse (N_train, M)
        Barycentric projection matrix.
    G : sparse (M, M)
        FEM stiffness matrix from assemble_spde_matrices(mesh_nodes).
    C : sparse (M, M)
        FEM mass matrix from assemble_spde_matrices(mesh_nodes).
    gamma : float, default 0.1
        Left from earlier experiments (weight of a spatial gradient term that
        is switched off). It has no effect, except as the default for alpha
        when alpha is None. The paper runs use gamma=0 with an explicit alpha.
    alpha : float or None, default None
        Spatial penalty lambda_spa, same for all levels.
        If None, gamma is used.
    alpha_start_scale : float, default 1.0
        Cosine decay multiplier. 1.0 = flat schedule (no decay).
    n_estimators : int, default 100
        Total boosting rounds (freeze + all level rounds).
    learning_rate : float, default 0.1
        Shrinkage eta_tree on the tree contributions (also the spatial step
        when w_mode="uniform").
    n_freeze : int, default 5
        Tree-only rounds before the spatial updates start.
        If kappa=None, the variogram is estimated at the end of this phase.
        Must be >= 1.
    n_levels : int, default 3
        Number of resolution levels. Must match len(kappa) when kappa is a list.
    kappa : float, list of float, or None, default None
        Spatial range specification. See module docstring for the three modes.
        None       -> variogram estimated from freeze-phase residuals.
        float      -> kappa for level 1 (coarsest); ratio applied for finer levels.
        list       -> explicit kappa per level; ratio ignored; len must equal n_levels.
    ratio : float, default 2.0
        Geometric factor between level kappas when kappa is None or float.
        kappa_l = kappa_1 * ratio^(l-1).
        Ignored when kappa is a list.
    kappa_fallback : float or None, default None
        Fallback kappa_1 when kappa=None and the variogram fit is unreliable.
        None -> use sqrt(8) / (domain_diagonal / 4) as heuristic.
        Only used when kappa=None.
    n_sub_variogram : int or None, default None
        Subsample size for the variogram. None -> min(N, 10_000).
        Only used when kappa=None.
    variogram_model : str, default "matern25"
        Parametric variogram model. Only used when kappa=None.
    n_stagnation_window : int, default 5
        Number of rounds L used to compute the OLS slope of the covariate
        loss at each node. The slope measures how fast the covariate model
        is improving there over the last L rounds:
            slope_j = OLS slope of L_cov_nodes_j over L rounds,
                      normalised by the mean loss in the window.
        Larger L gives a smoother, less reactive slope.
        Minimum useful value is 2. With 1, falls back to a simple difference.
    w_spatial_smooth : bool, default False
        If True, apply one spatial smoothing pass (A then A^T normalised) to
        L_cov_nodes before storing in the slope history. Smooths noisy
        per-node estimates in sparse areas but blurs sharp boundaries.
        False is recommended when the spatial effect has sharp structure.
    w_focus_pct : float, default 0.5
        Fraction of nodes (ranked by covariate progress) that receive the
        full spatial learning rate. The bottom w_focus_pct nodes by progress
        get lr = learning_rate. The top (1 - w_focus_pct) get lr scaled by
        w_slow_factor. Default 0.5 = median split.
    w_slow_factor : float, default 0.5
        In "full" and "no_eta" modes: lower bound of the w_nodes ratio range.
            w_nodes in [w_slow_factor, 1.0]
            rank=0 (most stagnated) -> w = 1.0
            rank=1 (most improved)  -> w = w_slow_factor
        In "binary" mode: multiplier on learning_rate for fast-progress nodes.
            w_nodes in {lr*w_slow_factor, lr}
        1.0 disables geographic discrimination.
    w_mode : {"full","no_eta","no_geo","uniform","unit","binary"}, default "full"
        How the spatial step is scaled. See the module docstring.
    penalize_z : bool, default True
        If True, the penalty acts on the accumulated field (-alpha Q z is added
        to the right-hand side), as in the paper. If False, on the increment only.
    center_spatial : bool, default True
        If True, after each spatial update the mean of A @ dz over the training
        points is moved from z to F0_, so that the spatial effect has mean zero
        on the training data and does not absorb the intercept.
    spatial_lr_cap : float or None, default None
        Only used in "full" and "no_geo" modes.
        If set, clips max(|committed|) to this value after applying eta.
        None = no cap.
    max_depth_stage1 : int, default 3
    min_samples_leaf : int, default 20
    subsample : float, default 1.0
    reg_lambda : float, default 1e-9
        L2 penalty lambda_tree in the leaf value -G / (H + reg_lambda).
    solver : {"direct", "cg"}, default "direct"
    cg_tol : float, default 1e-5
    cg_maxiter : int or None, default None
    random_state : int or None
    """

    def __init__(
        self,
        A,
        G,
        C,
        gamma: float = 0.1,
        alpha: float = None,
        alpha_start_scale: float = 1.0,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        n_freeze: int = 5,
        n_levels: int = 3,
        kappa=None,
        ratio: float = 2.0,
        kappa_fallback: float = None,
        n_sub_variogram: int = None,
        variogram_model: str = "matern25",
        n_stagnation_window: int = 5,
        w_spatial_smooth: bool = False,
        w_focus_pct: float = 0.5,
        w_slow_factor: float = 0.5,
        w_mode: str = "full",
        penalize_z: bool = True,
        center_spatial: bool = True,
        spatial_lr_cap: float = None,
        max_depth_stage1: int = 3,
        min_samples_leaf: int = 20,
        subsample: float = 1.0,
        reg_lambda: float = 1e-9,
        solver: str = "direct",
        cg_tol: float = 1e-5,
        cg_maxiter: int = None,
        random_state=None,
    ):
        self.A                   = A
        self.G                   = G
        self.C                   = C
        self.gamma               = gamma
        self.alpha               = alpha
        self.alpha_start_scale   = alpha_start_scale
        self.n_estimators        = n_estimators
        self.learning_rate       = learning_rate
        self.n_freeze            = n_freeze
        self.n_levels            = n_levels
        self.kappa               = kappa
        self.ratio               = ratio
        self.kappa_fallback      = kappa_fallback
        self.n_sub_variogram     = n_sub_variogram
        self.variogram_model     = variogram_model
        self.n_stagnation_window = n_stagnation_window
        self.w_spatial_smooth    = w_spatial_smooth
        self.w_focus_pct         = w_focus_pct
        self.w_slow_factor       = w_slow_factor
        self.w_mode              = w_mode
        self.penalize_z          = penalize_z
        self.center_spatial      = center_spatial
        self.spatial_lr_cap      = spatial_lr_cap
        self.max_depth_stage1    = max_depth_stage1
        self.min_samples_leaf    = min_samples_leaf
        self.subsample           = subsample
        self.reg_lambda          = reg_lambda
        self.solver              = solver
        self.cg_tol              = cg_tol
        self.cg_maxiter          = cg_maxiter
        self.random_state        = random_state

        if solver not in ("direct", "cg"):
            raise ValueError(f"solver must be 'direct' or 'cg', got '{solver}'")
        _VALID_MODES = ("full", "no_eta", "no_geo", "uniform", "unit", "binary")
        if w_mode not in _VALID_MODES:
            raise ValueError(
                f"w_mode must be one of {_VALID_MODES}, got '{w_mode}'"
            )
        if n_freeze < 1:
            raise ValueError("n_freeze must be >= 1")
        if n_levels < 1:
            raise ValueError("n_levels must be >= 1")
        if isinstance(kappa, list):
            if len(kappa) != n_levels:
                raise ValueError(
                    f"kappa is a list of length {len(kappa)} but "
                    f"n_levels={n_levels}. They must match."
                )

        # Per-node observation counts for loss projection: O(3N) once.
        self._node_obs_counts = np.asarray(
            A.T @ np.ones(A.shape[0])
        ).ravel().clip(min=1e-10)

        # Placeholder - overwritten in fit() once kappa is resolved.
        self._diag_h_spatial = np.zeros(A.shape[0])
        self._Q_active        = None

    # >> Kappa resolution

    def _resolve_kappa_list(
        self,
        coords: np.ndarray,
        F_cov: np.ndarray,
        y: np.ndarray,
    ) -> tuple:
        """
        Return (kappa_array, rho_array, kappa_source, variogram_result).

        Three modes:
          user_list    kappa is a list -> use directly, ratio ignored
          user_float   kappa is a float -> apply ratio to build the list
          variogram    kappa is None -> estimate from freeze-phase residuals
        """
        L = self.n_levels

        # Mode 1: explicit list
        if isinstance(self.kappa, list):
            kappa_arr = np.array(self.kappa, dtype=float)
            rho_arr   = np.sqrt(8.0) / kappa_arr
            print(f"  [Kappa] user-supplied list: "
                  f"{', '.join(f'L{l+1}={kappa_arr[l]:.4f}' for l in range(L))}")
            return kappa_arr, rho_arr, "user_list", None

        # Mode 2: single float, finer levels via ratio
        if self.kappa is not None:
            kappa_1   = float(self.kappa)
            kappa_arr = np.array([kappa_1 * (self.ratio ** l) for l in range(L)])
            rho_arr   = np.sqrt(8.0) / kappa_arr
            print(f"  [Kappa] user-supplied float kappa_1={kappa_1:.4f}, "
                  f"ratio={self.ratio}: "
                  f"{', '.join(f'L{l+1}={kappa_arr[l]:.4f}' for l in range(L))}")
            return kappa_arr, rho_arr, "user_float", None

        # Mode 3: variogram
        residuals = y - expit(F_cov)
        N         = len(y)
        n_sub     = self.n_sub_variogram if self.n_sub_variogram else min(N, 10_000)

        print(f"  [Kappa] fitting variogram on {n_sub:,} points...")
        variogram_result = None
        kappa_1          = None

        try:
            variogram_result = fit_variogram(
                coords       = coords,
                residuals    = residuals,
                n_sub        = n_sub,
                n_bins       = 20,
                model        = self.variogram_model,
                random_state = (self.random_state or 0),
            )
        except Exception as e:
            warnings.warn(f"Variogram failed: {e}. Using fallback.", RuntimeWarning)

        reliable = (
            variogram_result is not None
            and variogram_result.fit_success
            and (variogram_result.nugget / (variogram_result.sill + 1e-12)) < 0.95
            and variogram_result.r2_fit >= 0.3
        )

        if reliable:
            kappa_1 = variogram_result.kappa_optimal
            print(f"  [Kappa] variogram: range={variogram_result.range_optimal:.4f} "
                  f"kappa_1={kappa_1:.4f} "
                  f"nugget/sill={variogram_result.nugget/variogram_result.sill:.3f} "
                  f"R^2={variogram_result.r2_fit:.3f}")
        else:
            # Fallback
            if self.kappa_fallback is not None:
                kappa_1 = float(self.kappa_fallback)
                reason  = f"kappa_fallback={kappa_1:.4f}"
            else:
                diag    = float(np.sqrt(
                    (coords[:,0].max() - coords[:,0].min())**2 +
                    (coords[:,1].max() - coords[:,1].min())**2
                ))
                kappa_1 = np.sqrt(8.0) / (diag / 4.0)
                reason  = f"heuristic domain_diag/4, kappa_1={kappa_1:.4f}"

            nug_sill = (variogram_result.nugget / variogram_result.sill
                        if variogram_result else float("nan"))
            r2       = variogram_result.r2_fit if variogram_result else float("nan")
            warnings.warn(
                f"Variogram unreliable (nugget/sill={nug_sill:.3f}, R^2={r2:.3f}). "
                f"Using {reason}.",
                RuntimeWarning,
            )
            print(f"  [Kappa] unreliable variogram - {reason}")

        kappa_arr = np.array([kappa_1 * (self.ratio ** l) for l in range(L)])
        rho_arr   = np.sqrt(8.0) / kappa_arr
        print(f"  [Kappa] levels: "
              f"{', '.join(f'L{l+1}: kappa={kappa_arr[l]:.4f} rho={rho_arr[l]:.4f}' for l in range(L))}")
        return kappa_arr, rho_arr, "variogram", variogram_result

    # >> Build Q matrices from resolved kappa array

    def _build_level_Qs(self, kappa_arr: np.ndarray) -> list:
        """
        Build one Q_l per level from the resolved kappa array.

        tau is derived from kappa using the marginal variance normalisation:
            tau_l = 1 / (sqrt(4*pi) * kappa_l)
        This ensures the marginal variance of the SPDE field is approximately
        1 regardless of the range, making alpha comparable across levels.
        """
        Q_list = []
        for l in range(self.n_levels):
            kappa_l = float(kappa_arr[l])
            tau_l   = 1.0 / (np.sqrt(4.0 * np.pi) * kappa_l)
            Q_list.append(
                compute_Q(self.G, self.C, kappa=kappa_l, tau=tau_l)
            )
        return Q_list

    def _update_diag_h_spatial(self, Q_l):
        """Recompute diag(AQA^T) for the newly active level Q_l."""
        AQ = self.A @ Q_l
        self._diag_h_spatial = np.asarray(
            self.A.multiply(AQ).sum(axis=1)
        ).ravel()
        self._Q_active = Q_l

    # >> Unlock schedule

    def _build_unlock_schedule(self, T: int) -> list:
        """Divide Stage 2 rounds into n_levels equal windows."""
        stage2 = T - self.n_freeze
        window = max(1, stage2 // self.n_levels)
        return [self.n_freeze + l * window for l in range(self.n_levels)]

    # >> Alpha cosine decay

    def _alpha_at_round(self, t: int, T: int, alpha_end: float) -> float:
        if self.alpha_start_scale == 1.0 or T == 1:
            return alpha_end
        alpha_start   = alpha_end * self.alpha_start_scale
        cosine_factor = 0.5 * (1.0 + np.cos(np.pi * t / (T - 1)))
        return float(alpha_end + (alpha_start - alpha_end) * cosine_factor)

    # >> Covariate loss projection

    def _project_loss_to_nodes(self, F_cov, y):
        """
        Project binary log-loss from observations to mesh nodes.

        L_nodes_j = mean of L_obs_i for observations near node j.

        If w_spatial_smooth=True, applies one spatial smoothing pass
        (A then A^T normalised) before returning, reducing noise at
        sparsely covered nodes at the cost of blurring sharp boundaries.
        """
        p     = np.clip(expit(F_cov), 1e-12, 1 - 1e-12)
        L_obs = -xlogy(y, p) - xlogy(1 - y, 1 - p)
        L_nodes = np.asarray(self.A.T @ L_obs).ravel() / self._node_obs_counts
        if self.w_spatial_smooth:
            L_nodes = self._smooth_nodes(L_nodes)
        return L_nodes

    # >> Spatial smoothing helper (optional)

    def _smooth_nodes(self, L_nodes: np.ndarray) -> np.ndarray:
        """
        One spatial smoothing pass through the mesh: A then A^T normalised.

        Each node receives the average of its neighbouring nodes' values,
        weighted by how many observations fall in shared triangles.
        Cost: O(3N) - two sparse matrix-vector products.

        Only called when w_spatial_smooth=True.
        """
        L_obs_back = np.asarray(self.A @ L_nodes).ravel()         # (N,)
        L_smooth   = np.asarray(self.A.T @ L_obs_back).ravel()    # (M,)
        return L_smooth / self._node_obs_counts

    # >> Geographic weights

    def _compute_w_nodes(
        self,
        loss_deque: deque,
        is_final_round: bool,
    ) -> np.ndarray:
        """
        Compute per-node geographic weights w_nodes.

        Returns a (M,) array used in the fit loop as:
            committed = eta * w_nodes * delta_z
        where eta depends on w_mode (1.0 or Newton-optimal).

        Mode dispatch
        -------------
        "full", "no_eta"   -> adaptive slope-based weights in [w_slow_factor, 1.0]
        "no_geo"           -> 1.0 everywhere  (Newton step active in fit loop)
        "uniform"          -> 1.0 everywhere  (committed = lr * dz in fit loop)
        "unit"             -> 1.0 everywhere  (committed = dz in fit loop)
        "binary"           -> two levels in {lr*slow, lr}

        Final round always returns 1.0 (or lr for binary) for a full clean step.
        """
        M  = self.A.shape[1]
        lr = self.learning_rate

        # no_geo, uniform, unit: no per-node weights
        if self.w_mode in ("no_geo", "uniform", "unit"):
            return np.ones(M, dtype=np.float64)

        # Last round or not enough history: full step
        if is_final_round or len(loss_deque) < 2:
            if self.w_mode == "binary":
                return np.full(M, lr, dtype=np.float64)
            return np.ones(M, dtype=np.float64)    # "full" or "no_eta"

        # Relative slope of the node loss over the window (progress > 0 means improving)
        window    = np.array(list(loss_deque), dtype=np.float64)  # (L, M)
        L_win     = len(window)
        x         = np.arange(L_win, dtype=float)
        x         = x - x.mean()
        x_dot_x   = float((x ** 2).sum())
        abs_slope = (x[:, None] * window).sum(axis=0) / x_dot_x  # (M,)

        mean_loss      = window.mean(axis=0) + 1e-10
        relative_slope = abs_slope / mean_loss
        progress       = -relative_slope              # positive = improving

        # full, no_eta: the weight decreases with the rank of the progress
        if self.w_mode in ("full", "no_eta"):
            if np.max(progress) < 1e-6:              # blind spot guard
                return np.ones(M, dtype=np.float64)
            rank    = np.argsort(np.argsort(progress)).astype(float)
            rank   /= max(M - 1, 1)
            w_nodes = self.w_slow_factor + (1.0 - self.w_slow_factor) * (1.0 - rank)
            return w_nodes.astype(np.float64)

        # binary: two levels, split at the w_focus_pct quantile
        if self.w_focus_pct <= 0.0:
            return np.full(M, lr * self.w_slow_factor, dtype=np.float64)
        if self.w_focus_pct >= 1.0:
            return np.full(M, lr, dtype=np.float64)
        threshold = np.quantile(progress, self.w_focus_pct)
        w_nodes   = np.where(
            progress <= threshold,
            lr,
            lr * self.w_slow_factor,
        )
        return w_nodes.astype(np.float64)

    # >> Newton-optimal step size

    def _newton_step(
        self,
        delta_z: np.ndarray,
        hess2: np.ndarray,
        alpha_t: float,
        Q_l,
    ) -> float:
        """
        Compute the Newton-optimal step size eta along direction A*delta_z.

        Derivation
        ----------
        Minimise the double-penalty surrogate S'(eta) along A*delta_z.
        Using the normal equations to substitute g'A*dz, the cross-term
        (dz'Qz) cancels exactly, giving the same formula whether
        penalize_z is True or False:

            b = dz' A'HA dz     (data curvature along dz)
            q = dz' Q dz        (SPDE roughness of dz)

            eta* = (b + alpha*q) / (b + 2*alpha*q)  in (0.5, 1.0]

        Bounds:
            alpha -> 0:  eta* -> 1.0   (no regularisation: full step)
            alpha -> inf: eta* -> 0.5  (strong regularisation: half step)

        Cost: O(N + M) - one sparse mat-vec (Q*dz) + two dot products.
        """
        A_dz      = np.asarray(self.A @ delta_z).ravel()
        data_term = float(np.dot(hess2 * A_dz, A_dz))                    # b
        q_term    = float(np.dot(delta_z,
                                  np.asarray(Q_l @ delta_z).ravel()))    # q
        num = data_term + alpha_t * q_term
        den = data_term + 2.0 * alpha_t * q_term
        return float(np.clip(num / (den + 1e-12), 0.0, 1.0))

    # >> Spatial grad/hess

    def _spatial_grad_hess(self, F, grad_logit, hess_logit):
        grad_spatial = 0
        logit_mag    = 0
        spat_mag     = 0
        norm         = 0
        # The gamma term is switched off: grad2 = grad and hess2 = hess.
        grad2 = grad_logit + self.gamma * grad_spatial * norm
        hess2 = hess_logit + self.gamma * self._diag_h_spatial * norm
        return grad2, hess2

    # >> WLS solve

    def _wls_solve(self, r2, hess2, alpha_t, Q_l, warm_start, z_current=None):
        """
        Solve the WLS normal equations for delta_z.

        penalize_z=False:
            (A'HA + aQ) dz = A'H r    with r = -g/h  -> A'Hr = -A'g

        penalize_z=True:
            (A'HA + aQ) dz = -A'g - a*Q*z_current

        The system matrix is the same in both cases.
        """
        H   = diags(hess2)
        AtH = self.A.T @ H
        K   = AtH @ self.A + alpha_t * Q_l
        rhs = AtH @ r2                                   # = -A'g  (since r2 = -g/h)
        if self.penalize_z and z_current is not None:
            rhs = rhs - alpha_t * (Q_l @ z_current)     # pull-back term
        return self._solve(K, rhs, warm_start)

    # >> Solvers

    def _solve_direct(self, K, rhs, _ws):
        return spsolve(K, rhs)

    def _solve_cg(self, K, rhs, ws):
        dz, info = sparse_cg(K, rhs, x0=ws, rtol=self.cg_tol, maxiter=self.cg_maxiter)
        if info > 0:
            warnings.warn(f"CG did not converge (info={info}).",
                          RuntimeWarning, stacklevel=3)
            return ws
        return dz

    def _solve(self, K, rhs, ws):
        return (self._solve_direct(K, rhs, ws) if self.solver == "direct"
                else self._solve_cg(K, rhs, ws))

    # >> Tree helpers

    def _make_tree(self, rng):
        return DecisionTreeRegressor(
            max_depth        = self.max_depth_stage1,
            min_samples_leaf = self.min_samples_leaf,
            random_state     = rng.randint(0, 2**31),
        )

    def _newton_leaf_update(self, tree, X, grad, hess):
        leaf_ids = tree.apply(X)
        for leaf in np.unique(leaf_ids):
            mask = leaf_ids == leaf
            tree.tree_.value[leaf, 0, 0] = (
                -np.sum(grad[mask]) / (np.sum(hess[mask]) + self.reg_lambda)
            )

    # >> Fit

    def fit(self, X, y, coords=None, sample_weight=None):
        """
        Fit the model.

        Parameters
        ----------
        X      : (N, p)  Non-spatial features.
        y      : (N,)    Binary labels {0, 1}.
        coords : (N, 2)  Observation coordinates. Required when kappa=None
                         (variogram estimation). Must be in the same units
                         as the mesh. Optional when kappa is user-supplied.
        sample_weight : (N,) or None
                         Passed to the tree split search only; the leaf values
                         and the spatial update are unweighted.
        """
        if coords is None and self.kappa is None:
            raise ValueError(
                "coords is required when kappa=None (variogram estimation). "
                "Pass coords or set kappa to a float or list."
            )
        rng = np.random.RandomState(self.random_state)
        T   = self.n_estimators
        L   = self.n_levels
        M   = self.A.shape[1]

        alpha_end = self.gamma if self.alpha is None else self.alpha

        X      = (X.astype(np.float32) if issparse(X)
                  else np.asarray(X, dtype=np.float32))
        y      = np.asarray(y, dtype=np.float64)
        coords = np.asarray(coords, dtype=np.float64)
        N      = len(y)

        p0       = np.clip(y.mean(), 1e-6, 1 - 1e-6)
        self.F0_ = float(np.log(p0 / (1.0 - p0)))
        self.F0_original_ = self.F0_   # before spatial centering corrections

        F     = np.full(N, self.F0_)
        F_cov = np.full(N, self.F0_)

        z_levels      = [np.zeros(M, dtype=np.float64) for _ in range(L)]
        cg_warm_starts = [np.zeros(M, dtype=np.float64) for _ in range(L)]

        self.stage1_trees_ = []

        loss_history = deque(maxlen=self.n_stagnation_window)

        unlock_rounds = self._build_unlock_schedule(T)
        print(f"Unlock schedule (n_levels={L}): "
              f"{dict(zip([f'L{l+1}' for l in range(L)], unlock_rounds))}")

        alpha_hist        = np.empty(T, dtype=np.float64)
        w_nodes_hist      = np.full((T, M), np.nan)
        eta_hist          = np.full(T, np.nan, dtype=np.float64)
        all_znodes        = np.zeros((T, L, M), dtype=np.float64)
        loss_nodes_hist   = np.empty((T, M))
        active_level_hist = np.full(T, -1, dtype=int)

        # Q is built now if kappa is given, otherwise at the end of the freeze.
        Q_list         = None
        kappa_per_level = None
        rho_per_level   = None
        variogram_result = None
        kappa_source     = None

        if self.kappa is not None:
            print(f"\nResolving kappa before training (user-supplied)...")
            kappa_per_level, rho_per_level, kappa_source, variogram_result = \
                self._resolve_kappa_list(coords, F_cov, y)
            Q_list = self._build_level_Qs(kappa_per_level)
            self._update_diag_h_spatial(Q_list[0])

        # Per-round history (loss, step size, size of the field)
        self.history_ = []

        for t in range(T):

            alpha_t = self._alpha_at_round(t, T, alpha_end)
            alpha_hist[t] = alpha_t

            # Stage 1: tree on the negative gradient, Newton leaf values
            p          = expit(F)
            grad_logit = p - y
            hess_logit = p * (1.0 - p)

            mask = (np.ones(N, dtype=bool) if self.subsample >= 1.0
                    else rng.rand(N) < self.subsample)
            sw   = sample_weight[mask] if sample_weight is not None else None

            tree_t = self._make_tree(rng)
            tree_t.fit(X[mask], -grad_logit[mask], sample_weight=sw)
            self._newton_leaf_update(tree_t, X, grad_logit, hess_logit)
            self.stage1_trees_.append(tree_t)

            pred   = tree_t.predict(X)
            F     += self.learning_rate * pred
            F_cov += self.learning_rate * pred

            L_nodes = self._project_loss_to_nodes(F_cov, y)
            loss_history.append(L_nodes)
            loss_nodes_hist[t] = L_nodes

            # End of the freeze: estimate kappa from a variogram if kappa is None
            if t == self.n_freeze - 1 and self.kappa is None:
                kappa_per_level, rho_per_level, kappa_source, variogram_result = \
                    self._resolve_kappa_list(coords, F_cov, y)
                Q_list = self._build_level_Qs(kappa_per_level)
                self._update_diag_h_spatial(Q_list[0])

            # Freeze rounds: trees only
            if t < self.n_freeze:
                for l in range(L):
                    all_znodes[t, l] = z_levels[l]
                p_t     = expit(F)
                loss_t  = float(-np.mean(xlogy(y, p_t) + xlogy(1 - y, 1 - p_t)))
                z_total = sum(z_levels)
                self.history_.append({
                    "round"        : t,
                    "phase"        : "freeze",
                    "loss"         : loss_t,
                    "alpha"        : alpha_t,
                    "eta"          : np.nan,
                    "active_level" : -1,
                    "z_norm"       : float(np.linalg.norm(z_total)),
                    "z_std"        : float(np.std(np.asarray(self.A @ z_total).ravel())),
                    "F_cov_std"    : float(np.std(F_cov)),
                })
                continue

            active_l = L - 1
            for l in range(L - 1, -1, -1):
                if t >= unlock_rounds[l]:
                    active_l = l
                    break

            active_level_hist[t] = active_l

            # Update Q when a new level activates
            if t == unlock_rounds[active_l] and active_l > 0:
                self._update_diag_h_spatial(Q_list[active_l])

            is_final = (t == T - 1)
            w_nodes  = self._compute_w_nodes(loss_history, is_final)
            w_nodes_hist[t] = w_nodes

            # Stage 2: gradient and Hessian at the updated score, penalised WLS for the active level
            p_upd          = expit(F)
            grad_logit_upd = p_upd - y
            hess_logit_upd = p_upd * (1.0 - p_upd)

            grad2, hess2 = self._spatial_grad_hess(F, grad_logit_upd, hess_logit_upd)

            # Pseudo-residuals: r = -g/h  (element-wise Newton direction)
            # A'H r = A' diag(h) (-g/h) = -A'g  -> H cancels in RHS
            # but still weights the system matrix A'HA.
            r2 = -grad2 / (hess2 + 1e-12)

            # z_current for active level (used when penalize_z=True)
            z_current = z_levels[active_l] if self.penalize_z else None

            delta_z = self._wls_solve(
                r2, hess2, alpha_t,
                Q_list[active_l],
                cg_warm_starts[active_l],
                z_current=z_current,
            )
            cg_warm_starts[active_l] = delta_z.copy()

            # Spatial step per w_mode:
            #  "full"    : committed = eta_newton * w_geo * delta_z
            #  "no_eta"  : committed = 1.0        * w_geo * delta_z
            #  "no_geo"  : committed = eta_newton * 1.0   * delta_z
            #  "uniform" : committed = lr                 * delta_z
            #  "unit"    : committed =                      delta_z
            #  "binary"  : committed = w_binary           * delta_z

            if self.w_mode == "full":
                eta           = self._newton_step(delta_z, hess2, alpha_t,
                                                  Q_list[active_l])
                eta_hist[t]   = eta
                scaled_dz     = eta * w_nodes * delta_z
                if self.spatial_lr_cap is not None:
                    max_step = np.abs(scaled_dz).max()
                    if max_step > self.spatial_lr_cap:
                        scaled_dz *= self.spatial_lr_cap / max_step

            elif self.w_mode == "no_eta":
                scaled_dz = w_nodes * delta_z              # eta = 1.0

            elif self.w_mode == "no_geo":
                eta           = self._newton_step(delta_z, hess2, alpha_t,
                                                  Q_list[active_l])
                eta_hist[t]   = eta
                scaled_dz     = eta * delta_z              # w = 1.0
                if self.spatial_lr_cap is not None:
                    max_step = np.abs(scaled_dz).max()
                    if max_step > self.spatial_lr_cap:
                        scaled_dz *= self.spatial_lr_cap / max_step

            elif self.w_mode == "uniform":
                scaled_dz = self.learning_rate * delta_z   # eta=1, w=1

            elif self.w_mode == "unit":
                scaled_dz = delta_z                        # eta=1, w=1, lr=1

            else:  # "binary"
                scaled_dz = w_nodes * delta_z

            A_scaled_dz         = np.asarray(self.A @ scaled_dz).ravel()  # (N,) computed once
            z_levels[active_l] += scaled_dz
            F                  += A_scaled_dz

            # Centering: c = mean(A dz) over the training points is moved from z to F0_.
            # Each row of A sums to 1, so A z drops by c everywhere and F0_ + A z, hence
            # every prediction, is unchanged. The running training score F is shifted by c.
            if self.center_spatial:
                correction         = float(np.mean(A_scaled_dz))   # reuses cached result
                z_levels[active_l] -= correction
                F                  -= correction
                self.F0_           += correction

            for l in range(L):
                all_znodes[t, l] = z_levels[l]

            p_t     = expit(F)
            loss_t  = float(-np.mean(xlogy(y, p_t) + xlogy(1 - y, 1 - p_t)))
            z_total = sum(z_levels)
            self.history_.append({
                "round"        : t,
                "phase"        : "active",
                "loss"         : loss_t,
                "alpha"        : alpha_t,
                "eta"          : float(eta_hist[t]) if not np.isnan(eta_hist[t]) else np.nan,
                "active_level" : active_l,
                "z_norm"       : float(np.linalg.norm(z_total)),
                "z_std"        : float(np.std(np.asarray(self.A @ z_total).ravel())),
                "F_cov_std"    : float(np.std(F_cov)),
            })

        self.z_levels_              = z_levels
        self.alpha_                 = alpha_end
        self.alpha_start_           = alpha_end * self.alpha_start_scale
        self.alpha_schedule_        = alpha_hist
        self.w_nodes_schedule_      = w_nodes_hist
        self.eta_schedule_          = eta_hist           # (T,) NaN=freeze/binary/uniform
        self.all_znodes_            = all_znodes          # (T, L, M)
        self.loss_nodes_schedule_   = loss_nodes_hist
        self.kappa_per_level_       = kappa_per_level
        self.rho_per_level_         = rho_per_level
        self.tau_per_level_         = (
            1.0 / (np.sqrt(4.0 * np.pi) * kappa_per_level)
            if kappa_per_level is not None else None
        )  # tau_l = 1 / (sqrt(4*pi) * kappa_l)
        self.kappa_source_          = kappa_source
        self.variogram_result_      = variogram_result
        self.active_level_per_round_ = active_level_hist
        self.unlock_rounds_         = unlock_rounds
        self.n_features_in_         = X.shape[1]
        self.classes_               = np.array([0, 1])
        return self

    # >> Predict

    def _z_total(self):
        return sum(self.z_levels_)

    def decision_function(self, X, A_new):
        check_is_fitted(self, ["stage1_trees_", "z_levels_"])
        X = (X.astype(np.float32) if issparse(X)
             else np.asarray(X, dtype=np.float32))
        F = np.full(X.shape[0], self.F0_)
        for tree in self.stage1_trees_:
            F += self.learning_rate * tree.predict(X)
        F += np.asarray(A_new @ self._z_total()).ravel()
        return F

    def predict_proba(self, X, A_new):
        p = expit(self.decision_function(X, A_new))
        return np.column_stack([1 - p, p])

    def predict(self, X, A_new):
        return (self.decision_function(X, A_new) >= 0.0).astype(int)

    # >> Decomposition

    def spatial_effect(self, A_new):
        """Total spatial contribution: A_new @ (z_1 + ... + z_L)."""
        check_is_fitted(self, "z_levels_")
        return np.asarray(A_new @ self._z_total()).ravel()

    def spatial_effect_at_level(self, A_new, level: int):
        """Spatial contribution from one level (1-indexed)."""
        check_is_fitted(self, "z_levels_")
        if not 1 <= level <= self.n_levels:
            raise ValueError(f"level must be in [1, {self.n_levels}]")
        return np.asarray(A_new @ self.z_levels_[level - 1]).ravel()

    def covariate_effect(self, X):
        """Pure covariate contribution: lr * sum of trees (no intercept).
        The intercept (F0_) is a nuisance term that belongs to neither the
        covariate nor the spatial decomposition."""
        check_is_fitted(self, "stage1_trees_")
        X = (X.astype(np.float32) if issparse(X)
             else np.asarray(X, dtype=np.float32))
        F = np.zeros(X.shape[0])
        for tree in self.stage1_trees_:
            F += self.learning_rate * tree.predict(X)
        return F

    @property
    def z_nodes(self):
        """Total spatial field across all levels (M,)."""
        check_is_fitted(self, "z_levels_")
        return self._z_total()

    def level_summary(self):
        check_is_fitted(self, "z_levels_")
        lines = [
            "", "=" * 65,
            f"  SAMBoost - level summary  [source: {self.kappa_source_}]",
            "=" * 65,
            f"  {'Level':<8} {'rho':>8} {'kappa':>8} {'tau':>8} "
            f"{'std(z_l)':>10} {'||z_l||':>10}",
            "  " + "-" * 58,
        ]
        for l in range(self.n_levels):
            z_l   = self.z_levels_[l]
            tau_l = self.tau_per_level_[l]
            lines.append(
                f"  L{l+1:<7} {self.rho_per_level_[l]:>8.4f} "
                f"{self.kappa_per_level_[l]:>8.4f} "
                f"{tau_l:>8.4f} "
                f"{z_l.std():>10.4f} {np.linalg.norm(z_l):>10.4f}"
            )
        lines += ["=" * 65, ""]
        return "\n".join(lines)