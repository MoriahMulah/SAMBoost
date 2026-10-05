"""
model_continuous.py
===================
SAMBoostContinuous: spatial gradient boosting for continuous outcomes.
This is the regression model of the paper.

The algorithm, the notation and the options are the same as in model.py,
which documents them. Only the likelihood-specific parts differ, and they
come from a pluggable objective (squared error by default):

  - F0                    objective init(y)
  - gradient and Hessian  objective grad_hess(y, F)
  - loss for the history  objective loss(y, F)
  - loss at the nodes     objective loss_elementwise(y, F_cov)
  - variogram residuals   y - F_cov
  - predict()             returns the score itself (identity link)
"""

import warnings
import numpy as np
from collections import deque

from scipy.sparse import issparse, diags
from scipy.sparse.linalg import spsolve, cg as sparse_cg
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted

from utils.mesh_building import compute_Q, assemble_spde_matrices
from utils.variogram import fit_variogram


# >> Built-in objectives

def _mse_init(y):
    """Optimal constant prediction for squared error: mean(y)."""
    return float(np.mean(y))


def _mse_grad_hess(y, F):
    """Squared error: L = 0.5*(F-y)^2, g = F-y, h = 1."""
    grad = F - y
    hess = np.ones_like(y)
    return grad, hess


def _mse_loss(y, F):
    """Mean squared error (scalar)."""
    return float(0.5 * np.mean((F - y) ** 2))


def _mse_loss_elementwise(y, F):
    """Per-observation squared error for node projection."""
    return 0.5 * (F - y) ** 2


_BUILTIN_OBJECTIVES = {
    "reg:squarederror": {
        "init":              _mse_init,
        "grad_hess":         _mse_grad_hess,
        "loss":              _mse_loss,
        "loss_elementwise":  _mse_loss_elementwise,
    },
}


class SAMBoostContinuous(BaseEstimator, RegressorMixin):
    """
    Spatial gradient boosting for continuous outcomes.

    Objective-agnostic: the algorithm consumes (grad, hess) from a
    pluggable objective.  Squared error is the default.

    Parameters
    ----------
    A, G, C : sparse matrices
        SPDE projection (N,M), stiffness (M,M), mass (M,M).
    objective : str or dict, default "reg:squarederror"
        If str: key in the built-in objective registry.
        If dict: must contain four callables:
            "init"             : (y,)      -> float  (F0)
            "grad_hess"        : (y, F)    -> (grad, hess)  both (N,)
            "loss"             : (y, F)    -> float  (scalar loss)
            "loss_elementwise" : (y, F)    -> (N,)   (per-obs loss)

    All other parameters are identical to SAMBoost (Bernoulli).
    See the Bernoulli class docstring for full documentation.
    """

    def __init__(
        self,
        A,
        G,
        C,
        objective="reg:squarederror",
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
        self.objective           = objective
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

        if isinstance(objective, str):
            if objective not in _BUILTIN_OBJECTIVES:
                raise ValueError(
                    f"Unknown objective '{objective}'. "
                    f"Available: {list(_BUILTIN_OBJECTIVES.keys())}"
                )
            obj = _BUILTIN_OBJECTIVES[objective]
        elif isinstance(objective, dict):
            required = {"init", "grad_hess", "loss", "loss_elementwise"}
            missing  = required - set(objective.keys())
            if missing:
                raise ValueError(
                    f"Custom objective dict missing keys: {missing}"
                )
            obj = objective
        else:
            raise TypeError(
                "objective must be a str or dict, "
                f"got {type(objective).__name__}"
            )

        self._obj_init             = obj["init"]
        self._obj_grad_hess        = obj["grad_hess"]
        self._obj_loss             = obj["loss"]
        self._obj_loss_elementwise = obj["loss_elementwise"]

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

        self._node_obs_counts = np.asarray(
            A.T @ np.ones(A.shape[0])
        ).ravel().clip(min=1e-10)

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
        # Identity link: residuals are y - F_cov directly
        residuals = y - F_cov
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

    # >> Build Q matrices

    def _build_level_Qs(self, kappa_arr: np.ndarray) -> list:
        Q_list = []
        for l in range(self.n_levels):
            kappa_l = float(kappa_arr[l])
            tau_l   = 1.0 / (np.sqrt(4.0 * np.pi) * kappa_l)
            Q_list.append(
                compute_Q(self.G, self.C, kappa=kappa_l, tau=tau_l)
            )
        return Q_list

    def _update_diag_h_spatial(self, Q_l):
        AQ = self.A @ Q_l
        self._diag_h_spatial = np.asarray(
            self.A.multiply(AQ).sum(axis=1)
        ).ravel()
        self._Q_active = Q_l

    # >> Unlock schedule

    def _build_unlock_schedule(self, T: int) -> list:
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

    # >> Covariate loss projection (objective-agnostic)

    def _project_loss_to_nodes(self, F_cov, y):
        """
        Project per-observation loss from observations to mesh nodes.
        Uses the objective's elementwise loss - not hardcoded to any
        specific likelihood.
        """
        L_obs   = self._obj_loss_elementwise(y, F_cov)
        L_nodes = np.asarray(self.A.T @ L_obs).ravel() / self._node_obs_counts
        if self.w_spatial_smooth:
            L_nodes = self._smooth_nodes(L_nodes)
        return L_nodes

    # >> Spatial smoothing helper

    def _smooth_nodes(self, L_nodes: np.ndarray) -> np.ndarray:
        L_obs_back = np.asarray(self.A @ L_nodes).ravel()
        L_smooth   = np.asarray(self.A.T @ L_obs_back).ravel()
        return L_smooth / self._node_obs_counts

    # >> Geographic weights

    def _compute_w_nodes(
        self,
        loss_deque: deque,
        is_final_round: bool,
    ) -> np.ndarray:
        M  = self.A.shape[1]
        lr = self.learning_rate

        if self.w_mode in ("no_geo", "uniform", "unit"):
            return np.ones(M, dtype=np.float64)

        if is_final_round or len(loss_deque) < 2:
            if self.w_mode == "binary":
                return np.full(M, lr, dtype=np.float64)
            return np.ones(M, dtype=np.float64)

        window    = np.array(list(loss_deque), dtype=np.float64)
        L_win     = len(window)
        x         = np.arange(L_win, dtype=float)
        x         = x - x.mean()
        x_dot_x   = float((x ** 2).sum())
        abs_slope = (x[:, None] * window).sum(axis=0) / x_dot_x

        mean_loss      = window.mean(axis=0) + 1e-10
        relative_slope = abs_slope / mean_loss
        progress       = -relative_slope

        if self.w_mode in ("full", "no_eta"):
            if np.max(progress) < 1e-6:
                return np.ones(M, dtype=np.float64)
            rank    = np.argsort(np.argsort(progress)).astype(float)
            rank   /= max(M - 1, 1)
            w_nodes = self.w_slow_factor + (1.0 - self.w_slow_factor) * (1.0 - rank)
            return w_nodes.astype(np.float64)

        # binary mode
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
        eta* = (b + alpha*q) / (b + 2*alpha*q)  in (0.5, 1.0]

        b = dz' A'HA dz     (data curvature along dz)
        q = dz' Q dz        (SPDE roughness of dz)
        """
        A_dz      = np.asarray(self.A @ delta_z).ravel()
        data_term = float(np.dot(hess2 * A_dz, A_dz))
        q_term    = float(np.dot(delta_z,
                                  np.asarray(Q_l @ delta_z).ravel()))
        num = data_term + alpha_t * q_term
        den = data_term + 2.0 * alpha_t * q_term
        return float(np.clip(num / (den + 1e-12), 0.0, 1.0))

    # >> Spatial grad/hess

    def _spatial_grad_hess(self, F, grad, hess):
        # The gamma term is switched off: grad2 = grad and hess2 = hess.
        grad_spatial = 0
        logit_mag    = 0
        spat_mag     = 0
        norm         = 0
        grad2 = grad + self.gamma * grad_spatial * norm
        hess2 = hess + self.gamma * self._diag_h_spatial * norm
        return grad2, hess2

    # >> WLS solve

    def _wls_solve(self, r2, hess2, alpha_t, Q_l, warm_start, z_current=None):
        H   = diags(hess2)
        AtH = self.A.T @ H
        K   = AtH @ self.A + alpha_t * Q_l
        rhs = AtH @ r2
        if self.penalize_z and z_current is not None:
            rhs = rhs - alpha_t * (Q_l @ z_current)
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
        y      : (N,)    Continuous target.
        coords : (N, 2)  Observation coordinates. Required when kappa=None.
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

        self.F0_ = self._obj_init(y)
        self.F0_original_ = self.F0_

        F     = np.full(N, self.F0_)
        F_cov = np.full(N, self.F0_)

        z_levels       = [np.zeros(M, dtype=np.float64) for _ in range(L)]
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
        Q_list           = None
        kappa_per_level  = None
        rho_per_level    = None
        variogram_result = None
        kappa_source     = None

        if self.kappa is not None:
            print(f"\nResolving kappa before training (user-supplied)...")
            kappa_per_level, rho_per_level, kappa_source, variogram_result = \
                self._resolve_kappa_list(coords, F_cov, y)
            Q_list = self._build_level_Qs(kappa_per_level)
            self._update_diag_h_spatial(Q_list[0])

        self.history_ = []

        for t in range(T):

            alpha_t = self._alpha_at_round(t, T, alpha_end)
            alpha_hist[t] = alpha_t

            # Stage 1: tree on the negative gradient, Newton leaf values
            grad, hess = self._obj_grad_hess(y, F)

            mask = (np.ones(N, dtype=bool) if self.subsample >= 1.0
                    else rng.rand(N) < self.subsample)
            sw   = sample_weight[mask] if sample_weight is not None else None

            tree_t = self._make_tree(rng)
            tree_t.fit(X[mask], -grad[mask], sample_weight=sw)
            self._newton_leaf_update(tree_t, X, grad, hess)
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
                loss_t  = self._obj_loss(y, F)
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

            if t == unlock_rounds[active_l] and active_l > 0:
                self._update_diag_h_spatial(Q_list[active_l])

            is_final = (t == T - 1)
            w_nodes  = self._compute_w_nodes(loss_history, is_final)
            w_nodes_hist[t] = w_nodes

            # Stage 2: gradient and Hessian at the updated score, penalised WLS for the active level
            grad_upd, hess_upd = self._obj_grad_hess(y, F)

            grad2, hess2 = self._spatial_grad_hess(F, grad_upd, hess_upd)

            r2 = -grad2 / (hess2 + 1e-12)

            z_current = z_levels[active_l] if self.penalize_z else None

            delta_z = self._wls_solve(
                r2, hess2, alpha_t,
                Q_list[active_l],
                cg_warm_starts[active_l],
                z_current=z_current,
            )
            cg_warm_starts[active_l] = delta_z.copy()

            # Spatial step per w_mode:
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
                scaled_dz = w_nodes * delta_z

            elif self.w_mode == "no_geo":
                eta           = self._newton_step(delta_z, hess2, alpha_t,
                                                  Q_list[active_l])
                eta_hist[t]   = eta
                scaled_dz     = eta * delta_z
                if self.spatial_lr_cap is not None:
                    max_step = np.abs(scaled_dz).max()
                    if max_step > self.spatial_lr_cap:
                        scaled_dz *= self.spatial_lr_cap / max_step

            elif self.w_mode == "uniform":
                scaled_dz = self.learning_rate * delta_z

            elif self.w_mode == "unit":
                scaled_dz = delta_z

            else:  # "binary"
                scaled_dz = w_nodes * delta_z

            A_scaled_dz         = np.asarray(self.A @ scaled_dz).ravel()
            z_levels[active_l] += scaled_dz
            F                  += A_scaled_dz

            # Centering: c = mean(A dz) over the training points is moved from z to F0_.
            # Each row of A sums to 1, so A z drops by c everywhere and F0_ + A z, hence
            # every prediction, is unchanged. The running training score F is shifted by c.
            if self.center_spatial:
                correction         = float(np.mean(A_scaled_dz))
                z_levels[active_l] -= correction
                F                  -= correction
                self.F0_           += correction

            for l in range(L):
                all_znodes[t, l] = z_levels[l]

            loss_t  = self._obj_loss(y, F)
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
        self.eta_schedule_          = eta_hist
        self.all_znodes_            = all_znodes
        self.loss_nodes_schedule_   = loss_nodes_hist
        self.kappa_per_level_       = kappa_per_level
        self.rho_per_level_         = rho_per_level
        self.tau_per_level_         = (
            1.0 / (np.sqrt(4.0 * np.pi) * kappa_per_level)
            if kappa_per_level is not None else None
        )
        self.kappa_source_          = kappa_source
        self.variogram_result_      = variogram_result
        self.active_level_per_round_ = active_level_hist
        self.unlock_rounds_         = unlock_rounds
        self.n_features_in_         = X.shape[1]
        return self

    # >> Predict (regression - identity link)

    def _z_total(self):
        return sum(self.z_levels_)

    def decision_function(self, X, A_new):
        """Raw prediction: F0 + lr*trees + A_new @ z_total."""
        check_is_fitted(self, ["stage1_trees_", "z_levels_"])
        X = (X.astype(np.float32) if issparse(X)
             else np.asarray(X, dtype=np.float32))
        F = np.full(X.shape[0], self.F0_)
        for tree in self.stage1_trees_:
            F += self.learning_rate * tree.predict(X)
        F += np.asarray(A_new @ self._z_total()).ravel()
        return F

    def predict(self, X, A_new):
        """Continuous prediction (identity link)."""
        return self.decision_function(X, A_new)

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
        """Pure covariate contribution: lr * sum of trees (no intercept)."""
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
            f"  SAMBoostContinuous - level summary  [source: {self.kappa_source_}]",
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
