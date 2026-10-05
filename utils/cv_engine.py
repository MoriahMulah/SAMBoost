"""
cv_engine.py
============
K-fold cross-validation for the classification model (SAMBoost), used to
tune Sim 1.

The mesh, G, C and Q are built once from all training points and stay fixed
across folds. The projection matrix A has one row per observation, so each
fold uses the rows of its own observations:

    A_fold_train = A_global[train_idx]   passed to the model as A
    A_fold_val   = A_global[val_idx]     used for prediction
"""

import time
import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score

# Plot style of the per-seed diagnostic plots of the simulation runs
plt.style.use("seaborn-v0_8-whitegrid")


class SAMBoostCV:
    """
    Stratified K-fold cross-validator for SAMBoost.

    The model is fitted on A_global[train_idx] and evaluated with
    predict_proba(X_val, A_new=A_global[val_idx]). Q is stored but not used:
    the model builds its own Q from G, C and kappa.
    """

    def __init__(
        self,
        A_global,           # (N, M) sparse - full dataset A
        Q,                  # (M, M) sparse - SPDE precision (constant)
        G,
        C,
        n_splits: int = 5,
        random_state: int = 0,
    ):
        self.A_global     = A_global
        self.Q            = Q
        self.G            = G
        self.C            = C
        self.n_splits     = n_splits
        self.random_state = random_state

    def score_params(
        self,
        X: np.ndarray,
        y: np.ndarray,
        coords: np.ndarray,
        params: dict,
        verbose: bool = False,
    ) -> dict:
        """
        Run K-fold CV for one hyperparameter configuration.

        Returns a dict with mean/std of AUC, log-loss, accuracy,
        spatial_effect_std (how large the spatial surface is), and fit time.
        """
        skf = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )

        fold_aucs, fold_lls, fold_accs = [], [], []
        fold_spat_stds, fold_times     = [], []

        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            A_tr  = self.A_global[tr_idx]    # (n_train_fold, M)
            A_val = self.A_global[val_idx]   # (n_val_fold,   M)

            X_tr, y_tr = X[tr_idx], y[tr_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            coords_tr, coords_val = coords[tr_idx], coords[val_idx]

            from samboost.model import SAMBoost
            model = SAMBoost(A=A_tr, G=self.G, C=self.C, **params)

            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_tr, y_tr, coords=coords_tr)
            fold_times.append(time.perf_counter() - t0)

            proba = model.predict_proba(X_val, A_new=A_val)[:, 1]
            preds = (proba >= 0.5).astype(int)

            fold_aucs.append(roc_auc_score(y_val, proba))
            fold_lls.append(log_loss(y_val, proba))
            fold_accs.append(accuracy_score(y_val, preds))

            # Standard deviation of the spatial effect on the validation points
            spat = model.spatial_effect(A_val)
            fold_spat_stds.append(spat.std())

            if verbose:
                print(f"  fold {fold+1}/{self.n_splits}  "
                      f"auc={fold_aucs[-1]:.4f}  "
                      f"ll={fold_lls[-1]:.4f}  "
                      f"time={fold_times[-1]:.1f}s")

        return {
            "mean_auc"       : np.mean(fold_aucs),
            "std_auc"        : np.std(fold_aucs),
            "mean_logloss"   : np.mean(fold_lls),
            "std_logloss"    : np.std(fold_lls),
            "mean_acc"       : np.mean(fold_accs),
            "std_acc"        : np.std(fold_accs),
            "mean_spat_std"  : np.mean(fold_spat_stds),
            "std_spat_std"   : np.std(fold_spat_stds),
            "mean_fit_time"  : np.mean(fold_times),
            "fold_aucs"      : fold_aucs,
            "fold_lls"       : fold_lls,
        }
