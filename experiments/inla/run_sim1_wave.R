################################################################################
# run_sim1_wave.R
# INLA (R-INLA, SPDE) on Simulation 1 (classification), binomial likelihood, logit link.
#
# Model: intercept + covariate effects + 2D Matern SPDE spatial field on an
# inla.mesh.2d built from the training locations.
#   linear : covariates as linear fixed effects
#   spline : each covariate as a 1D-SPDE spline on quantile knots
# Variants run: linear and spline (Table 1 of the paper reports the spline one).
#
# Prior selection: empirical variogram of y on the training data -> range estimate;
# grid of PC priors (range0 = range estimate x {0.5, 1, 2}, sigma0 in {1, 2});
# each combination fitted with CPO, the one with the largest sum of log CPO kept;
# final fit with the selected priors (empirical Bayes integration).
#
# Called by run_pipeline/run_multi_seed_inla.R (function run_sim). Outputs in
# out_dir: tuning_grid.csv, best_params.json, scores.csv, dissociation.csv,
# predictions.csv, grid_spatial_field.csv and plots.
################################################################################

library(INLA)
library(dplyr)
library(ggplot2)
library(jsonlite)

# >> Paths

REPO_ROOT <- normalizePath(file.path(getwd()), mustWork = FALSE)
while (!dir.exists(file.path(REPO_ROOT, "simulations")) && nchar(REPO_ROOT) > 3) {
  REPO_ROOT <- dirname(REPO_ROOT)
}

SIM_DIR <- file.path(REPO_ROOT, "outputs", "data", "sim1_wave")


# >> Simulation settings

SIM_NAME      <- "sim1_wave"
COORD_COLS    <- c("s1", "s2")
TARGET_COL    <- "y"
NON_FEAT_COLS <- c("s1", "s2", "F_effect", "spatial_effect", "m", "y")

# likelihood and link of the simulation
LIKELIHOOD    <- "binomial"
LINK          <- "logit"


# >> Tuning constants

# variogram settings
N_VARIOGRAM_SAMPLE <- 200    # points subsampled for variogram (speed)
N_VARIOGRAM_BINS   <- 15     # distance bins

# Prior grid: multipliers of the variogram range (HARD_* values override it)
RANGE_MULTIPLIERS  <- c(0.5, 1.0, 2.0)   # used when HARD_RANGE_VALUES = NULL
SIGMA_VALUES       <- c(1.0, 2.0)         # used when HARD_SIGMA_VALUES = NULL

HARD_RANGE_VALUES  <- NULL   # e.g. c(0.05, 0.10, 0.20) to hard-code range grid
HARD_SIGMA_VALUES  <- NULL   # e.g. c(0.5, 1.0, 2.0)   to hard-code sigma grid

# spline settings
N_SPLINE_KNOTS <- 3   # interior knots per covariate spline (try 5 if underfitting)

# INLA compute options
INLA_VERBOSE   <- FALSE
N_THREADS      <- parallel::detectCores() - 1

# Change the path if needed.
LOCAL_TMP <- "C:/inla_tmp"
dir.create(LOCAL_TMP, showWarnings = FALSE, recursive = TRUE)
inla.setOption("working.directory", LOCAL_TMP)

# Fallback prior grid, used only if every combination fails
FALLBACK_RANGE_VALUES <- c(0.10, 0.20, 0.40)
FALLBACK_SIGMA_VALUES <- c(0.5,  1.0,  2.0)


# >> Utility functions

# metrics

auc_score <- function(y_true, proba) {
  # Wilcoxon-Mann-Whitney AUC
  n1 <- sum(y_true == 1); n0 <- sum(y_true == 0)
  if (n1 == 0 || n0 == 0) return(NA_real_)
  r  <- rank(proba)
  (sum(r[y_true == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}

log_loss_score <- function(y_true, proba) {
  eps <- 1e-15
  p   <- pmin(pmax(proba, eps), 1 - eps)
  -mean(y_true * log(p) + (1 - y_true) * log(1 - p))
}

brier_score <- function(y_true, proba) {
  mean((y_true - proba)^2)
}

score_split <- function(name, y_true, proba) {
  pred <- as.integer(proba >= 0.5)
  data.frame(
    split    = name,
    n        = length(y_true),
    y1_freq  = round(mean(y_true), 4),
    error    = round(mean(pred != y_true), 4),
    auc      = round(auc_score(y_true, proba), 4),
    log_loss = round(log_loss_score(y_true, proba), 4),
    brier    = round(brier_score(y_true, proba), 4),
    stringsAsFactors = FALSE
  )
}

dissociation_row <- function(split_name, b_true, b_hat, F_true, F_hat) {
  data.frame(
    split        = split_name,
    r_spatial    = round(cor(b_true, b_hat, use = "complete.obs"), 4),
    rmse_spatial = round(sqrt(mean((b_true - b_hat)^2, na.rm = TRUE)), 4),
    r_cov        = round(cor(F_true, F_hat,  use = "complete.obs"), 4),
    rmse_cov     = round(sqrt(mean((F_true - F_hat)^2, na.rm = TRUE)), 4),
    stringsAsFactors = FALSE
  )
}


# variogram

estimate_spatial_range <- function(coords, y,
                                   n_sample = N_VARIOGRAM_SAMPLE,
                                   n_bins   = N_VARIOGRAM_BINS) {
  # Subsample to keep O(n^2) pairs manageable
  idx  <- sample(nrow(coords), min(n_sample, nrow(coords)))
  locs <- coords[idx, , drop = FALSE]
  vals <- y[idx]

  D    <- as.matrix(dist(locs))
  V    <- outer(vals, vals, function(a, b) (a - b)^2 / 2)

  upper  <- upper.tri(D)
  dists  <- D[upper]
  svars  <- V[upper]

  breaks  <- quantile(dists, probs = seq(0, 1, length.out = n_bins + 1))
  bin_id  <- cut(dists, breaks = breaks, include.lowest = TRUE)
  gamma   <- as.numeric(tapply(svars, bin_id, mean))
  centers <- as.numeric(tapply(dists, bin_id, mean))

  # range = first distance where semivariance exceeds 95% of sill
  sill      <- max(gamma, na.rm = TRUE)
  above_idx <- which(gamma >= 0.95 * sill)
  range_est <- if (length(above_idx) > 0) centers[above_idx[1]] else centers[length(centers)]

  list(range_est = range_est, centers = centers, gamma = gamma)
}

plot_variogram <- function(vario, out_dir) {
  df <- data.frame(distance = vario$centers, semivariance = vario$gamma)
  p  <- ggplot(df, aes(x = distance, y = semivariance)) +
    geom_point(size = 3, colour = "#1f77b4") +
    geom_line(colour = "#1f77b4") +
    geom_vline(xintercept = vario$range_est, linetype = "dashed",
               colour = "red", linewidth = 0.8) +
    annotate("text", x = vario$range_est, y = max(df$semivariance) * 0.9,
             label = sprintf("range = %.3f", vario$range_est),
             hjust = -0.1, colour = "red", size = 3.5) +
    labs(title = "Empirical variogram (training y)",
         x = "Distance", y = "Semivariance") +
    theme_bw()
  ggsave(file.path(out_dir, "variogram.png"), p, width = 6, height = 4, dpi = 150)
  cat(sprintf("  -> variogram.png  (estimated range = %.4f)\n", vario$range_est))
}


# mesh

build_spatial_mesh <- function(coords, range_est) {
  # max.edge: inner zone = range/3, outer zone = range * 2
  # cutoff  : min node spacing = max.edge.inner / 5
  max_edge_inner <- range_est / 3
  max_edge_outer <- range_est * 2
  cutoff         <- max_edge_inner / 5

  mesh <- inla.mesh.2d(
    loc      = coords,
    max.edge = c(max_edge_inner, max_edge_outer),
    cutoff   = cutoff,
    offset   = c(range_est * 0.5, range_est * 1.5)
  )
  cat(sprintf("  Mesh nodes = %d  (max.edge inner=%.4f outer=%.4f)\n",
              mesh$n, max_edge_inner, max_edge_outer))
  mesh
}


# spline knots

make_1d_mesh <- function(x_train, n_knots = N_SPLINE_KNOTS) {
  # quantile-based interior knots + boundary at observed range
  probs  <- seq(0, 1, length.out = n_knots + 2)
  knots  <- quantile(x_train, probs = probs)
  bounds <- c(knots[1], knots[length(knots)])
  inla.mesh.1d(knots, interval = bounds, degree = 2,
               boundary = c("neumann", "neumann"))
}


# >> Load data

load_data <- function(sim_dir) {
  train_df <- read.csv(file.path(sim_dir, "train_df.csv"))
  test_df  <- read.csv(file.path(sim_dir, "test_df.csv"))
  cat(sprintf("  Train : %d rows\n", nrow(train_df)))
  cat(sprintf("  Test  : %d rows\n", nrow(test_df)))
  list(train = train_df, test = test_df)
}


# >> Prior grid

build_prior_grid <- function(range_est) {
  # Use hard-coded values if provided, otherwise multiply variogram estimate
  r_vals <- if (!is.null(HARD_RANGE_VALUES)) HARD_RANGE_VALUES else range_est * RANGE_MULTIPLIERS
  s_vals <- if (!is.null(HARD_SIGMA_VALUES)) HARD_SIGMA_VALUES else SIGMA_VALUES

  cat("  Prior grid:\n")
  cat(sprintf("    range0 values : %s\n", paste(round(r_vals, 4), collapse = ", ")))
  cat(sprintf("    sigma0 values : %s\n", paste(round(s_vals, 4), collapse = ", ")))

  expand.grid(range0 = r_vals, sigma0 = s_vals,
              stringsAsFactors = FALSE)
}


# >> INLA stack (linear and spline variants)

build_stack <- function(y, coords, feat_df, mesh_spatial, spde_spatial,
                        spde_list = NULL,
                        mesh_list = NULL,
                        feat_cols = NULL,
                        tag) {
  n_obs <- nrow(feat_df)

  # A matrix for spatial effect: maps observation locations to mesh nodes
  A_spatial <- inla.spde.make.A(mesh_spatial, loc = as.matrix(coords))
  spat_idx  <- inla.spde.make.index("spatial", n.spde = spde_spatial$n.spde)

  # Intercept as a fixed-effect covariate equal to 1 (projection A = 1)
  intercept_list <- list(intercept = rep(1, n_obs))

  if (is.null(spde_list)) {
    # Variant A: linear fixed effects
    # intercept and covariate values share the same A = 1 block
    covariate_list <- c(intercept_list, as.list(feat_df))
    A_list   <- list(A_spatial, 1)
    eff_list <- list(spat_idx, covariate_list)
  } else {
    # Variant B: spline (1D SPDE) effects
    # One A matrix per covariate spline + 2D spatial A + A = 1 for intercept
    A_splines  <- lapply(seq_along(feat_cols), function(j) {
      inla.spde.make.A(mesh_list[[j]], loc = feat_df[[feat_cols[j]]])
    })
    spline_idx <- lapply(seq_along(feat_cols), function(j) {
      inla.spde.make.index(feat_cols[j], n.spde = spde_list[[j]]$n.spde)
    })
    A_list   <- c(list(A_spatial), A_splines, list(1))
    eff_list <- c(list(spat_idx), spline_idx, list(intercept_list))
  }

  inla.stack(
    data    = list(y = y),
    A       = A_list,
    effects = eff_list,
    tag     = tag
  )
}


# >> Formulas

make_formula_linear <- function(feat_cols) {
  # y ~ -1 + intercept + x1 + ... + xp + f(spatial, model = spde)
  fixed_part <- paste(feat_cols, collapse = " + ")
  as.formula(paste(
    "y ~ -1 + intercept +", fixed_part,
    "+ f(spatial, model = spde_spatial)"
  ))
}

make_formula_spline <- function(feat_cols) {
  # y ~ -1 + intercept + f(x1, model=spde_x1) + ... + f(spatial, model=spde)
  spline_terms <- paste(
    sprintf("f(%s, model = spde_list[[\"%s\"]])", feat_cols, feat_cols),
    collapse = " + "
  )
  as.formula(paste(
    "y ~ -1 + intercept +", spline_terms,
    "+ f(spatial, model = spde_spatial)"
  ))
}


# >> Effect extraction

extract_spatial_effect <- function(fit, mesh_spatial, coords) {
  # Posterior mean of the spatial field at the given locations
  proj  <- inla.mesh.projector(mesh_spatial, loc = as.matrix(coords))
  b_hat <- inla.mesh.project(proj, fit$summary.random$spatial$mean)
  as.numeric(b_hat)
}

extract_covariate_effect_linear <- function(fit, feat_df, feat_cols) {
  # Sum of beta_j * x_ij for all fixed effects j
  # fit$summary.fixed rows are named by variable (after intercept)
  betas   <- fit$summary.fixed$mean
  names(betas) <- rownames(fit$summary.fixed)
  # exclude intercept from covariate effect
  betas   <- betas[names(betas) %in% feat_cols]
  F_hat   <- as.matrix(feat_df[, names(betas)]) %*% betas
  as.numeric(F_hat)
}

extract_covariate_effect_spline <- function(fit, feat_df, feat_cols,
                                            mesh_list) {
  # For each covariate, project its 1D SPDE posterior mean to observation values
  # then sum across all covariates
  F_hat <- rep(0, nrow(feat_df))
  for (j in seq_along(feat_cols)) {
    col     <- feat_cols[j]
    proj_1d <- inla.mesh.projector(mesh_list[[j]], loc = feat_df[[col]])
    contrib <- inla.mesh.project(proj_1d, fit$summary.random[[col]]$mean)
    F_hat   <- F_hat + as.numeric(contrib)
  }
  F_hat
}

extract_predictions <- function(fit, join_stack, tag) {
  # Posterior mean fitted values (probability scale)
  idx  <- inla.stack.index(join_stack, tag)$data
  phat <- fit$summary.fitted.values$mean[idx]

  # Check that the fitted values lie in [0, 1] (probability scale)
  if (any(phat < 0 | phat > 1, na.rm = TRUE)) {
    warning(sprintf("[%s] some fitted values outside [0,1] - check link function",
                    tag))
  }
  as.numeric(phat)
}


# >> Fit one prior combination (CPO grid search)

fit_prior_combo <- function(range0, sigma0,
                             join_stack, formula_obj, mesh_spatial,
                             spde_list = NULL, mesh_list = NULL,
                             use_fallback_options = FALSE) {
  # Rebuild the SPDE with the new priors in the formula's environment, where INLA looks it up
  new_spde <- inla.spde2.pcmatern(
    mesh        = mesh_spatial,
    alpha       = 2,
    prior.range = c(range0, 0.05),
    prior.sigma = c(sigma0, 0.05),
    constr      = TRUE
  )
  environment(formula_obj)$spde_spatial <- new_spde
  if (!is.null(spde_list)) {
    environment(formula_obj)$spde_list <- spde_list
  }

  # int.strategy: "eb" (empirical Bayes) is fast; "grid" is more robust
  # when "eb" causes convergence failures
  int_strategy <- if (use_fallback_options) "grid" else "eb"

  # Fit with CPO and config = TRUE (without config, inla.cpo() returns NA)
  fit_ok <- tryCatch({
    inla(
      formula_obj,
      family             = LIKELIHOOD,
      control.family     = list(link = LINK),
      data               = inla.stack.data(join_stack),
      control.predictor  = list(A = inla.stack.A(join_stack),
                                compute = TRUE, link = 1),
      control.compute    = list(cpo = TRUE, dic = TRUE, config = TRUE),
      control.inla       = list(int.strategy = int_strategy),
      working.directory  = LOCAL_TMP,
      # verbose            = INLA_VERBOSE,
      verbose            = TRUE,
      num.threads        = N_THREADS
    )
  }, error = function(e) {
    cat(sprintf("    [base fit failed: %s]", conditionMessage(e)))
    NULL
  })

  if (is.null(fit_ok)) {
    return(list(fit = NULL, sum_log_cpo = -Inf, failed = TRUE))
  }

  # Score = sum of log CPO over the training points; -DIC if CPO is unavailable
  sum_log_cpo <- tryCatch({
    if (!is.null(fit_ok$cpo$failure) &&
        any(fit_ok$cpo$failure > 0, na.rm = TRUE)) {
      fit_ok <- inla.cpo(fit_ok, verbose = FALSE)
    }
    cpo_vals <- fit_ok$cpo$cpo
    cpo_vals <- cpo_vals[is.finite(cpo_vals) & cpo_vals > 0]
    if (length(cpo_vals) == 0) {
      if (!is.null(fit_ok$dic$dic)) -fit_ok$dic$dic else -Inf
    } else {
      sum(log(cpo_vals))
    }
  }, error = function(e) {
    cat(sprintf("    [CPO failed, using DIC fallback]"))
    if (!is.null(fit_ok$dic$dic)) -fit_ok$dic$dic else -Inf
  })

  list(fit = fit_ok, sum_log_cpo = sum_log_cpo, failed = FALSE)
}


# >> CPO grid search

run_one_grid <- function(prior_grid, join_stack, formula_obj,
                          mesh_spatial, spde_list, mesh_list,
                          use_fallback_options, label) {
  # Internal helper: run a single grid pass and return a results data.frame
  n_combos <- nrow(prior_grid)
  cat(sprintf("  %s - %d combinations\n", label, n_combos))

  rows <- vector("list", n_combos)
  for (i in seq_len(n_combos)) {
    r0 <- prior_grid$range0[i]
    s0 <- prior_grid$sigma0[i]
    cat(sprintf("  [%2d/%2d]  range0=%.4f  sigma0=%.4f", i, n_combos, r0, s0))

    result <- fit_prior_combo(r0, s0, join_stack, formula_obj,
                               mesh_spatial, spde_list, mesh_list,
                               use_fallback_options = use_fallback_options)
    cpo_val <- result$sum_log_cpo
    cat(sprintf("  score = %.4f\n", cpo_val))

    rows[[i]] <- data.frame(range0 = r0, sigma0 = s0,
                             sum_log_cpo = cpo_val,
                             failed = result$failed,
                             grid_pass = label,
                             stringsAsFactors = FALSE)
  }
  do.call(rbind, rows)
}


run_cpo_grid_search <- function(prior_grid, join_stack, formula_obj,
                                 mesh_spatial,
                                 spde_list = NULL, mesh_list = NULL,
                                 out_dir) {

  # Pass 1: variogram-derived (or user-specified) prior grid
  grid_df <- run_one_grid(prior_grid, join_stack, formula_obj,
                           mesh_spatial, spde_list, mesh_list,
                           use_fallback_options = FALSE,
                           label = "variogram grid")

  # Pass 2: if ALL combinations failed, retry with hard-coded fallback
  all_failed <- all(grid_df$failed)
  if (all_failed) {
    cat("
  [!] All variogram-grid combinations failed.
")
    cat("  [!] Retrying with fallback priors and robust integration (int.strategy=grid).
")
    cat(sprintf("  [!] Fallback range values : %s\n",
                paste(FALLBACK_RANGE_VALUES, collapse = ", ")))
    cat(sprintf("  [!] Fallback sigma values : %s\n",
                paste(FALLBACK_SIGMA_VALUES, collapse = ", ")))

    fallback_grid <- expand.grid(range0 = FALLBACK_RANGE_VALUES,
                                 sigma0 = FALLBACK_SIGMA_VALUES,
                                 stringsAsFactors = FALSE)
    grid_fb <- run_one_grid(fallback_grid, join_stack, formula_obj,
                             mesh_spatial, spde_list, mesh_list,
                             use_fallback_options = TRUE,
                             label = "fallback grid")

    # combine both passes in the saved CSV for transparency
    grid_df <- rbind(grid_df, grid_fb)
  }

  # Select best non-failed combination
  grid_df <- grid_df[order(-grid_df$sum_log_cpo), ]
  grid_df$rank <- seq_len(nrow(grid_df))
  write.csv(grid_df, file.path(out_dir, "tuning_grid.csv"), row.names = FALSE)
  cat(sprintf("  Tuning grid saved -> %s\n",
              file.path(out_dir, "tuning_grid.csv")))

  # If everything still failed, raise an informative error
  valid <- grid_df[!grid_df$failed, ]
  if (nrow(valid) == 0) {
    stop(paste(
      "All CPO/DIC grid combinations failed for both variogram and fallback grids.",
      "Check your INLA installation, formula, or stack construction.",
      "Try setting INLA_VERBOSE <- TRUE to see the internal error messages."
    ))
  }

  best <- valid[1, ]
  cat(sprintf("  Best prior : range0=%.4f  sigma0=%.4f  score=%.4f  [%s]\n",
              best$range0, best$sigma0, best$sum_log_cpo, best$grid_pass))

  list(grid_df    = grid_df,
       best_range0 = best$range0,
       best_sigma0 = best$sigma0,
       best_log_cpo = best$sum_log_cpo)
}


# >> Final fit with the selected priors

fit_final_model <- function(best_range0, best_sigma0,
                             join_stack, formula_obj,
                             mesh_spatial,
                             spde_list = NULL, mesh_list = NULL) {
  # As in fit_prior_combo: SPDE with the selected priors in the formula's environment
  environment(formula_obj)$spde_spatial <- inla.spde2.pcmatern(
    mesh        = mesh_spatial,
    alpha       = 2,
    prior.range = c(best_range0, 0.05),
    prior.sigma = c(best_sigma0, 0.05),
    constr      = TRUE
  )
  if (!is.null(spde_list)) {
    environment(formula_obj)$spde_list <- spde_list
  }

  t0  <- proc.time()["elapsed"]
  fit <- inla(
    formula_obj,
    family               = LIKELIHOOD,
    control.family       = list(link = LINK),
    data                 = inla.stack.data(join_stack),
    control.predictor    = list(A = inla.stack.A(join_stack),
                                compute = TRUE, link = 1),
    control.compute      = list(cpo = TRUE, dic = TRUE, waic = TRUE,
                                config = TRUE),
    control.inla         = list(int.strategy = "eb"),
    working.directory    = LOCAL_TMP,
    # verbose              = INLA_VERBOSE,
    verbose              = TRUE,
    num.threads          = N_THREADS
  )
  fit_time <- proc.time()["elapsed"] - t0
  cat(sprintf("  Final model fit time: %.1f s\n", fit_time))
  list(fit = fit, fit_time = fit_time)
}


# >> Plots

plot_dissociation <- function(split_name, df, b_hat, F_hat,
                               sim_name, out_dir) {
  s1    <- df$s1;   s2    <- df$s2
  b_true <- df$spatial_effect;  F_true <- df$F_effect

  r_s <- cor(b_true, b_hat, use = "complete.obs")
  r_f <- cor(F_true, F_hat, use = "complete.obs")

  # helper: spatial scatter coloured by value
  make_scatter <- function(vals, title, low = "blue", high = "red") {
    ggplot(data.frame(s1 = s1, s2 = s2, v = vals),
           aes(x = s1, y = s2, colour = v)) +
      geom_point(size = 1.2, alpha = 0.8) +
      scale_colour_gradient2(low = low, mid = "white", high = high,
                             midpoint = 0) +
      labs(title = title, colour = "") +
      theme_bw() + theme(legend.position = "right")
  }

  p1 <- make_scatter(b_true, "True  b(s)")
  p2 <- make_scatter(b_hat,  "INLA  b_hat(s)")
  p3 <- make_scatter(b_true - b_hat, "Residual  b - b_hat",
                     low = "blue", high = "red")
  p4 <- make_scatter(F_true, "True  F(X)", low = "green4", high = "purple")
  p5 <- make_scatter(F_hat,  "INLA  F_hat(X)", low = "green4", high = "purple")
  p6 <- ggplot(data.frame(F_true = F_true, F_hat = F_hat),
               aes(x = F_true, y = F_hat)) +
    geom_point(size = 0.8, alpha = 0.4, colour = "#d62728") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
    labs(title = "True vs INLA F(X)",
         x = "True F(X)", y = "INLA F_hat(X)") +
    theme_bw()

  png(file.path(out_dir, sprintf("dissociation_%s.png", split_name)),
      width = 1800, height = 1200, res = 150)
  gridExtra::grid.arrange(p1, p2, p3, p4, p5, p6, ncol = 3,
    top = sprintf("INLA [%s] [%s]  r_spatial=%.3f  r_cov=%.3f",
                  sim_name, split_name, r_s, r_f))
  dev.off()
  cat(sprintf("  -> dissociation_%s.png\n", split_name))
}

plot_predicted_proba <- function(train_df, test_df,
                                  p_train, p_test,
                                  sim_name, variant_name, out_dir) {
  make_panel <- function(df, proba, title) {
    ggplot(data.frame(s1 = df$s1, s2 = df$s2, p = proba),
           aes(x = s1, y = s2, colour = p)) +
      geom_point(size = 1.2, alpha = 0.8) +
      scale_colour_viridis_c(limits = c(0, 1), name = "P(y=1)") +
      labs(title = title, x = "s1", y = "s2") +
      theme_bw()
  }
  p1 <- make_panel(train_df, p_train, "Train")
  p2 <- make_panel(test_df,  p_test,  "Test")

  png(file.path(out_dir, "predicted_proba.png"),
      width = 1400, height = 700, res = 150)
  gridExtra::grid.arrange(p1, p2, ncol = 2,
    top = sprintf("INLA (%s) [%s] - Predicted Probability",
                  variant_name, sim_name))
  dev.off()
  cat("  -> predicted_proba.png\n")
}

plot_tuning_heatmap <- function(grid_df, out_dir) {
  p <- ggplot(grid_df, aes(x = factor(round(range0, 4)),
                            y = factor(round(sigma0, 4)),
                            fill = sum_log_cpo)) +
    geom_tile() +
    geom_text(aes(label = round(sum_log_cpo, 1)), size = 3) +
    scale_fill_viridis_c(name = "sum log CPO") +
    labs(title = "Prior grid - sum log CPO",
         x = "prior range0", y = "prior sigma0") +
    theme_bw()
  ggsave(file.path(out_dir, "tuning_heatmap.png"), p,
         width = 6, height = 4, dpi = 150)
  cat("  -> tuning_heatmap.png\n")
}


# >> Run one variant

run_variant <- function(variant,       # "linear" or "spline"
                         train_df, test_df,
                         feat_cols, coords_train, coords_test,
                         mesh_spatial, spde_spatial,
                         prior_grid, out_dir_variant) {

  dir.create(out_dir_variant, showWarnings = FALSE, recursive = TRUE)
  cat(sprintf("\n--- Variant: %s ---\n", variant))

  y_train <- train_df[[TARGET_COL]]
  y_test  <- rep(NA_real_, nrow(test_df))

  spde_list <- NULL
  mesh_list <- NULL

  if (variant == "spline") {
    mesh_list  <- lapply(feat_cols, function(col)
                    make_1d_mesh(train_df[[col]]))
    names(mesh_list) <- feat_cols

    d      <- 1; nu <- 1
    alpha_ <- nu + d / 2
    spde_list <- lapply(feat_cols, function(col)
                   inla.spde2.pcmatern(
                     mesh_list[[col]], alpha = alpha_,
                     prior.sigma = c(2, 0.05), prior.range = c(1, 0.05),
                     constr = TRUE))
    names(spde_list) <- feat_cols
  }

  # stacks
  stack_train <- build_stack(y_train, coords_train,
                              train_df[, feat_cols, drop = FALSE],
                              mesh_spatial, spde_spatial,
                              spde_list, mesh_list, feat_cols, tag = "train")
  stack_test  <- build_stack(y_test, coords_test,
                              test_df[, feat_cols, drop = FALSE],
                              mesh_spatial, spde_spatial,
                              spde_list, mesh_list, feat_cols, tag = "test")

  join_stack <- inla.stack(stack_train, stack_test)

  # formula
  formula_obj <- if (variant == "linear") {
    make_formula_linear(feat_cols)
  } else {
    make_formula_spline(feat_cols)
  }
  environment(formula_obj) <- environment()

  # CPO grid search
  cat("\n[CPO grid search]\n")
  cpo_result <- run_cpo_grid_search(
    prior_grid, join_stack, formula_obj,
    mesh_spatial, spde_list, mesh_list, out_dir_variant
  )
  plot_tuning_heatmap(cpo_result$grid_df, out_dir_variant)

  # final model
  cat("\n[Final model]\n")
  final <- fit_final_model(
    cpo_result$best_range0, cpo_result$best_sigma0,
    join_stack, formula_obj, mesh_spatial,
    spde_list, mesh_list
  )
  fit      <- final$fit
  fit_time <- final$fit_time

  # predictions
  p_train <- extract_predictions(fit, join_stack, "train")
  p_test  <- extract_predictions(fit, join_stack, "test")

  # spatial effect at observation locations
  b_train_hat <- extract_spatial_effect(fit, mesh_spatial, coords_train)
  b_test_hat  <- extract_spatial_effect(fit, mesh_spatial, coords_test)

  # covariate effect at observation locations
  if (variant == "linear") {
    F_train_hat <- extract_covariate_effect_linear(fit, train_df, feat_cols)
    F_test_hat  <- extract_covariate_effect_linear(fit, test_df,  feat_cols)
  } else {
    F_train_hat <- extract_covariate_effect_spline(fit, train_df, feat_cols, mesh_list)
    F_test_hat  <- extract_covariate_effect_spline(fit, test_df,  feat_cols, mesh_list)
  }

  # sanity check: linear predictor decomposition
  eta_approx <- F_train_hat + b_train_hat
  if (!is.null(fit$summary.fixed["intercept", ])) {
    eta_approx <- eta_approx + fit$summary.fixed["intercept", "mean"]
  }
  p_check <- plogis(eta_approx)
  check_cor <- cor(p_train, p_check, use = "complete.obs")
  cat(sprintf("  Decomposition check: cor(p_hat, Phi(F+b+intercept)) = %.4f\n",
              check_cor))
  if (check_cor < 0.95) {
    warning("Decomposition check failed - verify effect extraction")
  }

  # scores
  scores_df <- rbind(
    score_split("train", train_df[[TARGET_COL]], p_train),
    score_split("test",  test_df[[TARGET_COL]],  p_test)
  )
  write.csv(scores_df, file.path(out_dir_variant, "scores.csv"),
            row.names = FALSE)
  cat("\n  Prediction scores:\n")
  print(scores_df)

  # dissociation
  dissoc_df <- rbind(
    dissociation_row("train", train_df$spatial_effect, b_train_hat,
                              train_df$F_effect,        F_train_hat),
    dissociation_row("test",  test_df$spatial_effect,  b_test_hat,
                              test_df$F_effect,         F_test_hat)
  )
  dissoc_df$fit_time <- round(fit_time, 2)
  write.csv(dissoc_df, file.path(out_dir_variant, "dissociation.csv"),
            row.names = FALSE)
  cat("\n  Dissociation:\n")
  print(dissoc_df[, c("split", "r_spatial", "rmse_spatial",
                       "r_cov", "rmse_cov")])

  # predictions CSV
  make_pred_rows <- function(df, p_hat, b_hat, F_hat, split) {
    data.frame(s1 = df$s1, s2 = df$s2,
               y = df[[TARGET_COL]],
               F_true = df$F_effect, b_true = df$spatial_effect,
               p_hat = p_hat, b_hat = b_hat, F_hat = F_hat,
               split = split, stringsAsFactors = FALSE)
  }
  pred_df <- rbind(
    make_pred_rows(train_df, p_train, b_train_hat, F_train_hat, "train"),
    make_pred_rows(test_df,  p_test,  b_test_hat,  F_test_hat,  "test")
  )
  write.csv(pred_df, file.path(out_dir_variant, "predictions.csv"),
            row.names = FALSE)

  # grid spatial field (50x50 grid)
  grid_1d <- seq(0, 1, length.out = 50)
  grid_locs <- as.matrix(expand.grid(s1 = grid_1d, s2 = grid_1d))
  b_hat_grid <- extract_spatial_effect(fit, mesh_spatial, grid_locs)
  grid_df <- data.frame(s1 = grid_locs[, 1], s2 = grid_locs[, 2],
                         b_hat = b_hat_grid)
  # Compute true wave spatial effect on grid
  b_true_grid <- 1.2 * sin(2*pi*grid_locs[, 1]) * cos(1.5*pi*grid_locs[, 2]) +
                 0.7 * cos(3*pi*grid_locs[, 1] * grid_locs[, 2])
  grid_df$b_true <- b_true_grid
  write.csv(grid_df, file.path(out_dir_variant, "grid_spatial_field.csv"),
            row.names = FALSE)
  cat(sprintf("  Grid spatial field saved -> %s\n",
              file.path(out_dir_variant, "grid_spatial_field.csv")))

  # mesh geometry
  write.csv(data.frame(s1 = mesh_spatial$loc[, 1], s2 = mesh_spatial$loc[, 2]),
            file.path(out_dir_variant, "mesh_nodes.csv"), row.names = FALSE)
  write.csv(data.frame(v1 = mesh_spatial$graph$tv[, 1],
                        v2 = mesh_spatial$graph$tv[, 2],
                        v3 = mesh_spatial$graph$tv[, 3]),
            file.path(out_dir_variant, "mesh_triangles.csv"), row.names = FALSE)
  cat("  Mesh geometry saved -> mesh_nodes.csv, mesh_triangles.csv\n")

  # plots
  plot_dissociation("train", train_df, b_train_hat, F_train_hat,
                    SIM_NAME, out_dir_variant)
  plot_dissociation("test",  test_df,  b_test_hat,  F_test_hat,
                    SIM_NAME, out_dir_variant)
  plot_predicted_proba(train_df, test_df,
                       p_train, p_test,
                       SIM_NAME, variant, out_dir_variant)

  # best params JSON
  best_params <- list(
    range0          = cpo_result$best_range0,
    sigma0          = cpo_result$best_sigma0,
    variant         = variant,
    n_spline_knots  = N_SPLINE_KNOTS,
    n_mesh_nodes    = mesh_spatial$n
  )
  write_json(best_params,
             file.path(out_dir_variant, "best_params.json"),
             pretty = TRUE, auto_unbox = TRUE)

  # summary dict
  d <- function(split, col) {
    dissoc_df[[col]][dissoc_df$split == split]
  }
  s <- function(split, col) {
    scores_df[[col]][scores_df$split == split]
  }

  list(
    sim        = SIM_NAME,
    model      = paste0("inla_", variant),
    log_cpo_cv = round(cpo_result$best_log_cpo, 4),

    auc_train  = s("train", "auc"),
    auc_test   = s("test",  "auc"),
    ll_test    = s("test",  "log_loss"),

    r_spatial_train    = d("train", "r_spatial"),
    rmse_spatial_train = d("train", "rmse_spatial"),
    r_cov_train        = d("train", "r_cov"),
    rmse_cov_train     = d("train", "rmse_cov"),

    r_spatial_test     = d("test", "r_spatial"),
    rmse_spatial_test  = d("test", "rmse_spatial"),
    r_cov_test         = d("test", "r_cov"),
    rmse_cov_test      = d("test", "rmse_cov"),

    fit_time          = round(fit_time, 2),
    best_range0       = cpo_result$best_range0,
    best_sigma0       = cpo_result$best_sigma0,
    best_n_mesh_nodes = mesh_spatial$n,
    best_variant      = variant
  )
}


# >> Main entry point

run_sim <- function(out_dir, dev_folder = NULL, fixed_priors = NULL) {

  out_dir <- file.path(out_dir)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  cat(sprintf("\n%s\n  INLA-SPDE  [%s]\n%s\n",
              strrep("=", 60), SIM_NAME, strrep("=", 60)))

  # [1] load data
  cat("\n[1] Loading simulation data ...\n")
  data_list <- load_data(SIM_DIR)
  train_df  <- data_list$train
  test_df   <- data_list$test

  feat_cols    <- setdiff(names(train_df), NON_FEAT_COLS)
  coords_train <- as.matrix(train_df[, COORD_COLS])
  coords_test  <- as.matrix(test_df[, COORD_COLS])

  cat(sprintf("  Feature cols (%d): %s\n",
              length(feat_cols), paste(feat_cols, collapse = ", ")))
  cat(sprintf("  Coords passed to SPDE mesh: %s\n",
              paste(COORD_COLS, collapse = ", ")))

  # [2] variogram
  cat("\n[2] Estimating spatial range from empirical variogram ...\n")
  vario <- estimate_spatial_range(coords_train, train_df[[TARGET_COL]])
  cat(sprintf("  Estimated range = %.4f\n", vario$range_est))
  plot_variogram(vario, out_dir)

  # [3] build spatial mesh
  cat("\n[3] Building 2D INLA mesh from training point cloud ...\n")
  mesh_spatial <- build_spatial_mesh(coords_train, vario$range_est)

  # [4] build prior grid
  cat("\n[4] Building prior grid ...\n")
  if (!is.null(fixed_priors)) {
    prior_grid <- data.frame(range0 = fixed_priors$range0,
                              sigma0 = fixed_priors$sigma0)
    cat(sprintf("  Using fixed priors: range0=%.4f, sigma0=%.4f\n",
                fixed_priors$range0, fixed_priors$sigma0))
  } else {
    prior_grid <- build_prior_grid(vario$range_est)
  }

  # placeholder SPDE
  spde_spatial <- inla.spde2.pcmatern(
    mesh        = mesh_spatial,
    alpha       = 2,
    prior.range = c(vario$range_est, 0.05),
    prior.sigma = c(1.0, 0.05),
    constr      = TRUE
  )

  # [5] run both variants
  summaries <- list()

  cat("\n[5a] Running Variant A (linear fixed effects) ...\n")
  summaries[["linear"]] <- run_variant(
    "linear", train_df, test_df,
    feat_cols, coords_train, coords_test,
    mesh_spatial, spde_spatial, prior_grid,
    out_dir_variant = file.path(out_dir, "linear")
  )

  cat("\n[5b] Running Variant B (covariate splines) ...\n")
  summaries[["spline"]] <- run_variant(
    "spline", train_df, test_df,
    feat_cols, coords_train, coords_test,
    mesh_spatial, spde_spatial, prior_grid,
    out_dir_variant = file.path(out_dir, "spline")
  )

  cat(sprintf("\n  Done. All outputs in %s\n", out_dir))
  summaries
}


# >> Standalone entry

if (!interactive()) {
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  out_dir   <- file.path(REPO_ROOT, "outputs", "results", "inla",
                         paste0("run_", timestamp), SIM_NAME)
  run_sim(out_dir)
}