# run_inla.R
# INLA on the house-price data, one or more test years per run (folds from
# data_folds.R). Gaussian likelihood on log_price with:
#   - a 1D SPDE spline for each numeric covariate (age, TLA, beds, baths,
#     halfbaths, frontage, depth, garagesqft, rooms, lotsize), covariates
#     standardised on the training data;
#   - stories, wall and garage as fixed effects;
#   - the sale year as a linear term (when HOUSE_INCLUDE_SYEAR is set);
#   - a 2D SPDE spatial field on a mesh built from a variogram range estimate.
# The spatial PC prior is tuned on the validation year (9 combinations, lowest
# val RMSE), then the model is refit on train and scored on test.
# Outputs per test year in final_syear/fold_{t}/inla_spline/: scores.csv,
# predictions.csv, best_params.json, tuning_grid.csv, inla_coverage.csv.
#
# Run from the repository root, one test year per R process (INLA can leave the
# R session unable to write files after the first year on Windows):
#   bash:       HOUSE_INCLUDE_SYEAR=1 INLA_FOLDS=1996 Rscript real_data/house/run_inla.R
#   PowerShell: $env:HOUSE_INCLUDE_SYEAR=1; $env:INLA_FOLDS=1996; Rscript real_data/house/run_inla.R

library(INLA)
library(jsonlite)

# The default inla.mode ("compact") can fail at result collection on Windows
inla.setOption(inla.mode = "classic")

REPO_ROOT <- normalizePath(file.path(getwd()), mustWork = FALSE)
while (!dir.exists(file.path(REPO_ROOT, "simulations")) && nchar(REPO_ROOT) > 3) {
  REPO_ROOT <- dirname(REPO_ROOT)
}
source(file.path(REPO_ROOT, "real_data", "house", "data_folds.R"))


# >> Settings

SPLINE_COLS <- c("age", "TLA", "beds", "baths", "halfbaths",
                 "frontage", "depth", "garagesqft", "rooms", "lotsize")
FACTOR_COLS <- c("stories", "wall", "garage")

# Sale year as a linear term (centred on 1993), on when HOUSE_INCLUDE_SYEAR is set;
# results then go to final_syear/. It carries the price level forward to the
# unseen test year.
INCLUDE_SYEAR <- nzchar(Sys.getenv("HOUSE_INCLUDE_SYEAR"))
SYEAR_ORIGIN  <- 1993

# Number of spline knots per covariate, set from the shape of log_price against
# each covariate: more where the effect bends (age, frontage, lotsize), fewer where
# it is close to linear or noisy. Knots are placed at quantiles (make_1d_mesh).
SPLINE_KNOTS <- c(age = 4, TLA = 3, beds = 3, baths = 3, halfbaths = 2,
                  frontage = 4, depth = 2, garagesqft = 3, rooms = 2, lotsize = 4)

# Heavy-tailed covariates: spline domain limited to the 1% - 99% quantiles, so
# that a single outlier does not set the outer knot (values beyond are flat).
CAP_COLS     <- c("frontage", "depth", "garagesqft", "lotsize")
SPLINE_CAP_Q <- c(0.01, 0.99)

# Spatial PC prior (range0, sigma0). TRUE = tune it on the validation year
# (9 combinations); FALSE = one fit with range0 = variogram range, sigma0 = sd(y).
GRID_SEARCH_INLA <- TRUE

# Prior grid: multipliers of the variogram range and of sd(y)
RANGE_MULTIPLIERS <- c(0.5, 1.0, 2.0)
SIGMA_MULTIPLIERS <- c(0.5, 1.0, 2.0)

N_VARIOGRAM_SAMPLE <- 300
N_VARIOGRAM_BINS   <- 15

OUT_BASE <- file.path(REPO_ROOT, "outputs", "results", "house_price")
if (INCLUDE_SYEAR) OUT_BASE <- file.path(OUT_BASE, "final_syear")

# Keep INLA's default temporary folder (a custom one can fail on Windows)
N_THREADS <- max(1, parallel::detectCores() - 1)


# >> Metrics

score_split_reg <- function(name, y_true, y_hat) {
  data.frame(
    split = name,
    n     = length(y_true),
    rmse  = round(sqrt(mean((y_true - y_hat)^2, na.rm = TRUE)), 4),
    mae   = round(mean(abs(y_true - y_hat), na.rm = TRUE), 4),
    r2    = round(1 - sum((y_true - y_hat)^2, na.rm = TRUE) /
                       sum((y_true - mean(y_true))^2, na.rm = TRUE), 4),
    stringsAsFactors = FALSE
  )
}


# >> Variogram range estimate (sets the mesh size and centres the prior grid)
# Uses a random subsample of N_VARIOGRAM_SAMPLE points (no fixed seed)

estimate_spatial_range <- function(coords, y,
                                    n_sample = N_VARIOGRAM_SAMPLE,
                                    n_bins   = N_VARIOGRAM_BINS) {
  idx  <- sample(nrow(coords), min(n_sample, nrow(coords)))
  locs <- coords[idx, , drop = FALSE]
  vals <- y[idx]

  D <- as.matrix(dist(locs))
  V <- outer(vals, vals, function(a, b) (a - b)^2 / 2)

  upper <- upper.tri(D)
  dists <- D[upper]
  svars <- V[upper]

  breaks  <- quantile(dists, probs = seq(0, 1, length.out = n_bins + 1))
  bin_id  <- cut(dists, breaks = breaks, include.lowest = TRUE)
  gamma   <- as.numeric(tapply(svars, bin_id, mean))
  centers <- as.numeric(tapply(dists, bin_id, mean))

  sill      <- max(gamma, na.rm = TRUE)
  above_idx <- which(gamma >= 0.95 * sill)
  range_est <- if (length(above_idx) > 0) centers[above_idx[1]] else centers[length(centers)]
  range_est
}


# >> Meshes

build_spatial_mesh <- function(coords, range_est) {
  max_edge_inner <- range_est / 3
  max_edge_outer <- range_est * 2
  cutoff         <- max_edge_inner / 5

  mesh <- inla.mesh.2d(
    loc      = coords,
    max.edge = c(max_edge_inner, max_edge_outer),
    cutoff   = cutoff,
    offset   = c(range_est * 0.5, range_est * 1.5)
  )
  cat(sprintf("  Spatial mesh nodes = %d  (max.edge inner=%.1f outer=%.1f)\n",
              mesh$n, max_edge_inner, max_edge_outer))
  mesh
}

make_1d_mesh <- function(x_train, n_knots, lo_q = 0, hi_q = 1) {
  # Knots at quantiles over [lo_q, hi_q] (cap for heavy-tailed covariates).
  probs <- seq(lo_q, hi_q, length.out = n_knots + 2)
  knots <- sort(unique(as.numeric(quantile(x_train, probs = probs, na.rm = TRUE))))
  if (length(knots) < 2) knots <- range(x_train, na.rm = TRUE)  # degenerate guard
  bounds <- c(knots[1], knots[length(knots)])
  inla.mesh.1d(knots, interval = bounds, degree = 2,
               boundary = c("neumann", "neumann"))
}


# >> Stack and formula (splines, fixed effects, spatial field)

build_stack <- function(y, coords, feat_df, mesh_spatial,
                        spde_list, mesh_list, tag) {
  n_obs <- nrow(feat_df)

  A_spatial <- inla.spde.make.A(mesh_spatial, loc = as.matrix(coords))
  # n.spde equals mesh$n for a stationary Matern SPDE
  spat_idx  <- inla.spde.make.index("spatial", n.spde = mesh_spatial$n)

  A_splines  <- lapply(SPLINE_COLS, function(col)
                  inla.spde.make.A(mesh_list[[col]], loc = feat_df[[col]]))
  spline_idx <- lapply(SPLINE_COLS, function(col)
                  inla.spde.make.index(col, n.spde = spde_list[[col]]$n.spde))

  intercept_and_factors <- c(
    list(intercept = rep(1, n_obs)),
    as.list(feat_df[, FACTOR_COLS, drop = FALSE])
  )
  if (INCLUDE_SYEAR) {
    intercept_and_factors$syear <- as.numeric(feat_df[["year"]]) - SYEAR_ORIGIN
  }

  A_list   <- c(list(A_spatial), A_splines, list(1))
  eff_list <- c(list(spat_idx), spline_idx, list(intercept_and_factors))

  inla.stack(data = list(y = y), A = A_list, effects = eff_list, tag = tag)
}

make_formula_spline <- function() {
  spline_terms <- paste(
    sprintf('f(%s, model = spde_list[["%s"]])', SPLINE_COLS, SPLINE_COLS),
    collapse = " + "
  )
  factor_terms <- paste(FACTOR_COLS, collapse = " + ")
  syear_term <- if (INCLUDE_SYEAR) "+ syear " else ""
  as.formula(paste(
    "y ~ -1 + intercept +", factor_terms, "+", spline_terms, syear_term,
    "+ f(spatial, model = spde_spatial)"
  ))
}


# >> Effects (spatial part b_hat, covariate part F_hat)

extract_spatial_effect <- function(fit, mesh_spatial, coords) {
  proj  <- inla.mesh.projector(mesh_spatial, loc = as.matrix(coords))
  b_hat <- inla.mesh.project(proj, fit$summary.random$spatial$mean)
  as.numeric(b_hat)
}

extract_covariate_effect <- function(fit, feat_df, mesh_list) {
  # Spline contributions (numeric covariates)
  F_hat <- rep(0, nrow(feat_df))
  for (col in SPLINE_COLS) {
    proj_1d <- inla.mesh.projector(mesh_list[[col]], loc = feat_df[[col]])
    contrib <- inla.mesh.project(proj_1d, fit$summary.random[[col]]$mean)
    F_hat   <- F_hat + as.numeric(contrib)
  }
  # Factor contributions (stories, wall, garage): model.matrix() gives the same
  # dummy columns as INLA, so names match the rows of fit$summary.fixed
  X_factors <- model.matrix(
    as.formula(paste("~", paste(FACTOR_COLS, collapse = " + "))),
    data = feat_df
  )
  X_factors <- X_factors[, colnames(X_factors) != "(Intercept)", drop = FALSE]
  matched   <- intersect(colnames(X_factors), rownames(fit$summary.fixed))
  if (length(matched) > 0) {
    betas <- fit$summary.fixed[matched, "mean"]
    F_hat <- F_hat + as.numeric(X_factors[, matched, drop = FALSE] %*% betas)
  }
  # Sale-year contribution (the predictions already include it)
  if (INCLUDE_SYEAR && "syear" %in% rownames(fit$summary.fixed)) {
    F_hat <- F_hat + (as.numeric(feat_df[["year"]]) - SYEAR_ORIGIN) *
             fit$summary.fixed["syear", "mean"]
  }
  F_hat
}

extract_predictions <- function(fit, join_stack, tag) {
  idx <- inla.stack.index(join_stack, tag)$data
  as.numeric(fit$summary.fitted.values$mean[idx])
}


# >> Model pieces (built once per fit set, reused for every prior candidate)

# Standardise the spline covariates on the fit set, build the spatial mesh and
# the 1D spline meshes, and assemble a joint stack; eval_df gets y = NA so it is
# predicted.
build_pieces <- function(fit_df, eval_df) {
  coords_fit  <- as.matrix(fit_df[,  COORD_COLS])
  coords_eval <- as.matrix(eval_df[, COORD_COLS])
  y_fit       <- fit_df[[TARGET_COL]]

  # Standardise on the fit set, same transform for eval
  means <- sapply(SPLINE_COLS, function(c) mean(fit_df[[c]], na.rm = TRUE))
  sds   <- sapply(SPLINE_COLS, function(c) sd(fit_df[[c]],   na.rm = TRUE))
  sds[sds == 0 | is.na(sds)] <- 1
  for (col in SPLINE_COLS) {
    fit_df[[col]]  <- (fit_df[[col]]  - means[col]) / sds[col]
    eval_df[[col]] <- (eval_df[[col]] - means[col]) / sds[col]
  }

  range_est <- estimate_spatial_range(coords_fit, y_fit)
  sigma_est <- sd(y_fit)
  cat(sprintf("  Variogram range_est=%.1f   sd(y)=%.4f\n", range_est, sigma_est))

  mesh_spatial <- build_spatial_mesh(coords_fit, range_est)
  mesh_list <- lapply(SPLINE_COLS, function(col) {
    q <- if (col %in% CAP_COLS) SPLINE_CAP_Q else c(0, 1)
    make_1d_mesh(fit_df[[col]], SPLINE_KNOTS[[col]], q[1], q[2])
  })
  names(mesh_list) <- SPLINE_COLS

  # PC priors of the 1D splines, on the standardised scale
  d <- 1; nu <- 1; alpha_ <- nu + d / 2
  spde_list <- lapply(SPLINE_COLS, function(col)
    inla.spde2.pcmatern(mesh_list[[col]], alpha = alpha_,
                        prior.range = c(1.0, 0.5),
                        prior.sigma = c(1.0, 0.05),
                        constr = TRUE))
  names(spde_list) <- SPLINE_COLS

  stack_fit  <- build_stack(y_fit, coords_fit, fit_df, mesh_spatial,
                            spde_list, mesh_list, tag = "fit")
  stack_eval <- build_stack(rep(NA_real_, nrow(eval_df)), coords_eval, eval_df,
                            mesh_spatial, spde_list, mesh_list, tag = "eval")
  join <- inla.stack(stack_fit, stack_eval)

  formula_obj <- make_formula_spline()
  environment(formula_obj) <- environment()
  environment(formula_obj)$spde_list <- spde_list

  list(join = join, formula = formula_obj, mesh_spatial = mesh_spatial,
       mesh_list = mesh_list, spde_list = spde_list,
       range_est = range_est, sigma_est = sigma_est,
       coords_fit = coords_fit, coords_eval = coords_eval,
       fit_df = fit_df, eval_df = eval_df, y_fit = y_fit)
}

# Fit INLA on `pieces` with the given spatial PC-prior median (range0, sigma0).
fit_prior <- function(pieces, range0, sigma0, verbose = FALSE) {
  environment(pieces$formula)$spde_spatial <- inla.spde2.pcmatern(
    mesh        = pieces$mesh_spatial, alpha = 2,
    prior.range = c(range0, 0.05),
    prior.sigma = c(sigma0, 0.05),
    constr      = TRUE
  )
  inla(
    pieces$formula, family = "gaussian",
    data              = inla.stack.data(pieces$join),
    control.predictor = list(A = inla.stack.A(pieces$join), compute = TRUE),
    # Only fitted values and random effects are used: no CPO, DIC, WAIC
    control.compute   = list(cpo = FALSE, dic = FALSE, waic = FALSE, config = FALSE),
    control.inla      = list(int.strategy = "eb", strategy = "gaussian"),
    verbose           = verbose,
    num.threads       = N_THREADS
  )
}


# >> One test year

run_fold <- function(fold, t, out_dir) {
  # INLA can leave R in a removed temporary folder; go back to the repository
  # root before writing files
  setwd(REPO_ROOT)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  cat(sprintf("\n[t=%d] train=%d  test=%d  inner_train=%d  val=%d\n",
              t, nrow(fold$train), nrow(fold$test),
              nrow(fold$inner_train), nrow(fold$val)))

  # Tune the spatial prior: fit on inner_train, RMSE on val. The selected
  # multipliers of range_est and sd(y) are then applied on train.
  best_mr <- 1.0; best_ms <- 1.0
  if (GRID_SEARCH_INLA) {
    tune   <- build_pieces(fold$inner_train, fold$val)
    y_val  <- fold$val[[TARGET_COL]]
    combos <- expand.grid(m_r = RANGE_MULTIPLIERS, m_s = SIGMA_MULTIPLIERS)
    cat(sprintf("  Prior grid: %d combos (fit inner_train, RMSE on val)\n",
                nrow(combos)))
    rows <- vector("list", nrow(combos)); best_rmse <- Inf
    for (i in seq_len(nrow(combos))) {
      mr <- combos$m_r[i]; ms <- combos$m_s[i]
      r0 <- tune$range_est * mr; s0 <- tune$sigma_est * ms
      rmse_i <- tryCatch({
        f  <- fit_prior(tune, r0, s0)
        pv <- extract_predictions(f, tune$join, "eval")
        sqrt(mean((y_val - pv)^2))
      }, error = function(e) {
        cat(sprintf("    [combo failed: %s]\n", conditionMessage(e))); NA_real_
      })
      cat(sprintf("  [%d/%d]  m_r=%.1f  m_s=%.1f  val_rmse=%.4f\n",
                  i, nrow(combos), mr, ms, rmse_i))
      rows[[i]] <- data.frame(m_r = mr, m_s = ms, range0 = r0, sigma0 = s0,
                              val_rmse = rmse_i)
      if (!is.na(rmse_i) && rmse_i < best_rmse) {
        best_rmse <- rmse_i; best_mr <- mr; best_ms <- ms
      }
    }
    setwd(REPO_ROOT); dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
    grid_df <- do.call(rbind, rows)
    grid_df <- grid_df[order(grid_df$val_rmse), ]
    write.csv(grid_df, file.path(out_dir, "tuning_grid.csv"), row.names = FALSE)
    cat(sprintf("  Best: m_r=%.1f  m_s=%.1f  val_rmse=%.4f\n",
                best_mr, best_ms, best_rmse))
  } else {
    cat("  Prior grid OFF -> range0=range_est, sigma0=sd(y) on train\n")
  }

  # Final model: refit on train with the selected prior, predict test
  pieces <- build_pieces(fold$train, fold$test)
  r0 <- pieces$range_est * best_mr
  s0 <- pieces$sigma_est * best_ms
  t0  <- proc.time()["elapsed"]
  fit <- fit_prior(pieces, r0, s0, verbose = TRUE)
  fit_time <- proc.time()["elapsed"] - t0
  cat(sprintf("  Final fit time: %.1fs  (range0=%.1f  sigma0=%.4f)\n",
              fit_time, r0, s0))

  setwd(REPO_ROOT); dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  # Predictions and effects (standardised covariates for the splines, raw
  # coordinates for the spatial field)
  y_train     <- pieces$y_fit
  y_train_hat <- extract_predictions(fit, pieces$join, "fit")
  y_test_hat  <- extract_predictions(fit, pieces$join, "eval")
  b_train_hat <- extract_spatial_effect(fit, pieces$mesh_spatial, pieces$coords_fit)
  b_test_hat  <- extract_spatial_effect(fit, pieces$mesh_spatial, pieces$coords_eval)
  F_train_hat <- extract_covariate_effect(fit, pieces$fit_df,  pieces$mesh_list)
  F_test_hat  <- extract_covariate_effect(fit, pieces$eval_df, pieces$mesh_list)

  scores_df <- rbind(
    score_split_reg("train", y_train, y_train_hat),
    score_split_reg("test",  fold$test[[TARGET_COL]], y_test_hat)
  )
  scores_df$fit_time <- round(fit_time, 2)
  write.csv(scores_df, file.path(out_dir, "scores.csv"), row.names = FALSE)
  cat("\n  Scores:\n"); print(scores_df)

  # Same columns as the Python methods
  pred_df <- rbind(
    data.frame(s1 = fold$train$s1, s2 = fold$train$s2, log_price = y_train,
               y_hat = y_train_hat, F_hat = F_train_hat, b_hat = b_train_hat,
               split = "train", stringsAsFactors = FALSE),
    data.frame(s1 = fold$test$s1, s2 = fold$test$s2,
               log_price = fold$test[[TARGET_COL]],
               y_hat = y_test_hat, F_hat = F_test_hat, b_hat = b_test_hat,
               split = "test", stringsAsFactors = FALSE)
  )
  write.csv(pred_df, file.path(out_dir, "predictions.csv"), row.names = FALSE)

  # Coverage of the 90% and 95% predictive intervals on test: posterior sd of
  # the linear predictor plus the Gaussian noise (variance 1 / precision)
  idx_eval <- inla.stack.index(pieces$join, "eval")$data
  sd_test  <- as.numeric(fit$summary.fitted.values$sd[idx_eval])
  prow     <- grep("Precision for the Gaussian", rownames(fit$summary.hyperpar))
  prec     <- fit$summary.hyperpar[prow[1], "mean"]
  noise_sd <- 1 / sqrt(prec)
  pred_sd  <- sqrt(sd_test^2 + 1 / prec)
  y_test   <- fold$test[[TARGET_COL]]
  err      <- abs(y_test - y_test_hat)
  cov_df <- data.frame(
    test_year = t,
    cov90 = mean(err <= 1.6448536 * pred_sd),
    cov95 = mean(err <= 1.9599640 * pred_sd),
    cov90_mean = mean(err <= 1.6448536 * sd_test),
    pred_sd_mean = mean(pred_sd), fitted_sd_mean = mean(sd_test),
    noise_sd = noise_sd, n_test = length(y_test))
  write.csv(cov_df, file.path(out_dir, "inla_coverage.csv"), row.names = FALSE)
  cat(sprintf("  INLA held-out coverage: cov90=%.3f cov95=%.3f (pred_sd=%.3f noise_sd=%.3f)\n",
              cov_df$cov90, cov_df$cov95, cov_df$pred_sd_mean, noise_sd))

  best_params <- list(m_r = best_mr, m_s = best_ms, range0 = r0, sigma0 = s0,
                      spline_knots = as.list(SPLINE_KNOTS),
                      n_mesh_nodes = pieces$mesh_spatial$n)
  write_json(best_params, file.path(out_dir, "best_params.json"),
             pretty = TRUE, auto_unbox = TRUE)

  list(year = t, method = "inla_spline",
       rmse_test = scores_df$rmse[scores_df$split == "test"],
       mae_test  = scores_df$mae[scores_df$split == "test"],
       r2_test   = scores_df$r2[scores_df$split == "test"],
       fit_time  = round(fit_time, 2))
}


# >> Main

main <- function() {
  # INLA_FOLDS: comma-separated test years to fit (default: all). On Windows,
  # run one year per R process (see the header). grand_summary_inla.csv is
  # merged across runs.
  folds_env    <- Sys.getenv("INLA_FOLDS")
  folds_to_run <- if (nzchar(folds_env)) {
    as.integer(strsplit(folds_env, ",")[[1]])
  } else TEST_YEARS

  cat(strrep("=", 60), "\n")
  cat("  House Price (Lucas County) - R-INLA (spline variant)\n")
  cat(sprintf("  Folds this run: %s\n", paste(folds_to_run, collapse = ", ")))
  cat(sprintf("  Prior grid search (GRID_SEARCH_INLA): %s\n", GRID_SEARCH_INLA))
  cat(strrep("=", 60), "\n")

  df <- load_house_sigrist()

  all_summaries <- list()
  for (t in folds_to_run) {
    fold  <- get_fold(df, t)
    out_t <- file.path(OUT_BASE, sprintf("fold_%d", t), "inla_spline")
    result <- tryCatch(run_fold(fold, t, out_t), error = function(e) {
      cat(sprintf("  FAILED (t=%d): %s\n", t, conditionMessage(e)))
      NULL
    })
    if (!is.null(result)) all_summaries[[length(all_summaries) + 1]] <- result
  }

  if (length(all_summaries) > 0) {
    setwd(REPO_ROOT)
    grand_df   <- do.call(rbind, lapply(all_summaries, as.data.frame))
    grand_path <- file.path(OUT_BASE, "grand_summary_inla.csv")
    # Keep the years already on disk from earlier runs
    if (file.exists(grand_path)) {
      prev <- read.csv(grand_path, stringsAsFactors = FALSE)
      prev <- prev[!(prev$year %in% grand_df$year), , drop = FALSE]
      grand_df <- rbind(prev, grand_df)
    }
    grand_df <- grand_df[order(grand_df$year), ]
    cat("\n", strrep("=", 60), "\n")
    cat("  GRAND SUMMARY (test set)\n")
    cat(strrep("=", 60), "\n")
    print(grand_df)
    write.csv(grand_df, grand_path, row.names = FALSE)
    cat(sprintf("\n  -> %s\n", grand_path))
  }
}

if (!interactive()) {
  main()
}