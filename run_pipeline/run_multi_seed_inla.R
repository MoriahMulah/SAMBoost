# run_multi_seed_inla.R
# Runs the INLA part of the simulation study over several seeds. For each seed:
# generate the data with the Python simulation generator, run the INLA runner
# (experiments/inla/run_<sim>.R) and collect the test metrics.
# Output: outputs/results/multi_seed/<sim>/run_<timestamp>/ with
# all_seeds_raw_inla.csv and summary_inla.csv.
#
# Usage (from the repository root, after setting the switches below):
#   Rscript run_pipeline/run_multi_seed_inla.R

library(jsonlite)

# >> Settings

# Which simulations to run
RUN_SIM1                <- FALSE  # Simulation 1 (classification, no confounding)
RUN_SIM2        <- FALSE  # Simulation 2 (regression, confounded river covariate)
RUN_SIM2_ABLATION <- FALSE  # ablation (Simulation 2 with a nearly additive covariate effect)

# Seeds
SEEDS <- 1:30

# Tuning mode: "first_seed" (CPO grid on seed 1, priors reused for seeds 2-30)
# or "per_seed" (CPO grid on every seed). first_seed reads the seed-1 priors of the
# spline variant: with the hybrid variant only (Simulation 2, ablation), every seed
# is tuned.
TUNING_MODE <- "first_seed"

# Python executable. Change the path if needed.
PYTHON_EXE <- "C:/Users/mulah.moriah/AppData/Local/anaconda3/python.exe"

# Repo root (auto-detect from working directory)
REPO_ROOT <- normalizePath(file.path(getwd()), mustWork = FALSE)
while (!dir.exists(file.path(REPO_ROOT, "simulations")) && nchar(REPO_ROOT) > 3) {
  REPO_ROOT <- dirname(REPO_ROOT)
}

MULTI_SEED_DIR <- file.path(REPO_ROOT, "outputs", "results", "multi_seed")

# >> Simulations

SIM_CONFIG <- list(
  sim1_wave = list(
    gen_script = file.path(REPO_ROOT, "simulations", "generate_sim1_wave.py"),
    data_dir   = file.path(REPO_ROOT, "outputs", "data", "sim1_wave"),
    runner     = file.path(REPO_ROOT, "experiments", "inla", "run_sim1_wave.R"),
    sim_name   = "sim1_wave"
  ),
  sim2_hotspot_river = list(
    gen_script = file.path(REPO_ROOT, "simulations", "generate_sim2_hotspot_river.py"),
    data_dir   = file.path(REPO_ROOT, "outputs", "data", "sim2_hotspot_river_inla"),  # separate from the Python driver, both can run at once
    runner     = file.path(REPO_ROOT, "experiments", "inla", "run_sim2_hotspot_river.R"),
    sim_name   = "sim2_hotspot_river"
  ),
  sim2_ablation_simpleF = list(
    gen_script = file.path(REPO_ROOT, "simulations", "generate_sim2_ablation_simpleF.py"),
    data_dir   = file.path(REPO_ROOT, "outputs", "data", "sim2_ablation_simpleF_inla"),  # separate from the Python driver
    runner     = file.path(REPO_ROOT, "experiments", "inla", "run_sim2_ablation_simpleF.R"),
    sim_name   = "sim2_ablation_simpleF"
  )
)

# >> Helpers

load_metrics <- function(folder) {
  out <- list()

  scores_path <- file.path(folder, "scores.csv")
  if (file.exists(scores_path)) {
    scores <- read.csv(scores_path, stringsAsFactors = FALSE)
    test_row <- scores[scores$split == "test", ]
    if (nrow(test_row) > 0) {
      # classification + regression metrics (whichever the runner wrote)
      for (col in c("auc", "log_loss", "brier", "rmse", "r2", "mae")) {
        if (col %in% names(test_row)) out[[col]] <- test_row[[col]][1]
      }
    }
  }

  dissoc_path <- file.path(folder, "dissociation.csv")
  if (file.exists(dissoc_path)) {
    dissoc <- read.csv(dissoc_path, stringsAsFactors = FALSE)
    test_row <- dissoc[dissoc$split == "test", ]
    if (nrow(test_row) > 0) {
      for (col in c("r_spatial", "rmse_spatial", "r_cov", "rmse_cov")) {
        if (col %in% names(test_row)) out[[col]] <- test_row[[col]][1]
      }
    }
    train_row <- dissoc[dissoc$split == "train", ]
    if (nrow(train_row) > 0 && "fit_time" %in% names(train_row)) {
      out[["fit_time"]] <- train_row[["fit_time"]][1]
    }
  }

  out
}

# >> Main loop

run_sim_multi_seed <- function(sim_key) {
  config   <- SIM_CONFIG[[sim_key]]
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  out_base <- file.path(MULTI_SEED_DIR, sim_key, paste0("run_", timestamp))
  dir.create(out_base, showWarnings = FALSE, recursive = TRUE)

  cat(sprintf("\n%s\n  INLA Multi-Seed: %s (%d seeds)\n%s\n",
              strrep("=", 60), sim_key, length(SEEDS), strrep("=", 60)))

  # Source the runner (defines run_sim function)
  source(config$runner, local = TRUE)

  all_rows <- list()
  fixed_priors <- NULL  # will be set after seed 1 if TUNING_MODE == "first_seed"

  for (i in seq_along(SEEDS)) {
    seed <- SEEDS[i]
    cat(sprintf("\n%s\n  Seed %d  (%d/%d)\n%s\n",
                strrep("-", 60), seed, i, length(SEEDS), strrep("-", 60)))

    # 1. Regenerate data using Python
    cmd <- sprintf('"%s" "%s" --seed %d --output_dir "%s"', PYTHON_EXE, config$gen_script, seed, config$data_dir)
    cat(sprintf("  [data] %s\n", cmd))
    system(cmd, ignore.stdout = TRUE, ignore.stderr = TRUE)

    # 2. Run INLA (the variants set in the runner)
    seed_out <- file.path(out_base, sprintf("seed_%04d", seed), config$sim_name)
    dir.create(seed_out, showWarnings = FALSE, recursive = TRUE)

    # Pass fixed_priors (NULL on seed 1, set after seed 1)
    priors_to_use <- if (TUNING_MODE == "first_seed" && !is.null(fixed_priors)) {
      fixed_priors
    } else {
      NULL
    }

    seed_t0 <- proc.time()
    tryCatch({
      cat("  [INLA] Running ...\n")
      summaries <- run_sim(out_dir = seed_out, fixed_priors = priors_to_use)

      seed_time <- (proc.time() - seed_t0)["elapsed"]

      # Collect metrics from spline variant (primary)
      spline_dir <- file.path(seed_out, "spline")
      if (dir.exists(spline_dir)) {
        m <- load_metrics(spline_dir)
        m$method <- "INLA_spline"
        m$seed   <- seed
        m$total_run_time <- round(seed_time, 2)
        all_rows[[length(all_rows) + 1]] <- m
        cat(sprintf("  [INLA spline] AUC=%.4f  r_sp=%.4f  time=%.1fs\n",
                    m$auc %||% NA, m$r_spatial %||% NA, seed_time))
      }

      # Also collect linear variant
      linear_dir <- file.path(seed_out, "linear")
      if (dir.exists(linear_dir)) {
        m <- load_metrics(linear_dir)
        m$method <- "INLA_linear"
        m$seed   <- seed
        m$total_run_time <- round(seed_time, 2)
        all_rows[[length(all_rows) + 1]] <- m
        cat(sprintf("  [INLA linear] AUC=%.4f  r_sp=%.4f\n",
                    m$auc %||% NA, m$r_spatial %||% NA))
      }

      # Also collect hybrid variant (linear binary + spline continuous), if present
      hybrid_dir <- file.path(seed_out, "hybrid")
      if (dir.exists(hybrid_dir)) {
        m <- load_metrics(hybrid_dir)
        m$method <- "INLA_hybrid"
        m$seed   <- seed
        m$total_run_time <- round(seed_time, 2)
        all_rows[[length(all_rows) + 1]] <- m
        cat(sprintf("  [INLA hybrid] r_sp=%.4f  rmse_cov=%.4f\n",
                    m$r_spatial %||% NA, m$rmse_cov %||% NA))
      }

      # After seed 1: extract best priors for first_seed mode
      if (i == 1 && TUNING_MODE == "first_seed") {
        # Read best_params.json from spline variant (primary)
        bp_path <- file.path(spline_dir, "best_params.json")
        if (file.exists(bp_path)) {
          bp <- fromJSON(bp_path)
          fixed_priors <- list(range0 = bp$range0, sigma0 = bp$sigma0)
          cat(sprintf("  [tuning] First-seed priors: range0=%.4f, sigma0=%.4f\n",
                      fixed_priors$range0, fixed_priors$sigma0))
          cat("  [tuning] Using these priors for all remaining seeds.\n")
        }
      }
    }, error = function(e) {
      cat(sprintf("  [INLA] FAILED: %s\n", e$message))
    })
  }

  # Restore default seed
  cat("\n  Restoring default seed=42 data ...\n")
  cmd <- sprintf('"%s" "%s" --seed 42 --output_dir "%s"', PYTHON_EXE, config$gen_script, config$data_dir)
  system(cmd, ignore.stdout = TRUE, ignore.stderr = TRUE)

  if (length(all_rows) == 0) {
    cat("  No results collected.\n")
    return(NULL)
  }

  # Build results dataframe
  df <- do.call(rbind, lapply(all_rows, function(row) {
    as.data.frame(row, stringsAsFactors = FALSE)
  }))
  df$simulation <- sim_key

  write.csv(df, file.path(out_base, "all_seeds_raw_inla.csv"), row.names = FALSE)

  # Summary
  cat(sprintf("\n%s\n  SUMMARY - %s (%d seeds)\n%s\n",
              strrep("=", 60), sim_key, length(SEEDS), strrep("=", 60)))

  metric_cols <- c("auc", "log_loss", "brier", "r_spatial", "rmse_spatial",
                   "r_cov", "rmse_cov", "fit_time")
  metric_cols <- intersect(metric_cols, names(df))

  for (method in unique(df$method)) {
    sub <- df[df$method == method, ]
    cat(sprintf("  %s (n=%d):\n", method, nrow(sub)))
    for (col in metric_cols) {
      vals <- sub[[col]]
      vals <- vals[!is.na(vals)]
      if (length(vals) > 0) {
        cat(sprintf("    %s: %.4f +/- %.4f\n", col, mean(vals), sd(vals)))
      }
    }
  }

  # Save summary
  summary_rows <- list()
  for (method in unique(df$method)) {
    sub <- df[df$method == method, ]
    row <- list(Method = method, n_seeds = nrow(sub))
    for (col in metric_cols) {
      vals <- sub[[col]][!is.na(sub[[col]])]
      if (length(vals) > 0) {
        row[[paste0(col, "_mean")]] <- round(mean(vals), 4)
        row[[paste0(col, "_std")]]  <- round(sd(vals), 4)
      }
    }
    summary_rows[[length(summary_rows) + 1]] <- as.data.frame(row, stringsAsFactors = FALSE)
  }
  summary_df <- do.call(rbind, summary_rows)
  write.csv(summary_df, file.path(out_base, "summary_inla.csv"), row.names = FALSE)

  cat(sprintf("\n  Raw -> %s\n", file.path(out_base, "all_seeds_raw_inla.csv")))
  cat(sprintf("  Summary -> %s\n", file.path(out_base, "summary_inla.csv")))

  df
}

# >> Run

main <- function() {
  t0 <- proc.time()

  sims <- c()
  if (RUN_SIM1)                sims <- c(sims, "sim1_wave")
  if (RUN_SIM2)        sims <- c(sims, "sim2_hotspot_river")
  if (RUN_SIM2_ABLATION) sims <- c(sims, "sim2_ablation_simpleF")

  cat(sprintf("%s\n  INLA Multi-Seed Runner\n  Seeds: %d\n  Simulations: %s\n%s\n",
              strrep("=", 60), length(SEEDS), paste(sims, collapse = ", "),
              strrep("=", 60)))

  for (sim_key in sims) {
    run_sim_multi_seed(sim_key)
  }

  elapsed <- (proc.time() - t0)["elapsed"]
  cat(sprintf("\n  Total time: %.0fs (%.1fmin)\n", elapsed, elapsed / 60))
}

main()