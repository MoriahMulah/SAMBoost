# bench_inla.R
# Fit-time scaling study for INLA: fit time as a function of N on the datasets
# saved by run_scaling.py. Hybrid model (river linear, x1..x5 as 1D-SPDE splines,
# 2D SPDE field), fixed 2D mesh on the 40x40 reference grid, fixed 1D spline meshes,
# fixed PC priors, empirical Bayes integration, no CPO / DIC / prior grid.
# Output: results_inla.csv. Run from the repository root:
#   Rscript validation/scaling/bench_inla.R
suppressMessages(library(INLA))
# Change the path if needed.
LOCAL_TMP <- "C:/inla_tmp"; unlink(file.path(LOCAL_TMP,"*"), recursive=TRUE, force=TRUE)
dir.create(LOCAL_TMP, showWarnings=FALSE, recursive=TRUE); inla.setOption("working.directory", LOCAL_TMP)

HERE   <- normalizePath(file.path(getwd(), "validation", "scaling"))
DATA   <- file.path(HERE, "data"); OUTCSV <- file.path(HERE, "results_inla.csv")
NS     <- c(500, 1000, 2000, 4000, 8000, 16000, 32000, 64000)
SEEDS  <- c(1, 2, 3)
SPLINE_COLS <- c("x1","x2","x3","x4","x5")   # continuous -> splines
LINEAR_COLS <- c("x_hotspot")                # river -> linear

# Fixed 2D mesh on the 40x40 reference grid
gg <- seq(0,1,length.out=40); ref <- as.matrix(expand.grid(gg, gg))
mesh_spatial <- inla.mesh.2d(loc=ref, max.edge=c(0.05,0.15), cutoff=0.01)
spde_spatial <- inla.spde2.pcmatern(mesh_spatial, prior.range=c(0.3,0.5), prior.sigma=c(1,0.5))
spat_idx <- inla.spde.make.index("spatial", spde_spatial$n.spde)

# Fixed 1D spline meshes and SPDEs (fixed knots, x ~ N(0,1))
knots1d <- c(-2.5,-1,0,1,2.5)
mesh_list <- setNames(lapply(SPLINE_COLS, function(j)
  inla.mesh.1d(knots1d, degree=2, boundary=c("neumann","neumann"))), SPLINE_COLS)
spde_list <- setNames(lapply(SPLINE_COLS, function(j)
  inla.spde2.pcmatern(mesh_list[[j]], prior.range=c(1,0.5), prior.sigma=c(1,0.5))), SPLINE_COLS)
cat(sprintf("FIXED: 2D mesh %d nodes + 5 spline meshes; hybrid config\n", mesh_spatial$n))

spline_terms <- paste(sprintf('f(%s, model=spde_list[["%s"]])', SPLINE_COLS, SPLINE_COLS), collapse=" + ")
FORM <- as.formula(paste("y ~ -1 + intercept +", paste(LINEAR_COLS, collapse=" + "),
                         "+", spline_terms, "+ f(spatial, model=spde_spatial)"))

append_row <- function(N, seed, fit_s) {
  row <- data.frame(N=N, seed=seed, mesh_nodes=mesh_spatial$n, method="INLA_spde", fit_s=fit_s)
  write.table(row, OUTCSV, sep=",", row.names=FALSE, col.names=!file.exists(OUTCSV), append=file.exists(OUTCSV))
  cat(sprintf("   N=%d seed=%d fit=%.1fs\n", N, seed, fit_s))
}

bench_one <- function(N, seed) {
  fp <- file.path(DATA, sprintf("N%d_s%d", N, seed), "train_df.csv")
  if (!file.exists(fp)) { cat("   [skip] no data\n"); return(invisible()) }
  df <- read.csv(fp); coords <- as.matrix(df[,c("s1","s2")]); y <- df$y; n <- nrow(df)
  A_spatial <- inla.spde.make.A(mesh_spatial, loc=coords)
  A_splines <- lapply(SPLINE_COLS, function(j) inla.spde.make.A(mesh_list[[j]], loc=df[[j]]))
  spline_idx <- lapply(SPLINE_COLS, function(j) inla.spde.make.index(j, spde_list[[j]]$n.spde))
  eff_fixed <- c(list(intercept=rep(1,n)), as.list(df[, LINEAR_COLS, drop=FALSE]))
  stk <- inla.stack(data=list(y=y), A=c(list(A_spatial), A_splines, list(1)),
                    effects=c(list(spat_idx), spline_idx, list(eff_fixed)), tag="est")
  t0 <- Sys.time()
  fit <- try(inla(FORM, family="gaussian", data=inla.stack.data(stk),
                  control.predictor=list(A=inla.stack.A(stk), compute=TRUE),
                  control.compute=list(config=TRUE),
                  control.inla=list(int.strategy="eb"),
                  working.directory=LOCAL_TMP, verbose=TRUE), silent=TRUE)
  fit_s <- as.numeric(difftime(Sys.time(), t0, units="secs"))
  if (inherits(fit,"try-error")) { fit_s <- NA; cat("   FAIL:", conditionMessage(attr(fit,"condition")), "\n") }
  append_row(N, seed, fit_s)
}

for (N in NS) for (s in SEEDS) { cat(sprintf("=== INLA N=%d seed=%d ===\n", N, s)); bench_one(N, s) }
cat("DONE ->", OUTCSV, "\n")
