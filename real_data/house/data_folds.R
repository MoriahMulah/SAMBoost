# data_folds.R
# Loads house_sigrist.csv and defines the expanding-window folds, identical to
# data_folds.py, for INLA (run_inla.R). For each test year t in {1996, 1997, 1998}:
#   train       : year <  t
#   test        : year == t
#   inner_train : year <  t - 1
#   val         : year == t - 1

REPO_ROOT <- normalizePath(file.path(getwd()), mustWork = FALSE)
while (!dir.exists(file.path(REPO_ROOT, "simulations")) && nchar(REPO_ROOT) > 3) {
  REPO_ROOT <- dirname(REPO_ROOT)
}

DATA_DIR <- file.path(REPO_ROOT, "outputs", "data", "house")

COORD_COLS       <- c("s1", "s2")
TARGET_COL       <- "log_price"
NON_FEAT_COLS    <- c("log_price", "price", "year", "s1", "s2")
CATEGORICAL_COLS <- c("stories", "wall", "garage")

TEST_YEARS <- c(1996, 1997, 1998)


load_house_sigrist <- function() {
  path <- file.path(DATA_DIR, "house_sigrist.csv")
  df   <- read.csv(path, stringsAsFactors = FALSE)
  for (col in CATEGORICAL_COLS) {
    df[[col]] <- as.factor(df[[col]])
  }
  cat(sprintf("  Loaded house_sigrist.csv: %d rows, years %d-%d\n",
              nrow(df), min(df$year), max(df$year)))
  df
}


get_fold <- function(df, t) {
  if (!(t %in% TEST_YEARS)) {
    stop(sprintf("t must be one of %s, got %d",
                 paste(TEST_YEARS, collapse = ", "), t))
  }
  list(
    train       = df[df$year <  t,     , drop = FALSE],
    test        = df[df$year == t,     , drop = FALSE],
    inner_train = df[df$year <  t - 1, , drop = FALSE],
    val         = df[df$year == t - 1, , drop = FALSE]
  )
}


if (sys.nframe() == 0) {
  df <- load_house_sigrist()
  feat_cols <- setdiff(names(df), NON_FEAT_COLS)
  cat(sprintf("Feature cols: %s\n", paste(feat_cols, collapse = ", ")))
  for (t in TEST_YEARS) {
    fold <- get_fold(df, t)
    cat(sprintf(
      "  t=%d: train=%d  test=%d  inner_train=%d  val=%d\n",
      t, nrow(fold$train), nrow(fold$test),
      nrow(fold$inner_train), nrow(fold$val)
    ))
  }
}
