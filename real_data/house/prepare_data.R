#!/usr/bin/env Rscript
# prepare_data.R
# Builds the house-price dataset from spData::house (Lucas County, Ohio: 25,357
# single-family home sales, 1993-1998), as used by Sigrist (2022).
# Outputs, in outputs/data/house/:
#   house_baseline.csv : all spData columns, coordinates as s1, s2, plus log_price
#   house_sigrist.csv  : the variables used by the methods (see below)
# Run from the repository root:
#   Rscript real_data/house/prepare_data.R

suppressMessages({
  library(spData)
  library(sf)
})

data(house)
h_sf <- st_as_sf(house)

cat("CRS: ", st_crs(h_sf)$input, "\n")
cat("N rows: ", nrow(h_sf), "\n\n")

coords <- st_coordinates(h_sf)
df <- st_drop_geometry(h_sf)
df$s1 <- coords[, 1]
df$s2 <- coords[, 2]
df$log_price <- log(df$price)

cat("Column summary:\n")
print(str(df))

cat("\nNA counts (columns with NA only):\n")
na_counts <- sapply(df, function(x) sum(is.na(x)))
print(na_counts[na_counts > 0])

cat("\nprice summary:\n")
print(summary(df$price))

cat("\nCoordinate ranges:\n")
cat("s1:", range(df$s1), "\n")
cat("s2:", range(df$s2), "\n")

out_dir <- file.path("outputs", "data", "house")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

baseline_path <- file.path(out_dir, "house_baseline.csv")
write.csv(df, baseline_path, row.names = FALSE)
cat("\nSaved baseline ->", baseline_path, "\n")

# >> Variables of Sigrist (2022)
# Response: log(price). Predictors: age, stories, TLA, wall, beds, baths,
# halfbaths, frontage, depth, garage, garagesqft, rooms, lotsize, year (sale
# year), s1, s2. The transformed columns age2, age3, log_TLA and log_lotsize
# are kept in the file but not used by the methods.
sigrist <- data.frame(
  log_price  = df$log_price,
  price      = df$price,
  year       = as.integer(as.character(df$syear)),
  age        = df$age,
  age2       = df$age^2,
  age3       = df$age^3,
  stories    = as.character(df$stories),
  TLA        = df$TLA,
  log_TLA    = log(df$TLA),
  wall       = as.character(df$wall),
  beds       = df$beds,
  baths      = df$baths,
  halfbaths  = df$halfbaths,
  frontage   = df$frontage,
  depth      = df$depth,
  garage     = as.character(df$garage),
  garagesqft = df$garagesqft,
  rooms      = df$rooms,
  lotsize    = df$lotsize,
  log_lotsize= log(df$lotsize),
  s1         = df$s1,
  s2         = df$s2
)

cat("\nyear distribution (sigrist version):\n")
print(table(sigrist$year))

sigrist_path <- file.path(out_dir, "house_sigrist.csv")
write.csv(sigrist, sigrist_path, row.names = FALSE)
cat("\nSaved sigrist version ->", sigrist_path, "\n")
