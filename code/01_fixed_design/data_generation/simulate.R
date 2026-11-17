# Fixed-rule design (Table 1, Panel A; Figure 3(a), fixed-rule bars): the Monte Carlo.
#
# For each original dataset (case 1..400): six independent N(0, 1) outcomes on n = 1,000
# observations; two-sided known-variance z-tests of zero means; a selector that reports the
# smallest p-value with probability 0.8 and a uniformly chosen one otherwise; a reference bank of
# J = 50,000 subsamples of m = 31 observations scored on the selected test; random orderings of
# the bank's tail (p <= u) and body (p > u) members; the selector replayed on every ordered
# subsample; and, where the bank holds fewer than 1,000 tail members, further uniform subsamples
# drawn in chunks of 10,000 and kept when they fall in the tail until the tail has 1,000. Replay
# budgets of 25, 100, 400 and 1,000 per stratum are nested prefixes of the same orderings.
# Intervals come from code/lib/replay.R: two-sided 95% (gamma = 0.05) and the one-sided 95% upper
# bound (upper end of the two-sided 90% interval, gamma = 0.10).
#
# Seeds reproduce the run of record (Great Lakes, 22 October 2026): base seeds 2026102200 +
# offset + case for the data, selection, bank, orderings and replays, and 2026172200,
# 2026182200 + case for the tail extension. R's default RNG (Mersenne-Twister, Inversion,
# Rejection).
#
# Input:  code/lib/replay.R
# Output: CSV with one row per case and budget: case_id, b, the design (n, m, k, epsilon), q, u,
#         x1, b1, x0, b0, lower95, upper95, upper_one95
#         (data/intermediate/01_fixed_design/replay_draws.csv for cases 1..400; see its CODEBOOK.md)
# Usage:  Rscript simulate.R FIRST_CASE LAST_CASE OUT.csv [CORES]
# Runtime: about 5,800 CPU-seconds for all 400 cases (under an hour on 2 cores), almost all of it
# the intervals.

args <- commandArgs(TRUE)
stopifnot(length(args) %in% 3:4)
first <- as.integer(args[1]); last <- as.integer(args[2]); out <- args[3]
cores <- if (length(args) == 4) as.integer(args[4]) else 2L
stopifnot(1L <= first, first <= last, last <= 400L)
script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
script <- gsub("~+~", " ", script, fixed = TRUE)  # Rscript encodes spaces in the path this way
root <- normalizePath(file.path(dirname(script), "..", "..", ".."))
source(file.path(root, "code", "lib", "replay.R"))

n <- 1000L; m <- 31L; k <- 6L; epsilon <- .2; J <- 50000L; B <- c(25L, 100L, 400L, 1000L)
RNGkind("Mersenne-Twister", "Inversion", "Rejection")

# The selector: a uniformly chosen test with probability epsilon, otherwise the smallest p-value.
select_q <- function(p) {
  if (runif(1) < epsilon) sample.int(length(p), 1L) else which.min(p)
}

# z-test p-values of all six outcomes on the subsamples in the columns of `indices`.
bank_scores <- function(data, indices) {
  mm <- nrow(indices)
  vapply(seq_len(ncol(data)), function(j)
    2 * pnorm(-abs(colSums(matrix(data[as.vector(indices), j], nrow = mm)) / sqrt(mm))),
    numeric(ncol(indices)))
}

# Further tail subsamples for a bank with fewer than max(B) tail members.
extend_tail <- function(d, q, u, need, case) {
  set.seed(2026172200L + case)
  kept <- matrix(numeric(), nrow = 0, ncol = k)
  while (nrow(kept) < need) {
    idx <- vapply(seq_len(10000L), function(j) sample.int(n, m), integer(m))
    pq <- 2 * pnorm(-abs(colSums(matrix(d[as.vector(idx), q], nrow = m)) / sqrt(m)))
    take <- head(which(pq <= u), need - nrow(kept))
    if (length(take))
      kept <- rbind(kept, matrix(bank_scores(d, idx[, take, drop = FALSE]), nrow = length(take), ncol = k))
  }
  set.seed(2026182200L + case)
  vapply(seq_len(need), function(i) select_q(kept[i, ]), integer(1))
}

simulate_case <- function(case) {
  base <- 2026102200L
  set.seed(base + case); d <- matrix(rnorm(n * k), ncol = k)
  pfull <- 2 * pnorm(-abs(colMeans(d) * sqrt(n)))
  set.seed(base + 10000L + case); q <- select_q(pfull); u <- pfull[q]
  set.seed(base + 20000L + case); idx <- vapply(seq_len(J), function(j) sample.int(n, m), integer(m))
  pbank <- bank_scores(d, idx)
  tail <- which(pbank[, q] <= u); body <- which(pbank[, q] > u)
  stopifnot(length(body) >= max(B))
  set.seed(base + 30000L + case)
  ti <- if (length(tail)) tail[sample.int(length(tail), min(max(B), length(tail)))] else integer()
  set.seed(base + 40000L + case); bi <- body[sample.int(length(body), max(B))]
  set.seed(base + 50000L + case); qt <- vapply(ti, function(j) select_q(pbank[j, ]), integer(1))
  set.seed(base + 60000L + case); qb <- vapply(bi, function(j) select_q(pbank[j, ]), integer(1))
  if (length(qt) < max(B)) qt <- c(qt, extend_tail(d, q, u, max(B) - length(qt), case))
  do.call(rbind, lapply(B, function(b) {
    x1 <- sum(qt[seq_len(b)] == q); x0 <- sum(qb[seq_len(b)] == q)
    two <- replay_interval_lr(u, x1, b, x0, b, gamma = .05)$p_interval
    one <- replay_interval_lr(u, x1, b, x0, b, gamma = .10)$p_interval
    data.frame(case_id = case, b = b, n = n, m = m, k = k, epsilon = epsilon, q = q, u = u,
               x1 = x1, b1 = b, x0 = x0, b0 = b,
               lower95 = two[["lower"]], upper95 = two[["upper"]], upper_one95 = one[["upper"]])
  }))
}

rows <- do.call(rbind, parallel::mclapply(first:last, simulate_case, mc.cores = cores))
num <- c("u", "lower95", "upper95", "upper_one95")
rows[num] <- lapply(rows[num], sprintf, fmt = "%.17g")
write.csv(rows, out, row.names = FALSE, quote = FALSE)
