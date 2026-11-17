# Corrected p-value, multiplier and replay intervals for each original at 40 + 200 replays and at 240 + 400.
#
# Inputs:  data/agent_runs/open_ended_luna/originals.csv, data/derived/02_open_ended/replay_counts.csv
# Output:  data/derived/02_open_ended/per_original.csv

script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
script <- gsub("~+~", " ", script, fixed = TRUE)  # Rscript writes spaces in the path as ~+~
root <- normalizePath(file.path(dirname(script), "..", ".."))
source(file.path(root, "code", "lib", "replay.R"))
cohort <- file.path(root, "data", "agent_runs", "open_ended_luna")
out <- file.path(root, "data", "derived", "02_open_ended")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

originals <- read.csv(file.path(cohort, "originals.csv"), encoding = "UTF-8")
originals <- originals[originals$usable, c("slot", "u")]
counts <- read.csv(file.path(out, "replay_counts.csv"))
base <- counts[counts$component == "base", ]
ext1 <- counts[counts$component == "ext1", ]
stopifnot(identical(base$slot, originals$slot), identical(ext1$slot, originals$slot))
sum_counts <- function(a, b) { for (v in c("attempted", "x1", "b1", "x0", "b0")) a[[v]] <- a[[v]] + b[[v]]; a }
budgets <- list(base = base, pooled = sum_counts(base, ext1))

rows <- do.call(rbind, lapply(names(budgets), function(budget) {
  d <- merge(originals, budgets[[budget]], by = "slot")
  do.call(rbind, lapply(seq_len(nrow(d)), function(i) {
    r <- d[i, ]
    est <- replay_estimate(r$u, r$x1, r$b1, r$x0, r$b0)
    two <- replay_interval_lr(r$u, r$x1, r$b1, r$x0, r$b0, gamma = .05)$p_interval
    one <- replay_interval_lr(r$u, r$x1, r$b1, r$x0, r$b0, gamma = .10)$p_interval
    # With a body match and no tail match the point estimate is 0 and is not reported; only the bound is used.
    status <- if (est$status == "unavailable") "unavailable" else if (r$x1 == 0) "body_only" else "estimated"
    p <- if (status == "body_only") NA else est$p_value
    data.frame(budget = budget, slot = r$slot, u = r$u, attempted = r$attempted, x1 = r$x1, b1 = r$b1,
               x0 = r$x0, b0 = r$b0, status = status, p_corrected = p,
               multiplier = if (status == "estimated") p / r$u else NA,
               lower95 = two[["lower"]], upper95 = two[["upper"]], upper_one95 = one[["upper"]])
  }))
}))
num <- c("u", "p_corrected", "multiplier", "lower95", "upper95", "upper_one95")
rows[num] <- lapply(rows[num], function(x) ifelse(is.na(x), "", sprintf("%.17g", x)))
write.csv(rows, file.path(out, "per_original.csv"), row.names = FALSE, quote = FALSE)
