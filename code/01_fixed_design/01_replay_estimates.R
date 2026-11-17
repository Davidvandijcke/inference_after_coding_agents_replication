# Corrected p-value, exact selective p-value H(u) = epsilon u + (1 - epsilon) {1 - (1 - u)^k} and the 5%
# decisions of the reported p-value, the oracle and the replay test (one-sided 95% upper bound below 0.05),
# for every dataset and replay budget. Column in_figure_3a marks the budget Figure 3(a) shows.
#
# Input:  data/intermediate/01_fixed_design/replay_draws.csv
# Output: data/derived/01_fixed_design/fixed_rejections.csv

FIGURE_BUDGET <- 100L  # replays per stratum shown in Figure 3(a)

script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
script <- gsub("~+~", " ", script, fixed = TRUE)  # Rscript writes spaces in the path as ~+~
root <- normalizePath(file.path(dirname(script), "..", ".."))
source(file.path(root, "code", "lib", "replay.R"))
out <- file.path(root, "data", "derived", "01_fixed_design")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

d <- read.csv(file.path(root, "data", "intermediate", "01_fixed_design", "replay_draws.csv"))
stopifnot(nrow(d) == 1600L, all(d$b1 == d$b), all(d$b0 == d$b))
with(d, stopifnot(length(unique(n)) == 1, length(unique(m)) == 1, length(unique(k)) == 1,
                  length(unique(epsilon)) == 1))
d$p_corrected <- mapply(function(u, x1, b1, x0, b0) replay_estimate(u, x1, b1, x0, b0)$p_value,
                        d$u, d$x1, d$b1, d$x0, d$b0)
d$oracle <- d$epsilon * d$u + (1 - d$epsilon) * (-expm1(d$k * log1p(-d$u)))
d$reject_agent_05 <- d$u <= .05
d$reject_oracle_05 <- d$oracle <= .05
d$reject_replay_05 <- d$upper_one95 < .05
d$in_figure_3a <- d$b == FIGURE_BUDGET
stopifnot(sum(d$in_figure_3a) == length(unique(d$case_id)))

d <- d[order(d$case_id, d$b), c("case_id", "b", "n", "m", "k", "epsilon", "q", "u", "oracle", "x1", "b1", "x0", "b0", "p_corrected",
                                "lower95", "upper95", "upper_one95", "reject_agent_05", "reject_oracle_05",
                                "reject_replay_05", "in_figure_3a")]
num <- c("u", "oracle", "p_corrected", "lower95", "upper95", "upper_one95")
d[num] <- lapply(d[num], sprintf, fmt = "%.17g")
write.csv(d, file.path(out, "fixed_rejections.csv"), row.names = FALSE, quote = FALSE)
