# Replay correction of GPT-6.1-Sol's Card-Krueger original, with its two-sided 95% signed-LR interval, on the
# replays the declaration review classified as match or nonmatch.
#
# Inputs:  data/agent_runs/card_sol/original.json, replays.csv.gz (module 09)
# Output:  data/derived/10_self_correction/card_sol_interval.csv

script <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE))
script <- gsub("~+~", " ", script, fixed = TRUE)  # Rscript writes spaces in the path as ~+~
root <- normalizePath(file.path(dirname(script), "..", ".."))
source(file.path(root, "code", "lib", "replay.R"))
cohort <- file.path(root, "data", "agent_runs", "card_sol")
out <- file.path(root, "data", "derived", "10_self_correction")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

u <- jsonlite::fromJSON(file.path(cohort, "original.json"))$p_value
stopifnot(is.numeric(u), length(u) == 1)
replays <- read.csv(gzfile(file.path(cohort, "replays.csv.gz")), encoding = "UTF-8")
classified <- replays[replays$review_label %in% c("match", "nonmatch"), ]
n <- function(stratum, label = c("match", "nonmatch")) sum(classified$stratum == stratum & classified$review_label %in% label)
x1 <- n("tail", "match"); b1 <- n("tail"); x0 <- n("body", "match"); b0 <- n("body")
est <- replay_estimate(u, x1, b1, x0, b0)
ci <- replay_interval_lr(u, x1, b1, x0, b0, gamma = .05)$p_interval
row <- data.frame(u = u, x1 = x1, b1 = b1, x0 = x0, b0 = b0,
                  unclassified = sum(!replays$review_label %in% c("match", "nonmatch")),
                  p_corrected = est$p_value, multiplier = est$multiplier,
                  lower95 = ci[["lower"]], upper95 = ci[["upper"]])
num <- c("u", "p_corrected", "multiplier", "lower95", "upper95")
row[num] <- lapply(row[num], function(x) sprintf("%.17g", x))
write.csv(row, file.path(out, "card_sol_interval.csv"), row.names = FALSE, quote = FALSE)
