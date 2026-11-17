# Corrected p-value and multiplier per cohort on the reviewer's labels, with the one-sided 95% bounds (the ends of
# the two-sided 90% interval).
#
# Input:  data/derived/09_card_agents/counts.csv
# Output: data/derived/09_card_agents/estimates.csv (Table 8's rows)
# Rscript writes spaces in the script path as "~+~".
script <- gsub("~+~", " ", sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)), fixed = TRUE)
root <- normalizePath(file.path(dirname(script), "..", ".."))
source(file.path(root, "code/lib/replay.R"))
out <- file.path(root, "data/derived/09_card_agents")
d <- read.csv(file.path(out, "counts.csv"), stringsAsFactors = FALSE)
d <- d[, c("cohort", "u", "x1", "b1", "x0", "b0")]
est <- as.data.frame(t(sapply(seq_len(nrow(d)), function(i) with(d[i, ], {
  point <- replay_estimate(u, x1, b1, x0, b0)
  one <- replay_interval_lr(u, x1, b1, x0, b0, gamma = 0.10)$p_interval
  c(p_corrected = point$p_value, multiplier = point$multiplier, upper_one95 = one[["upper"]],
    multiplier_lower_one95 = one[["lower"]] / u)
}))))
cat(sprintf("%-19s p %.4f (<= %.4f)  lambda %.3f (>= %.3f)\n", d$cohort, est$p_corrected,
            est$upper_one95, est$multiplier, est$multiplier_lower_one95), sep = "")
d$u <- sprintf("%.17g", d$u)
d <- cbind(d, lapply(est, function(v) sprintf("%.17g", v)))
write.csv(d, file.path(out, "estimates.csv"), row.names = FALSE, quote = FALSE)
