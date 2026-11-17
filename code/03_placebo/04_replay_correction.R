# Corrected p-value, two-sided 95% signed-LR interval and one-sided 95% upper bound of each placebo original, from
# the replays' own verdicts. With no match, or an empty stratum (slot 75), the correction is unavailable (p = 1);
# with body matches only there is no point estimate. The replay test rejects when the upper bound is below 0.05.
#
# Input:   data/derived/03_placebo/replay_counts.csv
# Outputs: data/derived/03_placebo/per_original.csv (read by module 02), replay_intervals.csv
args <- commandArgs(FALSE)
# Rscript writes spaces in the script path as "~+~".
here <- dirname(normalizePath(gsub("~+~", " ", sub("^--file=", "", args[grep("^--file=", args)]), fixed = TRUE)))
root <- normalizePath(file.path(here, "..", ".."))
source(file.path(root, "code", "lib", "replay.R"))
out <- file.path(root, "data", "derived", "03_placebo")
dir.create(out, showWarnings = FALSE, recursive = TRUE)

d <- read.csv(file.path(out, "replay_counts.csv"), colClasses = c(u = "character"))
stopifnot(identical(d$slot, 0:101))
u_recorded <- d$u  # written back as recorded: R may parse a 17-digit decimal one unit off in the last place
d$u <- as.numeric(d$u)

interval <- function(i, gamma) {
  unname(replay_interval_lr(d$u[i], d$tail_matches[i], d$tail_replays[i], d$body_matches[i], d$body_replays[i],
                            gamma = gamma)$p_interval)
}
rows <- lapply(seq_len(nrow(d)), function(i) {
  est <- replay_estimate(d$u[i], d$tail_matches[i], d$tail_replays[i], d$body_matches[i], d$body_replays[i])
  two <- interval(i, 0.05)
  status <- if (est$status == "unavailable") "unavailable" else if (d$tail_matches[i] == 0) "body_only" else "estimated"
  p <- if (status == "body_only") NA_real_ else est$p_value
  data.frame(status = status, p_corrected = p, multiplier = if (status == "estimated") p / d$u[i] else NA_real_,
             lower95 = two[1], upper95 = two[2], p_upper_one95 = interval(i, 0.10)[2])
})
r <- cbind(d, do.call(rbind, rows))

full <- function(x) ifelse(is.na(x), "", sprintf("%.17g", x))
per_original <- data.frame(study = r$study, original_id = r$slot, u = u_recorded,
                           x1 = r$tail_matches, b1 = r$tail_replays, x0 = r$body_matches, b0 = r$body_replays,
                           status = r$status, p_corrected = full(r$p_corrected), p_upper_one95 = full(r$p_upper_one95),
                           reject_agent_05 = as.integer(r$u <= 0.05), reject_replay_05 = as.integer(r$p_upper_one95 < 0.05))
write.csv(per_original, file.path(out, "per_original.csv"), row.names = FALSE, quote = FALSE)
write.csv(data.frame(original_id = r$slot, multiplier = full(r$multiplier), lower95 = full(r$lower95),
                     upper95 = full(r$upper95)),
          file.path(out, "replay_intervals.csv"), row.names = FALSE, quote = FALSE)
cat("replay test rejects", sum(per_original$reject_replay_05), "of", nrow(per_original), "at 5%\n")
