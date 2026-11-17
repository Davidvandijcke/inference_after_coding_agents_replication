# Two-sided 95% and one-sided 95% upper replay bounds for the corrected p-value of each usable original.
#
# Inputs:  data/derived/04_power/replay_counts.csv, data/agent_runs/power_luna_login/originals.csv
# Output:  data/derived/04_power/bounds.csv

# Rscript writes spaces in the path as ~+~
self <- gsub("~+~", " ", sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)), fixed = TRUE)
root <- normalizePath(file.path(dirname(self), "..", ".."))
source(file.path(root, "code", "lib", "replay.R"))
suppressMessages(library(parallel))

d <- read.csv(file.path(root, "data", "derived", "04_power", "replay_counts.csv"), colClasses = c(u = "character"))
arms <- read.csv(file.path(root, "data", "agent_runs", "power_luna_login", "originals.csv"))[, c("slot", "arm")]
d <- merge(arms, d, by = "slot")
d <- d[order(d$slot), c("slot", "arm", "u", "x1", "b1", "x0", "b0")]
u <- as.numeric(d$u)  # u is written back as read, since R can misparse the last digit of a 17-digit decimal

interval <- function(u, x1, b1, x0, b0, gamma) {
  unname(replay_interval_lr(u, x1, b1, x0, b0, gamma = gamma)$p_interval)
}
res <- mclapply(seq_len(nrow(d)), function(i) {
  one <- interval(u[i], d$x1[i], d$b1[i], d$x0[i], d$b0[i], 0.10)
  two <- interval(u[i], d$x1[i], d$b1[i], d$x0[i], d$b0[i], 0.05)
  c(lower95 = two[1], upper95 = two[2], upper_one95 = one[2])
}, mc.cores = 2)
out <- cbind(d, apply(do.call(rbind, res), 2, function(x) sprintf("%.17g", x)))
dir.create(file.path(root, "data", "derived", "04_power"), recursive = TRUE, showWarnings = FALSE)
write.csv(out, file.path(root, "data", "derived", "04_power", "bounds.csv"), row.names = FALSE, quote = FALSE)
