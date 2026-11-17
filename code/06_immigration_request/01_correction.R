# Corrected p-value, multiplier and intervals of the realistic request. A replay matches on its own verdict or when
# the second reading found its stated differences from the original to be wording only (reviewed_match).
#
# Inputs:  data/agent_runs/issp_request_luna/originals.csv, replays.csv,
#          data/intermediate/06_immigration_request/bank_p.csv
# Output:  data/derived/06_immigration_request/correction.csv
args <- commandArgs(trailingOnly = FALSE)
# Rscript writes spaces in the script's path as "~+~".
here <- dirname(normalizePath(gsub("~+~", " ", sub("^--file=", "", args[grep("^--file=", args)]), fixed = TRUE)))
root <- dirname(dirname(here))
source(file.path(root, "code", "lib", "replay.R"))

originals <- read.csv(file.path(root, "data", "agent_runs", "issp_request_luna", "originals.csv"),
                      colClasses = "character")
u <- as.numeric(originals$program_p[originals$selected_as_original == "1"])
replays <- read.csv(file.path(root, "data", "agent_runs", "issp_request_luna", "replays.csv"),
                    colClasses = c(session_id = "character", subsample_p = "character"))
bank <- read.csv(file.path(root, "data", "intermediate", "06_immigration_request", "bank_p.csv"),
                 colClasses = c(p = "character"))

stopifnot(nrow(bank) == 4000, nrow(replays) == 1000, all(replays$status == "completed"),
          all(replays$subsample_p == bank$p[match(replays$bank_index, bank$bank_index)]),
          all((as.numeric(replays$subsample_p) <= u) == (replays$stratum == "tail")))
is_match <- replays$reviewed_match == 1
tail <- replays$stratum == "tail"; body <- replays$stratum == "body"
x1 <- sum(is_match[tail]); b1 <- sum(tail); x0 <- sum(is_match[body]); b0 <- sum(body)
est <- replay_estimate(u, x1, b1, x0, b0)
two <- replay_interval_lr(u, x1, b1, x0, b0, gamma = 0.05)$p_interval
one <- replay_interval_lr(u, x1, b1, x0, b0, gamma = 0.10)$p_interval  # its ends are the one-sided 95% bounds
out <- data.frame(u = u, bank_size = nrow(bank), bank_tail = sum(as.numeric(bank$p) <= u), x1 = x1, b1 = b1, x0 = x0,
                  b0 = b0, corrected_p = est$p_value, multiplier = est$multiplier, p_lower95 = two[[1]],
                  p_upper95 = two[[2]], p_lower_one95 = one[[1]], p_upper_one95 = one[[2]])
dir.create(file.path(root, "data", "derived", "06_immigration_request"), recursive = TRUE, showWarnings = FALSE)
write.csv(out, file.path(root, "data", "derived", "06_immigration_request", "correction.csv"), row.names = FALSE)
