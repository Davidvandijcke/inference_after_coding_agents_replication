# Replay match counts, corrected p-value, multiplier and signed-LR bounds of each fixed-menu workflow. A replay is
# classified when it recorded a selection and matches when it selected the original's menu model.
#
# Inputs:  data/agent_runs/issp_menu_{anti,pro}/replays.csv and design.json,
#          data/intermediate/07_immigration_menu/bank_{anti,pro}.csv.gz
# Output:  data/derived/07_immigration_menu/replay.csv

args <- commandArgs(trailingOnly = FALSE)
# Rscript writes spaces in the script's path as "~+~".
here <- dirname(normalizePath(gsub("~+~", " ", sub("^--file=", "", args[grep("^--file=", args)]), fixed = TRUE)))
root <- dirname(dirname(here))
source(file.path(root, "code", "lib", "replay.R"))

rows <- list()
for (arm in c("anti", "pro")) {
  folder <- file.path(root, "data", "agent_runs", paste0("issp_menu_", arm))
  design <- jsonlite::fromJSON(file.path(folder, "design.json"))
  replays <- read.csv(file.path(folder, "replays.csv"), colClasses = "character")
  bank <- read.csv(gzfile(file.path(root, "data", "intermediate", "07_immigration_menu",
                                    paste0("bank_", arm, ".csv.gz"))), colClasses = "character")
  for (w in c("fixed_reference", "frontier", "client_pressure")) {
    target <- design[[w]]$target; u <- design[[w]]$u
    all_w <- replays[replays$workflow == w, ]
    p_col <- if (arm == "anti") "p_target" else if (w == "client_pressure") "p_second" else "p_primary"
    p_bank <- as.numeric(bank[[p_col]][match(all_w$bank_index, bank$bank_index)])
    stopifnot(!anyNA(p_bank), all((p_bank <= u) == (all_w$stratum == "tail")))
    d <- all_w[all_w$selected_model_id != "", ]
    x1 <- sum(d$selected_model_id[d$stratum == "tail"] == target); b1 <- sum(d$stratum == "tail")
    x0 <- sum(d$selected_model_id[d$stratum == "body"] == target); b0 <- sum(d$stratum == "body")
    est <- replay_estimate(u, x1, b1, x0, b0)
    two <- replay_interval_lr(u, x1, b1, x0, b0, gamma = 0.05)$p_interval
    one <- replay_interval_lr(u, x1, b1, x0, b0, gamma = 0.10)$p_interval
    rows[[length(rows) + 1]] <- data.frame(
      arm = arm, workflow = w, target = target, u = u, attempted = nrow(all_w),
      followed_rule = sum(d$selected_model_id == d$rule_pick), x1 = x1, b1 = b1, x0 = x0, b0 = b0,
      corrected_p = est$p_value, multiplier = est$multiplier, p_lower95 = two[[1]], p_upper95 = two[[2]],
      p_lower_one95 = one[[1]], p_upper_one95 = one[[2]])
  }
}
dir.create(file.path(root, "data", "derived", "07_immigration_menu"), recursive = TRUE, showWarnings = FALSE)
write.csv(do.call(rbind, rows), file.path(root, "data", "derived", "07_immigration_menu", "replay.csv"),
          row.names = FALSE)
