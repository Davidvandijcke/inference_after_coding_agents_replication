# Replay correction of the original analysis: matches on the fact labels in the tail and the body (unresolved pairs
# excluded), corrected p-value and multiplier at u, and their one-sided 95% bounds.
#
# Inputs:  data/agent_runs/card_luna/replays.csv.gz, data/derived/08_card/original.csv
# Output:  data/derived/08_card/correction.csv
f <- gsub("~+~", " ", sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)), fixed = TRUE)
root <- normalizePath(file.path(dirname(f), "..", ".."))
source(file.path(root, "code/lib/replay.R"))
out <- file.path(root, "data/derived/08_card")
original <- read.csv(file.path(out, "original.csv"))
u <- original$value[original$name == "u"]
r <- read.csv(gzfile(file.path(root, "data/agent_runs/card_luna/replays.csv.gz")), stringsAsFactors = FALSE)
stopifnot(all(r$status == "completed"))
classified <- r[r$review_label %in% c("match", "nonmatch"), ]
count <- function(s, matches) sum(classified$stratum == s & (!matches | classified$review_label == "match"))
x1 <- count("tail", TRUE); b1 <- count("tail", FALSE); x0 <- count("body", TRUE); b0 <- count("body", FALSE)

point <- replay_estimate(u, x1, b1, x0, b0)
one <- replay_interval_lr(u, x1, b1, x0, b0, gamma = 0.10)$p_interval  # its ends are the one-sided 95% bounds
values <- c(u = u, x1 = x1, b1 = b1, x0 = x0, b0 = b0, p_corrected = point$p_value, multiplier = point$multiplier,
            upper_one95 = one[["upper"]], multiplier_lower_one95 = one[["lower"]] / u)
write.csv(data.frame(name = names(values), value = sprintf("%.17g", values)), file.path(out, "correction.csv"),
          row.names = FALSE, quote = FALSE)
