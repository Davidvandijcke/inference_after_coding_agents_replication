# Test whether a review failure can change the reported result: rerun the replay's analysis(data)
# on its own subset, and again with the stores the objection concerns removed. Equal results mean
# the objection has no effect on these data. Run from ../card-replay-estimates-20261029/ inside
# code/sandbox.sb: sandbox-exec -f code/sandbox.sb Rscript ../card-soundness-review-20261029/code/audit_failures.R
root <- "../card-mn-robustness-20261027"
data <- read.csv(file.path(root, "original/data.csv"), check.names = FALSE, stringsAsFactors = FALSE)
subsets <- read.csv(file.path(root, "processed/card-bank-subsets.csv.gz"), stringsAsFactors = FALSE)
rows <- lapply(strsplit(subsets$rows, " "), as.integer)
replays <- read.csv(file.path(root, "processed/card-replays.csv"), stringsAsFactors = FALSE)
pk <- jsonlite::stream_in(file("../card-soundness-review-20261029/inputs/packets.jsonl"), verbose = FALSE)
rev <- read.csv("../card-soundness-review-20261029/results/card-replay-review.csv", stringsAsFactors = FALSE)

tmp <- tempfile("audit-"); dir.create(tmp)
run <- function(id, d) {
  packet <- file.path(tmp, "data.csv"); write.csv(d, packet, row.names = FALSE)
  script <- file.path(tmp, "analysis.R"); writeLines(pk$code[match(id, pk$replay_id)], script)
  env <- new.env(parent = globalenv()); sys.source(script, envir = env)
  r <- env$analysis(read.csv(packet))
  c(estimate = as.numeric(r[["estimate"]])[1], se = as.numeric(r[["se"]])[1], p = as.numeric(r[["p_value"]])[1])
}
closure <- rev$replay_id[rev$ground == "declaration" & grepl("clos", rev$reason, ignore.case = TRUE)]
tests <- c(setNames(rep("closed_missing", length(closure)), closure),
           setNames("emppt_missing", rev$replay_id[grepl("omits emppt", rev$reason)]))
# closed_missing: closed stores whose follow-up employment count is missing; emppt_missing: stores missing part-time counts
drop <- list(closed_missing = function(d) d$status2 %in% c(2, 3, 4, 5) & (is.na(d$empft2) | is.na(d$emppt2)),
             emppt_missing = function(d) is.na(d$emppt) | is.na(d$emppt2))
cat("replay_id,test,stores_removed,estimate,se,p,estimate_without,se_without,p_without\n")
for (id in names(tests)) {
  d <- data[rows[[replays$bank_id[match(id, replays$replay_id)]]], , drop = FALSE]
  gone <- drop[[tests[[id]]]](d)
  a <- run(id, d); b <- if (any(gone)) run(id, d[!gone, , drop = FALSE]) else a
  cat(id, tests[[id]], sum(gone), sprintf("%.10g", c(a, b)), sep = ",", fill = FALSE); cat("\n")
}
