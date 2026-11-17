# Run each replay's own final analysis(data) on the 150-store subset it analyzed.
# The subset is written and read back as the replay's data.csv was. No model calls.
# Usage: Rscript code/run_replay_functions.R <rebuilt.jsonl> <out.csv>
# Run inside the sandbox in run_all.sh (no network, writes only to a temp folder).
args <- commandArgs(TRUE)
root <- "../card-mn-robustness-20261027"
data <- read.csv(file.path(root, "original/data.csv"), check.names = FALSE, stringsAsFactors = FALSE)
subsets <- read.csv(file.path(root, "processed/card-bank-subsets.csv.gz"), stringsAsFactors = FALSE)
rows <- lapply(strsplit(subsets$rows, " "), as.integer)
replays <- read.csv(file.path(root, "processed/card-replays.csv"), stringsAsFactors = FALSE)
rebuilt <- jsonlite::stream_in(file(args[1]), verbose = FALSE)

tmp <- tempfile("replay-run-"); dir.create(tmp)
run_one <- function(code, bank_id) {
  packet <- file.path(tmp, "data.csv")
  write.csv(data[rows[[bank_id]], , drop = FALSE], packet, row.names = FALSE)
  script <- file.path(tmp, "analysis.R"); writeLines(code, script)
  env <- new.env(parent = globalenv())
  out <- tryCatch({
    setTimeLimit(elapsed = 60, transient = TRUE)
    sys.source(script, envir = env)
    r <- env$analysis(read.csv(packet))
    setTimeLimit(elapsed = Inf)
    num <- function(x) sprintf("%.17g", as.numeric(x)[1])
    c(estimate = num(r[["estimate"]]), se = num(r[["se"]]), p_value = num(r[["p_value"]]), error = NA_character_)
  }, error = function(e) { setTimeLimit(elapsed = Inf); c(estimate = NA_character_, se = NA_character_, p_value = NA_character_, error = conditionMessage(e)) })
  out
}
res <- t(vapply(seq_len(nrow(rebuilt)), function(i) {
  b <- replays$bank_id[match(rebuilt$replay_id[i], replays$replay_id)]
  code <- rebuilt$analysis_R[i]
  if (is.na(b) || is.null(code) || is.na(code)) return(c(estimate = NA_character_, se = NA_character_, p_value = NA_character_, error = "no script or subset"))
  run_one(code, b)
}, character(4)))
out <- data.frame(replay_id = rebuilt$replay_id, rebuild_status = rebuilt$status, res, stringsAsFactors = FALSE)
write.csv(out, args[2], row.names = FALSE)
cat(sprintf("%d replays, %d with a result, %d errors\n", nrow(out), sum(!is.na(out$estimate)), sum(!is.na(out$error))))
