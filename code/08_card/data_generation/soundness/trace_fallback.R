root <- "../card-mn-robustness-20261027"
data <- read.csv(file.path(root, "original/data.csv"), check.names = FALSE, stringsAsFactors = FALSE)
subsets <- read.csv(file.path(root, "processed/card-bank-subsets.csv.gz"), stringsAsFactors = FALSE)
rows <- lapply(strsplit(subsets$rows, " "), as.integer)
replays <- read.csv(file.path(root, "processed/card-replays.csv"), stringsAsFactors = FALSE)
pk <- jsonlite::stream_in(file("../card-soundness-review-20261029/inputs/packets.jsonl"), verbose = FALSE)
rev <- read.csv("../card-soundness-review-20261029/results/card-replay-review.csv", stringsAsFactors = FALSE)
ids <- rev$replay_id[rev$ground == "silent_fallback"]
fits <- list()
suppressMessages(trace("lm", where = asNamespace("stats"), print = FALSE, exit = quote({
  m <- returnValue(); fits[[length(fits) + 1L]] <<- c(formula = paste(deparse(m$call$formula), collapse = ""), rank = m$rank, cols = ncol(stats::model.matrix(m)))
})))
tmp <- tempfile("fb-"); dir.create(tmp)
for (id in ids) {
  b <- replays$bank_id[match(id, replays$replay_id)]
  packet <- file.path(tmp, "data.csv"); write.csv(data[rows[[b]], , drop = FALSE], packet, row.names = FALSE)
  script <- file.path(tmp, "analysis.R"); writeLines(pk$code[match(id, pk$replay_id)], script)
  env <- new.env(parent = globalenv()); fits <- list()
  sys.source(script, envir = env); r <- env$analysis(read.csv(packet))
  cat(substr(id, nchar(id) - 7, nchar(id)), "estimate", signif(r[["estimate"]], 5), "p", signif(r[["p_value"]], 4), "\n")
  for (f in fits) cat("   lm:", f[["formula"]], "rank", f[["rank"]], "of", f[["cols"]], "\n")
}
