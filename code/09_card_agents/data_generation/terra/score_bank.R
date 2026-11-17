# Score the Terra original's emitted program on the 20,000 reference subsets of
# the luna Card bank (150 whole stores each, both waves), with the same per-world
# semantics as the luna bank: a fresh environment per world, analysis(data) on
# the subset, failures recorded. Tail means p <= the original's full-data p.
# No model calls. Run from the pilot root: Rscript code/score_bank.R
library(parallel)
data <- read.csv("inputs/data.csv", check.names = FALSE, stringsAsFactors = FALSE)
program <- "runs/original/emitted-analysis.R"
subsets <- read.csv("../card-mn-robustness-20261027/processed/card-bank-subsets.csv.gz", stringsAsFactors = FALSE)
rows <- lapply(strsplit(subsets$rows, " "), as.integer)

score <- function(index) tryCatch({
  env <- new.env(parent = globalenv())
  sys.source(program, envir = env)
  z <- env$analysis(data[index, , drop = FALSE])
  c(p_value = as.numeric(z$p_value), estimate = as.numeric(z$estimate))
}, error = function(e) c(p_value = NA_real_, estimate = NA_real_))

full <- score(seq_len(nrow(data)))
u <- unname(full["p_value"])
p <- do.call(rbind, mclapply(rows, score, mc.cores = 8))
bank <- data.frame(bank_id = subsets$bank_id, p_value = p[, "p_value"], estimate = p[, "estimate"])
write.csv(bank, "bank/terra-bank-p.csv", row.names = FALSE)
summary <- list(u = u, full_data_estimate = unname(full["estimate"]), J = nrow(bank),
                failures = sum(!is.finite(bank$p_value)),
                tail = sum(bank$p_value <= u, na.rm = TRUE), body = sum(bank$p_value > u, na.rm = TRUE),
                tail_share = mean(bank$p_value <= u, na.rm = TRUE))
jsonlite::write_json(summary, "bank/summary.json", auto_unbox = TRUE, digits = I(15), pretty = TRUE)
str(summary)
