# Rebuilds the Card--Krueger reference bank as the cluster run built it: 20,000 subsets of 150 of the 410 stores
# (whole stores, both waves), drawn without replacement with R's L'Ecuyer-CMRG generator from seed 2026101731 in one
# stream, and the original program's p-value on each (each subset scored in a fresh environment). The replay subsets
# are the first 1,142 of the tail subsets (p <= u) and the first 1,858 of the body subsets, each stratum permuted with
# its own seed (2026101732, 2026101733); their CSV packets are what the replay agents received as data.csv. No model
# calls; about 45 seconds.
#
# The rebuild goes to its own folder, never over the shipped bank, and is then compared with the shipped files:
# the subsets and the roster must be identical and the p-values equal to 1e-12 (on another platform than the cluster's
# R 4.5.1 on Linux they differ by about 5e-15). The scripts that ran on the cluster are archived in bank_cluster/.
#
# Usage:   Rscript build_bank.R [OUT]   (OUT defaults to data/derived/08_card/bank)
# Inputs:  data/agent_runs/card_luna/packet/data.csv, programs/original-analysis.R, replays.csv.gz;
#          data/intermediate/08_card/card-bank-subsets.csv.gz, card-bank-p.csv (for the comparison)
# Outputs: OUT/card-bank-subsets.csv.gz (bank_id, rows: 1-based row indices into data.csv, in draw order),
#          OUT/card-bank-p.csv (bank_id, p_value), OUT/packets/<bank_id>.csv (the 3,000 replay datasets)
f <- gsub("~+~", " ", sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)), fixed = TRUE)
root <- normalizePath(file.path(dirname(f), "..", "..", ".."))
runs <- file.path(root, "data/agent_runs/card_luna")
inter <- file.path(root, "data/intermediate/08_card")
out <- commandArgs(trailingOnly = TRUE)
out <- if (length(out)) out[1] else file.path(root, "data/derived/08_card/bank")
dir.create(file.path(out, "packets"), recursive = TRUE, showWarnings = FALSE)
stopifnot(normalizePath(out) != normalizePath(inter))
data <- read.csv(file.path(runs, "packet/data.csv"), check.names = FALSE, stringsAsFactors = FALSE)
program <- file.path(runs, "programs/original-analysis.R")
score <- function(rows) {
  env <- new.env(parent = globalenv())
  sys.source(program, envir = env)
  env$analysis(data[rows, , drop = FALSE])$p_value
}

RNGkind("L'Ecuyer-CMRG", "Inversion", "Rejection")
set.seed(2026101731)
draws <- lapply(seq_len(20000), function(i) sample.int(nrow(data), 150L, replace = FALSE))
p <- vapply(draws, score, 0)
u <- score(seq_len(nrow(data)))

subsets <- data.frame(bank_id = seq_along(draws), rows = vapply(draws, paste, "", collapse = " "))
con <- gzfile(file.path(out, "card-bank-subsets.csv.gz"), "w")
write.csv(subsets, con, row.names = FALSE)
close(con)
write.csv(data.frame(bank_id = seq_along(draws), p_value = sprintf("%.17g", p)), file.path(out, "card-bank-p.csv"),
          row.names = FALSE, quote = FALSE)

permute <- function(ids, seed) {
  set.seed(seed)
  sample(ids, length(ids), replace = FALSE)
}
roster <- c(head(permute(which(p <= u), 2026101732), 1142), head(permute(which(p > u), 2026101733), 1858))
for (id in roster) write.csv(data[draws[[id]], , drop = FALSE], file.path(out, "packets", paste0(id, ".csv")), row.names = FALSE)

shipped <- read.csv(file.path(inter, "card-bank-p.csv"))
replays <- read.csv(gzfile(file.path(runs, "replays.csv.gz")), stringsAsFactors = FALSE)
stopifnot(identical(read.csv(file.path(inter, "card-bank-subsets.csv.gz"), stringsAsFactors = FALSE)$rows, subsets$rows),
          identical(replays$bank_id[order(replays$position)], roster),
          max(abs(p - shipped$p_value)) < 1e-12)
cat(sprintf("%d subsets, %d in the tail (p <= %.6f); roster of %d replay subsets; largest p-value difference from the shipped bank %.1e\n",
            length(p), sum(p <= u), u, length(roster), max(abs(p - shipped$p_value))))
