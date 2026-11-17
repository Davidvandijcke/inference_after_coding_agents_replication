# Checks the shipped reference bank: redraws the 20,000 subsets of 150 stores from their seed, reruns the original
# program on every subset against the shipped p-values, and redraws the replay roster (1,142 tail, 1,858 body).
#
# Inputs:  data/intermediate/08_card/card-bank-subsets.csv.gz, card-bank-p.csv
#          data/agent_runs/card_luna/packet/data.csv, programs/original-analysis.R, replays.csv.gz
#          data/derived/08_card/original.csv
# Output:  data/derived/08_card/bank.csv
f <- gsub("~+~", " ", sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)), fixed = TRUE)
root <- normalizePath(file.path(dirname(f), "..", ".."))
source(file.path(root, "code/lib/replay.R"))
inter <- file.path(root, "data/intermediate/08_card")
runs <- file.path(root, "data/agent_runs/card_luna")
out <- file.path(root, "data/derived/08_card")
data <- read.csv(file.path(runs, "packet/data.csv"), check.names = FALSE, stringsAsFactors = FALSE)
program <- new.env()
sys.source(file.path(runs, "programs/original-analysis.R"), envir = program)
original <- read.csv(file.path(out, "original.csv"))
u <- original$value[original$name == "u"]

RNGkind("L'Ecuyer-CMRG", "Inversion", "Rejection")
set.seed(2026101731)
draws <- lapply(seq_len(20000), function(i) sample.int(nrow(data), 150L, replace = FALSE))
subsets <- read.csv(file.path(inter, "card-bank-subsets.csv.gz"), stringsAsFactors = FALSE)
stopifnot(identical(subsets$bank_id, seq_len(20000)), identical(lapply(strsplit(subsets$rows, " "), as.integer), draws))

bank <- read.csv(file.path(inter, "card-bank-p.csv"))
stopifnot(identical(bank$bank_id, seq_len(20000)))
p <- vapply(draws, function(rows) program$analysis(data[rows, , drop = FALSE])$p_value, 0)
stopifnot(max(abs(p - bank$p_value)) < 1e-12)

# Neyman split of the 3,000 from an earlier, unshipped cohort of 743 replays (11 of 272 tail, 7 of 458 body matched).
stopifnot(identical(unname(neyman_allocation(3000L, 11L, 272L, 7L, 458L, minimum = 300L)), c(1142, 1858)))

permute <-function(ids, seed) {
  set.seed(seed)
  sample(ids, length(ids), replace = FALSE)
}
tail_ids <- bank$bank_id[bank$p_value <= u]
body_ids <- bank$bank_id[bank$p_value > u]
roster <- c(head(permute(tail_ids, 2026101732), 1142), head(permute(body_ids, 2026101733), 1858))
replays <- read.csv(gzfile(file.path(runs, "replays.csv.gz")), stringsAsFactors = FALSE)
replays <- replays[order(replays$position), ]
stopifnot(identical(replays$bank_id, roster), identical(replays$stratum, rep(c("tail", "body"), c(1142, 1858))),
          all((replays$bank_p <= u) == (replays$stratum == "tail")),
          max(abs(replays$bank_p - bank$p_value[replays$bank_id])) < 1e-12)

values <- c(subsets = nrow(subsets), subset_size = length(draws[[1]]), replays = nrow(replays),
            tail_replays = sum(replays$stratum == "tail"), body_replays = sum(replays$stratum == "body"))
write.csv(data.frame(name = names(values), value = values), file.path(out, "bank.csv"), row.names = FALSE, quote = FALSE)
