# Sealed roster for the 200-replay masked pilot (../PLAN.md): 100 tail and 100 body
# subsets of the Sonnet-scored bank (all tail subsets and the rest body if fewer than 100 are in the
# tail), in one random run order. Packets are written as in the Terra and Sol cohorts; the roster keeps
# each packet's sha256. No model calls. Run from the run folder: Rscript code/make_roster.R
packet_dir <- file.path(getwd(), "packets")
stopifnot(!file.exists("roster.json"), !dir.exists(packet_dir))
summary_bank <- jsonlite::fromJSON("bank/summary.json")
stopifnot(is.finite(summary_bank$u), summary_bank$failures <= 200)
data <- read.csv("inputs/data.csv", check.names = FALSE, stringsAsFactors = FALSE)
subsets <- read.csv("inputs-bank/card-bank-subsets.csv.gz", stringsAsFactors = FALSE)
bank <- read.csv("bank/bank-p.csv")
u <- jsonlite::fromJSON("bank/summary.json")$u
stopifnot(identical(bank$bank_id, subsets$bank_id), is.finite(u))
ok <- is.finite(bank$p_value)
tail_ids <- bank$bank_id[ok & bank$p_value <= u]
body_ids <- bank$bank_id[ok & bank$p_value > u]
n_tail <- min(100, length(tail_ids)); n_body <- 200 - n_tail
stopifnot(length(body_ids) >= n_body)

set.seed(2026110401)
tail <- if (n_tail == length(tail_ids)) tail_ids else sample(tail_ids, n_tail)
body <- sample(body_ids, n_body)
set.seed(2026110402)
order <- sample(c(tail, body))

dir.create(packet_dir)
cases <- lapply(seq_along(order), function(i) {
  j <- order[i]
  id <- substr(digest::digest(paste0("sonnet-masked-pilot-", j), algo = "sha256"), 1, 20)
  path <- file.path(packet_dir, paste0(id, ".csv"))
  write.csv(data[as.integer(strsplit(subsets$rows[j], " ")[[1]]), , drop = FALSE], path, row.names = FALSE)
  list(position = i, case_id = id, bank_id = j, stratum = if (j %in% tail) "tail" else "body",
       bank_p = bank$p_value[j], packet = path, packet_sha256 = digest::digest(file = path, algo = "sha256"))
})
jsonlite::write_json(list(u = u, n_tail = n_tail, n_body = n_body, failures = sum(!ok),
                          draw_seed = 2026110401, order_seed = 2026110402, cases = cases),
                     "roster.json", auto_unbox = TRUE, digits = I(15), pretty = TRUE)
cat(length(cases), "cases:", n_tail, "tail,", n_body, "body\n")
