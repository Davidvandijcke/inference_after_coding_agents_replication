# Second extension of the masked pilot (../PLAN.md, "Extension to 3,000"): 1,100 more tail and 1,100 more body
# subsets of the same Sonnet-scored bank, drawn from the subsets neither the pilot nor the first extension used,
# then one pending list for card_claude.py replays holding the first extension's 600 cases and these 2,200 in one
# random run order (cases with a record are skipped at launch). Packets as before. No model calls.
# Run from the run folder: Rscript code/make_ext2_roster.R
stopifnot(!file.exists("roster-ext2.json"), !file.exists("pending-E2-b.json"))
data <- read.csv("inputs/data.csv", check.names = FALSE, stringsAsFactors = FALSE)
subsets <- read.csv("inputs-bank/card-bank-subsets.csv.gz", stringsAsFactors = FALSE)
bank <- read.csv("bank/bank-p.csv")
u <- jsonlite::fromJSON("bank/summary.json")$u
pilot <- jsonlite::fromJSON("roster.json", simplifyVector = FALSE)
ext1 <- jsonlite::fromJSON("roster-ext.json", simplifyVector = FALSE)
stopifnot(identical(bank$bank_id, subsets$bank_id), isTRUE(all.equal(u, pilot$u)), isTRUE(all.equal(u, ext1$u)))
used <- vapply(c(pilot$cases, ext1$cases), function(x) x$bank_id, numeric(1))
ok <- is.finite(bank$p_value) & !(bank$bank_id %in% used)
tail_ids <- bank$bank_id[ok & bank$p_value <= u]
body_ids <- bank$bank_id[ok & bank$p_value > u]
stopifnot(length(tail_ids) >= 1100, length(body_ids) >= 1100)

set.seed(2026110503)
tail <- sample(tail_ids, 1100)
body <- sample(body_ids, 1100)
set.seed(2026110504)
order <- sample(c(tail, body))

cases <- lapply(seq_along(order), function(i) {
  j <- order[i]
  id <- substr(digest::digest(paste0("sonnet-masked-ext2-", j), algo = "sha256"), 1, 20)
  path <- file.path(getwd(), "packets", paste0(id, ".csv"))
  stopifnot(!file.exists(path))
  write.csv(data[as.integer(strsplit(subsets$rows[j], " ")[[1]]), , drop = FALSE], path, row.names = FALSE)
  list(position = 800 + i, case_id = id, bank_id = j, stratum = if (j %in% tail) "tail" else "body",
       bank_p = bank$p_value[j], packet = path, packet_sha256 = digest::digest(file = path, algo = "sha256"))
})
jsonlite::write_json(list(u = u, n_tail = 1100, n_body = 1100, draw_seed = 2026110503, order_seed = 2026110504,
                          excluded = "the 200 pilot and 600 first-extension subsets", cases = cases),
                     "roster-ext2.json", auto_unbox = TRUE, digits = I(15), pretty = TRUE)
both <- c(ext1$cases, cases)
set.seed(2026110505)
both <- both[sample(length(both))]
jsonlite::write_json(list(launch = "E2", account = "b", accounts = list("b"), cases = both),
                     "pending-E2-b.json", auto_unbox = TRUE, digits = I(15), pretty = TRUE)
cat(length(cases), "new cases: 1100 tail, 1100 body;", length(both), "in the pending list\n")
