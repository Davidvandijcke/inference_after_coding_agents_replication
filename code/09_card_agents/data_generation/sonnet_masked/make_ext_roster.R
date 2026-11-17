# Extension of the masked pilot (../PLAN.md, "Extension to 800"): 300 more tail and 300 more body subsets of
# the same Sonnet-scored bank, drawn from the subsets the pilot roster did not use, in one random run order.
# Packets are written as for the pilot. Writes roster-ext.json and the pending list pending-E1-b.json for
# card_claude.py replays. No model calls. Run from the run folder: Rscript code/make_ext_roster.R
stopifnot(!file.exists("roster-ext.json"), !file.exists("pending-E1-b.json"))
data <- read.csv("inputs/data.csv", check.names = FALSE, stringsAsFactors = FALSE)
subsets <- read.csv("inputs-bank/card-bank-subsets.csv.gz", stringsAsFactors = FALSE)
bank <- read.csv("bank/bank-p.csv")
u <- jsonlite::fromJSON("bank/summary.json")$u
pilot <- jsonlite::fromJSON("roster.json", simplifyVector = FALSE)
stopifnot(identical(bank$bank_id, subsets$bank_id), isTRUE(all.equal(u, pilot$u)))
used <- vapply(pilot$cases, function(x) x$bank_id, numeric(1))
ok <- is.finite(bank$p_value) & !(bank$bank_id %in% used)
tail_ids <- bank$bank_id[ok & bank$p_value <= u]
body_ids <- bank$bank_id[ok & bank$p_value > u]
stopifnot(length(tail_ids) >= 300, length(body_ids) >= 300)

set.seed(2026110501)
tail <- sample(tail_ids, 300)
body <- sample(body_ids, 300)
set.seed(2026110502)
order <- sample(c(tail, body))

cases <- lapply(seq_along(order), function(i) {
  j <- order[i]
  id <- substr(digest::digest(paste0("sonnet-masked-ext-", j), algo = "sha256"), 1, 20)
  path <- file.path(getwd(), "packets", paste0(id, ".csv"))
  stopifnot(!file.exists(path))
  write.csv(data[as.integer(strsplit(subsets$rows[j], " ")[[1]]), , drop = FALSE], path, row.names = FALSE)
  list(position = 200 + i, case_id = id, bank_id = j, stratum = if (j %in% tail) "tail" else "body",
       bank_p = bank$p_value[j], packet = path, packet_sha256 = digest::digest(file = path, algo = "sha256"))
})
jsonlite::write_json(list(u = u, n_tail = 300, n_body = 300, draw_seed = 2026110501, order_seed = 2026110502,
                          excluded = "the 200 pilot subsets", cases = cases),
                     "roster-ext.json", auto_unbox = TRUE, digits = I(15), pretty = TRUE)
jsonlite::write_json(list(launch = "E1", account = "b", accounts = list("b"), cases = cases),
                     "pending-E1-b.json", auto_unbox = TRUE, digits = I(15), pretty = TRUE)
cat(length(cases), "cases: 300 tail, 300 body\n")
