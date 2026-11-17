# Sealed roster for the fixed 3,000-replay Sol cohort (MAIN_PLAN.md): 1,500 tail
# and 1,500 body subsets drawn from the Sol-scored bank, excluding the 20 pilot
# subsets, in one random run order. Packets are written as in the Terra cohort to
# a local folder outside the synced project folder; the roster keeps each packet's sha256. No model
# calls. Run from the pilot root: Rscript code/make_main_roster.R
packet_dir <- path.expand("<packet-cache>/sol-card-main/packets")
stopifnot(!file.exists("main-roster.json"), !dir.exists(packet_dir))
data <- read.csv("inputs/data.csv", check.names = FALSE, stringsAsFactors = FALSE)
subsets <- read.csv("../card-mn-robustness-20261027/processed/card-bank-subsets.csv.gz", stringsAsFactors = FALSE)
bank <- read.csv("bank/sol-bank-p.csv")
u <- jsonlite::fromJSON("bank/summary.json")$u
pilot <- vapply(jsonlite::fromJSON("roster.json", simplifyVector = FALSE)$cases, `[[`, 0, "bank_id")
stopifnot(identical(bank$bank_id, subsets$bank_id), all(is.finite(bank$p_value)))

a1 <- (0 + .5) / (10 + 1); a0 <- (0 + .5) / (10 + 1)
ratio <- a0 * sqrt(a1 * (1 - a1)) / (a1 * sqrt(a0 * (1 - a0)))
n_tail <- round(3000 * ratio / (1 + ratio)); n_body <- 3000 - n_tail
stopifnot(n_tail == 1500, n_body == 1500)

eligible <- setdiff(bank$bank_id, pilot)
set.seed(2026103003)
tail <- sample(eligible[bank$p_value[eligible] <= u], n_tail)
body <- sample(eligible[bank$p_value[eligible] > u], n_body)
set.seed(2026103004)
order <- sample(c(tail, body))

dir.create(packet_dir, recursive = TRUE)
cases <- lapply(seq_along(order), function(i) {
  j <- order[i]
  id <- substr(digest::digest(paste0("sol-main-", j), algo = "sha256"), 1, 20)
  path <- file.path(packet_dir, paste0(id, ".csv"))
  write.csv(data[as.integer(strsplit(subsets$rows[j], " ")[[1]]), , drop = FALSE], path, row.names = FALSE)
  list(position = i, case_id = id, bank_id = j, stratum = if (j %in% tail) "tail" else "body",
       bank_p = bank$p_value[j], packet = path, packet_sha256 = digest::digest(file = path, algo = "sha256"))
})
jsonlite::write_json(list(u = u, n_tail = n_tail, n_body = n_body, planning_rates = c(a1 = a1, a0 = a0),
                          draw_seed = 2026103003, order_seed = 2026103004, excluded_pilot_bank_ids = pilot,
                          cases = cases),
                     "main-roster.json", auto_unbox = TRUE, digits = I(15), pretty = TRUE)
cat(length(cases), "cases:", n_tail, "tail,", n_body, "body\n")
