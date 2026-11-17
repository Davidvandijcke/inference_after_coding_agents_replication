# Reruns each of the four originals' programs on the reference subsets its replays ran on (with the argument "all",
# on all 20,000) and checks the p-values, estimates, u and tail set against the saved bank.
#
# Inputs:  data/agent_runs/card_luna/packet/data.csv, card_sonnet_masked/packet/data.csv,
#          data/agent_runs/card_{terra,sol,sonnet,sonnet_masked}/ (programs/original-analysis.R, original.json,
#          replays.csv.gz), data/intermediate/08_card/card-bank-subsets.csv.gz, data/intermediate/09_card_agents/
# Outputs: data/derived/09_card_agents/{terra,sol,sonnet,sonnet_masked}_bank_p_rescored.csv
# Rscript writes spaces in the script path as "~+~".
script <- gsub("~+~", " ", sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)), fixed = TRUE)
root <- normalizePath(file.path(dirname(script), "..", ".."))
full <- identical(commandArgs(trailingOnly = TRUE), "all")
packets <- c(terra = "card_luna", sol = "card_luna", sonnet = "card_luna", sonnet_masked = "card_sonnet_masked")
subsets <- read.csv(file.path(root, "data/intermediate/08_card/card-bank-subsets.csv.gz"), stringsAsFactors = FALSE)
stopifnot(identical(subsets$bank_id, seq_len(nrow(subsets))))
out <- file.path(root, "data/derived/09_card_agents")
dir.create(out, showWarnings = FALSE, recursive = TRUE)

for (agent in names(packets)) {
  cohort <- file.path(root, "data/agent_runs", paste0("card_", agent))
  data <- read.csv(file.path(root, "data/agent_runs", packets[[agent]], "packet/data.csv"), check.names = FALSE, stringsAsFactors = FALSE)
  program <- file.path(cohort, "programs/original-analysis.R")
  score <- function(index) tryCatch({
    env <- new.env(parent = globalenv())
    sys.source(program, envir = env)
    z <- env$analysis(data[index, , drop = FALSE])
    c(p_value = as.numeric(z$p_value), estimate = as.numeric(z$estimate))
  }, error = function(e) c(p_value = NA_real_, estimate = NA_real_))
  ids <- if (full) subsets$bank_id else
    sort(read.csv(gzfile(file.path(cohort, "replays.csv.gz")), stringsAsFactors = FALSE)$bank_id)
  p <- do.call(rbind, parallel::mclapply(strsplit(subsets$rows[ids], " "), function(r) score(as.integer(r)), mc.cores = 2))
  bank <- data.frame(bank_id = ids, p_value = p[, "p_value"], estimate = p[, "estimate"])
  write.csv(bank, file.path(out, paste0(agent, "_bank_p_rescored.csv")), row.names = FALSE)
  saved <- read.csv(file.path(root, "data/intermediate/09_card_agents", paste0(agent, "_bank_p.csv")))
  saved <- saved[match(bank$bank_id, saved$bank_id), ]
  stopifnot(nrow(bank) == length(ids), !anyNA(saved$bank_id), !anyNA(bank$p_value), !anyNA(saved$p_value),
            max(abs(bank$p_value - saved$p_value)) < 1e-12, max(abs(bank$estimate - saved$estimate)) < 1e-12)
  u <- unname(score(seq_len(nrow(data)))["p_value"])
  stopifnot(abs(u - jsonlite::fromJSON(file.path(cohort, "original.json"))$p_value) < 1e-12)
  stopifnot(identical(which(bank$p_value <= u), which(saved$p_value <= u)))
  cat(sprintf("%-13s u = %.10f; %d subsets rescored, %d in the tail; equal to the saved bank\n", agent, u, nrow(bank),
              sum(bank$p_value <= u, na.rm = TRUE)))
}
