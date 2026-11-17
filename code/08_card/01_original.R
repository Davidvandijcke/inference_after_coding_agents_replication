# Builds the agents' data.csv from Card's public.dat and checks it byte for byte; runs the original agent's program on
# all 410 stores (its p-value u splits the bank into tail and body); fits Card and Krueger's Table 4, column (ii).
#
# Inputs:  data/raw/card_krueger/public.dat, codebook
#          data/agent_runs/card_luna/packet/data.csv, programs/original-analysis.R
# Output:  data/derived/08_card/original.csv
f <- gsub("~+~", " ", sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)), fixed = TRUE)
root <- normalizePath(file.path(dirname(f), "..", ".."))
raw <- file.path(root, "data/raw/card_krueger")
runs <- file.path(root, "data/agent_runs/card_luna")
out <- file.path(root, "data/derived/08_card")
dir.create(out, recursive = TRUE, showWarnings = FALSE)

codebook <-readLines(file.path(raw, "codebook"), encoding = "latin1")
columns <- tolower(sub(" .*", "", grep("^[A-Z][A-Z_0-9]+ +[0-9]+ +[0-9]+ +[0-9.]+ ", codebook, value = TRUE)))
built <- read.table(file.path(raw, "public.dat"), header = FALSE, na.strings = ".", col.names = columns)
built$store_id <- paste(built$state, built$sheet)
tmp <- tempfile(fileext = ".csv")
write.csv(built, tmp, row.names = FALSE)
packet <- file.path(runs, "packet/data.csv")
stopifnot(identical(readBin(tmp, "raw", file.size(tmp)), readBin(packet, "raw", file.size(packet))))

data <- read.csv(packet, check.names = FALSE, stringsAsFactors = FALSE)
program <- new.env()
sys.source(file.path(runs, "programs/original-analysis.R"), envir = program)
fit <- program$analysis(data)
used <- sum(complete.cases(data[c("state", "chain", "empft", "emppt", "empft2", "emppt2")]))

# Stores with employment and starting wages in both waves, keeping permanently closed stores (status2 == 3).
change <-with(data, (empft2 + nmgrs2 + .5 * emppt2) - (empft + nmgrs + .5 * emppt))
ck <- !is.na(change) & !is.na(data$wage_st) & (!is.na(data$wage_st2) | data$status2 == 3)
ck_fit <- summary(lm(change ~ state + factor(chain) + co_owned, data = cbind(data, change)[ck, ]))$coefficients
values <- c(stores = nrow(data), estimate = fit$estimate, u = fit$p_value, stores_used = used,
            ck_estimate = ck_fit["state", "Estimate"], ck_p = ck_fit["state", "Pr(>|t|)"], ck_stores = sum(ck))
write.csv(data.frame(name = names(values), value = sprintf("%.17g", values)), file.path(out, "original.csv"),
          row.names = FALSE, quote = FALSE)
