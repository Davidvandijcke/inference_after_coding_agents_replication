# p-values on the subset of each replay the first declaration review classified, with the original program's
# estimator (OLS on the change in employment, HC2 standard errors): p_<measure> for each employment measure with
# chain indicators; p_nj_only and p_nj_additive_chain for the replay's coded measure without and with them; s_rule
# adds its sample rule, of which only respondents_only changes the sample.
#
# Inputs:  data/agent_runs/card_luna/packet/data.csv, replays.csv.gz, choice-codes.csv;
#          data/intermediate/08_card/card-bank-subsets.csv.gz
# Output:  data/derived/08_card/choice-pvalues.csv
f <- gsub("~+~", " ", sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)), fixed = TRUE)
root <- normalizePath(file.path(dirname(f), "..", ".."))
runs <- file.path(root, "data/agent_runs/card_luna")
data <- read.csv(file.path(runs, "packet/data.csv"), check.names = FALSE, stringsAsFactors = FALSE)
subsets <- read.csv(file.path(root, "data/intermediate/08_card/card-bank-subsets.csv.gz"), stringsAsFactors = FALSE)
rows <- lapply(strsplit(subsets$rows, " "), as.integer)
r <- read.csv(gzfile(file.path(runs, "replays.csv.gz")), stringsAsFactors = FALSE)
classified <- r[r$first_review_status == "classified", c("session_id", "bank_id", "bank_p", "stratum")]
r <- merge(classified, read.csv(file.path(runs, "choice-codes.csv"), stringsAsFactors = FALSE), by = "session_id")
stopifnot(nrow(r) == nrow(classified))

# Wave-1 and wave-2 employment under each measure; other_unclear is scored as nonmanager FTE
levels <- list(
  nonmanager_fte = list(function(s) s$empft + .5 * s$emppt, function(s) s$empft2 + .5 * s$emppt2),
  full_time_employees = list(function(s) s$empft, function(s) s$empft2),
  employee_headcount = list(function(s) s$empft + s$emppt + s$nmgrs, function(s) s$empft2 + s$emppt2 + s$nmgrs2),
  manager_inclusive_fte = list(function(s) s$empft + s$nmgrs + .5 * s$emppt, function(s) s$empft2 + s$nmgrs2 + .5 * s$emppt2))
level <- function(measure) levels[[if (measure %in% names(levels)) measure else "nonmanager_fte"]]
change <- function(s, measure) { f <- level(measure); f[[2]](s) - f[[1]](s) }

hc2_p <- function(y, s, chain) {
  ok <- complete.cases(y, s$state, s$chain); y <- y[ok]; s <- s[ok, ]
  X <- if (chain) model.matrix(~ state + factor(chain), s) else model.matrix(~ state, s)
  fit <- lm.fit(X, y); B <- solve(crossprod(X)); h <- rowSums((X %*% B) * X)
  V <- B %*% crossprod(X * fit$residuals / sqrt(1 - h)) %*% B
  unname(2 * pt(-abs(fit$coefficients[2] / sqrt(V[2, 2])), nrow(X) - ncol(X)))
}

p <- t(vapply(seq_len(nrow(r)), function(i) {
  s <- data[rows[[r$bank_id[i]]], ]
  ruled <- if (r$sample_rule[i] == "respondents_only") s[!is.na(s$status2) & s$status2 == 1, ] else s
  c(vapply(names(levels), function(k) hc2_p(change(s, k), s, TRUE), 0),
    nj_only = hc2_p(change(s, r$measure[i]), s, FALSE), nj_additive_chain = hc2_p(change(s, r$measure[i]), s, TRUE),
    s_rule = hc2_p(change(ruled, r$measure[i]), ruled, TRUE))
}, numeric(7)))
# Nonmanager FTE with chain indicators is the original program.
stopifnot(max(abs(p[, "nonmanager_fte"] - r$bank_p)) < 1e-10)
colnames(p)[1:6] <- paste0("p_", colnames(p)[1:6])
out <- data.frame(r[c("session_id", "stratum", "measure", "specification", "sample_rule")], p)
out[colnames(p)] <- lapply(out[colnames(p)], function(v) sprintf("%.17g", v))
write.csv(out[order(out$session_id), ], file.path(root, "data/derived/08_card/choice-pvalues.csv"), row.names = FALSE,
          quote = FALSE)
