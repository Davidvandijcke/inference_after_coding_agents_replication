# Separate data inspection and verification script. Run from /workspace with:
#   Rscript explore.R
# The input packet file is read here, never from analysis.R.

source("analysis.R")
dat <- read.csv("data.csv", na.strings = "NA")
cat("Input rows and columns:", nrow(dat), ncol(dat), "\n")
dat$chain <- factor(dat$chain)
dat$fte1 <- dat$empft + 0.5 * dat$emppt
dat$fte2 <- dat$empft2 + 0.5 * dat$emppt2
dat$head1 <- dat$empft + dat$emppt
dat$head2 <- dat$empft2 + dat$emppt2
dat$delta_fte <- dat$fte2 - dat$fte1
dat$delta_head <- dat$head2 - dat$head1
dat$delta_ft <- dat$empft2 - dat$empft
dat$delta_pt <- dat$emppt2 - dat$emppt
dat$delta_log_fte <- log1p(dat$fte2) - log1p(dat$fte1)
dat$state <- as.numeric(dat$state)

cat("Stores by state (0 = PA, 1 = NJ):\n")
print(table(dat$state))
cat("Wave-two status by state:\n")
print(table(dat$state, dat$status2, useNA = "ifany"))
cat("Paired FTE availability by state:\n")
paired <- complete.cases(dat[, c("state", "chain", "fte1", "fte2")])
print(table(dat$state, paired))
cat("Wave-two FTE among permanent closures (status 3):\n")
print(dat[dat$status2 == 3, c("state", "fte1", "fte2")], row.names = FALSE)

summarize_fit <- function(label, formula, subset = rep(TRUE, nrow(dat))) {
  formula <- stats::as.formula(formula)
  d <- dat[subset, , drop = FALSE]
  fit <- stats::lm(formula, data = d, na.action = stats::na.omit)
  v <- hc2_vcov(fit)
  b <- unname(stats::coef(fit)[["state"]])
  se <- sqrt(v["state", "state"])
  df <- stats::df.residual(fit)
  p <- 2 * stats::pt(abs(b / se), df = df, lower.tail = FALSE)
  ci <- b + c(-1, 1) * stats::qt(0.975, df = df) * se
  mf <- stats::model.frame(fit)
  cat(sprintf(
    "%-25s n=%3d PA=%2d NJ=%3d estimate=%8.4f SE=%7.4f CI=[%8.4f,%8.4f] p=%7.5f\n",
    label, stats::nobs(fit), sum(mf$state == 0), sum(mf$state == 1),
    b, se, ci[1], ci[2], p
  ))
  invisible(c(estimate = b, se = se, p_value = p))
}

cat("\nTwo-wave DID and sensitivity specifications (HC2, store-level t inference):\n")
main_independent <- summarize_fit("FTE, chain fixed effects", "delta_fte ~ state + chain")
summarize_fit("FTE, unadjusted", "delta_fte ~ state")
summarize_fit("FTE + baseline FTE", "delta_fte ~ state + chain + fte1")
summarize_fit("FTE, status 1 only", "delta_fte ~ state + chain", dat$status2 == 1)
summarize_fit("Headcount, unadjusted", "delta_head ~ state")
summarize_fit("Headcount, chain FE", "delta_head ~ state + chain")
summarize_fit("Headcount + baseline", "delta_head ~ state + chain + head1")
summarize_fit("Full-time employees", "delta_ft ~ state")
summarize_fit("Part-time employees", "delta_pt ~ state")
summarize_fit("Change in log(1+FTE)", "delta_log_fte ~ state")

cat("\nChange-score means and baseline/follow-up means, paired FTE sample:\n")
print(aggregate(cbind(fte1, fte2, delta_fte) ~ state, dat[paired, ], mean))

# Verify the reusable function agrees with the independent main-model fit.
function_result <- analysis(read.csv("data.csv", na.strings = "NA"))
stopifnot(
  identical(names(function_result), c("estimate", "se", "p_value")),
  isTRUE(all.equal(unname(function_result$estimate),
                   unname(main_independent[["estimate"]]), tolerance = 1e-12)),
  isTRUE(all.equal(unname(function_result$se),
                   unname(main_independent[["se"]]), tolerance = 1e-12)),
  isTRUE(all.equal(unname(function_result$p_value),
                   unname(main_independent[["p_value"]]), tolerance = 1e-12))
)
cat("\nanalysis(data) verification result:\n")
print(function_result)
