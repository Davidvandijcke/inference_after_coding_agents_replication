# Difference-in-differences analysis of New Jersey's April 1992 minimum-wage
# increase on fast-food employment (two-wave store survey, NJ treated, PA control).
#
# Main analysis (chosen from the study design, before comparing p-values):
#   outcome   : full-time-equivalent (FTE) employment = full-time + 0.5 * part-time + managers
#   estimand  : DiD in mean FTE, NJ vs PA, wave 2 (Nov-Dec 1992) minus wave 1 (Feb-Mar 1992)
#   estimator : OLS of the within-store change in FTE on the NJ indicator (identical to the
#               two-period DiD with store fixed effects); HC2 heteroskedasticity-robust SE
#   sample    : stores with FTE observed in wave 1 and in wave 2. Permanently closed stores
#               (status2 == 3) are kept with wave-2 employment = 0, so closures are not
#               dropped as attrition. Temporarily closed or non-responding stores with
#               missing employment are excluded.

# FTE employment from raw counts; missing if any component is missing.
fte_employment <- function(ft, pt, mgr) {
  ft + 0.5 * pt + mgr
}

# Build the store-level analysis frame from the supplied schema.
prepare_data <- function(data) {
  d <- as.data.frame(data, stringsAsFactors = FALSE)
  fte1 <- fte_employment(d$empft, d$emppt, d$nmgrs)
  fte2 <- fte_employment(d$empft2, d$emppt2, d$nmgrs2)
  closed <- !is.na(d$status2) & d$status2 == 3   # permanently closed => zero employment
  fte2[closed] <- 0
  d$fte1 <- fte1
  d$fte2 <- fte2
  d$dfte <- fte2 - fte1
  d$nj <- as.numeric(d$state == 1)
  d
}

# OLS with HC2 robust standard errors; returns coefficient table.
ols_hc2 <- function(y, X) {
  X <- as.matrix(X)
  n <- nrow(X); k <- ncol(X)
  XtXi <- solve(crossprod(X))
  b <- drop(XtXi %*% crossprod(X, y))
  e <- drop(y - X %*% b)
  h <- rowSums((X %*% XtXi) * X)
  meat <- crossprod(X * (e / sqrt(1 - h)))
  V <- XtXi %*% meat %*% XtXi
  se <- sqrt(diag(V))
  list(coef = b, se = se, df = n - k, n = n)
}

analysis <- function(data) {
  d <- prepare_data(data)
  d <- d[is.finite(d$dfte) & !is.na(d$nj), , drop = FALSE]
  fit <- ols_hc2(d$dfte, cbind(intercept = 1, nj = d$nj))
  est <- unname(fit$coef[2])
  se <- unname(fit$se[2])
  p <- 2 * stats::pt(abs(est / se), df = fit$df, lower.tail = FALSE)
  list(estimate = est, se = se, p_value = p)
}
