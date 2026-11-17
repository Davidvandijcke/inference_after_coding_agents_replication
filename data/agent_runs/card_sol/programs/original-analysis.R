# Reusable two-wave difference-in-differences tools; base R only.
# No data are read and no analyses are run when this file is sourced.

.did_validate <- function(data) {
  required <- c("state", "chain", "co_owned", "status2", "empft", "emppt",
                "nmgrs", "empft2", "emppt2", "nmgrs2")
  if (!is.data.frame(data)) stop("data must be a data.frame")
  absent <- setdiff(required, names(data))
  if (length(absent)) stop("Missing columns: ", paste(absent, collapse = ", "))
  for (v in required) {
    if (!is.numeric(data[[v]]) || any(!is.finite(data[[v]]) & !is.na(data[[v]])))
      stop(v, " must be numeric, with finite values or NA")
  }
  allowed <- list(state = 0:1, chain = 1:4, co_owned = 0:1, status2 = 0:5)
  for (v in names(allowed)) {
    if (any(!is.na(data[[v]]) & !data[[v]] %in% allowed[[v]]))
      stop("Invalid code in ", v)
  }
  for (v in c("empft", "emppt", "nmgrs", "empft2", "emppt2", "nmgrs2")) {
    if (any(data[[v]] < 0, na.rm = TRUE)) stop("Negative employment in ", v)
  }
  if ("store_id" %in% names(data) && anyDuplicated(data$store_id[!is.na(data$store_id)]))
    stop("store_id must identify unique stores")
  invisible(TRUE)
}

.did_prepare <- function(data, outcome, closure) {
  .did_validate(data)
  if (!closure %in% c("permanent_zero", "all_closed_zero", "respondents_only"))
    stop("Unknown closure rule")
  d <- data
  zero_status <- if (closure == "all_closed_zero") c(2, 3, 4, 5) else 3
  zero <- !is.na(d$status2) & d$status2 %in% zero_status
  for (v in c("empft2", "emppt2", "nmgrs2")) d[[v]][zero] <- 0
  # Refusals and unknown follow-up status do not supply a valid post outcome.
  unavailable <- is.na(d$status2) | d$status2 == 0
  if (closure == "respondents_only") unavailable <- is.na(d$status2) | d$status2 != 1
  for (v in c("empft2", "emppt2", "nmgrs2")) d[[v]][unavailable] <- NA_real_
  weights <- switch(outcome,
    fte = c(1, 0.5, 1), worker_fte = c(1, 0.5, 0),
    headcount = c(1, 1, 1), full_time = c(1, 0, 0),
    part_time = c(0, 1, 0), symmetric_fte = c(1, 0.5, 1),
    stop("Unknown outcome"))
  used <- which(weights != 0)
  pre <- as.matrix(d[c("empft", "emppt", "nmgrs")[used]]) %*% weights[used]
  post <- as.matrix(d[c("empft2", "emppt2", "nmgrs2")[used]]) %*% weights[used]
  change <- as.numeric(post - pre)
  if (outcome == "symmetric_fte") {
    denom <- as.numeric(pre + post)
    change <- ifelse(denom == 0, 0, 2 * change / denom)
  }
  data.frame(state = d$state, chain = d$chain, co_owned = d$co_owned,
             pre = as.numeric(pre), post = as.numeric(post), change = change)
}

.did_fit <- function(data, outcome, adjusted, closure) {
  d <- .did_prepare(data, outcome, closure)
  needed <- c("state", "change", if (adjusted) c("chain", "co_owned"))
  d <- d[stats::complete.cases(d[needed]), , drop = FALSE]
  if (length(unique(d$state)) != 2L) stop("Both states need complete observations")
  terms <- "state"
  if (adjusted && length(unique(d$chain)) > 1L) terms <- c(terms, "factor(chain)")
  if (adjusted && length(unique(d$co_owned)) > 1L) terms <- c(terms, "co_owned")
  fit <- stats::lm(stats::reformulate(terms, response = "change"), data = d)
  x <- stats::model.matrix(fit)
  if (fit$rank < ncol(x)) stop("Model is not identified: collinear regressors")
  if (fit$df.residual <= 0L) stop("Insufficient residual degrees of freedom")
  h <- stats::hatvalues(fit)
  if (any(h >= 1 - 1e-12)) stop("HC3 undefined: a store has unit leverage")
  bread <- solve(crossprod(x))
  score <- x * as.numeric(stats::residuals(fit) / (1 - h))
  covariance <- bread %*% crossprod(score) %*% bread
  estimate <- unname(stats::coef(fit)["state"])
  se <- sqrt(covariance["state", "state"])
  if (!is.finite(se) || se <= 0) stop("Nonpositive or undefined standard error")
  df <- fit$df.residual
  p <- 2 * stats::pt(-abs(estimate / se), df = df)
  critical <- stats::qt(0.975, df)
  list(estimate = estimate, se = unname(se), p_value = p,
       ci_low = estimate - critical * se, ci_high = estimate + critical * se,
       n = nrow(d), n_nj = sum(d$state == 1), n_pa = sum(d$state == 0),
       df = df, fit = fit, covariance = covariance, model_data = d)
}

# Finished main test (T08 in artifacts/all_analyses.csv):
# worker FTE = full-time workers + 0.5 * part-time workers; managers excluded.
# Regress the paired change on NJ, chain indicators, and baseline ownership.
# Permanent closures have zero post employment; other unavailable outcomes
# are not imputed. Complete cases are determined using this outcome/model.
# The returned p_value is the ordinary two-sided HC3 t p-value (residual df),
# NOT adjusted for the exploratory selection recorded in the accompanying report.
# This specification stays fixed when applied to new data; no re-selection occurs.
analysis <- function(data) {
  result <- .did_fit(data, outcome = "worker_fte", adjusted = TRUE,
                     closure = "permanent_zero")
  result[c("estimate", "se", "p_value")]
}
