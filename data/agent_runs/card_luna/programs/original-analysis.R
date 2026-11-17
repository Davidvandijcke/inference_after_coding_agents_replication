# Two-wave difference-in-differences analysis for the supplied store survey.
# This file only defines functions; it does not read data or run an analysis
# when sourced.

hc2_vcov <- function(model) {
  x <- stats::model.matrix(model)
  e <- stats::residuals(model)
  h <- stats::hatvalues(model)
  if (any(!is.finite(h)) || any(1 - h <= .Machine$double.eps^0.5)) {
    stop("HC2 covariance is undefined for this fitted model.", call. = FALSE)
  }
  bread <- solve(crossprod(x))
  meat <- crossprod(x, x * (e^2 / (1 - h)))
  bread %*% meat %*% bread
}

analysis <- function(data) {
  required <- c("state", "chain", "empft", "emppt", "empft2", "emppt2")
  absent <- setdiff(required, names(data))
  if (length(absent)) {
    stop("Missing required columns: ", paste(absent, collapse = ", "),
         call. = FALSE)
  }

  d <- data.frame(
    state = as.numeric(as.character(data$state)),
    chain = factor(data$chain),
    fte1 = data$empft + 0.5 * data$emppt,
    fte2 = data$empft2 + 0.5 * data$emppt2
  )
  d$change <- d$fte2 - d$fte1
  d <- d[stats::complete.cases(d), , drop = FALSE]
  d$chain <- droplevels(d$chain)

  if (!all(d$state %in% c(0, 1)) || length(unique(d$state)) != 2L) {
    stop("Complete cases must contain both state codes: 0 (PA) and 1 (NJ).",
         call. = FALSE)
  }

  fit <- stats::lm(change ~ state + chain, data = d)
  if (is.na(stats::coef(fit)[["state"]])) {
    stop("The NJ indicator is not estimable with the available state/chain overlap.",
         call. = FALSE)
  }

  vcov_hc2 <- hc2_vcov(fit)
  estimate <- unname(stats::coef(fit)[["state"]])
  se <- sqrt(vcov_hc2["state", "state"])
  statistic <- estimate / se
  p_value <- 2 * stats::pt(abs(statistic), df = stats::df.residual(fit),
                           lower.tail = FALSE)

  list(estimate = estimate, se = unname(se), p_value = unname(p_value))
}
