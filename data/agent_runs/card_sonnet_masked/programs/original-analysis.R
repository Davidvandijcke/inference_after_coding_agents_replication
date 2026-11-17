# Difference-in-differences estimate of the effect of a state minimum-wage
# increase on fast-food employment (two-wave store survey).
#
# Main specification (fixed on design grounds, not on results):
#   outcome  = full-time-equivalent (FTE) employment
#            = managers + full-time + 0.5 * part-time
#   sample   = stores with observed wave-1 FTE and either observed wave-2 FTE
#              or a permanent closure (status_w2 == 3; recorded employment 0)
#   estimand = DiD = (change in FTE, treated state) - (change in FTE, comparison)
#              estimated as the treated_state coefficient in
#              lm(fte_w2 - fte_w1 ~ treated_state), which is the two-period DiD.
#   SE       = heteroskedasticity-robust (HC1). Treatment varies only at the
#              state level (one treated, one comparison state), so cluster-robust
#              inference by state is not available; see limitations in REPORT.md.

library(sandwich)

fte_calc <- function(ft, pt, mg, pt_weight = 0.5) mg + ft + pt_weight * pt

prepare_did <- function(data) {
  d <- as.data.frame(data)
  d$fte_w1 <- fte_calc(d$fulltime_w1, d$parttime_w1, d$managers_w1)
  d$fte_w2 <- fte_calc(d$fulltime_w2, d$parttime_w2, d$managers_w2)
  # Permanently closed stores: employment is zero after closure.
  closed <- !is.na(d$status_w2) & d$status_w2 == 3
  d$fte_w2[closed] <- 0
  d$d_fte <- d$fte_w2 - d$fte_w1
  d
}

analysis <- function(data) {
  d <- prepare_did(data)
  d <- d[!is.na(d$d_fte) & !is.na(d$treated_state), ]
  fit <- lm(d_fte ~ treated_state, data = d)
  V <- sandwich::vcovHC(fit, type = "HC1")
  est <- unname(coef(fit)["treated_state"])
  se <- unname(sqrt(V["treated_state", "treated_state"]))
  df <- nrow(d) - 2
  p <- 2 * pt(abs(est / se), df = df, lower.tail = FALSE)
  list(estimate = est, se = se, p_value = p)
}
