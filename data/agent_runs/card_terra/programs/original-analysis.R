# Difference in mean store-level changes in non-manager FTE employment.
# Part-time employees are weighted by one half. Permanently closed stores have
# zero follow-up employment; other non-interviews are not treated as zero.
# Welch's test allows the two states to have different change variances.
analysis <- function(data) {
  required <- c('state', 'status2', 'empft', 'emppt', 'empft2', 'emppt2')
  if (!is.data.frame(data) || !all(required %in% names(data)))
    stop('data must be a data frame containing: ', paste(required, collapse = ', '))
  before <- data$empft + 0.5 * data$emppt
  after <- data$empft2 + 0.5 * data$emppt2
  after[!is.na(data$status2) & data$status2 == 3] <- 0
  eligible <- !is.na(data$status2) & data$status2 %in% c(1, 3) &
    !is.na(data$state) & data$state %in% c(0, 1) &
    is.finite(before) & is.finite(after)
  change <- after[eligible] - before[eligible]
  state <- data$state[eligible]
  nj <- change[state == 1]
  pa <- change[state == 0]
  if (length(nj) < 2 || length(pa) < 2)
    stop('At least two complete stores are required in each state.')
  test <- stats::t.test(nj, pa, var.equal = FALSE)
  list(estimate = unname(mean(nj) - mean(pa)),
       se = unname(sqrt(stats::var(nj)/length(nj) + stats::var(pa)/length(pa))),
       p_value = unname(test$p.value))
}
