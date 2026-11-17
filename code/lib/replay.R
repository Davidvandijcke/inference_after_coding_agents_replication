# The replay correction and its interval (base R only).
#
#   replay_estimate(u, x1, b1, x0, b0)        corrected p-value and multiplier
#   replay_interval_lr(u, x1, b1, x0, b0)     two-sided 1 - gamma interval (signed likelihood ratio, Berger-Boos)
#   neyman_allocation(total, x1, b1, x0, b0)  split of a replay budget between tail and body
#
# u is the original analysis's p-value on the full data; x1 of b1 classified tail replays (p <= u) and x0 of b0
# classified body replays (p > u) re-select the analysis. The replay test at level alpha rejects when the upper
# end of the two-sided 90% interval lies below alpha. With no tail match and a body match (x1 = 0 < x0)
# replay_estimate returns p = 0; the modules report no point estimate then and use the bound.

.scalar <- function(x, name, lower = -Inf, upper = Inf, integer = FALSE) {
  if (!is.numeric(x) || length(x) != 1L || is.na(x) || !is.finite(x) ||
      x < lower || x > upper || (integer && x != floor(x)))
    stop(name, " must be a finite ", if (integer) "integer " else "number ",
         "in [", lower, ", ", upper, "]", call. = FALSE)
  invisible(x)
}

.counts <- function(x, b, stratum) {
  .scalar(b, paste0("b", stratum), 0, integer = TRUE)
  .scalar(x, paste0("x", stratum), 0, b, integer = TRUE)
}

# Corrected p-value u*a1 / (u*a1 + (1-u)*a0) from the re-selection frequencies a1 = x1/b1 and a0 = x0/b0.
# An empty stratum or a zero denominator gives p = 1 with status "unavailable".
replay_estimate <- function(u, x1, b1, x0, b0, tail_mass = u,
                            failed1 = 0L, failed0 = 0L) {
  .scalar(u, "u", 0, 1); .scalar(tail_mass, "tail_mass", 0, 1)
  .counts(x1, b1, "1"); .counts(x0, b0, "0")
  .scalar(failed1, "failed1", 0, integer = TRUE)
  .scalar(failed0, "failed0", 0, integer = TRUE)
  a1 <- if (b1) x1 / b1 else NA_real_
  a0 <- if (b0) x0 / b0 else NA_real_
  den <- tail_mass * a1 + (1 - tail_mass) * a0
  boundary <- u == 0 || u == 1 || tail_mass == 0 || tail_mass == 1
  failed <- failed1 > 0 || failed0 > 0
  available <- !boundary && !failed && b1 > 0 && b0 > 0 &&
    is.finite(den) && den > 0
  p <- if (available) tail_mass * a1 / den else 1
  list(p_value = p, multiplier = if (available && u > 0) p / u else NA_real_,
       a1 = a1, a0 = a0, denominator = den, tail_mass = tail_mass,
       status = if (available) "estimated" else "unavailable",
       reason = if (available) NA_character_ else if (failed)
         "failed_replay" else if (boundary) "threshold_boundary" else if (!b1 || !b0)
         "missing_stratum" else "zero_denominator",
       failed1 = failed1, failed0 = failed0)
}

.cp <- function(x, b, tail_error) {
  c(lower = if (!x) 0 else stats::qbeta(tail_error, x, b - x + 1),
    upper = if (x == b) 1 else stats::qbeta(1 - tail_error, x + 1, b - x))
}

# Neyman allocation of a replay budget from pilot counts; the pilot only sets the allocation.
neyman_allocation <- function(total, x1, b1, x0, b0, available1 = total,
                              available0 = total, minimum = ceiling(sqrt(total))) {
  .scalar(total, "total", 2, integer = TRUE)
  .counts(x1, b1, "1"); .counts(x0, b0, "0")
  .scalar(minimum, "minimum", 1, integer = TRUE)
  .scalar(available1, "available1", b1, integer = TRUE)
  .scalar(available0, "available0", b0, integer = TRUE)
  lo <- max(minimum, b1, total - available0)
  hi <- min(available1, total - max(minimum, b0))
  if (lo > hi) stop("Budget/capacities cannot satisfy pilot and minimum counts")
  a1 <- (x1 + .5) / (b1 + 1); a0 <- (x0 + .5) / (b0 + 1)
  w1 <- a0 * sqrt(a1 * (1 - a1)); w0 <- a1 * sqrt(a0 * (1 - a0))
  n1 <- max(lo, min(hi, round(total * w1 / (w1 + w0))))
  c(tail = n1, body = total - n1)
}

# Signed LR / Berger--Boos, version 1.
.lr_method <- "signed_lr_berger_boos_cell_hull_v1"

.lr_xlog <- function(x, p) {
  out <- x * log(p)
  out[x == 0] <- 0
  out
}

# Ray coefficients are normalized to avoid infinite risk ratios.
.lr_fit <- function(i, n1, j, n0, t, w) {
  a <- t * (1-w); b <- (1-t) * w
  scale <- max(a,b); c1 <- a/scale; c0 <- b/scale
  k <- i+j
  A <- c1*c0*(n1+n0)
  B <- c1*(n1+j)+c0*(n0+i)
  disc <- pmax(0,B*B-4*A*k)
  z <- ifelse(k==0,0,2*k/(B+sqrt(disc)))
  z <- pmin(1,pmax(0,z))
  p1 <- c1*z; p0 <- c0*z
  ll <- .lr_xlog(i,p1)+.lr_xlog(n1-i,1-p1)+
    .lr_xlog(j,p0)+.lr_xlog(n0-j,1-p0)
  list(a1=p1,a0=p0,loglik=ll)
}

.lr_stat <- function(i,n1,j,n0,t,w) {
  # g(0,0) is undefined, but D=0 on every ray: define S=0 there.
  if (t==0) return(ifelse(i==0,0,Inf))
  if (t==1) return(ifelse(j==0,0,-Inf))
  fit <- .lr_fit(i,n1,j,n0,t,w)
  sat <- .lr_xlog(i,i/n1)+.lr_xlog(n1-i,1-i/n1)+
    .lr_xlog(j,j/n0)+.lr_xlog(n0-j,1-j/n0)
  direction <- sign((1-t)*w*(i/n1)-t*(1-w)*(j/n0))
  out <- direction*sqrt(pmax(0,2*(sat-fit$loglik)))
  out[i+j==0 | direction==0] <- 0
  out
}

# For ALL t in [lo,hi], enlarge the inclusive event using monotonicity of
# S_t(y) in t. S is increasing in i and decreasing in j, enabling row cuts.
.lr_event <- function(x1,n1,x0,n0,lo,hi,w,side,stat_guard) {
  greater <- side=="greater"
  obs <- .lr_stat(x1,n1,x0,n0,if(greater) hi else lo,w)
  tt <- if(greater) lo else hi
  j <- 0:n0
  predicate <- function(i) {
    s <- .lr_stat(i,n1,j,n0,tt,w)
    # Include equal infinities and all numerically unresolved comparisons.
    z <- if(greater) s >= obs-stat_guard else s <= obs+stat_guard
    z[is.na(z)] <- TRUE
    z
  }
  left <- rep(-1,n0+1); right <- rep(n1+1,n0+1)
  while(any(right-left>1)) {
    mid <- floor((left+right)/2)
    yes <- predicate(pmin(n1,pmax(0,mid)))
    active <- right-left>1
    if (greater) {
      right[active & yes] <- mid[active & yes]
      left[active & !yes] <- mid[active & !yes]
    } else {
      left[active & yes] <- mid[active & yes]
      right[active & !yes] <- mid[active & !yes]
    }
  }
  cut <- if(greater) right else left
  function(p1,p0) {
    tails <- if(greater) stats::pbinom(cut-1,n1,p1,lower.tail=FALSE)
             else stats::pbinom(cut,n1,p1)
    sum(stats::dbinom(j,n0,p0)*tails)
  }
}

# The event is monotone in counts. On each nuisance interval its probability
# is bounded above at the containing rectangle's favorable corner.
.lr_sup <- function(t,w,side,c1,c0,event,alpha,beta,max_boxes,tol,guard) {
  A <- (1-t)*w; B <- t*(1-w)
  if (A==0) {
    if (side=="greater") return(list(upper=min(1,event(c1[2],c0[1])+guard),boxes=1L,gap=0))
    if(c0[1]>0) return(list(upper=0,boxes=0L,gap=0))
    return(list(upper=min(1,event(c1[1],0)+guard),boxes=1L,gap=0))
  }
  r <- B/A
  if(!is.finite(r)) return(list(upper=1,boxes=0L,gap=1))
  if(side=="greater") {
    lo <- max(c0[1],if(r==0) {if(c1[1]>0) Inf else 0} else c1[1]/r)
    hi <- c0[2]; a1 <- function(v) pmin(c1[2],r*v)
  } else {
    lo <- c0[1]; hi <- min(c0[2],if(r==0) Inf else c1[2]/r)
    a1 <- function(v) pmax(c1[1],r*v)
  }
  if(lo>hi) return(list(upper=0,boxes=0L,gap=0))
  at <- function(v) event(a1(v),v)
  bound <- function(l,h) if(side=="greater") event(a1(h),l) else event(a1(l),h)
  ub <- bound(lo,hi); lower <- max(at(lo),at(hi),at((lo+hi)/2))
  repeat {
    upper <- min(1,max(ub)+guard)
    if(!is.finite(upper)) return(list(upper=1,boxes=length(ub),gap=1))
    if(upper+beta<alpha || lower-guard+beta>=alpha ||
       upper-lower<=tol || length(ub)>=max_boxes) break
    k <- which.max(ub); m <- (lo[k]+hi[k])/2
    if(m==lo[k] || m==hi[k]) break
    h <- hi[k]; hi[k] <- m; ub[k] <- bound(lo[k],m)
    lo <- c(lo,m); hi <- c(hi,h); ub <- c(ub,bound(m,h))
    lower <- max(lower,at(m))
  }
  list(upper=upper,boxes=length(ub),gap=max(0,upper-lower+guard))
}

# Search ALL cells to the left/right of the returned outward endpoint.
# A cell is discarded only when its uniform upper p bound is < alpha.
# Once the earliest retained cell is found, later cells cannot affect the hull.
.lr_hull_edge <- function(bound,side,alpha,tol,max_cells) {
  stack <- list(c(0,1)); seen <- 0L; excluded <- list()
  repeat {
    if(!length(stack)) return(list(endpoint=if(side=="greater") 1 else 0,
      evaluations=seen,reason="all_excluded",retained=NULL,excluded=excluded))
    cell <- stack[[length(stack)]]; stack[[length(stack)]] <- NULL
    if(seen>=max_cells) return(list(endpoint=if(side=="greater") cell[1] else cell[2],
      evaluations=seen,reason="cell_budget",retained=cell,excluded=excluded))
    upper <- bound(cell[1],cell[2]); seen <- seen+1L
    if(is.finite(upper) && upper<alpha) {
      excluded[[length(excluded)+1L]] <- c(cell,upper); next
    }
    m <- mean(cell)
    if(diff(cell)<=tol || m==cell[1] || m==cell[2])
      return(list(endpoint=if(side=="greater") cell[1] else cell[2],
        evaluations=seen,reason="unresolved_cell",retained=cell,excluded=excluded))
    children <- list(c(m,cell[2]),c(cell[1],m))
    if(side=="less") children <- rev(children)
    stack <- c(stack,children)
  }
}

# Interval for the corrected p-value: inverts one-sided signed-likelihood-ratio tests, each at gamma/2, maximizing
# over the nuisance parameter within a Clopper-Pearson rectangle of error beta (Berger-Boos).
replay_interval_lr <- function(u,x1,b1,x0,b0,gamma=.05,tail_mass=u,
  design="fixed",failed1=0L,failed0=0L,beta=gamma/20,root_tol=1e-5,
  nuisance_tol=1e-5,max_boxes=1024L,max_cells=512L,
  numerical_guard=1e-10,stat_guard=1e-7) {
  if(!identical(design,"fixed")) stop("LR intervals require fixed replay counts")
  .scalar(gamma,"gamma",.Machine$double.eps,1-.Machine$double.eps)
  .scalar(beta,"beta",.Machine$double.eps,gamma/2)
  if(beta>=gamma/2) stop("beta must be less than gamma/2")
  .scalar(root_tol,"root_tol",.Machine$double.eps,1)
  .scalar(nuisance_tol,"nuisance_tol",0,1)
  .scalar(max_boxes,"max_boxes",1,100000,integer=TRUE)
  .scalar(max_cells,"max_cells",1,100000,integer=TRUE)
  .scalar(numerical_guard,"numerical_guard",0,gamma/4)
  .scalar(stat_guard,"stat_guard",0,1)
  est <- replay_estimate(u,x1,b1,x0,b0,tail_mass,failed1,failed0)
  finish <- function(p,status,diagnostics=list(),raw=p,confidence=NULL) {
    names(p) <- names(raw) <- c("lower","upper")
    list(p_interval=p,multiplier_interval=p/if(u>0) u else 0,
      raw_directional_endpoints=raw,estimate=est,status=status,
      reason=est$reason,gamma=gamma,beta=beta,method=.lr_method,
      nuisance_confidence=confidence,diagnostics=c(diagnostics,
        list(floating_point_certified=FALSE,numerical_guard=numerical_guard,
          stat_guard=stat_guard,root_tol=root_tol,nuisance_tol=nuisance_tol)))
  }
  if(est$status!="estimated") {
    z <- finish(c(0,1),"vacuous")
    if(u==0) z$multiplier_interval <- c(lower=0,upper=Inf)
    return(z)
  }
  if(min(tail_mass,1-tail_mass)<.Machine$double.xmin^.25) {
    z <- finish(c(0,1),"numerically_unresolved",
      list(reason="extreme_tail_mass_retained"))
    return(z)
  }
  cp <- function(x,n) pmin(1,pmax(0,.cp(x,n,beta/4)+c(-1,1)*numerical_guard))
  c1 <- cp(x1,b1); c0 <- cp(x0,b0)
  boxes <- 0L; max_gap <- 0
  edge <- function(side) {
    bound <- function(lo,hi) {
      event <- .lr_event(x1,b1,x0,b0,lo,hi,tail_mass,side,stat_guard)
      z <- .lr_sup(if(side=="greater") hi else lo,tail_mass,side,
        c1,c0,event,gamma/2,beta,max_boxes,nuisance_tol,numerical_guard)
      boxes <<- boxes+z$boxes; max_gap <<- max(max_gap,z$gap)
      min(1,beta+z$upper)
    }
    .lr_hull_edge(bound,side,gamma/2,root_tol,max_cells)
  }
  lower <- edge("greater"); upper <- edge("less")
  raw <- c(lower$endpoint,upper$endpoint)
  finish(sort(raw),"computed",list(lower=lower,upper=upper,
    total_nuisance_boxes=boxes,max_supremum_gap=max_gap,
    endpoint_order_reversed=raw[1]>raw[2],
    cell_budget_exhausted=any(c(lower$reason,upper$reason)=="cell_budget")),
    raw,list(tail=c1,body=c0))
}
