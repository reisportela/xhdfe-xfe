# Analytic beta=0.7, with a correct fit before each adversarial grouped fit.
args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(args[1], args[2], .libPaths()))
library(xhdfe)
library(jsonlite)
backend <- args[3]
n <- if (length(args)>=5L) as.integer(args[5]) else 96L
stopifnot(length(n)==1L,!is.na(n),n>=4L)
T <- diag(n)
T[(col(T)-row(T)) %in% c(1L,3L)] <- 1
M <- matrix(0, 2L*n, 2L*n)
M[seq_len(n), seq_len(n)] <- T
M[n+seq_len(n), n+seq_len(n)] <- T
pattern <- rep(seq_len(2L*n), each=16L)
within_rep <- rep(0:15, 2L*n)
a <- 2*(within_rep %% 2)-1
b <- 2*((within_rep %/% 2) %% 2)-1
edges <- which(M[pattern,] != 0, arr.ind=TRUE)
group <- edges[,1]
individual <- edges[,2]
wave <- c(1,-1,1,rep(0,n-3L))
for (i in 4:n) wave[i] <- -wave[i-1]-wave[i-3]
wave <- wave/max(abs(wave))
rows <- list()

for (aggregation in c("sum","mean")) {
  scale <- if (aggregation == "sum") rep(1,2L*n) else rowSums(M)
  design <- M/scale
  for (weight in c("","analytic","frequency")) {
    w <- if (weight == "") rep(1,length(pattern)) else 1+(pattern-1L) %% n %% 3L
    pattern_weight <- if (weight == "") rep(1,n) else 1+(0:(n-1L)) %% 3L
    weak <- c(wave,-wave)*scale/rep(pattern_weight,2L)
    weak <- weak/sqrt(sum(weak^2)/2)
    good_x <- .2*a
    good_y <- .7*good_x+(design %*% (.2*cos(0:(2L*n-1L))))[pattern]+.113*b
    bad_x <- weak[pattern]+.2*a
    bad_y <- .7*bad_x+3*weak[pattern]+.113*b
    expected_rss <- sum(w*(.113*b)^2)
    if (weight == "analytic") expected_rss <- expected_rss*length(pattern)/sum(w)
    for (mode in c("reghdfe-comparable","strict-residual")) {
      options <- list(fes=list(individual,rep(0L,length(group))),
        group=group,individual=individual,aggregation=aggregation,backend=backend,
        threads=2,drop_singletons=FALSE,maxiter=100000,
        tolerance_mode=mode,tol=if (mode=="strict-residual") 1e-12 else 1e-8)
      if (weight != "") {
        options$weights <- w[group]
        options$weights_type <- weight
      }
      result <- tryCatch({
        good <- suppressWarnings(do.call(xhdfe_fit,c(list(y=good_y[group],X=good_x[group]),options)))
        stopifnot(good$converged,good$precision_certified,abs(good$coefficients[1]-.7)<1e-8)
        stopifnot(isTRUE(good$gpu_used)==(backend=="cuda"))
        bad <- tryCatch(suppressWarnings(do.call(xhdfe_fit,
            c(list(y=bad_y[group],X=bad_x[group]),options))),error=identity)
        if (inherits(bad,"error")) {
          stopifnot(grepl("no estimates",conditionMessage(bad),ignore.case=TRUE))
          list(status="SAFE_ERROR",error=conditionMessage(bad))
        } else {
          stopifnot(bad$converged,bad$precision_certified,
                    abs(bad$coefficients[1]-.7)<1e-8,abs(bad$rss-expected_rss)<1e-7)
          list(status="CORRECT_RESULT")
        }
      },error=function(error) list(status="FAIL",error=conditionMessage(error)))
      rows[[length(rows)+1L]] <- c(list(aggregation=aggregation,weight=weight,
                                       mode=mode,backend=backend,nodes=n,groups=length(pattern)),result)
    }
  }
}
write_json(rows,args[4],auto_unbox=TRUE,digits=NA,pretty=TRUE)
passed <- sum(vapply(rows,function(row) row$status %in% c("SAFE_ERROR","CORRECT_RESULT"),logical(1)))
cat("NEAR_NULL_R",backend,passed,"OF",length(rows),"\n")
quit(status=as.integer(passed!=length(rows)))
