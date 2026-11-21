# Real orchestration teams on the experimental bridge, synthetic data only.
args <- commandArgs(TRUE)
stopifnot(length(args)==2L)
lib <- normalizePath(args[1],mustWork=TRUE)
.libPaths(c(lib,"/home/mangelo/Documents/GitHub/xhdfe/r/Rlib",.libPaths()))
library(xhdfe,lib.loc=lib)
dll <- normalizePath(getLoadedDLLs()[["xhdfe"]][["path"]],mustWork=TRUE)
stopifnot(dll==normalizePath(file.path(lib,"xhdfe/libs/xhdfe.so"),mustWork=TRUE))
set.seed(2026111502)
n <- 49152L
fes <- list(rep(seq_len(n/24L),each=24L),sample.int(83L,n,TRUE),sample.int(13L,n,TRUE))
X <- matrix(rnorm(n*3L),n,3L)
y <- drop(X %*% c(.5,-.25,.01))+sin(fes[[1]])+cos(fes[[2]])+fes[[3]]/10+rnorm(n)
rows <- list()
for(team in c(1L,2L,8L,16L,24L,48L)) {
  model <- xhdfe_fit(y,X,fes=fes,cluster=list(fes[[1]]),backend="cuda",
    threads=team,tolerance_mode="reghdfe-comparable")
  stopifnot(model$converged,model$precision_certified,model$gpu_used,model$gpu_status_code==1,
    model$absorption_method_code==8L,model$reduced_pcg_status==1L,model$threads_used==team)
  if(team>1L) stopifnot(model$parallel_workers_active>1L)
  rows[[length(rows)+1L]] <- list(requested=team,used=model$threads_used,
    active=model$parallel_workers_active,method=model$absorption_method_code,
    tier_status=model$reduced_pcg_status,gpu_used=model$gpu_used)
}
dput(list(dll=dll,rows=rows),file=args[2])
cat("REDUCED_CUDA_R_THREADS_PASS",length(rows),"\n")
