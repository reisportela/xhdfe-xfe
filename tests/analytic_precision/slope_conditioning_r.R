# Exact dyadic model: beta=(.75,-.5), u=.125*e, with balanced sign columns.
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(args[1],args[2],.libPaths()))
library(xhdfe)
library(jsonlite)
backend <- args[3]
rows <- list()
for (n in c(1024L,8192L)) for (power in c(16L,18L,19L,20L,21L)) {
  i <- 0:(n-1L)
  delta <- 3*2^-power
  data <- data.frame(x1=2*(i%%2)-1, b=2*((i%/%2)%%2)-1,
                     e=2*((i%/%4)%%2)-1, fe=i%/%16, group=i, cl=i%/%2)
  data$x2 <- data$x1+delta*data$b
  data$y <- .75*data$x1-.5*data$x2+.125*data$e
  bread <- matrix(c(1+delta^2,-1,-1,1),2)/(n*delta^2)
  kinds <- c(if (backend=="cpu") "ols", "ordinary_fe","grouped_sum","grouped_mean")
  for (kind in kinds) for (vce in c("unadjusted","robust","cluster"))
    for (interface in c("matrix","formula")) {
      d <- data
      options <- list(backend=backend,threads=2,tol=1e-12,drop_singletons=FALSE,vcov=vce)
      K <- 2+if (kind=="ordinary_fe") n/16 else if (startsWith(kind,"grouped")) 1 else 0
      if (startsWith(kind,"grouped")) {
        d <- d[rep(seq_len(n),each=2),]
        d$individual <- rep(0:1,n)
        options$group <- d$group
        options$individual <- d$individual
        options$aggregation <- sub("grouped_","",kind)
        options$dof <- "exact"
      }
      if (vce=="cluster") options$cluster <- d["cl"]
      if (interface=="matrix") {
        options$y <- d$y;options$X <- as.matrix(d[c("x1","x2")])
        options$fit_intercept <- FALSE
        options$fes <- if (kind=="ols") NULL else d[if (kind=="ordinary_fe") "fe" else "individual"]
        estimator <- xhdfe_fit
      } else {
        options$data <- d
        options$fml <- if (kind=="ols") y~0+x1+x2 else if (kind=="ordinary_fe") y~0+x1+x2|fe else y~0+x1+x2|individual
        estimator <- xhdfe
      }
      V <- if (vce=="cluster") ((n-1)/(n-K))*((n/2)/(n/2-1))/(32*n*delta^2)*matrix(c(1,-1,-1,1),2) else n/(64*(n-K))*bread
      result <- tryCatch({
        model <- do.call(estimator,options)
        db <- max(abs(model$coefficients[1:2]-c(.75,-.5)))
        dv <- max(abs(model$vcov[1:2,1:2]-V))/max(abs(V))
        stopifnot(db<1e-10,dv<1e-8,abs(model$rss-n/64)<1e-9,
                  model$converged,model$precision_certified,isTRUE(model$gpu_used)==(backend=="cuda"))
        list(status="PASS",beta_error=db,covariance_relative_error=dv)
      },error=function(error) list(status="FAIL",error=conditionMessage(error)))
      rows[[length(rows)+1L]] <- c(list(n=n,power=power,kind=kind,vce=vce,interface=interface),result)
    }
}
write_json(rows,args[4],auto_unbox=TRUE,digits=NA,pretty=TRUE)
passed <- sum(vapply(rows,function(row) row$status=="PASS",logical(1)))
cat("SLOPE_CONDITIONING_R",backend,passed,"OF",length(rows),"\n")
quit(status=as.integer(passed!=length(rows)))
