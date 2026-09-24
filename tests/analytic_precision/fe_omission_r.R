# Exact FE redundancy and identified regressors with large level offsets.
args <- commandArgs(trailingOnly=TRUE)
.libPaths(c(args[1],args[2],.libPaths()))
library(xhdfe)
library(jsonlite)
backend <- args[3]
rows <- list()
check <- function(name,data,fes,weight="none",aggregation=NULL) {
  for (interface in c("matrix","formula")) {
    options <- list(backend=backend,threads=2,tol=1e-12,drop_singletons=FALSE)
    columns <- if ("x2" %in% names(data)) c("x1","x2","age") else "x1"
    beta <- if (length(columns)==1L) .75 else c(.75,-.5,0)
    if (weight!="none") {
      options$weights <- data$weight
      options$weights_type <- if (weight=="fw") "frequency" else "analytic"
    }
    if (!is.null(aggregation)) {
      options$group <- data$group;options$individual <- data$individual
      options$aggregation <- aggregation;options$dof <- "exact"
    }
    if (interface=="matrix") {
      options$y <- data$y;options$X <- as.matrix(data[columns]);options$fes <- data[fes]
      options$fit_intercept <- FALSE;estimator <- xhdfe_fit
    } else {
      options$data <- data
      options$fml <- as.formula(paste("y ~ 0 +",paste(columns,collapse=" + "),"|",paste(fes,collapse=" + ")))
      estimator <- xhdfe
    }
    result <- tryCatch({
      model <- do.call(estimator,options)
      error <- max(abs(model$coefficients-beta))
      stopifnot(error<1e-10,model$converged,model$precision_certified,
                isTRUE(model$gpu_used)==(backend=="cuda"))
      list(status="PASS",beta_error=error)
    },error=function(error) list(status="FAIL",error=conditionMessage(error)))
    rows[[length(rows)+1L]] <<- c(list(case=name,interface=interface),result)
  }
}
i <- 0:1023;a <- 2*(i%%2)-1;e <- 2*((i%/%4)%%2)-1
for (offset in c(1e6,1e10,1e12,1e14,1e15)) for (weight in c("none","aw","fw")) {
  data <- data.frame(y=.75*a+.125*e,x1=offset+a,fe=0L,weight=1+2*((i%/%8)%%2))
  check(paste("identified",offset,weight),data,"fe",weight)
}
i <- 0:4095;block <- i%/%8
data <- data.frame(x1=2*(i%%2)-1,x2=2*((i%/%2)%%2)-1,e=2*((i%/%4)%%2)-1,
                   f1=block%%7,f2=(block%/%7)%%11,f3=(block%/%77)%%5)
data$y <- .75*data$x1-.5*data$x2+.5*data$f1-.25*data$f2+.125*data$f3+.125*data$e
for (offset in c(0,1e6,1e14)) {
  data$age <- offset+3*data$f1-2*data$f2+data$f3
  check(paste("redundant_three_fe",offset),data,c("f1","f2","f3"))
}
teams <- lapply(0:39,function(p) (p+c(0L,1L,if (p%%2) c(3L,7L)))%%12)
pattern <- rep(0:39,each=8);group <- rep(seq_along(pattern),vapply(teams[pattern+1L],length,integer(1)))
individual <- unlist(teams[pattern+1L]);i <- seq_along(pattern)-1L
for (aggregation in c("sum","mean")) {
  component <- vapply(teams[pattern+1L],if (aggregation=="sum") sum else mean,numeric(1))
  base <- data.frame(x1=2*(i%%2)-1,x2=2*((i%/%2)%%2)-1,fe=0L)
  base$y <- .75*base$x1-.5*base$x2+component+.125*(2*((i%/%4)%%2)-1)
  for (offset in c(0,1e6,1e14)) {
    base$age <- component+offset;data <- base[group,]
    data$group <- group;data$individual <- individual
    check(paste("redundant_group",aggregation,offset),data,c("individual","fe"),aggregation=aggregation)
  }
}
write_json(rows,args[4],auto_unbox=TRUE,digits=NA,pretty=TRUE)
passed <- sum(vapply(rows,function(row) row$status=="PASS",logical(1)))
cat("FE_OMISSION_R",backend,passed,"OF",length(rows),"\n")
quit(status=as.integer(passed!=length(rows)))
