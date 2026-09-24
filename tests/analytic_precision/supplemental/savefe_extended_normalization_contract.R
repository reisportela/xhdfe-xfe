args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==6L)
xhdfe_library <- normalizePath(args[1],mustWork=TRUE)
dependency_libraries <- strsplit(args[2],.Platform$path.sep,fixed=TRUE)[[1L]]
dependency_libraries <- dependency_libraries[nzchar(dependency_libraries)]
dependency_libraries <- vapply(dependency_libraries,normalizePath,character(1),mustWork=TRUE)
.libPaths(c(xhdfe_library,dependency_libraries,.libPaths()))
expected_package <- normalizePath(args[5],mustWork=TRUE)
expected_dll <- normalizePath(args[6],mustWork=TRUE)
stopifnot(identical(normalizePath(find.package("xhdfe",lib.loc=xhdfe_library),mustWork=TRUE),expected_package))
library(xhdfe,lib.loc=xhdfe_library)
library(jsonlite)
backend <- args[3]
loaded_dll <- normalizePath(getLoadedDLLs()[["xhdfe"]][["path"]],mustWork=TRUE)
stopifnot(identical(loaded_dll,expected_dll))
i <- 0:191
data <- data.frame(worker=i%/%48,firm=(i%/%16)%%3,year=(i%/%8)%%2,
                   x=2*(i%%2)-1+.5*(i%/%48),z=2*((i%/%2)%%2)-1,
                   noise=.125*(2*((i%/%4)%%2)-1))
rows <- list()
for (family in c("one_fe","two_fe","three_fe","heterogeneous","pure_plus_fe","pure_only")) {
  fes <- if (family %in% c("one_fe","pure_only")) "worker" else c("worker","firm")
  if (family=="three_fe") fes <- c(fes,"year")
  mixed <- family %in% c("heterogeneous","pure_plus_fe","pure_only")
  include_worker <- !family %in% c("pure_plus_fe","pure_only")
  basis <- list()
  if (include_worker) basis <- lapply(0:3,function(g) as.numeric(data$worker==g))
  if (length(fes)>1) basis <- c(basis,lapply((if (include_worker) 1 else 0):2,function(g) as.numeric(data$firm==g)))
  if (family=="three_fe") basis <- c(basis,list(as.numeric(data$year==1)))
  if (mixed) basis <- c(basis,lapply(0:3,function(g) (data$worker==g)*data$z))
  dummies <- do.call(cbind,basis)
  data$y <- 3+.75*data$x+data$noise+if (include_worker) 2*data$worker else 0
  if (length(fes)>1) data$y <- data$y-data$firm
  if (family=="three_fe") data$y <- data$y+.5*data$year
  if (mixed) data$y <- data$y+(1+data$worker)*data$z
  for (weighted in c(FALSE,TRUE)) for (intercept in c(FALSE,TRUE)) {
    w <- if (weighted) 1+data$worker else rep(1,nrow(data))
    design <- dummies
    if (family=="pure_only" && intercept) design <- cbind(design,1)
    design <- cbind(design,x=data$x)
    reference <- lm(data$y ~ 0+design,weights=w)
    stopifnot(reference$rank==ncol(design))
    reference_b <- tail(coef(reference),1)
    reference_v <- tail(diag(vcov(reference)),1)
    model_has_constant <- include_worker || length(fes)>1 || intercept
    centered_y <- if (model_has_constant) data$y-weighted.mean(data$y,w) else data$y
    adjusted_r2 <- 1-(sum(w*residuals(reference)^2)/(nrow(data)-reference$rank))/
      (sum(w*centered_y^2)/(nrow(data)-as.integer(model_has_constant)))
    for (style in c("component","reghdfe")) for (retain in c(FALSE,TRUE)) for (interface in c("matrix","formula")) {
      Sys.setenv(XHDFE_FE_NORMALIZE=style)
      fit_args <- list(threads=2,maxiter=1000,tol=1e-8,backend=backend,
                       save_fe=retain,drop_singletons=FALSE,vcov="unadjusted")
      if (weighted) fit_args$weights <- w
      if (interface=="matrix") {
        fit_args$y <- data$y;fit_args$X <- as.matrix(data["x"]);fit_args$fes <- data[fes]
        fit_args$fit_intercept <- intercept
        if (mixed) fit_args$slopes <- list(list(fe=1L,values=data$z,include_intercept=include_worker))
        estimator <- xhdfe_fit
      } else {
        terms <- fes
        if (mixed) terms[1] <- if (include_worker) "worker[z]" else "worker[[z]]"
        fit_args$fml <- as.formula(paste("y ~",if (!intercept) "0 + x" else "x","|",paste(terms,collapse=" + ")))
        fit_args$data <- data;estimator <- xhdfe
      }
      record <- list(family=family,weighted=weighted,intercept=intercept,style=style,retain=retain,interface=interface)
      result <- tryCatch({
        model <- do.call(estimator,fit_args)
        errors <- c(beta=unname(abs(model$coefficients[1]-reference_b)),
                    variance=unname(abs(model$vcov[1,1]/reference_v-1)),
                    residual=max(abs(model$residuals-residuals(reference))),
                    adjusted_r2=abs(model$r2_a-adjusted_r2))
        if (retain) {
          fitted <- model$coefficients[1]*data$x+Reduce(`+`,model$fe_effects)
          if (intercept) fitted <- fitted+tail(model$coefficients,1)
          errors <- c(errors,reconstruction=max(abs(data$y-fitted-residuals(reference))))
        }
        passed <- errors[["beta"]]<=1e-9 && errors[["variance"]]<=1e-8 &&
          all(errors[setdiff(names(errors),c("beta","variance"))]<=1e-8) &&
          model$converged && model$precision_certified &&
          model$fe_recovery_converged && isTRUE(model$gpu_used)==(backend=="cuda")
        list(verdict=if (passed) "PASS" else "FAIL",errors=as.list(errors))
      },error=function(e) list(verdict="FAIL",error=conditionMessage(e)))
      rows[[length(rows)+1L]] <- c(record,result)
    }
  }
}
passed <- sum(vapply(rows,function(r) r$verdict=="PASS",logical(1)))
write_json(list(passed=passed,total=length(rows),rows=rows,
                package=expected_package,dll=loaded_dll),args[4],auto_unbox=TRUE,pretty=TRUE,digits=NA)
cat("SAVEFE_R",backend,passed,"OF",length(rows),"\n")
quit(status=as.integer(passed!=length(rows)))
