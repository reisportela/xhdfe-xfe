# Synthetic matrix/formula checks with a newly linked experimental bridge.
args <- commandArgs(TRUE)
stopifnot(length(args)==4L)
lib <- normalizePath(args[1],mustWork=TRUE)
inputs <- normalizePath(args[2],mustWork=TRUE)
reference <- args[3]
output <- args[4]
.libPaths(c(lib,"/home/mangelo/Documents/GitHub/xhdfe/r/Rlib",.libPaths()))
library(xhdfe,lib.loc=lib)
library(jsonlite)
dll <- normalizePath(getLoadedDLLs()[["xhdfe"]][["path"]],mustWork=TRUE)
stopifnot(dll==normalizePath(file.path(lib,"xhdfe/libs/xhdfe.so"),mustWork=TRUE))
cases <- fromJSON(file.path(inputs,"cases.json"),simplifyVector=FALSE)
refs <- fromJSON(reference,simplifyVector=FALSE)$rows
rows <- list()
for(case in cases) {
  d <- read.csv(file.path(inputs,"fixtures",paste0(case$data,".csv")),check.names=FALSE)
  xs <- unlist(case$xs); fes <- unlist(case$fes); cls <- unlist(case$clusters)
  w <- if(is.null(case$weight)) NULL else d$w
  wt <- if(is.null(case$weight)) "analytic" else case$weight
  for(interface in c("matrix","formula")) {
    if(interface=="matrix") {
      model <- xhdfe_fit(d$y,as.matrix(d[,xs,drop=FALSE]),fes=lapply(fes,function(v)d[[v]]),
        cluster=lapply(cls,function(v)d[[v]]),weights=w,weights_type=wt,
        threads=4L,backend="cuda",tolerance_mode="reghdfe-comparable")
    } else {
      fml <- as.formula(paste("y ~",paste(xs,collapse=" + "),"|",paste(fes,collapse=" + ")))
      model <- xhdfe(fml,data=d,cluster=lapply(cls,function(v)d[[v]]),weights=w,
        weights_type=wt,threads=4L,backend="cuda",tolerance_mode="reghdfe-comparable")
    }
    stopifnot(length(coef(model))==4L,all(dim(vcov(model))==4L),model$converged,
      model$precision_certified,model$gpu_used,model$gpu_status_code==1)
    eligible <- length(fes)>=2L && is.null(w)
    if(eligible) stopifnot(model$absorption_method_code==8L,model$reduced_pcg_status==1L)
    if(!eligible) stopifnot(model$reduced_pcg_status!=1L)
    ref <- Filter(function(r)r$name==case$name && r$interface==interface,refs)[[1]]
    b <- unname(coef(model)); V <- unname(vcov(model)); se <- unname(model$se)
    omitted <- model$omitted!=0L; use <- !omitted
    rb <- unlist(ref$b); rV <- do.call(rbind,lapply(ref$V,unlist))
    rs <- vapply(ref$se,function(v)if(is.null(v)) NA_real_ else as.double(v),0.)
    db <- max(abs(b[use]-rb[use])/pmax(1,abs(rb[use])))
    scale <- sqrt(diag(rV)[use]); scale <- outer(scale,scale)
    dv <- max(abs(V[use,use,drop=FALSE]-rV[use,use,drop=FALSE])/scale)
    ds <- max(abs(se[use]/rs[use]-1))
    stopifnot(db<=1e-8,dv<=1e-8,ds<=1e-7,
      all(V[omitted,,drop=FALSE]==0),all(V[,omitted,drop=FALSE]==0),all(is.na(se[omitted])))
    rows[[length(rows)+1L]] <- list(name=case$name,interface=interface,status="PASS",
      b_error=db,V_error=dv,se_error=ds,tier_entered=eligible,
      method=model$absorption_method_code,tier_status=model$reduced_pcg_status,
      gpu_used=model$gpu_used,gpu_status=model$gpu_status_code)
  }
}
write_json(list(dll=dll,package=normalizePath(find.package("xhdfe")),rows=rows),
  output,auto_unbox=TRUE,digits=17,pretty=TRUE)
cat("REDUCED_CUDA_R_CONTRACT_PASS",length(rows),"\n")
