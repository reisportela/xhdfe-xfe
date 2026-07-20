#!/usr/bin/env Rscript
## Absorbed-target allocation: a worker-invariant descriptive group gap.

library(xhdfe)
set.seed(20260719)

n_workers <- 200L
periods <- 5L
worker <- rep(seq_len(n_workers), each = periods)
group <- rep(sample(0:1, n_workers, replace = TRUE), each = periods)
experience <- rep(0:(periods - 1L), n_workers) + rnorm(length(worker), sd = 0.2)
job <- 0.3 * group + 0.2 * experience + rnorm(length(worker))
y <- (0.25 * group + 0.08 * experience + 0.5 * job +
      rep(rnorm(n_workers), each = periods) + rnorm(length(worker)))
x1 <- cbind(group = group, experience = experience)

## `group` is absorbed by worker FE. Its full coefficient is imposed at zero;
## it is not an estimated within-worker coefficient or causal mediation effect.
result <- xhdfe_gelbach(
  y, x1,
  x2_groups = list(job_covariate = job),
  fes = list(worker_fe = worker),
  vce = "cluster", cluster = worker,
  absorbed_targets = "group"
)

cat(result$estimand, "\n")
print(result$b_full_status)
print(result$delta["group", , drop = FALSE])
cat("total SE type:", result$total_se_type, "\n")
