* The shared parity comparator must reject material mutations and retain its
* ordinary tight-tolerance behavior for numerical noise.

matrix px_b = (1, -2)
matrix pr_b = px_b
matrix colnames px_b = x1 x2
matrix colnames pr_b = x1 x2
matrix px_V = (1, 0 \ 0, 4)
matrix pr_V = px_V
matrix colnames px_V = x1 x2
matrix rownames px_V = x1 x2
matrix colnames pr_V = x1 x2
matrix rownames pr_V = x1 x2

foreach s in N df_r df_m df_a N_clust rank {
    scalar px_`s' = 10
    scalar pr_`s' = 10
}
foreach s in r2 r2_a rmse tss rss mss {
    scalar px_`s' = 1
    scalar pr_`s' = 1
}
scalar px_F = 3
scalar pr_F = 3

xcert_parity_check, name("mutation-control")
assert r(fails) == 0

matrix px_b[1, 1] = pr_b[1, 1] + 1e-6
xcert_parity_check, name("coefficient-mutation-1e-6")
assert r(fails) > 0

matrix px_b[1, 1] = pr_b[1, 1] + 1e-12
xcert_parity_check, name("coefficient-noise-1e-12")
assert r(fails) == 0
matrix px_b = pr_b

matrix px_V[1, 1] = (1 + 1e-6)^2
xcert_parity_check, name("standard-error-mutation-1e-6")
assert r(fails) > 0

matrix px_V[1, 1] = (1 + 1e-12)^2
xcert_parity_check, name("standard-error-noise-1e-12")
assert r(fails) == 0

di as result "PASS: parity comparator mutation contract"
