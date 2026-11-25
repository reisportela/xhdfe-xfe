# xhdfe 2.35.0.20261125

- Eligible CPU Comparable fits can request finer blocks in the reduced-system
  operator with Stata `reducedpcgfine(auto)` or `reduced_pcg_apply_fine="auto"`
  in Python and R. Factors from 1 to 8, memory admission and the coarse-block
  default remain visible in the diagnostics. A transient allocation failure can
  retry with coarse blocks within the existing iteration and precision contract.
- The optional reduced solver for eligible heterogeneous slopes can be selected
  per fit in Stata and R, or per array regressor in Python. Schur is available
  on CPU and supported CUDA builds. The forest and sparse direct variants are
  CPU routes. The help covers the two optional covariance and sparse-assembly
  controls, eligibility and the stored outcome.
- An eligible CPU `savefe` fit can reuse its saved-effect absorption state
  with the matching input, solver and recovery settings and compute inference
  for the current covariance request. Version-6 entries record whether the
  writer captured fixed effects, and the reader checks provenance on eligible
  legacy entries. The shared core also checks mobility-graph dimensions
  before narrowing them to node indices.
- Stata adds `readpretouch(on|off)` for its eligible Linux parallel reader.
  The default remains caller first touch. The release includes
  `xfepout` 1.14.6, rebuilt from the updated shared sources.

The Ready panel timing is specific to the tested fits and the local
configuration. The validation record gives both the individual limits and the
inherited covariance discrepancy in the synthetic CUDA slope check.
[Validation and scope](VALIDATION_2.35.0.20261125.md)

Install the Stata CPU packages from the GitHub net-install site or from its
extracted `xhdfe_xfe-stata-netinstall-site.zip` snapshot. That snapshot provides
both package descriptors and the Stata catalogue. The flat `xhdfe-ssc-*`
archive includes those descriptors and all declared files and can be used
as an offline `net install` source after extraction. Run `net get xhdfe`
from the same source to copy the two examples, or use
`ssc install xhdfe, all` when the matching version is available on SSC.
