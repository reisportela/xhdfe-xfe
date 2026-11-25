# Serial negative control in the 2.35.0 preflight

**Demonstrated build error.** The macOS ARM64 preflight's serial negative
control for the Stata plugins failed to compile. The degrees-of-freedom
pair-loop referenced the allocation-failure test hook
`dof_pairs_fail_test()` although its declaration is compiled only with
`HDFE_USE_OPENMP`. The serial path sets `use_threads` to one, so the
parallel pair-loop cannot run, but the compiler still checks the reference.

The shared source now enables the call only in the OpenMP branch and uses
a false compile-time value in the serial branch. The OpenMP test injection
continues to exercise the exception path in the second pair. The production
OpenMP and macOS negative-control requirements remain in force.

Reproduce the serial syntax check with
`python3 -B tests/test_dof_serial_compile.py --cxx clang++`.
The check compares all five maintained source mirrors and compiles the
shared DoF translation unit plus the two Stata plugin frontends without
OpenMP.
