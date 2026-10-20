# Release validation lessons — 20 October 2026

## Source archive closure

**Class: evaluation-harness error.** The public source proposal omitted
`tools/build_macos_stata.sh`, although the archive builder and validator already
required it. The first preflight passed all native jobs and failed during source
archive assembly. Include the existing private helper with identical bytes;
its root-level `build_macos.sh` copy is generated inside the archive. Assembly
diagnostics may reuse unchanged successful native jobs; version tags rebuild
all artifacts. The assembly-reuse control distinguishes this offline helper
from the native runtime builder, which remains ineligible for reuse after a
change.

## Generator provenance is not the consumer environment

**Class: evaluation-harness error.** The newly enabled CTest run on Ubuntu passed
14 numerical/core tests, then rejected the Student-t Python consumer because
`/usr/bin/g++` differed from the compiler recorded when the reference fixtures
were generated on Athena. A reproduction using an altered producer fingerprint
fails before reaching any numerical comparison, despite identical reference
bytes and a correct numerical probe.

Retain the immutable generator, fixture and Stata-log hashes and the existing
`5e-12` numerical limits. Report a different consumer compiler/header as a
provenance note. `--verify-generator-host` remains a strict, explicit audit of
the recorded high-precision producer environment, including unavailable files.

The same inspection exposed a separate false acceptance: `--fixtures` could
point at a modified table while the consumer hashed the original table named
by the provenance record. Clearing the CLI table's p-value cells bypassed those
comparisons. Hash the bytes actually parsed and require their digest to match
the pinned table, as well as retaining every repository-file hash check.

`tests/test_student_t_provenance_contract.py`, registered in CTest, covers a
different consumer host, strict producer rejection, a forged CLI table and
incorrect numerical output. Real-probe positive and negative controls are in
the private release evidence directory's `student_t_portability/` folder.
No estimator source, tolerance or mathematical reference value changed.
