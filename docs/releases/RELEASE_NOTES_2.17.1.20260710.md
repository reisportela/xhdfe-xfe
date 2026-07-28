# xhdfe 2.17.1.20260710 — net-install plugin load fix (default-Unix)

Date: 2026-07-10

This is an ado-only maintenance release. It fixes a plugin load failure that
affected net-install users on a default Unix Stata install and changes no
estimand, default output, numerical result, convergence target, or compiled
artifact from 2.17.0.

## Fix

- `xhdfe`, `xfe`, `xhdfeakm`, `xhdfeconnected` and `xhdfegelbach` failed with

  ```
  xhdfe.plugin could not be loaded from ~/ado/plus/x/xhdfe.plugin      (r(601))
  ```

  when installed via `net install` on a default Unix Stata, where
  `c(sysdir_plus)` is `~/ado/plus`.
- Root cause: the loader derived the plugin path from `findfile` and passed it
  to `program ..., plugin using()`, which does not expand a leading `~` — unlike
  `confirm file`, which does, so the existence check passed but the load failed
  with file-not-found.
- The loader now expands a leading `~` to an absolute path before the file
  check, the stale-path guard and the load.

## Scope

- Ado-only. The compiled plugins, the C++/CUDA core, and the Python and R
  packages are byte-for-byte unchanged from 2.17.0.

## Validation

- Stata certification 26/26.
- The tilde `PLUS` scenario reproduced fixed: load `r(601)` → rc 0 with the
  patched ado; a real GPU run on the installed CUDA plugin reported
  `e(gpu_used)==1`, `e(gpu_backend)=="cuda"`, `e(gpu_status)=="used"`.

## Update

```
net install xhdfe, from("https://raw.githubusercontent.com/reisportela/xhdfe-xfe/gh-pages/stata") replace
```
