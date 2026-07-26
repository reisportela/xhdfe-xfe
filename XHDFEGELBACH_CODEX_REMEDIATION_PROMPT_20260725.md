# Prompt para o Codex — remediação final pré-freeze do `xhdfegelbach`

Cola integralmente a secção **Codex prompt** no Codex.

**Contexto.** A auditoria final independente de 25jul2026
(`Verifications/claude_xhdfegelbach_final_audit_20260725/`) terminou em
**CONDITIONAL GO funcional** e **NO-GO de release**. Quatro workstreams
independentes convergiram na mesma classe de defeito: **incerteza subestimada
apresentada como válida**. Este prompt fecha exatamente as condições do GO
funcional. A tranche de release (versões, ficheiros untracked, sync público,
core-23) fica **explicitamente fora** deste prompt.

**Porque é uma tranche pequena.** Quatro dos cinco itens são pós-processamento
nos wrappers — **não tocam em `src/`, `include/`, `python/`, `r/xhdfe/src/` nem
`stata/src/`**, logo não exigem rebuild do `.so` nem do `.plugin` e não abrem
superfície de regressão numérica. Só o P1-1 (α/2 do gate de regularidade) mexe
em C++, e por isso vai deliberadamente em último lugar.

**A cura já existe no produto.** O pairs bootstrap que já acompanha o comando
está bem calibrado exatamente nas células onde o intervalo analítico falha
(91–98% de cobertura contra 40–64%). Os P0 são *detetar, rotular e encaminhar* —
não é inferência nova. Em particular, **não** se implementam conjuntos de
Fieller / weak-inference, que continuam declarados fora de âmbito.

**Decisões de desenho já tomadas pela auditoria** (o Codex não as deve reabrir):
limiares `t_den >= 3.0` e `x1_fe_collinear_ratio <= 0.35`, reutilização do
template `include_other` existente, e o diagnóstico de que o loop de `tokenize`
do `labels()` está correto — o defeito é o `string asis`.

---

## Codex prompt

You are the **implementation engineer** for the `xhdfegelbach` feature of the
xhdfe package. Work in `/home/mangelo/Documents/GitHub/xhdfe` (private repo,
branch `main`, HEAD `04cdaeea`, working tree dirty — the Gelbach tranche is
uncommitted and must stay that way unless told otherwise).

Read `AGENTS.md` and `CLAUDE.md` completely and obey them — especially the
Global Non-Regression Requirement, the Two-Repository Alignment rules, the
Versioning and Release Dates rules, the Release/NDEBUG build gates, the local
`sm_90` CUDA policy and the OpenMP plugin guardrail.

You are not the auditor and you are not defending the previous implementation.
Treat every finding below as a **hypothesis to reproduce before you fix it**.
The audit was adversarial and thorough, and it self-corrected one of its own
findings mid-flight (see "Known auditor self-correction"). If you cannot
reproduce a finding, say so with evidence and do **not** "fix" it.

### 1. Read first

```text
Verifications/claude_xhdfegelbach_final_audit_20260725/
├── CLAUDE_XHDFEGELBACH_FINAL_INDEPENDENT_AUDIT_20260725.md   <- sections 7, 16, 17, 18
├── PRE_RECONCILIATION_VERDICT.md
├── claim_matrix.json
├── help_audit_matrix.csv                                      <- 505 rows, 20 release-blocking
├── evidence/                                                  <- all measurements
├── repros/                                                    <- runnable reproducers
└── workstream_memos/                                          <- WS0..WS4
```

The authoritative specification is **section 18** of the main report. Evidence
lives in `evidence/`, reproducers in `repros/`.

### 2. Scope

Implement **exactly** these five items, in this order, and nothing else:

| id | item | touches C++? |
|---|---|---|
| P0-1 | weak-denominator gate for shares | no |
| P0-2 | FE-variance validity gate | no |
| P0-3 | `include_other` for `etable` | no |
| P0-4 | Stata reporting: `labels()`, bootstrap CIs, provenance example | no |
| P1-1 | regularity gate α/2 | **yes — do last** |

**Do not** add features. **Do not** implement anything the report classifies as
category 4 (separate engine) or 5 (do not implement): IV/2SLS/LATE, dynamic
panel, nonlinear/distributional, multiway or wild-cluster inference,
nonconditional recovered-FE inference, automatic `connected(largest)`, a formula
layer, or a path-dependent narrative layer. **Do not** implement Fieller or
weak-inference confidence sets — the gates route users to the existing
bootstrap instead.

**Do not** perform the release tranche: no version bumps, no `git add` of the
untracked companions, no sync to `xhdfe-xfe`, no tag, no release.

### 3. Invariants — violating any means stop and revert

1. **No point estimate, covariance entry or existing SE value changes anywhere.**
   P0-1…P0-4 add metadata, labels, warnings and one aggregated table row only.
2. Any design **outside** a gate's trigger region must be **bit-identical** to
   today. Verify with a stored pre-patch baseline, not by eye.
3. `share_tol` keeps its semantics and its `1e-12` default. The `base_fixed`
   convention and its existing label are untouched.
4. Ordinary `xhdfe`, `xfe`, `xhdfeakm`, `xhdfeconnected`, FE recovery, caches,
   OpenMP and CUDA behaviour are untouched.
5. Python, R and Stata must expose **identical** new field names, identical
   status strings and identical default thresholds.
6. Existing labels are **extended, not replaced** (`conditional_gamma0`,
   `mixed_full_observed_conditional_fe`, `regular_inference_status`, …).

### 4. Known auditor self-correction — do not re-derive it wrongly

The auditor's **first** Monte Carlo for the FE-variance finding redrew the
fixed-effect *values* each replication. That targets a **super-population**
estimand the command never claims, and produced ~28% coverage — an
**overstatement, withdrawn**.

The correct test holds FE values fixed as population parameters and redraws only
`x` and the errors (the documented random-design contract). Under that contract:

| focal variance that is between-FE | reported SE ÷ true sampling SD | coverage |
|---|---|---|
| 0.05 | 1.060 | 96.0 % |
| 0.25 | 1.097 | 96.8 % |
| 0.50 | 1.052 | 96.5 % |
| 0.75 | 0.800 | 86.5 % |
| 0.90 | 0.491 | 65.5 % |
| 0.97 | 0.260 | 40.0 % |

If you re-run this, use the fixed-FE-value design. Do not "confirm" the
withdrawn 28% figure.

---

### 5. P0-1 — Weak-denominator gate for shares

**Finding:** CONV-02 (= LEAD-01 = ECON-02 = NUM-03; prior audit GEL-A09).
The only denominator guard is the **absolute** `share_tol` (default `1e-12`).
Nothing tests `|denominator| / se(denominator)`. Measured coverage of the
nominal-95% share interval: **64.4%** at `|t_den| ≈ 1`, 74.4% at 1.21, 84.8% at
2.04, 96.4% at 15.03. The pairs bootstrap gives 91.6 / 94.0 / 95.6 / 96.4.
Stata's default table prints `Share (%) 776.398` with **no uncertainty column**
under a header asserting "full delta method", at `|t_den| = 1.61`.

**Files and functions**

```text
xhdfe/gelbach.py         _share_rows() (933-1000); tidy(): warning block
                         (1039-1055), component rows (1099-1112), total row
                         (1133-1176)
r/xhdfe/R/gelbach.R      .gelbach_share_rows() (937-991); xhdfe_gelbach_tidy()
stata/xhdfegelbach.ado   share loop (497-560); notes (777); display block;
                         return list (868, 973-977)
```

**Patch boundary**

- Compute `t_den = |denominator| / se(denominator)`:
  - `share='base'` / `'base_fixed'` → `se = sqrt(base_cov[row,row])`
  - `share='movement'` → `se = sqrt(total_cov[row,row])`
  Both matrices are **already returned** in all three frontends. In Stata,
  `BASECOV` is already in scope and already used at `xhdfegelbach.ado:533`.
- Add returned fields `share_denominator_t` and `share_interval_status` with
  values `"valid_first_order"` / `"weak_denominator_delta_method_unreliable"`,
  mirroring the existing `regular_inference_status` vocabulary.
- Threshold: **`t_den >= 3.0`** for valid. Justification: measured coverage
  0.908 at t=4.01 and 0.848 at t=2.04; 3.0 is the smallest round cut keeping
  measured coverage ≥ 0.90. Expose as `share_t_min` (Python/R) /
  `SHARETMIN(real 3.0)` (Stata). **Do not** change `share_tol`.
- On failure: keep the point share, keep the SE, suffix `share_se_type` with
  `"_weak_denominator_diagnostic_only"`, emit **one** `RuntimeWarning` /
  `warning()` / Stata `notes` entry naming `gelbach.bootstrap` /
  `xhdfegelbachbootstrap` as the calibrated alternative.
- Stata display: print the qualifier in the default table and **suppress** the
  `"Share inference: full delta method ..."` header line when the gate fails.
- `share='movement'` must be covered as well as `share='base'`.

**Tests to add**

- `repros/lead_probe_02_fieller.py` (and the weak sweep in
  `evidence/lead_probe_02b_fieller_weak.log`) as a regression fixture: the gate
  fires for every `|t_den| < 1.96` row and is silent at `|t_den| = 15.03`.
- A reduced coverage harness (reps ≥ 250) asserting bootstrap coverage ≥ 0.90 at
  `t_den ≈ 1` while the analytic path is flagged.
- Three-frontend parity on `share_interval_status`.
- `VALIDATE_GELBACH_HELP.py`: assert the new fields are documented.

**Acceptance:** gate fires at `|t_den| ≈ 1` with warning and suffixed
`share_se_type`; at `|t_den| = 15` output is **bit-identical** to pre-patch;
`r(share)`, `r(share_se)`, `r(share_defined)` numerically unchanged.

---

### 6. P0-2 — FE-variance validity gate

**Finding:** CONV-01 (= ECON-01 = NUM-01; prior audit GEL-A05 / GEL-A11).
FE blocks receive the `gamma0` (auxiliary-regression-only) variance. That is
labelled, but the label is a string, not a gate, and the module contract asserts
a random-design variance. Coverage table in §4 above. It propagates to
`total_movement` — the headline number — whenever an added FE block is present.

This is a **gate**, not an implementation of nonconditional FE inference (which
stays category 4).

**Files and functions**

```text
xhdfe/gelbach.py         decompose() result assembly (631-660 se_type block;
                         873-889 fe_total block); tidy() interval_status (1072-1084)
r/xhdfe/R/gelbach.R      corresponding se_type / fe_total / tidy paths
stata/xhdfegelbach.ado   se_type + total_se_type assembly; display; return list
```

**Patch boundary**

- `x1_fe_collinear_ratio` is **already returned** per X1 column. Add
  `fe_variance_status` ∈ `{"valid_first_order",
  "conditional_only_between_fe_dominant"}` per focal row, keyed on
  **`ratio <= 0.35`**.
  Justification (measured, random-design contract, nominal 95%):
  ratio 0.783 → coverage 0.965; 0.302 → 0.890; 0.127 → 0.620; 0.039 → 0.400.
  0.35 is the smallest round cut capturing every measured cell below 0.90.
  Expose as `fe_variance_ratio_min` / `FEVARMIN(real 0.35)`.
- Note the existing near-collinearity warn band is `≤ 1e-4` — orders of
  magnitude below where the variance problem begins. **Do not reuse that band**
  and do not change it; it is calibrated for exact absorption, not variance
  validity.
- On failure: suffix the FE-block and total `se_type` with
  `"_conditional_only_diagnostic"`, set the tidy
  `confidence_interval_status` to `"diagnostic_only_between_fe_dominant"`, emit
  **one** warning naming the bootstrap.
- **Do not alter any variance value.**

**Tests to add**

- A between-FE sweep fixture (shares 0.05/0.25/0.50/0.75/0.90/0.97) asserting the
  gate fires at 0.75 and above and is silent at 0.50 and below.
- A bootstrap-coverage regression asserting ≥ 0.93 in the gated cells
  (measured: 0.975 / 0.985 / 0.955 / 0.930).
- Three-frontend parity on `fe_variance_status`.

**Acceptance:** no numerical output changes anywhere — new metadata and labels
only; gate fires for between-FE share ≥ 0.75; ordinary `xhdfe` unaffected.

---

### 7. P0-3 — `include_other` for `etable`

**Finding:** CONV-03 (= DOC-02). Reproduced by the lead auditor. With
`keep=['a']` the `share_movement` panel displays `a = 0.10621` against a printed
`total_movement = 1.000`; the `levels` panel shows an unexplained 0.5586 gap; no
`Other`/`omitted`/`filtered` row exists. Present in Python, R and Stata. No help
text discloses it.

**Root cause, already established:** `waterfall_data()` and `coefplot()` accept
`include_other=True` and implement the aggregation at
`xhdfe/_gelbach_features.py:995-996`. **`etable()` has no such parameter at
all** — `inspect.signature` confirms it and passing it raises `TypeError`. The
`omitted` list is already computed at `_gelbach_features.py:969-970`.

**Files and functions**

```text
xhdfe/_gelbach_features.py     _table_records() (557-740); etable() (853-949)
xhdfe/gelbach.py               etable() signature (1366-1398)
r/xhdfe/R/gelbach_features.R   xhdfe_gelbach_etable()
stata/xhdfegelbachetable.ado   row emission
```

**Patch boundary** — reuse the existing template verbatim; do not design new
behaviour. Add `include_other=True` to `etable` in all three frontends. When
`keep`/`drop` omit any component, emit one aggregated row labelled
`"Other (filtered)"` carrying the **sum** of the omitted contributions, in
**every** panel (`levels`, `share_base`, `share_movement`). Its SE must come
from the summed sub-block of the joint covariance — **never** from adding
component SEs.

**Tests to add:** for `keep` ∈ {None, `[a]`, `[a,b]`} and every panel,
`sum(displayed component rows) − printed total ≤ 1e-12`, in Python, R and Stata.

**Acceptance:** the lead reproducer
(`evidence/lead_verify_DOC02_etable_identity.log`) shows `share_movement` summing
to 1.000 with the `Other` row present; unfiltered output bit-identical;
`include_other=False` reproduces today's behaviour **and** warns that the
identity is not preserved.

---

### 8. P0-4 — Stata reporting correctness (three independent defects)

**(a) DOC-01 — `labels()` collapses on ≥2 entries.** Verified end-to-end: a
two-entry spec labels the first component `"Human capital : job = Job controls"`
and leaves the second unlabelled, in every panel. Affects
`xhdfegelbachetable.ado` and `xhdfegelbachcoefplot.ado`.

> **The tokenize/gettoken loop is CORRECT** — an isolated re-run on an unquoted
> local parses both entries properly. The defect is `LABELS(string asis)`
> (`xhdfegelbachetable.ado:50`) retaining the outer quotes, so
> `tokenize ..., parse(":")` at line 103 sees one quoted token. **Strip the
> enclosing quotes before `tokenize` (or drop `asis`) and leave the loop
> unchanged.** Do not rewrite the parser.

**(b) DOC-03 — bootstrap intervals loaded and never read.**
`xhdfegelbachetable.ado:23,26` load `r(bootstrap_delta_ci)` into `BDCI'`;
`awk 'NR>26 && /BDCI/'` over the file returns nothing. Every component row
silently falls back to the analytic interval while
`xhdfegelbachetable.sthlp:53-55` promises bootstrap intervals "wherever
available". **Prefer using `BDCI'`** (Python and R already do, so this also
closes a parity gap), stamping the interval type accordingly.

**(c) DOC-04 — shipped provenance example prints an empty hash.**
`xhdfegelbach.sthlp:741-743`: an intervening `tab` clears `r()`. Move the `tab`
after the hash is displayed/stored, or store `r()` into locals first, and add a
comment stating that `r()` is volatile.

**Tests:** a two-entry `labels()` yields exactly two correct labels (reuse
`repros/lead_verify_DOC01_endtoend.do`); bootstrap-sourced rows are stamped as
bootstrap intervals; the sthlp provenance example prints a non-empty 16-character
hash.

---

### 9. P1-1 — Regularity gate α/2 (the only C++ change)

**Finding:** CONV-04 (= NUM-02). The gate is the **union of two level-α tests**
(`src/akm_kss.cpp:6480` and `:6558`), so the declared
`regularity_test_alpha = 0.05` is not the family-wise size — measured **0.098**.

This was **not independently re-derived by the lead auditor**. Reproduce it
first; if you cannot, report that and stop.

**Patch:** compare each component test at `alpha/2`. Keep
`regularity_test_alpha` as the user-facing family-wise level.

**Unchanged:** every point estimate and covariance. Only the boolean gate and
its status strings may change, and only in the direction of firing slightly less
often.

**Tests:** a simulation asserting empirical family-wise size ≈ 0.05; the existing
`beta2=0 / Gamma>>0` fixture must still yield `regular_loading_nonzero` for the
`x` row and `nonregular_not_ruled_out` for `_cons`.

---

### 10. Slices and stop/go gates

Work in small verified slices. **Do not start a slice until the previous gate is
green.**

- **Slice 1 (P0-1).** Gate: Fieller-sweep fixture shows the gate firing for every
  `|t_den| < 1.96` row and silent at 15.03; a strong-denominator run is
  bit-identical to pre-patch; three frontends agree on `share_interval_status`.
- **Slice 2 (P0-2).** Gate: between-FE sweep fires at ≥ 0.75 and is silent at
  ≤ 0.50; no numerical output changed; three-frontend agreement.
- **Slice 3 (P0-3).** Gate: for every panel and every `keep`/`drop`,
  `sum(displayed) − printed total ≤ 1e-12`; unfiltered output bit-identical;
  three-frontend agreement.
- **Slice 4 (P0-4).** Gate: two-entry `labels()` correct; bootstrap rows stamped
  as bootstrap; provenance example prints a non-empty hash.
- **HARD GATE before Slice 5.** Slices 1–4 must touch **no** file under `src/`,
  `include/`, `python/`, `r/xhdfe/src/` or `stata/src/`. Confirm with
  `git diff --name-only`. **If any compiled source appears, stop and report.**
- **Slice 5 (P1-1).** After the change: refresh the mirrors, run
  `bash tools/check_cpp_core_alignment.sh`, rebuild the default `build/`
  (Release, `-march=native`) and `build_cuda/` (Release, `sm_90`), rebuild the
  Stata plugin with `--openmp`, verify `ldd stata/xhdfe.plugin | grep libgomp`,
  and run a CPU **and** a CUDA smoke requiring `e(gpu_used)==1`,
  `e(gpu_backend)=="cuda"`, `e(gpu_status)=="used"`.
  Gate: measured family-wise size in `[0.04, 0.06]`; existing regularity
  fixtures pass.

### 11. After all slices

- Update `xhdfe/help/gelbach.md`, the four `.sthlp`, the roxygen sources and the
  regenerated `.Rd`. Document `include_other` (currently undocumented even for
  `waterfall_data` — DOC-13) and remove the unconditional identity claim. Make
  the "Deliberate boundaries" entry on nonconditional FE uncertainty agree with
  the new runtime gate.
- Run `VALIDATE_GELBACH.py`, `VALIDATE_GELBACH_ADVERSARIAL.py`,
  `VALIDATE_GELBACH_PYFIXEST_FEATURES.py`,
  `VALIDATE_GELBACH_SAMPLE_PROVENANCE.py`, `VALIDATE_GELBACH_HELP.py`,
  `VALIDATE_GELBACH_FRONTENDS.py`, the R tests and the Stata certification tests.
- **Provenance trap (SYS-06):** `gelbach._core()` (`xhdfe/gelbach.py:141`)
  consults the **unqualified** `sys.modules["py_hdfe_v11"]` first, and
  `VALIDATE_GELBACH_SAMPLE_PROVENANCE.py:37-52` *prefers*
  `build_companion_candidate_cpu/` (`a50f5c63`) over the shipped module
  (`685b4aff`, the one `import xhdfe` loads) and injects it into `sys.modules`.
  Point the validator at the **shipped** module and hash whatever you actually
  loaded before claiming any result.

### 12. Report back

State, per slice: what you changed, the stop/go evidence, the bit-identity proof
for the untriggered regions, and anything you could **not** reproduce or verify.
If you disagree with a finding, say so with evidence rather than implementing it.

---

## Fora deste prompt (tranche de release, NÃO executar agora)

1. Uma única versão + data em todos os ficheiros de produção, nos dois repos.
2. `git add` dos companions e módulos R/Python que hoje só existem como
   untracked (6 ficheiros declarados em `stata/xhdfe.pkg`, 4 funções em
   `NAMESPACE`) — publicar como está dá 404 no `net install` e parte
   `R CMD check`.
3. Rebuild do `stata/xfe.plugin` (15 h mais antigo que as suas fontes).
4. Re-sync de `r/xhdfe/src/fe_absorption.cpp` (`minor` → `schur_minor`) e
   alargar `tools/check_cpp_core_alignment.sh` a **todos** os espelhos R — hoje
   declara 8 pares e imprime "R … mirrors match" (19/20 idênticos, 1 diferente).
5. Repor uma cache `build/` em Release até `tools/check_default_builds.sh` ficar
   verde e correr a matriz core-23 × 8 variantes.
6. Instalar e exercitar o pacote R; regenerar os quatro `.Rd` escritos à mão a
   partir do roxygen.
7. Re-varrer a identidade byte-a-byte contra a nova tag.
