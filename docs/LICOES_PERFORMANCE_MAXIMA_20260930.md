# Lições do ramo de performance máxima — 30set2026

Ramo local `perf/max-performance-20260929` (worktree
`build-worktrees/perf-max-20260929`), registo completo em
`Verifications/perf_max_20260929/PLANO_PERFORMANCE_MAXIMA_20260929.md`.
Cada entrada segue a classificação obrigatória: erro demonstrado do
estimador / recusa indevida do verificador / erro do harness de avaliação /
cobertura em falta.

## Um harness de identidade bit a bit precisa de mantissas completas

**Classe:** erro do harness de avaliação.

A primeira versão de `Verifications/perf_max_20260929/harness/packed_fused_bench.cpp`
comparava os núcleos Gauss-Seidel fundidos com os núcleos de duas passagens
sobre o `[y|X]` real do `ready` (salário em log, `age_sq`, `tenure`,
`tenure_sq`, `educ`) e combinações `0,75·a + 0,125·b`. Reportou "0 entradas
diferentes" para uma implementação cujas somas por nível eram sequenciais,
e a mesma implementação deu Δb 6,3e-16 e ΔV 7,6e-14 de ponta a ponta. Os
regressores são inteiros e as combinações preservam mantissas curtas; uma
soma de mantissas curtas é exacta em qualquer ordem, pelo que o teste não
distinguia associações diferentes.

**Correção:** entradas aleatórias com 53 bits de mantissa (`full_mantissa`,
derivada da posição) e um contador de controlo que conta as médias do núcleo
legado que não são a soma sequencial das suas linhas — no `ready`, S = 5,
`w_id`: 361 367 de 25 373 395. Um harness deste tipo só é evidência quando
esse contador é grande. Reprodutor: `h4/kernel_blocks_ready.log`.

## As somas por nível do núcleo contíguo não são sequenciais

**Classe:** facto do binário 2.28.1, não erro. Registado para que ninguém
volte a supor a associação.

`packed_demean_grouped<S,false>` (fe_absorption.cpp) é compilado com
`-ffast-math`; o GCC vectoriza a soma de um nível **entre linhas** (blocos de
8 linhas, 5 acumuladores zmm em S = 5) e usa outra ordem no resto escalar.
O núcleo CSR (`Random = true`) e o caso S = 8 (uma linha por registo) ficam
sequenciais. Consequência: qualquer reescrita "exacta" das somas por nível
muda os bits face a 2.28.1; para manter a identidade, a forma fundida por
blocos (H4) chama o próprio núcleo compilado sobre sub-intervalos de níveis,
o que dá os mesmos números por construção. Confirmado em 330 comparações
(`h4/kernel_blocks_sample10.log`, `h4/kernel_blocks_ready.log`).

## Achados pré-existentes em 2.28.1 CUDA, não tocados

**Classe:** por classificar (fora do âmbito deste ramo; ficam registados
com reprodutor).

- C++/Python, CUDA, `reghdfe-comparable`: três das 23 bases do núcleo são
  recusadas com "N1: original-design normal equation failed at regressor 1
  … no estimates returned; the single N1 refinement was exhausted", tanto
  no braço de referência 2.28.1 como nas candidatas (`h3_matrix_cuda/`,
  `final_matrix_cuda/`): `pf_difficult_10m_3fe` (defect 7,20e-01, limite
  1,87e-01), `pyfixest_difficult_1m_3fe` (3,34e-01, limite 1,96e-02) e
  `directors` (9,54e-02, limite 2,58e-02). Em CPU as mesmas células
  convergem (MLSMR, 37 / 23 / 21 iterações) e concordam com o oráculo; em
  CUDA `xhdfe-fast` também convergem. É uma recusa do verificador N1 sobre
  a solução CUDA comparable; falta classificar entre recusa indevida e erro
  demonstrado com um oráculo explícito sobre a solução CUDA. Não foi tocado
  neste ramo.
- Bateria `tests/analytic_precision` em CUDA com o módulo 2.28.1 congelado:
  `laplacian_sum__native__xhdfe__auto__strict__cuda` termina em
  ESTIMATION_ERROR (`battery_base_cpu_cuda/`); as restantes 312 linhas PASS
  e 97 "unsupported" esperadas.
- `tests/test_cuda_accuracy_retry_fail_closed` falha 4 vezes em 12 com o
  módulo sem alterações: instável, erro do harness, não da candidata.

## Optimizações para o grande custam nos pequenos: três portões de despacho

**Classe:** regressão de tempo (não de resultado) detectada pela matriz
completa; correcção por despacho dependente dos dados, não da equipa.

A matriz de 23 bases mostrou três células pequenas mais lentas com a
candidata H4 (`directors` +28 %, `enron` +18 %, ambos MLSMR comparable),
invisíveis nas 9 bases grandes onde as hipóteses foram medidas. Causas e
portões:

- lockstep: um lote resolvia as três lanes de um domínio de 45 000 níveis
  em sequência num só membro (`enron`); agora um lote por lane.
- Gauss-Seidel fundido: um buffer de 750 KB dava 2 blocos para 16 membros
  e mais regiões aninhadas do que trabalho (`directors`); agora só ≥ 64 MiB.
- fatorização approx-Cholesky em arena: blocos de 2^22 posições por factor
  e leitura mais lenta pelas lanes em 3 645 factores minúsculos
  (`directors`); agora só ≥ 65 536 arestas.

Lição: qualquer hipótese de performance tem de ser medida também nas bases
pequenas do núcleo antes de ser dada por validada; a bissecção pelos
módulos congelados (`build_h1_frozen`, …) e pelos interruptores de ambiente
localizou cada causa em minutos. `XHDFE_MLSMR_BUILD=reference` não repõe
o desempenho de 2.28.1 em grafos pequenos porque partilha o construtor de
domínios com a via nova.

## O plugin Stata é compilado sem FMA: um núcleo "exacto" não é um núcleo "igual"

**Classe:** erro do harness de avaliação / cobertura em falta (os núcleos
só tinham sido comparados com os legados na build `-march=native`).

`stata/tools/build-plugin.sh` compila o plugin para o x86-64 de base (sem
`-march=native`, por instabilidade observada com CUDA), logo sem a
instrução de multiplicação-adição fundida: aí um produto seguido de soma
arredonda duas vezes, enquanto `std::fma` arredonda uma. Os núcleos
`XHDFE_EXACT_FP` de H3/H4 escreviam `std::fma` para reproduzir a forma
contraída que o GCC gera com `-march=native`; no plugin davam outros bits
(Stata CPU fast: `ready` Δb 1e-16, AKM Δb 8e-14, `final_matrix_stata_plugin1adaa9e4_partial/`),
embora com uma precisão melhor. Correcção: `target_mul_add()` em
`include/hdfe/deterministic_parallel.hpp` — `std::fma` quando
`__FP_FAST_FMA` está definido (o alvo tem a instrução e o compilador
contrai), produto e soma separados caso contrário — nos núcleos que têm de
dar os bits de um núcleo de duas passagens em qualquer alvo (certificado e
passo de Irons-Tuck fundido). O módulo Python ficou byte-idêntico
(`8c8e4bd3…`); o harness `packed_fused_bench` passa também compilado sem
`-march` (`XHDFE_HARNESS_NO_MARCH=1`, `h4/kernel_blocks_sample10_packed_bench_nomarch.log`).
Nota: nesse alvo as somas do núcleo legado contíguo são vectorizadas a
duas linhas (SSE2), outra associação que a da build AVX-512 — os dois
binários 2.28.1 (Python e plugin) já não eram bit-idênticos entre si.

Depois da correcção, as células Stata de AKM e `simulated_panel` ainda
diferiam do plugin versionado em 1 ulp–4e-15, enquanto módulos Python
compilados sem `-march` (2.28.1 e candidata, `nomarch_matrix/`) eram
bit-idênticos. Um plugin 2.28.1 compilado agora a partir das fontes
versionadas (`build-stata-stage-2281/`, `f30de8c5…`) dá coeficientes
**iguais aos da candidata** nas três bases, e diferentes dos do plugin
versionado `stata/xhdfe.plugin` (`ae3bcc2d…`) em AKM e `simulated_panel`
(`final_matrix_stata/cpu_fast__fresh2281/`). Classificação: **proveniência
da build** — o plugin versionado não foi produzido com a cadeia actual
(`build-plugin.sh --linux --openmp`, GCC 11.5, sem `-march`); a candidata
é bit-idêntica à 2.28.1 compilada da mesma forma. Fica para o utilizador
apurar com que cadeia foi produzido o plugin versionado antes de o
substituir.

## Custo de regiões paralelas neste host

**Classe:** facto de ambiente, para desenho.

Com carga ≥ 10 e equipa de 48, uma região vazia custa ≈ 11 ms
(`harness/region_latency_probe.cpp`); uma região aninhada de um membro custa
0,44 µs (equipa 1), 1,6 µs (16), 2,1 µs (24), 3,9 µs (48). Um núcleo que abre
uma região por unidade de trabalho tem de ter unidades de centenas de
microssegundos; em H2 o número de regiões por iteração passou de 36 para 8,
e em H4 os blocos têm ≈ 512 KiB.
