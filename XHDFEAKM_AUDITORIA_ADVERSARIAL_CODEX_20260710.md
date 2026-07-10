# `xhdfeakm`: auditoria adversarial, otimização e não-regressão

Data de fecho: **10jul2026**. Base privada: `eb72d324bff9`; candidato local
2.16.0 descrito abaixo. A auditoria abrange `xhdfeakm`, `xhdfeconnected`,
`xhdfegelbach` e a matriz canónica `core23 × 8` do estimador principal.

## Veredicto executivo

O candidato é **aprovado em correção, features, precisão, convergência e uso
de backend**. A medição temporal agregada do `core23 × 8` continua
**não certificada** por carga externa, mas foi aceite como caveat explícito
para a release 2.16.0: o código alterado é AKM opt-in, os probes incrementais
apontam ganhos e não se usa a matriz contaminada para alegar speedup:

- o core AKM, os front-ends Python/R/Stata e o plugin final compilam e passam
  os seus oráculos;
- a matriz `23 datasets × 8 células` fechou com **184/184 runs**, **704/704
  coeficientes**, `success=1`, mesma amostra e GPU real em **92/92** linhas
  CUDA;
- os modos `fast` ficaram estáveis a aproximadamente `1e-11` nos coeficientes
  e `3.4e-13` nos erros-padrão; as diferenças `comparable` máximas (`5.18e-7`
  e `4.52e-7`) estão dentro da dispersão histórica dos problemas mal
  condicionados e não representam perda de precisão;
- os outputs AKM antigos e novos são bit-idênticos nos probes CPU PCG de 1 e
  2 draws, inclusive `Pii`, `sigma_i`, efeitos, amostra e componentes;
- no GPU, arrays primários são bit-idênticos e o pior desvio observado num
  componente derivado foi `4.77e-18`; a suite final mediu paridade CPU/GPU
  máxima de `1.11e-16`;
- o patch reduz cópias e picos de memória materialmente, além de reduzir
  sincronizações CUDA. Os ganhos temporais incrementais observados foram
  aproximadamente `1–11%` nos casos grandes/GPU, mas a máquina esteve com
  load average `57–73` durante a matriz e chegou a `80` durante a fase Stata.
  A ausência de regressão temporal não pode ser inferida de uma repetição
  nessas condições.

O salto de ordem de grandeza que ainda falta é algorítmico: substituir o
PCG-Jacobi do Laplaciano de firmas por approx-Cholesky/CMG. A sonda já
existente neste checkout encontrou cerca de `919 → ~20` iterações num grafo
difícil. Não o ativei por defeito porque muda o caminho numérico e exige uma
validação própria antes de cumprir o contrato de precisão.

## O que existe desde 3 de julho

Não houve criação destes comandos em 3–4 de julho. O inventário Git inclusivo
é:

| comando | introdução | versão atual | superfície preservada |
|---|---|---:|---|
| `xhdfeakm` | 05jul, `dd3f079a` | 1.6.0, 10jul | match/obs, exact/JLA, prune/noprune, controls/FWL, fweights, CPU/CUDA, efeitos, SE, CI/eigen, lowess, lincom no core/Python |
| `xhdfegelbach` | 05jul, `cc4ff1a5` | 1.0.2, 10jul | grupos X2, múltiplos FE, unadjusted/robust/cluster, aweights/fweights, `gamma0`, `cov0`, IDs grandes e stale-plugin guard |
| `xhdfeconnected` | 06jul, `64b16e32` | 1.1.1, 10jul | maior leave-one-out connected set, pruning iterativo, fweights, máscara na ordem original |

Entre 6 e 9 de julho foram acrescentados SE/CI/lincom, CUDA, fweights,
lowess, factor controls, recodificação de IDs e batching. A versão 2.15.0 já
trazia a ronda de otimização documentada em
`AKM_PERF_AUDIT_20260709_FABLE.md`: ganhos de `4–35×` abaixo de cerca de 10M
linhas e batching do bloco SE/CI. Esta auditoria parte desse estado e mede o
incremento adicional, não volta a atribuir a si os ganhos anteriores.

## Alterações implementadas

### 1. Remoção de buffers e cópias grandes no CPU/core

Em `src/akm_kss.cpp`:

- `solve_K_multi` deixou de construir um `scrw` de tamanho `N` por RHS e de
  voltar a copiá-lo para o tile; agora escreve `tw/Dw` diretamente em
  `packS`. Com 5M trabalhadores e 24 lanes, elimina cerca de **960 MB** de
  staging por chamada;
- o caso comum sem fweights deixou de materializar um vetor de uns. Num painel
  de 47,5M linhas, são cerca de **380 MB** evitados;
- `sample`, `Pii`, `sigma_i` e `row_weight` passam para o resultado por move,
  depois do último consumidor, em vez de cópias integrais;
- `sqrt(match_weight)` é cacheado apenas no modo match-level que realmente o
  reutiliza. O cache é lazy: observation-level e exact simples não pagam
  memória nem raízes adicionais;
- a soma total dos Rademacher usa uma redução OpenMP `int64` exata e
  determinística abaixo de `2^53`; no limiar ou acima mantém o loop FP legado;
- o limite OpenMP por chamada passou a RAII, restaurando o estado também em
  exceções e early returns;
- `solver_iterations` inclui agora corretamente as iterações CUDA acumuladas
  depois de SE e lincom.

### 2. Menos sincronizações CUDA

Em `src/akm_kss_cuda.cu`:

- um kernel fundido atualiza `x += alpha*p` e `r -= alpha*Ap`;
- a máscara ativa só volta ao device quando muda;
- `r'r` e `r'z` são enfileirados como duas reduções CUB e regressam num único
  D2H/sync;
- `d_mdots` e o staging pinned foram dimensionados para os dois vetores de
  escalares.

O caminho é bounds-safe e ordenado no default stream. Há um trade-off
localizado: a última iteração calcula especulativamente um hadamard+reduce que
o caminho antigo evitava. Num probe artificial com `J=100`, PCG forçado e
apenas 46 iterações agregadas, o tempo warmed foi aproximadamente `0.07s →
0.09s` contra um build 2.14 mais antigo. Esse cenário não é o default — com
`J=100`, o default usa Cholesky direto — e a perda absoluta é cerca de 20 ms.
É aceitável apenas como trade-off documentado face aos ganhos de 5–11% nos
workloads GPU relevantes; poderá ser eliminado no futuro com um gate por
tamanho/iterações.

### 3. Helper Python

`xhdfe/akm.py::subsampling_diagnostic` deixa de reconstruir `kf[order]` para
cada trabalhador; os arrays ordenados são calculados uma vez. Isto remove a
complexidade/acumulação temporária acidental sem mudar os outputs.

### 4. Espelhos e artefactos

Os ficheiros C++/CUDA foram propagados byte a byte para os mirrors R, Stata e
`share/`. `tools/check_cpp_core_alignment.sh` e `git diff --check` passam.

### 5. Hardening de `xhdfegelbach`

- FEs e clusters numéricos fora de int32, ou com labels numéricos não
  inteiros, são compactados no ado para `1..N`; a transformação preserva
  exatamente as classes e a decomposição;
- o plugin valida também o intervalo antes do cast, fechando o overflow em
  chamadas internas;
- o dispatcher aplica o mesmo guard de path fail-closed usado por
  `xhdfeakm`/`xhdfeconnected`;
- a certificação Stata testa invariância de `delta`, SE, total e connected set,
  e exige `_rc=498` perante um plugin de outro checkout.

Hashes finais relevantes:

```text
src/akm_kss.cpp       2be3cf4682ace4447bd20209f9988e433072ec6852894759024ff06ea77c7cac
src/akm_kss_cuda.cu  729d8c1436519010b1ad369d20219ccecfa2fda4587725779e01da4469cf55a2
build CPU module      f794aa54d083ff46d3f03b46d4b17a44690e3106720cd89f82436530a07265e3
build CUDA module     13e531bb72d56475204557805eb8f39d4462b241c0c264baccd85cf6d71bf364
Stata plugin sm_90    8665e6c03eae5e776bf20233494bd550ec1a1ec3cf183defafd5e1a39450cead
```

O plugin final liga `libgomp.so.1` e contém apenas cubins `sm_90` (três
translation units), como exigido para esta H100.

## Validação específica AKM e companions

| gate | resultado |
|---|---|
| `VALIDATE_AKM_KSS.py --module-dir build` | **ALL CHECKS PASSED**; GPU corretamente skipped no build CPU |
| `VALIDATE_AKM_KSS.py --module-dir build_cuda` | **ALL CHECKS PASSED**; CPU/GPU max `1.11e-16`; determinismo GPU |
| exact dense, match/obs, controls | passou em todas as células |
| JLA: convergência/determinismo | passou; 1 vs 8 threads idêntico |
| fweights vs row expansion | pior desvio `2.22e-16` |
| SE, CI/eigen, lowess, lincom | passou contra os oráculos/tolerâncias existentes |
| old vs new, CPU PCG, draws 1 e 2 | hashes e componentes bit-idênticos; mesmas 123/216 iterações |
| exact observation-level old vs new | hashes/componentes bit-idênticos; mediana sem regressão (`~0.125s` vs `~0.121s`) |
| `akm_bitcheck.py`, block 1 vs 8 | **BIT-IDENTICAL** |
| Stata AKM CPU/CUDA | ambos convergiram; 3171 iterações; CUDA `gpu_used=1`; `kss_var_psi=0.15969228710455` |
| exemplos Stata | `xhdfeconnected`, `xhdfeakm` e `xhdfegelbach`: **COMPANION_STATA_EXAMPLES_OK** |
| R `test-akm.R` | 54 checks/dots, sem falhas |
| R `test-gelbach.R` | 9 checks/dots, sem falhas |
| Stata certification pós-hardening | **26/26**; relabelling de IDs exata e stale-plugin `_rc=498` |
| Gelbach vs `b1x2` vivo | todos os checks passaram; deltas/SEs até cerca de `1e-16` |
| connected/Gelbach 1M old-vs-new | mesma amostra/delta; tempos sobrepostos (`~0.25–0.35s`, `~2.4–2.7s`) |

O smoke final V08/QP do plugin recompilado também passou: CPU convergiu em
20 iterações e 84,145 s; CUDA convergiu em 77 iterações, 17,279 s e
`gpu_used=1`. O CPU está acima do intervalo calmo de 60–68 s, mas o host tinha
load médio perto de 59. Por isso este smoke aprova funcionalidade/backend,
não o gate temporal de release.

## Matriz `core23 × 8`

Artefactos locais privados usados nesta auditoria:
`benchmarks/_out/core23_akm_adversarial_20260709/`.

A matriz completa correu no snapshot imediatamente anterior ao ajuste lazy
de `m_sqrt_c`. Esse ajuste final só altera uma alocação dentro da função AKM
opt-in e é inalcançável pelo `HdfeRegressor` do core23. Depois do rebuild
final, o V08/QP CPU+CUDA voltou a validar o plugin exato que ficou no
workspace.

Gate independente do harness:

```text
CORE23_GATE_OK runs=184 coefficients=704 datasets=23 cells=8
success: 184/184
CPU:  gpu_used=0, 92/92
CUDA: gpu_used=1, 92/92
nobs: idêntico à baseline, 184/184
```

O `xhdfe.ado` recusa publicar resultados não convergidos e devolve erro; por
isso `rc=0`/`success=1` em Stata é também um gate efetivo de convergência. Em
C++, `converged=1` foi gravado em 92/92 linhas.

### Precisão candidato vs baseline de 06jul

| célula | max `|Δβ|` | max `|ΔSE|` |
|---|---:|---:|
| C++ CPU fast | `1.20e-11` | `1.57e-13` |
| C++ CUDA fast | `9.14e-12` | `1.42e-13` |
| Stata CPU fast | `1.23e-11` | `1.86e-13` |
| Stata CUDA fast | `2.02e-11` | `3.40e-13` |
| C++ CPU comparable | `1.99e-7` | `1.18e-7` |
| C++ CUDA comparable | `2.05e-7` | `8.98e-8` |
| Stata CPU comparable | `2.41e-7` | `3.11e-7` |
| Stata CUDA comparable | `5.18e-7` | `4.52e-7` |

O pior caso é `workers`, Stata CUDA comparable, e permanece dentro da banda
histórica de 01/06jul. Só seis contagens de iterações mudaram; todas são
variações já observadas em `comparable` (`workers`, `directors` e
`pf_difficult_10m_3fe`). Não houve alteração de amostra ou estimador.

### Tempos — diagnóstico, não aceitação

| célula | atual (soma, s) | baseline (s) | variação |
|---|---:|---:|---:|
| C++ CPU fast | 498,644 | 311,929 | +59,9% |
| C++ CPU comparable | 925,799 | 511,572 | +81,0% |
| C++ CUDA fast | 98,314 | 106,075 | **−7,3%** |
| C++ CUDA comparable | 125,017 | 123,559 | +1,2% |
| Stata CPU fast | 546,398 | 418,416 | +30,6% |
| Stata CPU comparable | 864,683 | 621,099 | +39,2% |
| Stata CUDA fast | 242,656 | 226,763 | +7,0% |
| Stata CUDA comparable | 261,185 | 238,011 | +9,7% |

Estas aparentes regressões CPU não são causalmente atribuíveis ao patch:

1. as alterações estão em funções AKM opt-in e numa translation unit separada;
   o `HdfeRegressor` usado pelo core23 não as chama;
2. iterações, amostra e resultados fast ficaram essencialmente idênticos;
3. até datasets minúsculos, que não exercitam o código alterado, ficaram
   2–30× mais lentos em alguns momentos;
4. o `run.log` registou load average 57–73 e a máquina tinha outros jobs Stata
   com centenas de por cento de CPU;
5. um A/B AKM de 1M linhas, sem mudar qualquer hash/iteração, oscilou
   `23s → 42s → 52s → 29s` quando um job Stata externo arrancou.

Consequentemente, a matriz aprova **features, precisão, convergência e
backend**, mas o gate temporal fica **provisório**. A release 2.16.0 foi
autorizada com esta limitação declarada; deve repetir-se `REPS=3` em A-B-B-A
numa janela calma, com atenção a `workers`, `pf_difficult_10m_3fe`,
`secondreg` e Stata CUDA comparable, sem reescrever retroativamente a evidência
desta ronda.

## Performance incremental do patch AKM

Os A/B intercalados feitos antes da subida final de load deram:

| workload | baseline | candidato | ganho indicativo |
|---|---:|---:|---:|
| CPU PCG 1M | 17,282 s | 16,707 s | 3,3% |
| GPU 1M | 5,813 s | 5,174 s | 11,0% |
| GPU 4M | 23,578 s | 22,364 s | 5,1% |
| GPU 100k + SE(100) | 2,185 s | 1,936 s | 11,4% |
| painel QP real, team=1 | 307,415 s | 301,706 s | 1,9% |
| painel QP real, 16 threads | 242,326 s | 239,036 s | 1,4% |

No QP real: 43,7M observações mantidas, 10,18M matches, 4,79M
trabalhadores, 462.513 firmas e exatamente 88.547 iterações em ambos os
binários. Os componentes coincidem até cerca de `5e-18`.

Um ganho inicial de 18% no direct-100k não se repetiu no probe final: as
medianas foram ambas aproximadamente 0,612 s. Não é citado como ganho
robusto. Em suma, esta ronda é sobretudo uma redução substancial de memória
e uma melhoria GPU pequena/moderada; o ganho temporal de ordem de grandeza
continua dependente do novo precondicionador.

## Achados adversariais nos comandos associados

Os três bloqueadores concretos encontrados durante a auditoria foram fechados
antes da release: recodificação/range-check int32 de Gelbach, stale-plugin
guard e closure do bundle público. Permanecem itens de hardening futuro:

### High/Medium

1. O contrato `xhdfeconnected → xhdfeakm, noprune` só é válido com a mesma
   amostra completa e o mesmo peso. Missing em `y`/controls ou pesos
   diferentes podem quebrar a propriedade leave-out; a ajuda deve dizê-lo.
2. Gelbach precisa de gates explícitos para pesos finitos/positivos,
   integralidade de fweights, pelo menos dois clusters, `df_full>0`, rank e
   covariância finita. Hoje `identity_gap` não certifica essas condições.
3. `xhdfe.gelbach.decompose(..., tol=...)` aceita mas ignora `tol`;
   a ajuda Gelbach foi corrigida para apontar para essa API Python; `xhdfeakm.sthlp` mistura `e()` com um comando
   `rclass`.
4. O harness encompassing não é fail-closed: a função shell mascara o RC de
   cada step, o worker Python não exige `converged`, e o worker Stata não
   grava esse campo. Esta ronda compensou-o com validação independente, mas o
   harness deve ser corrigido antes da próxima certificação.

## Próximos alvos, por retorno esperado

1. **Approx-Cholesky/CMG para o Laplaciano AKM.** Primeiro fallback apenas em
   não-convergência (bit-safe nos casos atuais), depois opt-in, e só depois
   gate automático com matriz completa. É o único candidato plausível a
   `10–100×` nos grafos difíceis.
2. **Output mask para Stata.** Sem `generate()`, o plugin calcula e copia
   arrays row-level que o ado descarta; evitar esse trabalho reduz memória e
   bandwidth sem mudar a API.
3. **Connected set-only.** Saltar a finalização e o segundo sort que só o
   estimador AKM consome.
4. **Gelbach robust/cluster streaming.** Não materializar a matriz gigante de
   scores `n × K`; agregar o meat por blocos de forma determinística.
5. **Pruning ativo compacto.** Remover atomics por match e scans repetidos de
   matches já eliminados, preservando a ordem/tie-break atual.

## Decisão de release

O candidato foi versionado como `2.16.0.20260710`, com `xhdfeakm` 1.6.0 e
`xhdfegelbach` 1.0.2. A decisão de ship assenta nos gates de correção,
precisão, convergência e backend completos, na redução de memória demonstrada,
nos A/B incrementais favoráveis e na aceitação explícita de que o host não
permitiu certificar nesta janela os tempos agregados do core23. A release não
apresenta esses tempos contaminados como melhoria.

O caminho público é `reisportela/xhdfe-xfe`: fontes e artefactos Stata de
produção são sincronizados byte a byte (exceto plugins, por desenho), a tag é
presa ao commit exato, os três builds CPU de plataforma são obrigatórios, e a
release é preparada em draft antes de publicação. O rerun temporal calmo fica
como follow-up, não como condição escondida desta decisão.

## Artefactos de reprodução

Artefactos locais privados, não publicados no repositório público:

- matriz completa: `benchmarks/_out/core23_akm_adversarial_20260709/`;
- NPZ old/new e smokes Stata: `benchmarks/_out/akm_adversarial_20260709/`;
- probe determinístico/hashes:
  `benchmarks/_out/akm_adversarial_20260709/akm_module_probe.py`;
- wrapper dos exemplos Stata:
  `benchmarks/_out/akm_adversarial_20260709/stata_companion_examples.do`;
- auditoria matemática anterior:
  `AUDIT_ADVERSARIAL_AKM_CONNECTED_GELBACH_PYTWOWAY_20260709.md`;
- perfil/ronda 2.15.0 anterior: `AKM_PERF_AUDIT_20260709_FABLE.md`.

Artefactos reprodutíveis publicados com a release 2.16.0:

- `VALIDATE_AKM_KSS.py`;
- `tests/stata/part1/companions.do` e `tests/stata/testall.do`;
- testes R de AKM/Gelbach em `r/xhdfe/tests/testthat/`;
- `RELEASE_NOTES_2.16.0.20260710.md`;
- release assets públicos gerados pelo workflow `release.yml`.
