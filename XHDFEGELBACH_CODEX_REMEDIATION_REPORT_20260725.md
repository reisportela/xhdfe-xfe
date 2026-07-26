# Relatório de remediação final pré-freeze do `xhdfegelbach`

**Data:** 25jul2026

**Repositório:** `/home/mangelo/Documents/GitHub/xhdfe`

**Branch / HEAD de partida:** `main` / `04cdaeea`

**Especificação executada:** `XHDFEGELBACH_CODEX_REMEDIATION_PROMPT_20260725.md`

## Veredito

**GO funcional para a tranche estritamente delimitada P0-1...P1-1.**

Os cinco itens pedidos foram reproduzidos, implementados e validados na ordem
prescrita:

1. gate de denominador fraco para shares;
2. gate de validade da variância dos blocos FE;
3. `Other (filtered)` em `etable`;
4. correções de reporting Stata;
5. correção `alpha/2` do gate de regularidade.

Não encontrei evidência de alteração de estimativas pontuais, matrizes de
covariância ou erros-padrão existentes. Os P0 acrescentam exclusivamente
metadados, estados, avisos e uma linha agregada de reporting. O P1 altera apenas
as decisões booleanas do diagnóstico de regularidade.

**O release continua deliberadamente NO-GO e não foi executado.** Não houve
bump de versão/data, `git add`, sincronização com `xhdfe-xfe`, tag ou release.
Também não foi executada a matriz `core23 x 8`, nem foi reinstalado o pacote R:
essas operações pertencem à tranche de release que o prompt exclui
expressamente.

## Estado de partida, método e proteção da árvore suja

A árvore já estava extensamente dirty e continha alterações anteriores em
fontes compiladas e ficheiros de release. Por isso, um simples
`git diff --name-only` não permitia atribuir alterações a esta tranche.

Antes da primeira modificação foi criado um snapshot dos ficheiros relevantes,
com hashes. Em particular:

```text
src/akm_kss.cpp pre-P0/P1
77a0293c9041828b3c0312fd063e2ebd56808a5e248a1213fab69940f0132995
```

Durante P0-1...P0-4:

- `src/akm_kss.cpp` manteve exatamente esse hash;
- nenhum ficheiro sob `src/`, `include/`, `python/`, `r/xhdfe/src/` ou
  `stata/src/` ficou mais recente do que o manifesto pré-P0;
- as alterações foram limitadas aos wrappers, reporting, help e testes.

Só depois deste hard gate foi aplicado P1-1 aos cinco espelhos de
`akm_kss.cpp`.

## Reprodução dos achados antes da correção

| Achado | Reprodução |
|---|---|
| P0-1, share com denominador fraco | cobertura analítica 0,644 em `mean\|t_den\|=0,991`; bootstrap pairs 0,916 |
| P0-2, variância FE | undercoverage crescente nas células dominadas por variação between-FE; foi respeitado o desenho corrigido com valores FE fixos |
| P0-3, tabela filtrada | componentes visíveis não somavam ao total e não existia linha `Other` |
| P0-4(a), `labels()` | duas entradas eram colapsadas por `LABELS(string asis)` |
| P0-4(b), intervalos bootstrap | `BDCI` era carregada mas não usada nas linhas dos componentes |
| P0-4(c), proveniência | `tab` destruía `r(sample_hash)` antes da impressão |
| P1-1, regularidade | 49/500 ativações sob a nulidade, dimensão family-wise 0,098 |

Não reproduzi nem usei o valor retirado de aproximadamente 28% como evidência
do contrato relevante. O harness FE fixa os valores dos efeitos, o sinal
between-FE e a atribuição às categorias; só redesenha a covariável focal e os
erros.

## Slice 1 — P0-1: gate de denominador fraco

### Alterações

Nos frontends Python, R e Stata:

- novo limiar público `share_t_min=3.0` /
  `SHARETMIN(real 3.0)`;
- cálculo de
  `share_denominator_t = |denominator| / se(denominator)`;
- novo estado por linha:
  - `valid_first_order`;
  - `weak_denominator_delta_method_unreliable`;
- manutenção integral do ponto e SE da share;
- sufixo `_weak_denominator_diagnostic_only` no tipo de SE quando o gate ativa;
- exatamente um aviso que encaminha o investigador para
  `gelbach.bootstrap(method="pairs")`,
  `xhdfe_gelbach_bootstrap(method="pairs")` ou
  `xhdfegelbachbootstrap`;
- no output Stata, supressão da afirmação incondicional de “full delta method”
  e apresentação do qualificador diagnóstico.

`share_tol` permanece `1e-12`, com a semântica anterior. `base_fixed` não foi
alterado. O gate cobre `base`, `base_fixed` e `movement`.

### Stop/go

O novo validador
`VALIDATE_GELBACH_REMEDIATION_COVERAGE.py` obteve:

```text
Fieller sweep:
signal=-0.480  |t|=0.67  weak_denominator_delta_method_unreliable
signal=-0.470  |t|=0.91  weak_denominator_delta_method_unreliable
signal=-0.455  |t|=1.26  weak_denominator_delta_method_unreliable
signal=-0.435  |t|=1.74  weak_denominator_delta_method_unreliable
signal= 0.200  |t|=16.83 valid_first_order

Coverage:
outer=250, bootstrap draws=399
mean|t_den|=0.991
analytic=0.644
pairs bootstrap=0.916
weak status=0.992
status/threshold mismatches=0/250
```

Logo:

- todas as linhas observadas com `|t| < 1.96` ativaram o gate;
- o desenho forte, com `|t| > 15`, ficou silencioso;
- a cobertura bootstrap ultrapassou o mínimo 0,90;
- o estado devolvido seguiu exatamente o limiar em todas as 250 amostras.

`VALIDATE_GELBACH_FRONTENDS.py` confirmou igualdade dos nomes dos campos,
strings dos estados e limiar nos três frontends.

### Identidade

As estimativas, covariâncias e SEs pré-existentes ficaram exatamente iguais em
Python/R nos testes de arrays. A tabela não filtrada preservou as linhas e
valores anteriores; o confronto Stata com o baseline armazenado ficou dentro de
`1e-12` nos valores raw e `5e-12` no CSV, isto é, dentro da precisão de
serialização já usada pelo output. As únicas diferenças fora dos valores são os
novos campos de metadados.

**Gate do slice: GO.**

## Slice 2 — P0-2: gate de validade da variância FE

### Alterações

Nos três frontends:

- novo limiar `fe_variance_ratio_min=0.35` /
  `FEVARMIN(real 0.35)`;
- novo estado por focal:
  - `valid_first_order`;
  - `conditional_only_between_fe_dominant`;
- gate baseado exclusivamente em `x1_fe_collinear_ratio <= 0.35`;
- sufixo `_conditional_only_diagnostic` nos tipos de SE dos blocos FE e do
  total misto;
- `confidence_interval_status =
  diagnostic_only_between_fe_dominant` nas linhas afetadas;
- um aviso único que encaminha para o pairs bootstrap.

O band anterior de near-collinearity (`<=1e-4`) não foi reutilizado nem
alterado.

### Stop/go

Sweep pedido:

| share between-FE | `x1_fe_collinear_ratio` | estado |
|---:|---:|---|
| 0,05 | 0,958 | `valid_first_order` |
| 0,25 | 0,778 | `valid_first_order` |
| 0,50 | 0,557 | `valid_first_order` |
| 0,75 | 0,298 | `conditional_only_between_fe_dominant` |
| 0,90 | 0,117 | `conditional_only_between_fe_dominant` |
| 0,97 | 0,036 | `conditional_only_between_fe_dominant` |

O mesmo fixture foi executado com o gate neutralizado
(`fe_variance_ratio_min=0`). Em todas as seis células, `b_base`, `b_full`,
`cov` e o SE do bloco FE foram `array_equal`: o gate não recalcula variâncias.

Coverage do bootstrap no desenho corrigido, com valores FE e composição fixos:

| share between-FE | outer | draws | cobertura analítica | bootstrap, alvo MC | bootstrap, verdade conhecida | SE/MC sd |
|---:|---:|---:|---:|---:|---:|---:|
| 0,75 | 200 | 199 | 0,870 | 0,995 | 0,990 | 0,800 |
| 0,90 | 200 | 199 | 0,665 | 0,980 | 0,985 | 0,495 |
| 0,97 | 200 | 199 | 0,350 | 0,975 | 0,970 | 0,260 |

O gate ativou em 100% das amostras destas três células e a cobertura bootstrap
foi sempre superior a 0,93. O padrão reproduz a autocorreção do auditor: não
redesenha os valores FE e não sustenta a alegação retirada de 28%.

`VALIDATE_GELBACH_FRONTENDS.py` confirmou paridade dos estados Python/R/Stata.

**Gate do slice: GO.**

## Slice 3 — P0-3: `Other (filtered)` em `etable`

### Alterações

- Python e R expõem `include_other=True`.
- Stata usa a opção idiomática inversa `noother`; o default inclui o agregado.
- Quando `keep`/`drop` omite componentes, cada painel recebe uma única linha
  `Other (filtered)`.
- O ponto é a soma dos componentes omitidos.
- O SE é calculado pelo sub-bloco conjunto da covariância, incluindo todos os
  termos cruzados; nunca soma SEs.
- O comportamento aplica-se a `levels`, `share_base` e `share_movement`.
- `include_other=False` / `noother` mantém o comportamento antigo, mas emite um
  aviso explícito de que a identidade visível não é preservada.

### Stop/go

`VALIDATE_GELBACH_PYFIXEST_FEATURES.py` testou `keep=None`, um componente e dois
componentes nos três painéis:

```text
etable:identity:all:*       gap=0
etable:identity:human:*     gap=0
etable:identity:human-job:* gap=0
etable:other-se-uses-joint-covariance PASS
etable:include-other-false-warns PASS
```

Os testes R e o do-file Stata repetem a mesma identidade. O output não filtrado
mantém as linhas anteriores e não cria `Other`, porque não há componentes
omitidos.

**Gate do slice: GO.**

## Slice 4 — P0-4: reporting Stata

### `labels()`

Em `xhdfegelbachetable.ado` e `xhdfegelbachcoefplot.ado`,
`LABELS(string asis)` passou a `LABELS(string)`. O loop
`tokenize/gettoken` foi preservado, conforme o diagnóstico da auditoria.

O teste end-to-end confirma duas entradas e dois labels distintos.

### Intervalos bootstrap

As linhas de componentes agora preferem `r(bootstrap_delta_ci)` quando
disponível e identificam o método como bootstrap percentile. Para permitir que
o `etable` construa também o `Other` após um bootstrap,
`xhdfegelbachbootstrap.ado` passou a preservar/devolver as covariâncias do ponto
(`cov`, `total_cov`, `base_cov`, `cov_delta_bbase`). Isto é plumbing de
reporting; não recalcula nem altera qualquer valor.

### Proveniência

O exemplo em `xhdfegelbach.sthlp` guarda os elementos de `r()` em locals antes
de executar `tab`, imprime o hash de 16 caracteres e avisa que `r()` é volátil.

### Stop/go

```text
XHDFEGELBACH_REMEDIATION_P0_OK
XHDFEGELBACH_BOOTSTRAP_SMOKE_PASS
```

`bash tests/stata/run_stata_tests.sh` terminou sem erro e produziu os logs de
certificação e bootstrap em `tests/stata/output/`.

**Gate do slice: GO.**

## Hard gate antes de P1

O hash de `src/akm_kss.cpp` continuava
`77a0293c9041828b3c0312fd063e2ebd56808a5e248a1213fab69940f0132995`
depois de P0-1...P0-4. O manifesto temporal também não encontrou alterações
P0 em nenhum diretório de fonte compilada.

**Hard gate: GO.**

## Slice 5 — P1-1: correção `alpha/2` do gate de regularidade

### Reprodução

Antes do patch, o union gate ativou em 49 de 500 amostras sob a nulidade:

```text
family-wise size = 49/500 = 0.098
```

O achado CONV-04 foi, portanto, reproduzido antes da modificação.

### Alteração

Foi introduzido:

```cpp
const double regularity_component_alpha =
    res.regularity_test_alpha / 2.0;
```

Os dois testes componentes passam a comparar os respetivos p-values com
`regularity_component_alpha`. `regularity_test_alpha` continua a ser o nível
family-wise público.

A alteração foi replicada nos cinco espelhos:

```text
src/akm_kss.cpp
r/xhdfe/src/akm_kss.cpp
stata/src/akm_kss.cpp
share/xhdfe_estimation_cpp/src/akm_kss.cpp
share/xhdfe_estimation_cpp/stata/src/akm_kss.cpp
```

Todos têm atualmente o mesmo SHA-256:

```text
614e3a7604d2bd7964d7b1dc0210eb3ea6d24ce1e41c2f4ee39e0e338985c0c1
```

### Stop/go

Depois do patch:

```text
regular=24/500
family-wise size=0.048
intervalo de aceitação=[0.04, 0.06]
```

O fixture `beta2=0 / Gamma>>0` continua a devolver
`regular_loading_nonzero` para `x` e `nonregular_not_ruled_out` para `_cons`.
Os pontos e covariâncias não mudaram.

**Gate do slice: GO.**

## Builds e backends

### Integridade dos builds

```text
bash tools/check_cpp_core_alignment.sh
PASS — Python, Stata, R e share mirrors alinhados

bash tools/check_default_builds.sh
PASS
```

| artefacto | configuração | SHA-256 |
|---|---|---|
| `build/py_hdfe_v11...so` | Release, `-O3 -DNDEBUG -march=native -mtune=native` | `dc9ec9d6d170b964c31326332ea73b0059bcacb639d8f90ee34399e2787959ec` |
| `build_cuda/py_hdfe_v11...so` | Release, mesmas flags, `sm_90` | `01ced18d65c3f58b3678efac94c813e408db4542755b8521f3a6d09ddeafcb7e` |
| `stata/xhdfe.plugin` | CUDA `sm_90`, OpenMP | `d99aecd0399a5df5bd3d1871c9782459fa50fc6a1855612d9ebcc181ac904ab4` |

`ldd stata/xhdfe.plugin` resolve `libgomp.so.1`.

### Smokes

- Python CUDA: módulo `build_cuda` selecionado, `backend=cuda`,
  `status=used`, `gpu_used=1`, convergência em 2 iterações.
- Stata bootstrap CUDA: `pairs` e `cluster_pairs`, 3/3 repetições válidas em
  cada método, `gpu_used_point=1`, `gpu_used_all_valid=1`,
  `point_gpu_backend=cuda`, `point_gpu_status=used`.
- R Gelbach CUDA: `test-gelbach-features.R` com `XHDFE_TEST_CUDA=1` terminou
  integralmente verde sobre a H100.
- Smoke CPU grande do plugin, dataset QP V08:
  `success=1`, `converged=yes`, `elapsed=67.528s`, dentro da banda histórica
  60--68 s.

O row GPU do script grande V08 falhou com `rc=198` quando executado dentro do
sandbox sem dispositivo. Esta limitação ambiental não foi tratada como sucesso:
os smokes CUDA dedicados foram repetidos fora do sandbox e exigiram uso real da
H100.

## Validação funcional final

| Validador | Resultado |
|---|---|
| `VALIDATE_GELBACH.py --module-dir build` | `ALL CHECKS PASSED` |
| `VALIDATE_GELBACH_ADVERSARIAL.py --module-dir build` | `ALL ADVERSARIAL GELBACH CHECKS PASSED`; 24/500 = 0,048 |
| `VALIDATE_GELBACH_PYFIXEST_FEATURES.py --module-dir build` | `ALL PYFIXEST-DERIVED GELBACH FEATURE CHECKS PASSED` |
| `VALIDATE_GELBACH_SAMPLE_PROVENANCE.py` | PASS, 36 checks |
| `VALIDATE_GELBACH_HELP.py` | PASS, 107 checks |
| `VALIDATE_GELBACH_FRONTENDS.py` | `ALL FRONT-END PARITY CHECKS PASSED` |
| `VALIDATE_GELBACH_REMEDIATION_COVERAGE.py --module-dir build` | `ALL STRICT GELBACH REMEDIATION COVERAGE CHECKS PASSED` |
| `tests/stata/xhdfegelbach_remediation_20260725.do` | `XHDFEGELBACH_REMEDIATION_P0_OK` |
| `tests/stata/xhdfegelbach_bootstrap_smoke.do` | `XHDFEGELBACH_BOOTSTRAP_SMOKE_PASS` |
| `tests/stata/run_stata_tests.sh` | PASS |
| suite R completa, wrappers atuais + native library existente | PASS fora do sandbox; 1 skip esperado do bootstrap CUDA |
| teste R `gelbach-features` com `XHDFE_TEST_CUDA=1` | PASS, sem skips |
| `python -m py_compile` nos módulos/validadores alterados | PASS |
| whitespace/diff checks nos ficheiros da tranche | PASS |

O validador de proveniência carrega deliberadamente o módulo que `import xhdfe`
usa no pacote, não um candidato residual:

```text
xhdfe/py_hdfe_v11.cpython-312-x86_64-linux-gnu.so
SHA-256 685b4aff191dc16f7792ed121e66b220abcf5aaa5a675275dd760d5eba8f5c53
```

Os testes P1 que exigem o novo C++ usam explicitamente o build Release atual,
de hash `dc9ec...`; isto evita confundir o módulo shipped pré-release com o
build candidato.

## Help e documentação

Foram atualizados:

- `xhdfe/help/gelbach.md`;
- `stata/xhdfegelbach.sthlp`;
- `stata/xhdfegelbachbootstrap.sthlp`;
- `stata/xhdfegelbachetable.sthlp`;
- `stata/xhdfegelbachcoefplot.sthlp`;
- roxygen de `r/xhdfe/R/gelbach.R` e
  `r/xhdfe/R/gelbach_features.R`;
- outputs `.Rd` correspondentes.

A documentação agora:

- descreve os dois novos gates, limiares, estados e routing para bootstrap;
- deixa de afirmar incondicionalmente que todas as regiões têm inferência
  first-order válida;
- documenta `include_other` também em `waterfall_data`;
- explica que `Other` usa covariância conjunta;
- distingue intervalos analíticos e bootstrap;
- torna consistente a fronteira deliberada de incerteza FE não condicional;
- documenta a volatilidade de `r()` no exemplo Stata.

## Ficheiros principais alterados nesta tranche

### Produção

- `xhdfe/gelbach.py`
- `xhdfe/_gelbach_features.py`
- `r/xhdfe/R/gelbach.R`
- `r/xhdfe/R/gelbach_features.R`
- `stata/xhdfegelbach.ado`
- `stata/xhdfegelbachbootstrap.ado`
- `stata/xhdfegelbachetable.ado`
- `stata/xhdfegelbachcoefplot.ado`
- os cinco espelhos `akm_kss.cpp` listados acima
- os help files Python/R/Stata correspondentes

### Testes e certificação

- `VALIDATE_GELBACH.py`
- `VALIDATE_GELBACH_ADVERSARIAL.py`
- `VALIDATE_GELBACH_FRONTENDS.py`
- `VALIDATE_GELBACH_HELP.py`
- `VALIDATE_GELBACH_PYFIXEST_FEATURES.py`
- `VALIDATE_GELBACH_SAMPLE_PROVENANCE.py`
- `VALIDATE_GELBACH_REMEDIATION_COVERAGE.py`
- `r/xhdfe/tests/testthat/test-gelbach.R`
- `r/xhdfe/tests/testthat/test-gelbach-features.R`
- `tests/stata/xhdfegelbach_remediation_20260725.do`
- `tests/stata/part1/companions.do`

## Limites e trabalho deliberadamente não executado

Não ficou nenhum finding desta tranche por reproduzir ou remediar. Permanecem
fora deste trabalho, por ordem explícita do prompt:

- versionamento e data de release;
- inclusão dos companions untracked no índice;
- sincronização com o repositório público;
- rebuild de todos os binários de distribuição e instalação do pacote R;
- tag, assets e net-install site;
- matriz completa `core23 x 8`;
- qualquer motor IV, dinâmico, nonlinear/distributional, multiway/wild cluster
  ou de inferência FE não condicional;
- conjuntos de confiança Fieller/weak-inference.

Em particular, o `.so` atualmente colocado dentro do package tree continua a
ser o binário shipped pré-release de hash `685b4aff...`; a fonte e os builds
Release CPU/CUDA já contêm P1, mas a substituição desse artefacto pertence à
tranche de release. O mesmo se aplica à reinstalação do pacote R a partir das
fontes atualizadas.

## Conclusão

A remediação pedida está implementada e certificada como **funcionalmente
adequada**. O produto agora deteta as duas regiões de inferência analítica
perigosa, preserva todos os números existentes, encaminha para o bootstrap já
calibrado, mantém identidades em tabelas filtradas, corrige o reporting Stata e
controla corretamente a dimensão family-wise do gate de regularidade.

Este GO não deve ser confundido com autorização de publicação: o release só
pode avançar depois da tranche de versionamento, rebuild dos artefactos
shipped/R, sync público e certificação `core23 x 8`.
