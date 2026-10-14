# Lições da integração do PR #13 (Tiago) na candidata 2.29 — 12out2026

Base: candidata privada `test/tiago-2.29.0-20261012` (`94d77c11`, versão de
trabalho 2.29.0.20261006). Integrado: PR #13 do repositório público
(`codex/iv-first-stage-f`, seis commits sobre a 2.28.1, cópia privada
`55d46dd9`, versão de trabalho 2.29.5.20261009). Resultado: candidata
integrada, etiqueta provisória 2.29.6.20261012 (13out2026: renomeada
**2.29.0.20261013**, por ser a primeira 2.29 pública após a 2.28.1) no ramo `integrate/pr13-tiago-2.29-20261012`
(worktree `build-worktrees/pr13-integration-20261012`). Evidência em
`Verifications/pr13_integracao_2290_20261012/`. Pareceres: advisor interno e
Codex Astra (`astra_ronda1/ASTRA_REPLY_ROUND1.md`).

## 1. O merge automático anulava a política D-C na covariância IV

Classe: **erro de integração** (não é defeito do PR nem da candidata). O ramo
wide do PR (`if (full_psd_ready && iv_covariance_wide.values.size())`)
ficava antes do bloco da candidata e aplicava «corte completo + corte do
bloco» sem `preserve_iv_intercept` nem a reparação D-C da constante. Além
disso, o caminho retido do PR deixara de acumular `meat`/`cross`/`total`,
pelo que `wide_augmented_meat_positive_definite` recebia zeros e devolvia
sempre falso.

Reprodutor: etapa 1 (`etapa1_merge_nao_reconciliado_cpu/`): o controlo CGM
genuinamente indefinido de `equivalent_cluster_partition_contract.py`
devolve uma matriz corrigida ainda indefinida (valor próprio mínimo da
correlação −1,52e-11 contra folga 5,7e-14) — exactamente o defeito B1 que a
D-C corrige. Tudo o resto passava, incluindo `iv_weak_distinct_covariance`
(a via wide do PR elimina por si o corte espúrio).

Correção: um só ramo `full_psd_ready` com a regra de decisão da candidata
(`iv_single_partition` ou carne aumentada PD) e o input wide do PR quando
existe; D-C em `OlsWide` (`repair_intercept_covariance_wide`, template
generalizado sobre o tipo da matriz); o caminho retido volta a acumular a
carne aumentada (opção (a), recomendada pelo Astra: a congruência por T não
preserva o sinal de um pivô pequeno em precisão finita). Sem piso espectral,
sem tolerâncias novas. Etapa 2: 100 % das baterias IV das duas linhagens em
CPU e CUDA.

## 2. O oráculo PSD do PR codificava a política anterior à D-C

Classe: **evaluation-harness error** (política substituída pelo mantenedor a
10out2026). `iv_covariance_psd_contract.py::reference` aplicava «corte
completo padronizado, depois override do bloco dos declives» sem a condição
de Schur da constante. Astra verificou em memória (MP120) que a D-C altera
9 dos 44 casos acima de 1e-8 e conserva os 12 controlos positivos e 14
controlos detectáveis. Correção: D-C acrescentada à composição final de
`reference()` (não a `project()`, que também serve o bloco isolado), com o
espectro positivo do bloco original padronizado; fixtures Stata regeneradas
pelo mesmo gerador. Limites 1e-9/1e-8 e o controlo «≥ 12» inalterados. Controlo negativo no
módulo reconciliado: o oráculo original do PR falha 9/44 (sementes 2 e 11,
até 7,3e-2 escalado) e passa os 12 controlos positivos; o oráculo D-C passa
44/44 em CPU e CUDA, nos dois modos.

## 3. Expectativas Fast herdadas no certificado puro-FE

Classe: **undue verifier refusal** (na política anterior). Em
`supplemental/upstream_reghdfe_20261011/pure_fe_certificate.cpp`, os casos
Fast 2 (regressor) e 5 (pesos) com projeção exacta (erro directo 2,2e-16)
eram recusados pelo certificado por blocos, que o PR retira de Fast (o
contrato Fast é o certificado agregado histórico). Reproduzido na etapa 1
(`certified=1`, `expected=0`). Correção: expectativas Fast actualizadas e
dois controlos negativos novos (kinds 8 e 9: regressor/pesos com erro
material 2^-20) que o certificado agregado continua a recusar nos dois
modos. 20/20 linhas na etapa 2. B2 (`require_group_column_moments`) depende
de `canonical_group_scope`, não de `slope_block_scope`: Fast nunca o teve.

## 4. Harness de continuação dos declives sem o include da candidata

Classe: **evaluation-harness error**. `slope_continuation_acceptance_contract.py`
extrai o wrapper real de `fe_absorption.cpp`; a candidata usa
`ScopedAuditEnvironment` (`include/hdfe/audit_environment.hpp`) dentro
dele e o harness gerado não compilava. Correção: include acrescentado ao
harness. 32 casos PASS na etapa 2.

## 5. Primeiros estádios Stata com Driscoll–Kraay

Classe: **demonstrated error** da combinação (o PR antecede o DK da
candidata). `firststage_options` reconstruía `vce(cluster <tempo>)` a partir
de `se_type=="cluster"`, perdendo DK e a largura de banda; `testparm`
usaria uma VCE diferente da anunciada. Correção no ado: ramo DK antes do
ramo cluster (`vce(dkraay #)`/`vce(dkraay)`), e `absorb()` transportado como
o texto do utilizador (`absorb_vars`), que não depende de temporários
`tsrevar` da chamada original (suspeita do Astra para `estat firststage`
posterior com declives TS). Teste: `stata_extra/pr13_firststage_dk_ts.do`.

## 5b. `e(r2_a)` em modelo saturado (df_r ≤ 0) — correcção de 13out2026

Classe: **demonstrated error** do frontend da candidata (commit `94d77c11`),
não do PR. A guarda `statstyle != reghdfe` acrescentada à regra do modelo
saturado deixava `e(r2_a)` ausente em `statstyle(reghdfe)`, invocando
`docs/LICOES_UPSTREAM_REGHDFE_20261011.md` (ponto 3). Mas o reghdfe 6.14.1
devolve 1 nesse toy (RSS = 0 exacto, D < 0: a sua fórmula Mata dá 1), a
2.28.1 publicada devolvia 1, a ajuda promete a convenção do reghdfe e o teste
`part1/reghdfe-derived-statistics.do` exige 1. Correcção: guarda removida
(regra da 2.28.1 reposta nos dois estilos); reprodutor
`Verifications/pr13_integracao_2290_20261012/etapa3_stata_suite/r2a_reghdfe_probe/`.
O ponto 3 das lições do port upstream fica assim circunscrito ao F conjunto
com covariância singular.

## 6. O que não mudou

Tolerâncias, alvos de trabalho, método de absorção, amostra e contratos Fast
/Comparable. O ganho do PR em Fast com declives heterogéneos (AKM secondreg)
é a remoção de trabalho não exigido pelo contrato Fast; Comparable mantém o
certificado por blocos e a continuação. Recusas CUDA Comparable pré-existentes
(duas linhas de `fast_slope_fit_contract.py`) reproduzem-se no módulo CUDA da
candidata e não são introduzidas pelo merge.

## 7. Fixture de paridade R↔Python desactualizada pela política D-C (13out2026)

Classificação: erro do harness de avaliação. `test-parity-python.R` falhava em
`fe3_cluster2` (`se[_cons]`, 2,4e-5 relativo) porque a fixture datava da 2.26.0;
regenerada com o módulo CPU actual (determinística, duas execuções
byte-idênticas) e localizada com o módulo 2.28.1 congelado: só `se[_cons]` sob
multiway mudou entre 2.28.1 e 2.29.0, declives bit-idênticos — a assinatura da
D-C. O R nativo coincide com a fixture nova a 7e-15; «R = Python» é paridade,
não correcção: a evidência independente do valor novo está em
`intercept_psd_contract.py` e `iv_covariance_psd_contract.py`. A 2.28.1 pública
passa a fixture antiga, logo é uma mudança visível em R que as notas de release
registam. Proveniência (`parity_reference_current_provenance.json`) actualizada;
`parity_reference.json` intocada.

## 8. Objectos R reutilizados entre configurações (13out2026)

Classificação: defeito de build num caminho documentado (`R CMD INSTALL
r/xhdfe` a partir do clone). O harness de 13out instalou `portable`, `native`
e `cuda` no mesmo `r/xhdfe/src` sem limpar: o «native» ligou objectos portáteis
(0 compilações) e o `cuda` ligou objectos sem `-DHDFE_USE_CUDA` (símbolo
`make_n1_graph_preconditioner` indefinido). Correcção: carimbo de configuração
em `r/xhdfe/src/Makevars` (CUDA, arquitectura, `-march`, série) do qual todos os
objectos dependem, escrito em tempo de parse só quando muda; `.Rbuildignore` e
`.gitignore` ignoram-no. Lição de harness: nunca aceitar um testthat sem afirmar
`find.package()` e `packageVersion()` — o braço CUDA nulo testou o 2.28.0 do
`r/Rlib` do checkout principal e reportou 1213 PASS. Segunda lição, da
reexecução: nunca editar um script bash enquanto corre (o bash lê-o por offset
de ficheiro; a correcção do padrão `grep` do braço `portable`, feita durante o
braço `cuda`, abortou o `echo DONE` final sem afectar os resultados: 1210/1210/
1215 PASS, 0 falhas).

## 9. Recusas CUDA Comparable em declives heterogéneos (adjudicadas 13out2026)

Classificação: recusa legítima no contrato Comparable, conservadora quanto aos
coeficientes, herdada da 2.28.1 e circunscrita. Sonda directa de
`absorb_fixed_effects_v6` (`Verifications/pr13_integracao_2290_20261012/etapa6_cuda_adjudicacao/`):
o estado recusado tem coeficientes a 9,3e-15 do oráculo de dummies explícitas
(CPU certificada: 9,4e-15), mas um erro de projecção de ỹ de 1,1e-7 e resíduo por
blocos 2,8e-8 > 8e-9 que cinco etapas de continuação até 1e-15 não movem (piso
aritmético do solver CUDA com declives). Não é limite suficiente inconclusivo:
é um momento necessário falhado. Não se relaxa o contrato; a correcção exigiria
uma hipótese nova sobre o mecanismo CUDA. Fica documentada como limitação.

## 10. Cobertura acrescentada (13out2026)

Classificação prévia: `missing coverage`. Contratos novos, todos PASS no
módulo CPU e no pacote Stata (CPU e CUDA): cache de absorção em IV (acerto
bit-idêntico, 1 ULP em Z/y/X, amostra, modo), regressor omitido na via wide da
covariância IV (92 casos, erro ≤ 1,9e-15/1,6e-14) e IV fraca com
`vce(dkraay)` e `e(first)` contra oráculos explícitos (κ₂ até 67; erros
≤ 2,3e-14 com tolerância pré-declarada). Observação registada: a partição
«distinta» da fixture congelada torna-se indefinida com um quarto score; sem FE,
`gpubackend(cuda)` recusa por desenho (fail-closed), o que explica a exclusão
dos casos sem FE na campanha DK de 11out. Ver
`Verifications/pr13_integracao_2290_20261012/etapa7_cobertura/README.md`.

## 11. Harness ctest desactualizado (13out2026)

Classificação: erro do harness. `audit_20260804_contracts.py` esperava
fallback CPU após CUDA pedido (a 2.28.1 publicada falha fechado), aceitava
estimativas num encadeamento forçado que hoje recusa com «precision
certification failed» e usava uma fixture agrupada subdeterminada;
`test_student_t_precision.py` abortava por hash do executável Stata. Corrigidos
(expectativas fail-closed, recusa esperada, fixture sobredeterminada, NOTA em
vez de erro); `docs/gpu.md` deixou de prometer o fallback; ctest 10/10.

## 12. Lições do corte e da publicação da 2.29.0.20261014 (14out2026)

- **Propagação curada ao público a partir de `git ls-files`, não de uma lista
  redigida à mão.** A lista do plano (§4) cobria árvores e três ficheiros de
  teste de topo; o `CMakeLists.txt` propagado referenciava `tests/fma_liveness.cpp`
  e dois contratos novos que ficaram para trás, e o primeiro preflight falhou nos
  jobs Windows (`build-probe-inputs` e wheel). Regra: depois de copiar, correr um
  `cmake` de configuração no checkout público (build dir fora do repositório) e
  só depois etiquetar; a lista fica derivada dos ficheiros versionados.
- **Arrays vazios em bash 3.2.** O sonda de liveness FMA acrescentado na linha
  2.29 ligava com `"${fma_link_flags[@]}"`; com `set -u` no bash 3.2 do macOS um
  array vazio é variável indefinida e o job macOS arm64 falhou. O próprio
  script já usava o idioma seguro `${arr[@]+"${arr[@]}"}` para os outros arrays
  opcionais; corrigido nos dois scripts de build (`524b20c3`). A CI do macOS é a
  única execução do script em bash 3.2: nunca assumir que «corre em Linux» cobre
  o macOS.
- **Referência do contrato = release pública corrente.** A campanha mínima
  arrancou com a 2.26.2 como quarto braço porque o `CLAUDE.md` o exigia; o
  mantenedor fixou a referência na 2.28.1 a meio da campanha e o registo passou a
  tratar a 2.26.2 como anexo de precisão. Quando a release pública mudar, o
  baseline e os tempos de referência do ready mudam nos dois ficheiros de regras.
- **O checkout principal não é área de trabalho.** Tinha oito ficheiros
  modificados e nunca commitados (`AGENTS.md` de 03out, `.gitignore` de 11out,
  entre outros) que impediam o fast-forward do `main`; ficaram num stash
  etiquetado e num patch em `Verifications/`. Trabalho vive em worktrees; o
  checkout principal só recebe fast-forwards e rebuilds canónicos.
- **Um protocolo de medição por parágrafo de referência.** Os tempos Stata do
  harness das campanhas (`residuals()`) são 1,6–2× os do comando simples; os
  tempos de referência do ready no `CLAUDE.md` declaram agora os dois protocolos
  e proíbem misturá-los.
