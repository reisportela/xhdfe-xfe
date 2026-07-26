# Encerramento funcional do `xhdfegelbach`

**Data:** 25 de julho de 2026

**Decisão:** `FEATURE FREEZE` da tranche linear HDFE Gelbach

**Implementação local:** `GO`

**Release pública:** `GO` sujeito ao gate bifásico dos artefactos CI

## 1. Decisão

Esta tranche pode ser dada por encerrada.

O `xhdfegelbach` implementa agora o conjunto de funcionalidades que se adequa
ao seu estimando: decomposição exata, simultânea e order-invariant do movimento
de coeficientes entre uma especificação linear base e uma especificação linear
completa, com suporte HDFE e inferência explicitamente qualificada.

Continuar a acrescentar opções ao mesmo comando deixaria de ser consolidação do
Gelbach linear e passaria a misturar:

- novos estimadores, como IV, painel dinâmico ou GLM;
- novos problemas de identificação, como separação de três ou mais FE;
- novos métodos de inferência, como wild-cluster ou incerteza não condicional
  dos FE recuperados;
- uma camada narrativa/path-dependent distinta da identidade Gelbach global.

Esses desenvolvimentos podem ter valor, mas devem nascer como tranches ou
engines separados, motivados por uma aplicação empírica concreta. Não são
lacunas que devam bloquear o encerramento do `xhdfegelbach` atual.

## 2. Porque `b1x2` já não limita o produto

`b1x2` permanece como oracle de compatibilidade do subconjunto OLS clássico,
não como fronteira da API.

O candidato atual aceita:

- vários coeficientes focais na mesma execução;
- controlos comuns dentro de `x1`, com seleção separada por `focal`;
- qualquer número de blocos observados multicoluna em `x2_groups`;
- qualquer número de dimensões FE adicionadas, com subtotal
  normalization-safe;
- HDFE comuns à base e ao modelo completo;
- targets observados e targets totalmente absorvidos;
- seleção explícita do par FE usado no diagnóstico de conectividade;
- pesos analíticos e frequency weights;
- VCE não ajustada, robusta e cluster one-way;
- CPU/OpenMP e CUDA opt-in com diagnóstico de backend;
- múltiplos contrastes sem reestimação.

Assim, a antiga forma “um `b1`, um `x2`” é apenas um caso particular de uma
especificação muito mais ampla.

## 3. Superfície funcional concluída

### 3.1 Estimando e geometria HDFE

- identidade `b_base - b_full = sum(contribuições)`;
- blocos observados simultâneos e invariantes à sua ordem;
- FE adicionados como blocos de contribuição;
- FE comuns condicionados em base, full e regressões auxiliares;
- targets absorvidos como estimando distinto e explicitamente rotulado;
- subtotal FE agregado invariável à normalização;
- deteção de rank, colinearidade exata e near-FE-collinearity;
- remoção recursiva de singletons sobre a amostra full.

### 3.2 Identificação e inferência

- diagnóstico de conectividade e `connected(require)` fail-closed no domínio
  two-way certificado;
- rejeição explícita de uma falsa certificação multiway;
- covariance conjunta das contribuições;
- covariance da base e cross-covariances necessárias para shares;
- contrasts lineares conjuntos;
- diagnóstico conservador da não regularidade do produto
  `loading x beta2`;
- p-values/testes auxiliares e estados de validade por célula;
- advertência few-cluster;
- inferência de FE corretamente rotulada como condicional/mista.

### 3.3 Bootstrap e reporting aplicado

- pairs bootstrap;
- cluster-pairs bootstrap com unidade de reamostragem obrigatoriamente
  declarada;
- full refit em cada réplica;
- streams RNG independentes derivados de uma seed mestre;
- ledger completo de réplicas válidas e falhadas;
- mínimo de réplicas válidas com falha fechada;
- intervalos percentile e basic;
- `tidy`, shares do movimento e shares da base com covariance conjunta;
- `etable` em dataframe, Markdown, LaTeX, HTML e Great Tables;
- dados waterfall e `coefplot`;
- `keep`, `drop`, labels e agregação explícita das parcelas omitidas.

### 3.4 Auditabilidade

- contagens de input, amostra retida, effective N e singletons;
- índice zero-based e máscara opt-in da amostra efetiva;
- hash determinístico `fnv1a64-le-v1`, sensível à ordem das linhas;
- variável Stata com `1` retido, `0` singleton removido e missing fora da
  seleção inicial;
- metadados de estimando, VCE, regularidade, conectividade, FE e backend;
- contrato público alinhado em Python, R e Stata.

## 4. Estado dos principais achados da auditoria

| Tema | Decisão de encerramento |
|---|---|
| Conectividade two-way | Remediada no domínio certificado; falha fechada fora dele |
| Produto não regular | Falha silenciosa remediada; bootstrap disponível sem ser apresentado como cura universal |
| HDFE comuns | Implementadas e validadas contra LSDV/dummies explícitas |
| Rigidez `b1x2` | Removida da API e mantida apenas como oracle de regressão |
| Shares frágeis | Tolerância, estados undefined, warnings e covariance conjunta implementados |
| Bootstrap/relatórios PyFixest | Lacunas substantivas incorporadas |
| Ledger da amostra | Contagens, índice, máscara e fingerprint implementados |
| Multiway FE split | Deliberadamente não certificado; resultado agregado continua seguro |
| Inferência não condicional dos FE | Fora do contrato atual e visivelmente rotulada |
| IV, dinâmica e não linear | Novos estimandos; não pertencem ao engine OLS atual |

Esta matriz é uma decisão de produto, não uma alegação de que todo o universo
de decomposições econométricas foi implementado.

## 5. Funcionalidades que não devem ser acrescentadas agora

### 5.1 `connected(largest)` automático

Restringir à maior componente altera a amostra e o estimando. O pacote já
permite construir explicitamente uma amostra conectada antes da estimação, e a
nova proveniência permite auditar essa escolha. Integrar uma restrição
automática só se justifica quando uma aplicação exigir um workflow único de
seleção, reexecução de singletons e refit integral.

### 5.2 Fórmulas declarativas

Uma camada de fórmulas/model-matrix pode melhorar ergonomia, mas não aumenta a
capacidade econométrica do core. As APIs atuais de matrizes, listas/blocos,
nomes e seletores são explícitas e suficientemente gerais. Esta melhoria deve
ser reconsiderada apenas se workflows reais mostrarem erro humano recorrente.

### 5.3 IV, dinâmica e famílias não lineares

Estas extensões precisam de estimando, identificação, covariance, remainder,
oracles e documentação próprios. Devem ser comandos/engines separados, nunca
simples flags que reutilizem indevidamente a identidade OLS.

### 5.4 Multiway/wild-cluster e FE não condicional

São extensões inferenciais úteis, mas não necessárias para declarar concluído o
primitive Gelbach linear atual. Devem ser priorizadas por desenho empírico e
número de clusters, não por feature parity abstrata.

## 6. Gates finais reexecutados

Em 25 de julho de 2026 foram novamente executados:

| Gate | Resultado |
|---|---|
| `tools/check_default_builds.sh` | passou |
| `VALIDATE_GELBACH.py` | todos os checks passaram |
| `VALIDATE_GELBACH_ADVERSARIAL.py` | todos os checks adversariais passaram |
| `VALIDATE_GELBACH_PYFIXEST_FEATURES.py` | todas as funcionalidades passaram |
| `VALIDATE_GELBACH_SAMPLE_PROVENANCE.py` | 36 checks passaram |
| `VALIDATE_GELBACH_HELP.py` | 111 checks passaram |

Mantém-se ainda a evidência desta tranche:

- testes R Gelbach completos aprovados;
- 28 testes de certificação Stata aprovados;
- paridade CPU/CUDA Python–R–Stata aprovada, com uso real de CUDA;
- invariância numérica bit a bit do percurso default;
- smoke Stata/OpenMP grande com 20 iterações e 64,833 s, dentro da referência
  de 60–68 s;
- matriz `core23 × 8` com 184/184 sucessos e 92/92 células CUDA com
  `gpu_used=1`;
- builds CUDA exclusivamente `sm_90` inspecionados e executados no H100.

## 7. O trabalho restante pertence à publicação

A tranche local de release foi executada para `2.21.0.20260725`: versões,
headers, builds, instalações isoladas, matriz `core23 × 8`, CUDA real no H100,
documentação e empacotamento foram validados. O procedimento remoto é
deliberadamente bifásico:

1. o tag `v2.21.0.20260725` cria drafts e artefactos CI;
2. os binários CUDA exatos de ambos os CI são descarregados e executados no
   H100;
3. apenas depois dessa prova o marcador `publish-v2.21.0.20260725` publica os
   releases e atualiza o net-install/`gh-pages`.

O detalhe e a evidência quantitativa encontram-se em
`XHDFEGELBACH_RELEASE_CERTIFICATION_REPORT_20260725.md`.

## 8. Política a partir deste encerramento

Até existir uma aplicação concreta que exija outro estimando:

- congelar novas funcionalidades no `xhdfegelbach`;
- aceitar apenas correções de bugs, documentação, testes e release hardening;
- preservar a API e a compatibilidade `b1x2`;
- não enfraquecer warnings ou guards para acomodar casos não certificados;
- colocar novas famílias econométricas atrás de APIs separadas;
- manter o framing como coefficient-movement accounting, nunca mediação causal
  automática.

## 9. Veredicto final

**Sim, esta parte do Gelbach está adequadamente implementada e pode ser
encerrada.**

O produto não é “universal”, nem deve tentar sê-lo. É agora um primitive
Gelbach linear HDFE-aware substancialmente mais flexível que `b1x2`, com
inferência, bootstrap, reporting, conectividade, common FE e proveniência
auditável. Os limites restantes estão visíveis, falham de forma segura quando
necessário e pertencem a futuras linhas de investigação, não a uma lacuna
silenciosa da implementação atual.
