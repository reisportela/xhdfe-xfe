# Lições pós-2.26.2, segunda parte (20–22 set 2026)

Sucessor de `LICOES_POS_2.26.2_20260911.md`. Regista, como o `CLAUDE.md`
exige, cada erro demonstrado, recusa indevida e erro de avaliador com
reprodutor, revisão que o corrige e teste permanente. Contexto:
`AUDITORIA_POS_2.26.2_PLANO_E_CORRECOES_CLAUDE_20260920.md` (secções 5.8–5.10),
`Verifications/core24_matrix_20260921/`, ramo `claude/2270-fast-fixes-20260920`.

## 1. Erros demonstrados do estimador (autorizam mudar o estimador)

| Erro | Reprodutor | Correcção | Teste permanente |
|---|---|---|---|
| Refinamento parcial do resíduo (1.ª revisão): orçamento esgotado num RHS deixava y e X com erros de projecção inconsistentes; b passou de 3e-15 a 1e-8 no `laplacian_sum` em `fast` (a 2.26.2 estava certa). | bateria `laplacian_sum__native__xhdfe__auto__fast__cpu` | refinamento tudo-ou-nada em `comparable` (reversão com `A·δ` somado de volta), nunca em `fast` | `tests/analytic_precision/run.py` (bateria), `disconnected_chain_contract.py` |
| Refinamento "lavava" recusas: em `ic_tri_gap_c3e8` (cond 3e8, cancelamento 1e8 nos α) a 2.26.2 recusava; o refinamento baixava a medida dual até o certificado passar sem reparar o erro forward → `WRONG_SILENT`. | campanha 15set `ic_tri_gap_c3e8__{mean,sum}` | o refinamento só é mantido se a solução primária já passa o certificado dual por coluna; desenhos pequenos vão pela via directa (exacto) | campanha `x2270_*`; `ictri_fit.py` no scratchpad → a transformar em contrato |
| 2.26.2: cadeias desligadas com resíduo 1,3–1,9e-7·sd (18/36 linhas), causa: cegueira do teste dual do LSMR a direcções quase nulas, com estimativa de condição do LSMR 15 quando a real é 4e4. | `disconnected_chain_contract.py` (2.26.2 falha 18/36) | refinamento do resíduo (LSMR, CPU e CUDA) | idem, 36/36 |
| 2.26.2: desenhos near-null (cond 2e16) devolviam b = 1,32 para 0,7 com `converged=1`, também em `comparable`. | `near_null_contract.py` 96 e 600 nós; `part1/group-individual-near-null.do`; `near_null_r.R` | via directa comprimida por padrões + posto racional exacto + adjudicação pelos dados (`certify_group_forward_space` só para desenhos ambíguos) | os três acima, CPU e CUDA |
| Via directa (2.ª revisão): resíduo formado a partir dos coeficientes limitava os valores ajustados a eps·‖x‖ (α ≈ 6e4 → 6e-12) e falhava o gate strict de 1e-12 em `part1/group-individual-analytic` a 128 nós sob `mean`. | esse teste | resíduo comprimido = projecção de Householder `c − Q₁Q₁'c`; resíduo por linha em `long double` | `part1/group-individual-analytic.do` (64 e 128 nós) |
| 2.26.2 sobre-contava `df_a` no bloco `individual` quando a constante está no espaço das colunas de pertença (`mean`; `sum` com equipas iguais) com um FE ordinário absorvido; corrigido pela rc2 sem o declarar. | `part1/reghdfe-derived-statistics.do` (toy saturado: 2.26.2 `df_r=−2`, agora −1) | regra exacta em `individual_redundant` (rc2) | o mesmo teste, expectativa actualizada com justificação |
| Frontend Stata (rc2): `sortpreserve` nos dois programas aninhados custava uma geração de índice e uma ordenação por chamada (≈30–40 ns/linha; +7 a +15 % nas células Stata CUDA). Não é erro de estimativa. | matriz core-24 × 8, células `stata_*` vs `v2_stata_*` | removido do programa interior | células `v2_stata_*` da matriz; a matriz pré-release tem de incluir Stata |
| CUDA `xhdfe-fast` com `mean` em `ic_tri_gap_c3e8`: o solver do dispositivo devolvia b com erro 8,7e-7 e RSS com erro 1,3e-3 com certificado aprovado (resíduo relativo 1,4e-9); a via directa CPU é exacta. | `python3 -B ictri_fit_cuda.py <cuda .so> mean fast` com `XHDFE_GI_DIRECT=0` (mede o erro) e sem (recusa) | certificado de referência exacta no ramo CUDA (22set2026, auditoria 5.12 B): projecção directa completa como referência; recusa quando uma coluna retida desvia >10× a tolerância forward do modo ou uma coluna absorvida deixa de o ser | `ictri` CUDA em `validate23.log`; contratos CUDA repetidos |
| 2.26.2 e revisões 1–2: `aggregation(sum)` sem FE ordinário com equipas desiguais reportava `_cons` = média(y − Xb) com resíduos de média 0,23 (brinquedo), TSS centrada e R² de um modelo com intercepto que não foi ajustado (Achado 6). b, RSS e resíduos estavam certos. | `sum_nofe_toy.py`; `tests/test_group_sum_constant.py` | decisão constante-no-espaço por projecção do vector de uns (só nesse caso) e reporte `noconstant` quando não está (auditoria 5.12 A) | `tests/test_group_sum_constant.py`; `part1/group-sum-constant.do`; `r_sum_check.R` |
| 2.26.2: `savefe` em `credit2` devolvia efeitos fixos com erro até 5,6e-7 (resíduo relativo 5e-8, `precision_certified=0`): a varredura com captura de α não usa a via acelerada nem Krylov e pára pela norma da mudança. | `credit2_savefe.py` (scratchpad → `logs/blockB_20260922`) | projecção recomputada sem captura pela via do modo, certificada, efeitos recuperados de `y − Xb` (auditoria 5.14) | `tests/test_auto_routing_retry.py` cenário `retained` |

## 2. Recusas indevidas do verificador (autorizam afrouxar a verificação)

| Recusa | Onde | Correcção |
|---|---|---|
| Via directa (1.ª revisão) recusava todo o desenho near-null, incluindo o ajuste com dados bem condicionados que a 2.26.2 resolve; os contratos R/Stata e o README da bateria exigem que esse ajuste passe. | `near_null_r.R` 0/12; `part1/group-individual-near-null.do` r(198) | recusa só quando o resíduo ajustado depende da direcção não resolvida |
| Guarda forward recusava no piso FP64 (`test2 ≤ 64ε`) — majorante inconclusivo. | contratos de cadeias em strict | paragem dual vale; guarda fica em `krylov_max_condition*` |
| `strict-residual`: alvo do refinamento igual à tolerância dual (64ε), inatingível no orçamento; com a regra tudo-ou-nada o refinamento era revertido e o gate falhava (8 linhas). | `disconnected_chain_contract.py` cópias 9 | em strict cada estágio fica (o gate adjudica), alvo `max(tol, 64ε)`, orçamento maior |
| rc2: nível (iii) do certificado forward, overflow do posto racional, "omissão ambígua" (ver auditoria 3.3). | campanha 15set: 2 288 recusas | fora do caminho por omissão (`XHDFE_CERTIFY=1`) |
| Primeira métrica do certificado de referência exacta CUDA (desvio relativo à projecção exacta com piso 64ε·‖coluna‖) recusava colunas inteiramente absorvidas com *offsets* 1e6/1e14: a referência é ≈0 e o dispositivo pára à tolerância do input (`fe_omission_contract` CUDA 24 → falha). | `fe_omission_contract.py --backend cuda` | regra de omissão consistente: coluna absorvida pela projecção exacta só tem de ser absorvida também pelo dispositivo; colunas retidas comparam-se relativamente à projecção exacta |
| Sonda da constante anexada ao desenho (3.ª revisão, 1.ª forma): a coluna de uns, sendo um sistema consistente, falhava o teste dual do certificado (`test2` divide por arredondamento) e disparava a retentativa a 100× (fast: 63 → 144 it na Marta sem FE ordinário) e o descarte do refinamento (comparable). | `probe_marta_appended/` em `logs/blockA_20260922` | sonda como RHS separado fora do certificado/refinamento/retentativa (`group_constant_projection_residual_rel`); auditoria 5.13 |
| Candidata (até 22set): `savefe` em stall recusava sem tentar refinamento porque a retentativa automática excluía `retain` — perda de cobertura face à 2.26.2. | `credit2` comparable e fast | invólucro `absorb_with_precision_repair`: uma projecção sem captura + recuperação *map* (auditoria 5.14) |
| Protótipo opt-in `XHDFE_CUDA_AUTO_COMPARABLE_ACCURACY_RETRY`: quando nenhuma etapa da escada nem a retentativa fria atingem o alvo 1e-10, marcava o resultado como não convergido e recusava (`patents`: primária 4,0e-9 certificada, melhor etapa 1,1e-10). | `cudaladder/06_patents_ladder.json.log` | devolver o melhor candidato certificado (`event=best_certified`); `tests/test_cuda_accuracy_retry_fail_closed.py` reescrito para o contrato; auditoria 5.15 |
| Candidata (até 22set): declives heterogéneos sob `convergence(reghdfe)` explícito recusavam sem refinamento porque a continuação Option-A exigia critério `Auto`; a 2.26.2 devolvia não certificado com flag. | 2.ª regressão AKM, `akm2_slopes_reghdfe.py` (`logs/blockB_item3_20260922`) | a mesma regra continuada uma vez nas etapas da continuação, aceite na primeira etapa certificada (auditoria 5.16) |

## 3. Erros do avaliador e do harness

| Observação | Classificação | Tratamento |
|---|---|---|
| Campanha: `xf_consty_mix1248_nofe` e `xf_yspan_mix248_bridge_nofe` em CPU dão `WRONG_SILENT` com b = −2e-17 (oráculo 0) e `eta` 2e-8: a razão divide por uma escala de resposta ≈ 0. | escala de resposta nula no avaliador | registado; sem alteração ao estimador |
| Harness A/B rebentava num leg cujo ajuste falha (campo `elapsed` vazio) e abortava a célula. | fragilidade do harness | corrigido: leg registado como falha, célula continua |
| Harness A/B marcava todos os legs CUDA como falha funcional por haver um processo CUDA estrangeiro (llama-server ocioso do utilizador). | pré-condição do harness contrária à política de GPU partilhada | por omissão passa a observação registada e aviso; `--foreign-gpu-processes fail` mantém a regra antiga |
| Stata morre esporadicamente com r(5004) "Java installation not found" no `import delimited` antes de qualquer estimação. | erro transitório do host | o harness repete o leg uma vez quando esse texto aparece no log e não há resultado |
| `bateria run.py prepare` exige `--stata-package-root` = raiz do repositório, não `stata/`. | uso | registado |
| Desenho de teste "padrão {3t,3t+1},{3t,3t+1,3t+2}" para a constante no espaço: saturado (360 colunas para 240 grupos) e inteiramente podado pelos singletons ("must have at least one individual"). | erro do avaliador | substituído por desenho de *pool* (cada grupo com exactamente um membro de um pool de 30) |
| A guarda de pós-condição (`non-finite coefficient`) rejeitava o NaN da posição do intercepto de um modelo sem constante. | interacção guarda × reporte novo | excepção apenas nessa posição e apenas com `model_has_constant=false` |
| `pytest` em processo único: carregar `build/py_hdfe_v11` uma segunda vez dá `generic_type: type "HdfeRegressor" is already registered`; o teste novo reutiliza `sys.modules["py_hdfe_v11"]` e regista o módulo que carrega. | harness | `tests/test_group_sum_constant.py` |
| `sample_index_` do binding em modo group não é o conjunto de linhas longas retidas (1,104 M índices, máximo 2,76 M): a verdade "constante no espaço" calculada sobre `d.iloc[idx]` deu equipas de tamanho 1. | harness | verdade sobre os dados completos (transfere-se a subconjuntos de grupos) + convergência `test1` do LSMR |
| Campanha: `RESIDUALS_LOCAL` em 6 ajustes GPU das famílias de ajuste exacto (`xf_yspan_*`, `xf_consty_*`): resíduo exacto nulo, desvio do dispositivo 3e-10 absoluto (piso de reprodutibilidade CUDA), certificado de referência exacta a 4,6e-11. | escala de resíduo nula no avaliador | registado; sem alteração |
| `smoke_toy` sob `XHDFE_CERTIFY=1` pára na recusa "omission ambiguous" do cenário colinear: comportamento previsto do modo de auditoria, mas o script não distingue recusa esperada de falha. | harness | registado |
| O bloco-CG do dispositivo (`XHDFE_GPU_CG_AUTO=1`) nunca corre sob pedido CUDA: o selector põe `from_auto=false` e a porta `should_use_cuda_adaptive_cg` exige `from_auto`. Experiência sem efeito. | código morto sob GPU | registado (auditoria 5.15); a continuação de Krylov no dispositivo é a hipótese seguinte |

## 4. Lições de método

1. Toda a matriz pré-release inclui as células Stata: o custo do `sortpreserve` era invisível em C++.
2. Um refinamento aplica-se a todos os RHS ou a nenhum: os erros de projecção de y e X cancelam em b só se forem consistentes.
3. Um desenho ambíguo em FP64 adjudica-se pelos dados (o resíduo depende ou não da direcção não resolvida), nunca pelo desenho; recusar o ajuste bom é uma recusa indevida.
4. Refinar depois de o certificado passar pode "lavar" uma recusa legítima: o gate tem de julgar a solução primária.
5. Os resíduos de uma projecção directa vêm da factorização ortogonal, não dos coeficientes (que a incidência mal condicionada infla).
6. Mudanças de convenção herdadas (DoF) contam como mudança de saída por omissão e têm de ser declaradas nas notas, mesmo quando corrigem um erro.
7. `echo RC=$?` depois de `$(date)` reporta o `date`; `pkill -f <script>` mata a própria shell que o invoca.
