# Lições do trabalho posterior à 2.26.2 — 11 de setembro de 2026

Este documento regista aprendizagens e evidência. A orientação de execução é
o prompt de reorientação do utilizador de11set2026, reproduzido operacionalmente
no [STATE.md](../STATE.md). Substitui, para decisões futuras, as orientações
EXECUTION_CONTROL, PERFORMANCE_CYCLE_READY e PRECISION_PERFORMANCE_ASSESSMENT.
Nenhuma fonte, build, instalação, referência canónica ou publicação foi
alterada para preparar este checkpoint. Os scripts abaixo foram lidos; não
foram novamente executados.

Foi lida a auditoria completa
[Claude/Fable](../AUDITORIA_ADVERSARIAL_BECO_PRECISAO_POS_2.26.2_CLAUDE_20260911.md),
SHA-256 `7c31bec240e8be431ac4cf43be05bc17cc6708e5554397c7a9093deda76f46c9`.
As decisões fixas do utilizador são aceites: referência de produto2.26.2,
todas as funcionalidades e oito variantes, Fast com a filosofia original,
precisão Comparable da2.26.2, N1 necessário no fit e N2 apenas em auditoria.

**Regra de leitura das tabelas:** erro demonstrado significa discrepância
reproduzida contra uma referência pertinente, não falha de um majorante.
"Correção" distingue corrigir a estimativa de passar a recusar uma estimativa
errada. "Não medido" não significa custo zero. Uma recusa esperada não conta
como estimação numérica certificada. Um script preservado é um reprodutor
offline; quando não está no runner permanente, isso é indicado.

## (a) Erros demonstrados e correções a preservar

| Erro e classe | Evidência/reprodutor verificável | Revisão que corrigiu ou estado real | Custo medido e qualificação | Cobertura offline a manter em G1 |
|---|---|---|---|---|
| **Group/individual near-null:** b1,3198/1,2599/1,2022 contra0,7, com sucesso formal; RSS até491,25 contra78,45. **Solver group.** | [Checkpointv16](../Verifications/group_indiv/audit_20260906/CHECKPOINT_v16.md), [raw2.26.2](../Verifications/group_indiv/audit_20260906/near_null_active_2262.json). | v15–v42: bases/rank, contraste, deflação e verificação do espaço group. Preservar as correções de solução; não confundir uma recusa segura posterior com disponibilidade de estimativas corretas em todos os inputs. | [v38group3M](../Verifications/convergence_20260906/candidate/performance_v38_marta_cpu): aproximadamente4,8→10,2s,277→360iterações; custo agregado, não do hunk isolado. Grupo terá gate próprio de1,2×. | [near_null_contract.py](../tests/analytic_precision/near_null_contract.py), [disconnected_chain_contract.py](../tests/analytic_precision/disconnected_chain_contract.py), casos group da bateria1719. |
| **Refinamento strict group incompleto:** primeiras correções não transportavam todos os coeficientes FE usados na reconstrução. **Solver group/recomposição.** | [Veredito e controlos](../Verifications/convergence_20260906/VEREDICTO.md), [adversarial_contracts.cpp](../Verifications/convergence_20260906/adversarial_contracts.cpp). | v1 reprojeta; v2 compõe os witnesses/alphas CPU e CUDA; v3 conserva o orçamento/contagens. Na nova N1 não copiar os alvos internos mais apertados para o default. | Par Stata v3: CPU73,861→73,762s; CUDA16,529→16,316s, mesmas23/77iterações. Compatível com ausência de perda; não demonstra custo rigorosamente zero nem speedup. [Proveniência](../Verifications/algorithm_history_audit_20260909/timeline/V3_PERFORMANCE_PROVENANCE.json). | [Bateriav3](../Verifications/convergence_20260906/candidate/certification_v3/report.json), controlos de reconstrução e orçamento acima. |
| **Pedido CUDA executado em CPU**,24 casos. **Backend.** | [Comparação v3](../Verifications/convergence_20260906/candidate/certification_v3/comparison.json), [veredito](../Verifications/convergence_20260906/VEREDICTO.md). | v3 recusa explicitamente métodos CPU perante pedido CUDA e verifica uso efetivo. Não transformar fallback silencioso numa alegação de suporte GPU. | O par grande v3 acima não revelou regressão. O custo individual da validação backend não foi isolado. | [native.py](../tests/analytic_precision/native.py), [fail_closed.do](../tests/analytic_precision/fail_closed.do) e controlos adversariais CPU/CUDA. |
| **Estado antigo acessível depois de falha; proveniência incompleta da amostra group; falhas de recuperação mal transportadas.** **Lifecycle-pesos.** | [Veredito:17 casos de proveniência](../Verifications/convergence_20260906/VEREDICTO.md), [cronologia v4–v5](../Verifications/algorithm_history_audit_20260909/timeline/TIMELINE.md), [bindingsv14](../Verifications/convergence_20260906/candidate/bindings_v14). | v4–v5 e interfaces seguintes: transação de fit, limpeza das saídas, sample_index e erros informativos. Já incorporados emv48; não reaplicar patches por número de pasta. | Não há A/B isolado desta correção. Não atribuir os tempos das provas posteriores a estes guards. | [group_sample_contract.py](../tests/analytic_precision/group_sample_contract.py), [group_formula_contract.py](../tests/analytic_precision/group_formula_contract.py), [evaluate_fail_closed.py](../tests/analytic_precision/evaluate_fail_closed.py). |
| **i/pweights, regras estatísticas e transporte de pedidos; DoF group exato/rank racional.** **Lifecycle-pesos/OLS-rank.** Nem todos os itens são erros de estimativas: DoF exato é também funcionalidade explicitamente solicitada. | [Cronologia v6–v14](../Verifications/algorithm_history_audit_20260909/timeline/TIMELINE.md), [group_exact_rankv48](../Verifications/convergence_20260906/candidate/source_v48/group_exact_rank.hpp), [bindingPythonv14](../Verifications/convergence_20260906/candidate/bindings_v14/py_hdfe_v11.cpp). | Preservar os hunks já presentes emv48 e a semântica de pesos nas três interfaces. DoF exato continua com a opção própria; não passa a depender deCERTIFY. | Não há custo isolado medido por hunk. A opção de DoF exato não pode ser ativada por omissão para justificar um benchmark. | [weights_contract.py](../tests/analytic_precision/weights_contract.py), [weights_reference.do](../tests/analytic_precision/weights_reference.do), [casos permanentes](../tests/analytic_precision/cases.py). |
| **OLS quase colinear**, mesmo sem FE:0,750108518719/−0,500108518719 contra0,75/−0,5. **OLS-rank.** | [Log2.26.2](../Verifications/group_indiv/audit_20260906/active_2262_ordinary_analytic.log), [falhav42](../Verifications/convergence_20260906/candidate/certification_v42/SUPPLEMENTAL_OLS_FAILURE_20260907.json). | v43–v46 introduzem solução/rank OLS estáveis; v47 integra. Preservar na basev48. | O custo do hunk não foi isolado. Conjuntov47: readyCPUcomparable67,121→71,680s; Fast43,529→46,505s. [CSV](../Verifications/convergence_20260906/candidate/performance_v47_core22/cpp_cpu_comparable/runs.csv). Não chamar isto apenas "ms". | [ols_precision_contract.py](../tests/analytic_precision/ols_precision_contract.py), [ols_wide_arithmetic.py](../tests/analytic_precision/ols_wide_arithmetic.py). |
| **Omissão indevida de X=1e14+a/1e15+a na v47:** variável identificada omitida. **OLS-rank.** | [Falha v47](../Verifications/convergence_20260906/candidate/certification_v47/SUPPLEMENTAL_OMISSION_FAILURE.json). O [probe da 2.26.2](../Verifications/group_indiv/audit_20260906/covariance_guard_stress_20260909/product_probe_2262.json) demonstra outra omissão, numa construção Walsh quase colinear; não reproduz estes offsets e não permite atribuir literalmente o mesmo caso à 2.26.2. | v48 corrige o caso dos offsets: não trata um piso de arredondamento como prova de redundância. O helper de redundância/rank permanece no produto; não é a prova geral 1FE posterior. O controlo Walsh conserva adjudicação própria em G1. | Sem A/B isolado do hunk v48. A matriz v47 foi marcada [não aceitável após a falha](../Verifications/convergence_20260906/candidate/performance_v47_core22/NOT_ACCEPTANCE_EVIDENCE_AFTER_OMISSION_FAILURE.json). | [fe_omission_contract.py](../tests/analytic_precision/fe_omission_contract.py), contrato OLS acima e probe Walsh separado. |
| **savefe+noconstant:** b/RSS certos, resíduos deslocados em6. **Recomposição.** | [Do-file válido](../Verifications/group_indiv/audit_20260906/savefe_noconstant_active_2262_retry1.do), [log](../Verifications/group_indiv/audit_20260906/savefe_noconstant_active_2262_retry1.log). | v56: preservar numa dimensão âncora a média retirada quando não existe constante; corrigir normalização/recomposição. Cherry-pick funcional, sem o enforce que coexistia nesse snapshot. | Não há A/B isolado. A mudança acompanha operações de normalização já existentes; isso é análise de código, não medição de custo zero. | [savefe_no_intercept_contract.py](../Verifications/group_indiv/audit_20260906/savefe_no_intercept_contract.py), [savefe_extended_normalization_contract.py](../Verifications/group_indiv/audit_20260906/savefe_extended_normalization_contract.py); contratos preservados fora do runner principal, a incluir explicitamente. |
| **Slopes puros:** b1,95 e RSS1212,6 reportados, mas resíduos com erro2,1 e RSS recalculado365,88. **Recomposição.** | [Semântica Stata](../Verifications/group_indiv/audit_20260906/pure_slope_stata_semantics.log), [do-file](../Verifications/group_indiv/audit_20260906/pure_slope_stata_semantics.do). | v59 corrige presença de intercepto FE; v60 harmoniza a convenção Stata; v60_stats corrige indicação da constante emR. Preservar todos, não apenas o core. | Sem A/B isolado. Não extrapolar os timings de savefe/outros modelos. | [slope_conditioning_contract.py](../tests/analytic_precision/slope_conditioning_contract.py), [normalizaçãoPython](../Verifications/group_indiv/audit_20260906/savefe_extended_normalization_contract.py) e [R](../Verifications/group_indiv/audit_20260906/savefe_extended_normalization_contract.R). |
| **Nível2001 inexistente tratado como dummy a identificar.** **OLS-rank/representação do verificador**, não erro do estimadorv48. Registado aqui porquev103 é um donor pedido. | [Assessmentsv102](../Verifications/group_indiv/audit_20260906/ready_stata_v102_ab_20260909/ASSESSMENT.md), [controlo de gapsv103](../Verifications/group_indiv/audit_20260906/gapped_coarse_v103). | v103 compacta níveis observados no suporte3FE. Esse suporte não existe emv48; o hunk fica exclusivamente na auditoriaCERTIFY. Slopes puros pertencem à linha anterior, não av103. | Stata v102 recusou após284,617s; v103 concluiu140,852s. São estados/caminhos diferentes, não custo isolado nem ganho aceite do produto. | Reprodutor de gaps acima e regressão dos rótulos/níveis observados. |
| **V representável tornou-se zero por underflow intermédio.** **Gama numérica.** | [Contrato de gama](../tests/analytic_precision/ols_inference_range_contract.py), [comparação v129](../Verifications/implementation_plan_20260909/ready_ab_20260911/V129_RANGE_FIX_COMPARISON.json). | v129 corrige/recusa operações de gama; preservar zeros legítimos de cancelamento. Portar apenas esses hunks para o OLSv48, não copiar o OLS tardio inteiro. | Par ready85,520vs83,786s, diferença2,07%: abaixo de5%, não é resultado de performance. O caminho normal tem checks pequenos; custo das exceções não foi medido globalmente. | Mesmo contrato:19PASS+2recusas seguras no snapshotv129; casos HC, cluster, cancelamento e erro depois de sucesso. |
| **MLSMR em escalas mistas:** RSS9,75e87 em vez de128, embora os RHS originais fossem ortogonais aos FE. **Gama numérica/solver.** | [Rotas2.26.2](../Verifications/implementation_plan_20260909/final_inference_zero_20260911/ROUTES_2262.json), [oracle racional](../Verifications/implementation_plan_20260909/final_inference_zero_20260911/heterogeneous_scale.py), [rotasv134](../Verifications/implementation_plan_20260909/final_inference_zero_20260911/ROUTES_134_OFF.json). | v134 prova os momentos do RHS original na fatia delimitada e preserva os bits quando a projeção é zero. Não transportar os verificadoresv131–v137. | Sem timing A/B isolado; os casos corrigidos usam zero iterações. A restrição de dimensão evita a nova passagem na ready; não demonstra performance universal. | [mlsmr_orthogonality_contract.py](../tests/analytic_precision/mlsmr_orthogonality_contract.py); distinguir regressões do solver de recusas do verificador nos relatórios. |
| **Pesos heterogéneos1 vs1e-20:** b1,7 em vez de0,7 em64 linhas, dois FE. **Lifecycle-pesos/médias ponderadas.** | [Log2.26.2](../Verifications/group_indiv/audit_20260906/active_2262_ordinary_analytic.log), [oracle racional36casos](../Verifications/group_indiv/audit_20260906/exact_weighted_2x2_oracle.json), [v52 ainda falha](../Verifications/convergence_20260906/candidate/certification_v52/SUPPLEMENTAL_WEAK_WEIGHTS_FAILURE.json). | **Correção algorítmica mínima ainda não implementada.** Normalizar globalmente aweights não remove a razão relativa1e20. v55 acrescentou uma referência, não um hunk de projeção no estimador. Proposta localizada por massas/médias celulares noSTATE. Não importar a prova geral como "correção". | Não existe custo medido de uma correção isolada. A fórmula celular é O(Np), a validar CPU/CUDA e com transporte de FE. | [exact_weighted_2x2_oracle.py](../Verifications/group_indiv/audit_20260906/exact_weighted_2x2_oracle.py), [contrato preservadov60](../Verifications/group_indiv/audit_20260906/ordinary_guard_cpu_v60/ordinary_weak_weights_contract.json), controles homosk/HC1 e rescalas. |
| **Inferência em ruído local:** v63 com regressor bem condicionado chegou a285,1×V; uma ablação sem pesos devolveu V3,462% errada, com b correto. **Gama numérica/inferência.** | [v63](../Verifications/group_indiv/audit_20260906/ordinary_leverage_noise_cpu_v63.json), [referência](../Verifications/group_indiv/audit_20260906/ordinary_leverage_noise_exact_reference.json), [expansão literal sem pesos](../Verifications/group_indiv/audit_20260906/ordinary_layer_ablation_20260909/expanded_unweighted_FINDING.md). | Há retornos reais de fits nestes artefactos, não apenas arrays forjados. v64 passou a comparar/recusar; isso não é correção algorítmica isolada demonstrada. Não imputar a2.26.2/v48 sem o mesmo reprodutor nessa base. G1 terá de distinguir defeito herdado de efeito da experiência. | Sem custo isolado; não atribuir o custo da prova a uma correção ainda não identificada. | Reprodutores/leverage-noise e expansão acima entram no conjunto offline preservado; não constituem uma razão para manter N2 no default. |

| **Constante em slopes puros: um grau de liberdade em falta. OLS-rank/DoF.** | [Oracle e adjudicação](../build-worktrees/rebase_2262_v48_20260911/provenance/n01_execution/FAILURE_02_CLASSIFICATION.json): desenho rank6, df186; candidata devolvia df187 e V×186/187. O erro precede N01. | [Patch mínimo aplicado](../build-worktrees/rebase_2262_v48_20260911/provenance/n01_execution/PROPOSED_DF_FIX.patch); contar a constante explicitamente estimada uma vez. [Validação CPU](../build-worktrees/rebase_2262_v48_20260911/provenance/n01_execution/DF_FIX_CPU_TEST_RECEIPT.json). | Não medido isoladamente; não há alteração de solver ou tolerância. | 96 controlos de normalização e probe racional preservados na árvore isolada; incluir em T01. |
| **H02 recusa slope constante dentro dos grupos. OLS-rank/seleção.** | [Comparação com a 2.26.2](../build-worktrees/rebase_2262_v48_20260911/provenance/n01_execution/FAILURE_03_H02_ADJUDICATION.json): baseline passa rank5/df187; H02 mantém uma constante redundante e OLS recusa. | [Correção estrutural](../build-worktrees/rebase_2262_v48_20260911/provenance/h02_constant_span/H02_CONSTANT_SPAN.patch): slope não nulo e exatamente constante por grupo representa os níveis. Decidir na amostra retida; preservar constante com slopes variáveis. Validação em curso. | Não medido; verificação limitada ao caminho com slopes, sem tolerância nova. | [Reprodutor independente](../build-worktrees/rebase_2262_v48_20260911/provenance/n01_execution/df_scope_controls.py); acrescentar savefe e pesos no teste permanente. |

### Correção localizada de pesos/médias: o que está identificado

O exemplo tem quatro células worker×firm,16 réplicas/célula, massas positivas
Wc, médias ponderadas μc e contraste s=(+1,−1,−1,+1). Para cada RHS, a
projeção correta conserva a variação dentro da célula e o único contraste
ortogonal aos FE. Uma forma estável, evitando inversões de pesos globais, é
rc = sc·(m/Wc)·sum(sd·μd)/sum(m/Wd), m=min(Wc), seguida de
vtilde_i=(v_i−μc)+rc. Isto é uma operação de médias/projeção no estimador,
não um majorante de aceitação. Há material de médias ancoradas em
[v55,linhas44–65](../Verifications/convergence_20260906/candidate/source_v55/ordinary_cycle_reference.hpp).

N01 foi entretanto implementado na cópia isolada, com aplicação CPU/CUDA e
composição dos efeitos. A validação CPU passou 144 casos do contrato original
e 256 casos com variação intracélula; [recibo](../build-worktrees/rebase_2262_v48_20260911/provenance/n01_execution/CPU_RECEIPT.json).
A validação CUDA está pendente. O teste novo tinha um ID FE negativo: a
troca de -11 por 11 corrigiu a avaliação sem alterar a partição ou o oracle.
Nunca trocar silenciosamente CUDA por CPU. Variância e graus de
liberdade devem usar as observações e pesos originais, não tratar as quatro
médias celulares como quatro observações para calcularV. Esta correção ficará
condicionada ao seu reprodutor e às invariâncias; não autoriza generalizações
nem novos alvos internos.

## (b) Falsas recusas e ataques ao verificador

Uma amplificação bound/erro exige a mesma grandeza, norma, escala e ponto
candidato. Um ratio bound/target ou bound/ρ não é automaticamente essa
amplificação. Quando faltam os arrays recusados ou o bound comensurável, fica
"não quantificável"; não inventar um número para preencher a tabela.

| Caso e paths | O que falhou ou foi contornado | Amplificação medida/limite da evidência | Lição: por que não pode gate-ar um fit por omissão |
|---|---|---|---|
| **256/257**: [README](../Verifications/group_indiv/audit_20260906/covariance_guard_stress_20260909/README.md), [FINDING](../Verifications/group_indiv/audit_20260906/covariance_guard_stress_20260909/FINDING.json). | Arrays transformados forjados passam guard normal/direções/coeficiente. Não havia bound direto deV. | V22forjada/V22verdadeira=256/257; erro relativo1/257≈0,00389105. Normal1,5649e-20, direção9,2951e-10 e boundβ1,4074e-21 medem objetos diferentes deV. **Sem amplificação deV calculável.** | Revela uma lacuna lógica da composição de provas; não prova que o solver produziu esses arrays. Não apertar o solver para corrigir uma injeção no verificador. |
| **2·y e alphas±1e16**: [falhav38](../Verifications/convergence_20260906/candidate/certification_v38/SUPPLEMENTAL_AFFINE_FAILURE_20260907.json), [probe](../Verifications/group_indiv/audit_20260906/affine_budget_probe_v40.cpp). | O envelope afim crescia com a magnitude dos alphas e aceitava o resíduo forjado. | Erro de norma real2, controlo correto0. Não foi guardado um bound residual final comensurável; **razão não quantificável**. | Uma mudança nula de gauge dos FE não pode aumentar a tolerância do resultado. É ataque ao certifier, não reprodução end-user. |
| **Ano2001 ausente,v102**: [assessment](../Verifications/group_indiv/audit_20260906/ready_stata_v102_ab_20260909/ASSESSMENT.md). | coarse_rank25 contra required26 porque se usavam27 posições em vez de26 categorias observadas. MédiaFE4,194e-13 já abaixo de1e-9. | Não há output recusado para medir erro real. O fit anterior válido tinha db1,076e-16/SErel1,504e-12; são outros outputs. **Não é ratio de amplificação.** | Uma coluna dummy inexistente é um erro de representação; mais iterações não o corrigem. v103 pertence ao suporte de auditoria. |
| **three_fe strict v75**: [handoff](../Verifications/group_indiv/audit_20260906/HANDOFF_ORDINARY_REMEDIATION_20260908.md), [rawv76](../Verifications/group_indiv/audit_20260906/three_fe_strict_cpu_v76/raw.json), [aceitaçãov77](../Verifications/group_indiv/audit_20260906/three_fe_strict_cpu_v77/assessment.json). | Bound forward genérico com razão42,19633, apesar de médiaFE6,933e-16<1e-12. |42,19633 é bound/target, **não bound/erro real**. Na referência v77: erroβ1,039e-15,Vrel3,759e-13,resíduo3,768e-14. | Oito reprovações eram um modelo em oito superfícies. Uma condição suficiente inconclusiva não invalida outputs confirmados pela referência. |
| **64 de72 offsets**: [RESULT](../Verifications/group_indiv/audit_20260906/covariance_guard_real_offsets_20260909/RESULT.json), [auditoria numérica](../Verifications/algorithm_history_audit_20260909/numerics/AUDIT_NUMERICAL_ECONOMETRICS.md). | Auto: limites1000iterações comρ≈1,813e-16; MLSMR: ratios média/limite0,0784–2843,45; quatro razões forward2,399e12–1,087e13. | Oito retornos tinham b/resíduos/RSS exatos e Vrel3,470e-17. **Não há arrays dos64 recusados:** sem erro real comensurável. Fraction prova invariância do modelo, não a correção de todos os estados internos recusados. | Guardar como limitação/inconclusão da verificação, não declarar64 outputs errados nem64 outputs corretos. Não criar refinamento automático só por esses ratios. |
| **Seis recusas do contrato27**: [v135](../Verifications/implementation_plan_20260909/final_inference_zero_20260911/V135_ORTHOGONALITY_CONTRACT.json), [v137](../Verifications/implementation_plan_20260909/final_inference_zero_20260911/V137_ORTHOGONALITY_CONTRACT.json), [contrato](../tests/analytic_precision/mlsmr_orthogonality_contract.py). | Três execuções escala2^200/3FE recusadas por full_covariance_bound; três unit/nonzero-FE/2FE por comparação relativa contraV verdadeira zero. | Sem bound deV escalar no raw recusado. No controlo corrigido: b/RSS exatos; resíduo até1,110e-16; V00≈4,191e-34 contra zero, com allowance2,197e-28. **Erro relativo contra zero não está definido.** | São dois modelos/configurações em três execuções, não seis erros econométricos. O contrato para zero deve usar escala aritmética prévia; não exigir quociente impossível. |
| **DIAG139 ready**: [recibo](../Verifications/implementation_plan_20260909/ready_ab_20260911/diag139_auxiliary_point/RECEIPT.json), [comparação](../Verifications/implementation_plan_20260909/ready_ab_20260911/DIAG139_POINT_COMPARISON.json), [tracev139](../Verifications/implementation_plan_20260909/ready_ab_20260911/candidate_v139_auxiliary_work/stderr.txt). | Etapa auxiliar7iterações: full_covariance_bound. A recusa posterior do original é intercept_coefficient_bound. | b escalado3,998e-15<1e-9; V escalada2,301e-12<1e-8; RSS igual à referênciav138. Os majorantes não foram exportados na métrica final por célula; **sem ratio bound/erro real deV calculável**. | Confirma conservadorismo nesse ponto, auditado com o SSC correto. Não é solução analítica exata, não compara resíduos individuais/saveFE e não certifica o domínio inteiro. |

## (c) Erros da avaliação e do harness

| Erro e evidência | Efeito/correção | Regra concreta para impedir repetição |
|---|---|---|
| **Beta Stata emfloat**: [PHASE1](../Verifications/group_indiv/PHASE1_VERIFICATION.md), [grelha corrigida](../Verifications/group_indiv/audit_20260906/chain_grid_v2_report.json). | O armazenamento arredondava a grelha; os resultados antigos não sustentam precisão FP64. | Campos numéricos usados na adjudicação são explicitamente double; validar o formato armazenado antes de imputar erro ao estimador. |
| **Sintaxe savefe errada**: [original](../Verifications/group_indiv/audit_20260906/savefe_noconstant_active_2262.log), [retry válido](../Verifications/group_indiv/audit_20260906/savefe_noconstant_active_2262_retry1.log). | r(198) era erro de parser. Autoridade: `absorb(worker, savefe)`. | Confirmar que o comando estima o modelo solicitado; erro de sintaxe não é resultado econométrico. |
| **QR com514pivôs/rank511**: [adversarial_summary](../Verifications/convergence_20260906/candidate/adversarial_summary.json), [programa](../Verifications/convergence_20260906/adversarial_contracts.cpp). | O oracle de matriz singular usava a dimensão errada; passou a projetar na baseQ de rank identificado. | Dimensão da projeção vem do rank apurado, não do número de coeficientes/pivôs usados porQR.solve. |
| **aweights não normalizados no oracle**: [finding](../Verifications/group_indiv/audit_20260906/covariance_guard_stress_20260909/product_probe_FINDING.md), [raw corrigido](../Verifications/group_indiv/audit_20260906/covariance_guard_stress_20260909/product_probe_v92_v2.json). | RSS/bread preliminares tinham fator dois; não era erro do produto. | Aplicar a mesma normalização pública antes de comparar RSS, bread eV. Não confundir escala global com heterogeneidade1/1e-20 da tabela(a). |
| **Wrapper rc0 com fit falhado**: [v69](../Verifications/group_indiv/audit_20260906/core_credit2_v69/receipt.json), [log](../Verifications/group_indiv/audit_20260906/core_credit2_v69/run.log), [v70](../Verifications/group_indiv/audit_20260906/core_credit2_v70/run.log). | O agregador concluía apesar da linhaFAIL; razão forward8,105e5. | Exigir sucesso da célula, outputs, backend e convergência pertinentes. rc do processo agregador não é o gate. |
| **ABI/Eigen sem-march=native**: [falha](../Verifications/group_indiv/audit_20260906/mlsmr_cache_contract_v57/receipt.json), [ABI correspondente](../Verifications/group_indiv/audit_20260906/mlsmr_cache_contract_v57_matching_abi/receipt.json). | returncode−11 foi atribuído inicialmente ao cache; com flags coerentes passaram10/10. | Reproduzir as flags ABI/Eigen da biblioteca em probes ligados a objetos e guardar o comando completo. |
| **np.add.at em scores float**: [handoff](../Verifications/group_indiv/audit_20260906/HANDOFF_ORDINARY_REMEDIATION_20260908.md), [oracle corrigido](../Verifications/group_indiv/audit_20260906/ols_extreme_condition_contract.py). | Cancelamento numérico no oracle cluster; os scores inteiros foram acumulados antes dos pesos. | Quando a construção permite aritmética exata, usá-la antes da escala. np.add.at não é intrinsecamente inválido, mas a sua soma float não prova exatidão. |
| **Builderv99 repôs bytes antigos das interfaces**: [cronologia](../Verifications/algorithm_history_audit_20260909/timeline/TIMELINE.md), [resultado alinhado](../Verifications/group_indiv/audit_20260906/interface_validation_v99/RESULT_ALIGNED.md). | Core/DLLv99 corretos coexistiam com layersv38 antigas; ado82, pluginbinding72 eR60_stats corrigiram8/8 superfícies sem mudar o core. | Manifestar e conferir separadamente core, headers, bindings, wrappers e binários; hash do DLL sozinho não certifica a interface carregada. |
| **Exigir t/p finitos quando b=SE=0**: [harness preliminar preservado](../Verifications/implementation_plan_20260909/executor/v129_range_gate_initial.py), [contrato corrigido](../tests/analytic_precision/ols_inference_range_contract.py). | Uma inferência legitimamente indefinida foi inicialmente contada como falha. A correção distinguiu b/V/u válidos de estatísticas t/p não definidas. | Finitude aplica-se a quantidades matematicamente definidas; respeitar casos saturados e zero/zero. Não deformar o estimador para preencher um t-value impossível. |

## Achado no reghdfe e material para Sérgio

A investigação original está no [relatório de5–6set](../Verifications/marta_regression_20260905/RELATORIO.md).
Na versão auditada, `LSMR.mata:235` usa `rel_neq=norm_At_r/norm_A/norm_x`.
A experiência isolada substituiu apenas o denominador por `norm_r`; não
alterou o reghdfe canónico. A [proveniência](../Verifications/marta_regression_20260905/reference_normr_experiment/PROVENANCE.json)
e a [integridade](../Verifications/marta_regression_20260905/reghdfe_integrity_check.json)
ficam preservadas. No braided128, a alteração isolada reduziu o erro de
coeficientes de0,5600 para2,46e-11, com a mesma tolerância. É uma causa
suficiente naquele sintético, não uma atualização universal pronta a instalar.

[REPRO_REGHDFE_ONLY.do](../Verifications/marta_regression_20260905/REPRO_REGHDFE_ONLY.do)
é autónomo, gera dados emStata e já teve o marcador
REGHDFE_ONLY_REPRODUCER_PASS. [DEMO_GROUP_INDIV_SUM.do](../Verifications/marta_regression_20260905/DEMO_GROUP_INDIV_SUM.do)
e [OLS_ANALITICO_RESULTADOS.md](../Verifications/marta_regression_20260905/OLS_ANALITICO_RESULTADOS.md)
preservam a comparação independente. O
[UPSTREAM_REPORT_DRAFT.md](../Verifications/marta_regression_20260905/UPSTREAM_REPORT_DRAFT.md)
está preparado. O [STATUS local](../Verifications/marta_regression_20260905/STATUS.md)
diz que não foi submetido um issue; não encontrei comprovativo de envio ao
Sérgio. Logo o estado demonstrado é **preparado e reproduzido localmente;
envio/receção não confirmados**. Nenhuma mensagem foi enviada neste trabalho.

Nos dados da Marta, as imagens mostram20.618iterações e RMSE menor noxhdfe.
Contagens iguais e RSS menor não provam identidade da amostra/especificação
nem correção empírica. Sem os dados e diagnósticos dela, permanece pendente a
atribuição causal. A frente de condicionamento group/individual, incluindo
laplacian1024, é a prioridade de performance **depois dos gates**, como decidiu
o utilizador.

## Discordâncias factuais delimitadas com a auditoria Claude

Estas correções não põem em debate qualquer decisão da secção1 do utilizador.

1. **F2/"últimaCUDAreadyv3":** existem CSVv47 nas oito variantes,44linhas por
   variante,22datasets×A/B. Por exemplo, [C++CUDAcomparable](../Verifications/convergence_20260906/candidate/performance_v47_core22/cpp_cuda_comparable/runs.csv)
   regista ready9,892→14,162s comgpu_used1 e77iterações; [StataCUDAcomparable](../Verifications/convergence_20260906/candidate/performance_v47_core22/stata_cuda_comparable/runs.csv)
   regista17,129→25,264s. Assim, "desdev3" é inexato. Continua a faltar a
   matriz core23×8 atual, intercalada e repetida; v47 teve depois falha de
   omissão e não é aceitação de produto.
2. **"Toda adjudicação numa célula"** é demasiado absoluto. Houve a
   [campanha funcionalv129](../Verifications/implementation_plan_20260909/executor/V129_FULL_CAMPAIGN.json)
   emC++/Python,Stata eR CPU/CUDA e timingsStataCPU posteriores. Isso não
   substitui os timings completos exigidos e não enfraquece a crítica à
   otimização centrada na ready.
3. **F4 mistura duas recusasv139:** a etapa auxiliar de7iterações falhou
   full_covariance_bound; intercept_coefficient_bound pertence à tentativa
   posterior sobre o original. A amplificação1,6e6 apresentada como
   bound/ρ mistura um bound absoluto porRHS com um residual relativo. Sem
   converter para a mesma norma/escala e medir erro verdadeiro, não é o
   bound/erro exigido pela nova regra. O conservadorismo tem prova direta em
   DIAG139, sem precisar desse quociente.
4. **"Pesos corrigidos antesv49"** não corresponde à evidência: v52 ainda
   devolveu1,7. A correção localizada por pesos/médias está identificada como
   trabalho necessário, não como cherry-pick já pronto.
5. **Ataques e erros reais:** nem todos os exemplos posteriores são ataques ao
   certifier. A expansão sem pesos e a falha leverage/noise preservam retornos
   de fits; ficam em(a), com âmbito experimental explícito. Em contrapartida,
   os64offsets não provam64outputs corretos, porque os arrays recusados não
   foram guardados. Os extremos são testes úteis de representação; não
   justificam provas pesadas em cada estimação ordinária.
6. **Custos e garantias:** "ms/zero" em vários hunks é uma hipótese sem A/B
   isolado. v47 não demonstra que a nova base cumprirá automaticamente todas
   as metas absolutas. Os números são evidência histórica para planearG1–G3.
7. **Sérgio e Marta:** envio aoSérgio não confirmado pelos artefactos; caso
   empírico daMarta não encerrado. A hipótese de condicionamento é fundada e
   orienta a frente posterior, mas não substitui a reprodução empírica.
8. **Autoridade das instruções:** as orientações explícitas do utilizador
   deveriam ter prevalecido sobre uma leitura rígida dos ficheiros. A falta
   de atualização desses ficheiros não justifica ter continuado o ratchet.

## Conclusão operacional

O código de prova preservado é evidência e instrumento de auditoria, não a
base do produto. A nova base proposta é exclusivamente v48 verificada, com
correções localizadas, N1 necessário e todos os recursos da2.26.2. O ramo
`codex/convergence-inference-20260906` é rotulado neste checkpoint como
**v52 — experiência, não candidata**; a identidade do seu
`src/fe_absorption.cpp` comv52 foi confirmada por bytes. Não houve alteração
dos refsGit nem comunicação a terceiros no checkpoint. Implementação,
compilação e gates foram depois autorizados exclusivamente na cópia seletiva
`build-worktrees/rebase_2262_v48_20260911`; o STATE contém o âmbito vigente.
A proteção group permanece por omissão, a métrica conjunta de b é apenas
diagnóstico até H2 e T01 congela a adjudicação por modo antes de G1. A fórmula
2×2 preserva quaisquer pesos positivos e a variação dentro da célula;
razão/expoente não justificam recusa. Um defeito herdado realmente reproduzido
mantém G1 reprovado, enquanto cobertura ausente fora de 2×2 não demonstra erro
em toda a classe. Estas decisões seguem a conclusão Fable/Astra de 11set,
lida integralmente e ligada no STATE.

O primeiro diagnóstico da cópia autorizada produziu quatro pares completos
na ready: acréscimos de 4,54–5,55 s com as mesmas iterações; b/V coincidem
exatamente em CPU e dentro do contrato em CUDA. Como o baseline CUDA demora
8,67–10,09 s, esse custo equivale a +51–57%. N01, a nova N1 e o adaptador N2
ainda não existiam nesse build. Não podem ser responsabilizados pelo aumento.
O perfil posterior isolou o custo em `precise_ols_rank_mask`: 0,399→5,850 s,
com absorção e OLS estáveis. A otimização mecânica preservou as 240 máscaras
testadas e reduziu essa fase a 2,775 s; o fit ficou 11,119 s contra 8,996 s,
ainda +23,6%, sem aceitação de performance. [Perfil inicial](../build-worktrees/rebase_2262_v48_20260911/diagnostics/overhead_profile/RESULT.json),
[perfil após correção](../build-worktrees/rebase_2262_v48_20260911/diagnostics/rank_mechanical_profile/RESULT.json).
A carga
de fundo elevada impede aceitar estes pares como prova definitiva de
performance; a campanha foi parada para reportar o sinal. O
[recibo completo](../build-worktrees/rebase_2262_v48_20260911/diagnostics/early_ab/CHECKPOINT.json)
preserva o âmbito e as limitações. Naquele checkpoint G1–G3 continuavam por executar.

## N1 executado em staging — 12 setembro 2026

N1 foi integrado nos sete alvos autorizados, compilado com OpenMP em CPU e
CUDA sm_90 e testado em C++/Python, Stata e R. As fontes ficaram congeladas
desde os primeiros builds. Mantiveram-se os modos, tolerâncias, proteção
forward group, `fe_tolerance` e correções H02/N01. Os
[recibo e inventário finais](../build-worktrees/rebase_2262_v48_20260911/provenance/n1_execution_20260912/N1_RECEIPT.json)
e o [relatório](../build-worktrees/rebase_2262_v48_20260911/provenance/n1_execution_20260912/N1_REPORT.md)
documentam os resultados e a recuperação por ficheiro.

| Falha e classe | Reprodutor e correção | Validação permanente |
|---|---|---|
| Header N1 inerte: **missing coverage**. | Integração de momentos FE assinados, normal/finitude fundidas com u/RSS, identidade saveFE e uma continuação no orçamento restante. Nenhum alvo novo. | `n1_evidence_contract.cpp`, `n1_iv_cache_contract.py` e `n1_fault_injection_contract.py`, em `tests/analytic_precision/` da árvore seletiva. |
| 35 recusas previstas eram rotuladas como erro: **evaluation-harness error**. | Padrões de Schwarz ponderado CPU e LSMR/MLSMR CUDA não correspondiam à mensagem do método. H02/N1 devolveram a mesma recusa e estado limpo. Corrigidos catálogo e captura de diagnostics em `run.py`/`native.py`. | Reprodutores e negativos em `provenance/n1_execution_20260912/schwarz_harness/`; apenas 35 linhas repetidas, todas `UNSUPPORTED_EXPECTED`. Erros alheios e estimativas retidas continuam a falhar. |
| 26 normais agrupadas falhavam apenas no T01: **evaluation-harness error**. | O avaliador usava Xraw em lugar de Xtilde. Projeção ponderada independente no D explícito da mesma amostra fez os rácios passar de 1,019e-9…3,185e-9 para 1,53e-17…6,59e-17, com tau=1e-9 inalterado. Diagnóstico Xraw preservado. | `t01_normal_score_contract.py`: 539 reavaliações PASS e dois negativos efetivos. Recibo em `t01_n1_harness/RECEIPT_002.json`; nenhum allowance ou solver alterado. |
| Identidade ausente no erro R, quatro padrões agrupados sobrepostos e Strict contado como falha T01: **evaluation-harness error**. | O worker passou a conservar DLL/package também na recusa. O override Schwarz passou a respeitar `case.json.kind`; o Boolean T01 considera só `applicable`, conservando contagens externas e avaliação histórica Strict. | Par R sucesso/recusa, quatro reexecuções e negativos em `schwarz_harness/` e `r_harness_final/`. Falha histórica Strict continua a bloquear o resultado global. |

Preservação H02/N01: **623 CPU +454 CUDA PASS**. IV/cache: **22 CPU +16
CUDA PASS**. Condições N1 offline: **75 PASS**. Controlo de falha injetada
apenas no processo de teste: **6 CPU +6 CUDA PASS**; uma falha produz uma
continuação ao mesmo método/backend/tolerância, duas falhas deixam estado
`failed` e coeficientes, covariância, resíduos e amostra vazios. O controlo
encaminha os fits aceites para a função real; não representa erro natural do
estimador. Os dois ensaios naturais anteriores não acionaram N1 e permanecem
registados com essa limitação.

Após adjudicação dos erros de avaliação, a matriz principal contém **619
estimações xhdfe PASS +223 recusas previstas**; R matrix/formula contém
**626 PASS +194 recusas previstas**. Acrescem seis controlos Fast CUDA
C++/Stata e seis R. Reavaliações reutilizam os mesmos fits; recusas não são
estimações corretas. Reports e raws originais com os erros foram preservados.

O resultado é funcional e delimitado. N05/H09, o restante âmbito G1, a medição
do custo N1 em ready CPU16/CUDA e G2/G3 permanecem pendentes. Não houve
aceitação de performance, alteração de código ativo, publicação ou instalação
ativa. As tolerâncias e a discussão de precisão continuam fechadas.

## Continuação N05 e novos controlos G1 — 12 setembro 2026

O [checkpoint N05 nativo](../build-worktrees/rebase_2262_v48_20260911/provenance/n05_execution_20260912/NATIVE_FINAL_CHECKPOINT.json)
regista a auditoria observacional sob `XHDFE_CERTIFY=1`, sem refinamento nem
mudança do candidato. Recibos distinguem provas, limites inconclusivos e
âmbitos não suportados. Gate/counter, cache e identidade passaram em 36
controlos CPU, 29 CUDA e 34 C++ privados; preservação 623+454 PASS. Os builds
Stata/R finais foram ligados aos mesmos archives identificados; os controlos
low-noise default/audit nas interfaces passaram 192/192. A recusa CUDA/Strict
foi resolvida como descrito abaixo; a bateria funcional delimitada passou.

- **FE redundante — demonstrated estimator error.** O oracle Fraction na
  amostra representada prova b=3/4, rank4/df60; N1 devolvia aproximadamente
  7/4 ao acrescentar constantes/duplicados/recodificações FE, também com
  saveFE. Falharam 60/96 controlos N1 e 72/96 na pública 2.26.2. A correção
  algébrica estende a representação 2×2 apenas a partições comprovadamente
  redundantes, preservando slots, metadados e normalização. Passaram 96 CPU
  +96 CUDA e 24+24 negativos com interação/divisão intracélula. Evidência em
  `provenance/n05_execution_20260912/redundant_fe_fix`; testes permanentes
  `redundant_fe_exact_contract.py` e `redundant_fe_distinct_space_contract.py`.
- **HC1 low-noise — demonstrated estimator error.** A referência racional
  completa de rank8 confirma Gram256, meat512·small² e V=32·small²/4088.
  Os 15 inputs são exatamente representados; 11 resultados N1 falham também
  com a margem T01 preexistente, com erro relativo V máximo284,10. Não há
  margem derivada do erro observado nem motivo para polir tolerâncias.
  Referências e adjudicação em `baseline_n1_supplemental/LOW_NOISE_*` dentro
  da provenance N05; o contrato permanente existente é
  `supplemental/ordinary_leverage_noise_contract.py`. A correção algébrica
  de médias celulares aditivas passou 15+15 casos originais e 1336+1336
  estimações adicionais em CPU/CUDA. A elegibilidade, exclusões, threads e
  precisão efetiva dos workers passaram 447 controlos privados; não se muda
  o controlo de precisão dos processos. O primeiro build não entrava no ramo:
  `from_auto` já fora consumido pelo seletor. GDB confirmou a causa e uma
  intenção Auto privada com RAII corrigiu a ligação, preservando o seletor.
  A geração das caches foi avançada: dois erros antigos foram reproduzidos
  e guardados; oito chamadas da candidata confirmaram miss antigo, hit novo
  e miss perante alterações dos inputs. Não se reutilizam resultados errados
  só porque os dados coincidem.

Erros dos novos testes foram classificados antes de corrigir: igualdade de
floats entre processos CUDA, limite arbitrário1e-12 em partial_out e pedido
`dofadjustments(exact)` fora de group/individual. Os diagnósticos partial_out
N1/N05 foram idênticos (erro y1,09246e-12, rho4,22383e-14). As correções
ficaram no harness; os raws iniciais foram preservados e o estimador manteve
os seus contratos. Os timings de aceitação aguardam o fecho das falhas G1.

Dois outros **evaluation-harness errors** foram corrigidos antes da matriz:
`weights_contract.py` importava o checkout ativo e lia um fixture histórico
fixo, ignorando a raiz seletiva; passou a derivar a raiz do próprio ficheiro e
a receber `--fixture`. O teste `ols_wide_arithmetic.py` extraía helpers
adicionais depois do struct pretendido, causando erro de compilação; passou
a extrair apenas `OlsWide`. Nenhuma fórmula, tolerância ou operação aritmética
do estimador foi alterada. Passaram 42 linhas Stata e 28+28 Python nos pesos,
e 4800 referências racionais mais 12 controlos de zero na aritmética. Fontes
anteriores, classificação e falhas preservadas em `weights_harness_fix/` e
`g1_ols_wide/`, na provenance N05; ambos os testes são permanentes.

## Fecho N05/N1 — recusa indevida e avaliadores

**Undue verifier refusal.** No primeiro caso CUDA de `group_sample_contract.py`,
N1 recusava o momento FE combinado usando `tau*sqrt(D_g)*||u||`, embora o
certificado de absorção original passasse. Dois OLS e duas avaliações foram
capturados sem modificar o produto. A referência Fraction confirmou, antes
e depois da única continuação, erros b até 3,05e-16, V escalado até 1,58e-14
e u até 4,97e-13. O momento capturado concordava com a soma direta dentro
da margem aritmética; a escala adicional da recusa era o problema.

O hunk aprovado retirou apenas essa chamada de recusa. Os certificados de
absorção/FE, normal/finitude, reconstrução e orçamento N1 continuam. O checker
derivado fica offline. O teste permanente `n1_group_sample_refusal_contract.py`
e os 24 casos originais passaram em CPU/CUDA, sem bypass. A fonte principal
final é `ad4b82da6b7bf7552743dc5aa64406e1a9a6f7a6d4669269980059d30786f3e4`.
Reprodutor, referência, snapshots e adjudicação em
`provenance/n05_execution_20260912/g1_standalone_native/cuda_group_sample_diagnosis/`.
Não reintroduzir o requisito nem apertar o solver para o satisfazer.

Outros **evaluation-harness errors** ficaram separados do estimador:

- R saveFE construía nomes `beta.x`/`variance.x` e depois procurava `beta`/
  `variance`, falhando após fits válidos. `unname` corrigiu apenas os nomes;
  192/192 casos passaram por backend e modo de auditoria.
- O agregador aplicava a xhdfe as falhas dos comparadores reghdfe. O âmbito
  foi corrigido, preservando os relatórios originais; um erro xhdfe injetado
  continua a reprovar a agregação.
- O novo controlo de espaços FE exigia resíduo absoluto 1e-8, contra o
  `RESIDUAL_ABS=1e-7` já fixado em T01. b/V estavam corretos e a 2.26.2
  reproduzia a mesma aproximação sem saveFE. Usar o limite existente fez
  passar 24+24 controlos, sem modificar o estimador. Os erros grandes de
  saveFE da 2.26.2 permanecem distintos e corrigidos na candidata.

O [relatório final](../build-worktrees/rebase_2262_v48_20260911/provenance/n05_execution_20260912/N05_REPORT.md)
e o [recibo](../build-worktrees/rebase_2262_v48_20260911/provenance/n05_execution_20260912/N05_RECEIPT.json)
preservam os resultados e a custódia. Main: 619 PASS +223 recusas previstas;
R: 626 PASS +194 recusas previstas, em cada campanha default/audit.
Suplementos 13/13; guardas nativos 17/17. O smoke CPU/Stata convergiu em
23 iterações, com 16 threads, 46 156 187 observações e 84,266 s. É evidência
funcional; custo incremental N1, G2/G3 e promoção continuam pendentes.

## Custo N1 observado; decisão de timing inconclusiva — 12 setembro 2026

A [campanha N1](../build-worktrees/rebase_2262_v48_20260911/provenance/n1_cost_20260912/N1_COST_REPORT.md)
terminou 44/44 fits ready válidos em CPU/CUDA × Fast/Comparable, com 22 usos
reais da H100, zero refinamentos e zero trabalho N05. As quatro cópias CPP
instrumentadas ficaram em provenance; os 793 fontes/headers/frontends e os
binários finais permaneceram intactos. Os 16 controlos pequenos, 40
comparativos ready e cinco grupos de controlos do analisador passaram.

O custo é medido pelo intervalo inclusivo desde o primeiro certificado até
ao retorno do fit, incluindo retenção e destruição da evidência. Somar apenas
os checks ou chamar a todo o tempo OLS "custo N1" não identifica o incremento.
Os masks none/fe são contrafactuais privados, sem estatuto de produto.

Incrementos médios brutos: CPU Fast 0,419 s, CPU Comparable 1,125 s, CUDA
Fast 0,651 s e CUDA Comparable 0,357 s. Todos os blocos e referências tiveram
carga externa acima dos 10% fixados previamente (19,48–27,42%). Permanecem
excluídos da decisão CPU <2 s; diferenças negativas foram conservadas.
Classificação: **missing coverage** para uma medição com baixa carga. Não
houve erro do estimador nem justificação para alterar as tolerâncias. G2/G3
aguardam uma repetição válida deste diagnóstico com os mesmos bytes.

O primeiro adapter tentou chamar `as_list`/`slopes_of` como exports do módulo,
embora fossem helpers locais a `enc_worker.main`: **evaluation-harness error**,
com zero fits executados. A leitura direta dos campos do registry corrigiu
só o harness. Fonte anterior, erro e classificação ficaram em `ready/`;
`ready_001/` contém a repetição completa. O runner e o reprodutor preservados
permitem revalidar o mesmo caminho sem modificar o estimador.

## Continuação autorizada e fecho funcional G2 — 12 setembro 2026

Depois de comparar ready32 em CPU/CUDA Comparable, o utilizador aceitou
provisoriamente esse tipo de custo para preservar as correções e autorizou
G2/G3 sob a carga persistente. Isto supera o bloqueio de execução descrito
acima, sem reclassificar a antiga medição N1 nem aprovar custos por medir.
G2 concluiu 192 registos, 180 fits novos válidos e 12 reutilizados, em
32 workers nativos. A campanha completa e os reprodutores estão em
`provenance/g2_g3_20260912/` na árvore seletiva.

- **Evaluation-harness error, configuração Stata.** O primeiro wrapper
  pediu `set processors 32`, mas a licença Stata/MP permite 16. Falhou antes
  do fit. O wrapper conserva essa configuração e usa `numthreads(32)` no
  plugin OpenMP; os oito controlos C++/Stata confirmaram 32 workers reais,
  resultados e backend. Licença do Stata e equipa do plugin são distintos.
- **Evaluation-harness error, exportação.** Um fit simulated_panel válido
  terminou, mas a exportação de 173 163 263 linhas excedeu 300 s. A leitura
  repetida `st_data` por fatia era o custo dominante: em 20 milhões de linhas,
  ler uma vez reduziu 8,669 s a 0,356 s; a exportação completa demorou 2,953 s.
  `stata_worker_v4.do` conserva os bytes do bitmap e o bloco cronometrado.
  Reprodutores em `export_probe/run_001` e `run_002`; manifesto original,
  bitmap parcial e PROCESS falhado preservados. O manifesto de continuação
  retoma só os slots sem resultado válido.
- **Evaluation-harness error, referência residual Fast.** Os três pares
  Marta C++ CPU Fast diferiam da baseline nos resíduos amostrados em 2,02e-6,
  acima de 1e-7, com b/V/amostra/DoF/RSS aprovados. Os replays confirmaram
  identidade dos inputs e resultados. OLS explícito esparso sobre uma das
  seis réplicas completas e idênticas, usando LSQR e LSMR independentes,
  concordou nos resíduos em 4,44e-12. A candidata tem erro máximo 7,03e-12;
  a aproximação CPU Fast publicada tem 3,55e-6. Usar esta aproximação como
  referência exata de resíduos produziria um falso veredito de regressão.
  A decisão conserva os três REVIEW originais e adjudica PASS pela referência
  independente, com o limite original 1e-7. O oracle é de ponto flutuante;
  a covariância reportada é condicional ao DoF já comparado. Não se declara
  um novo erro de produto da 2.26.2 nem se altera o contrato Fast.
  Reprodutores congelados: `marta_residual_diagnosis/replay_worker.py` e
  `marta_oracle/oracle.py`; protocolo em `marta_oracle/PROTOCOL.md`.

O [recibo adjudicado](../build-worktrees/rebase_2262_v48_20260911/provenance/g2_g3_20260912/G2_ADJUDICATED_RECEIPT.json)
regista 93 pares PASS diretos e três PASS independentes, sem mudanças no
estimador ou tolerâncias. A correção forward group preservada explica a
continuação CPU Fast Marta de 235 para 462 iterações; o custo de cerca de
12 s contra 3,7 s é reportado separadamente. G3 pode executar; performance
global e publicação ainda não têm GO.

## G3 — diagnósticos delimitados e baseline sem convergência

Após 304 fits válidos, 145 dos 152 pares passaram diretamente e sete
exigiram classificação. O utilizador reiterou que o rigor não autoriza
loops de precisão irrazoáveis. Fez-se uma rodada fixa de referências
offline, sem alterar produto, tolerâncias, max_iter ou métodos.

- **Demonstrated estimator error preexistente: CUDA Comparable em directors
  e PF difficult 10M 3FE.** As referências independentes confirmam desvios
  tanto na 2.26.2 como na candidata. Em PF, a eliminação explícita dos
  indivíduos produz uma normal inteira esparsa; duas gauges concordam nos
  resíduos até 1,78e-15. O erro máximo b CUDA da candidata é cerca de 4,18e-7
  (0,00125 SE), contra cerca de 4,25e-7 na baseline; V escalada cerca de 6,9e-6.
  CPU Comparable concorda em b até 8,31e-14. Em directors, LSQR e LSMR
  independentes confirmam b CUDA com erro cerca de 5e-8 nas duas versões.
  Os quatro pares REVIEW não são convertidos em PASS, nem se atribuem esses
  erros a N1/N05. Fast conserva a aproximação publicada e não recebe o
  contrato Comparable por causa destes diagnósticos.
- **Missing coverage: workers.** A única rodada LSQR/LSMR divergiu em u até
  3,38e-7, acima da resolução predeclarada da referência. Os dois pares ficam
  inconclusivos; não houve retry, novo alvo ou polimento do estimador.
- **Missing coverage aritmética: zigzag.** O grafo de contrastes e Fraction
  provam b e V exatos, com V=0. Os valores Stata CPU Comparable são da ordem
  de 2e-26/6e-26 e compatíveis com o piso condicional de avaliação final
  derivado dos operandos. Esse piso não foi apresentado como enclosure do
  backend inteiro. Preservar esta limitação sem transformar valores de
  arredondamento em motivo para apertar o solver ou desenvolver mais provas.

Reprodutores permanentes na árvore seletiva:
`provenance/g2_g3_20260912/pf_direct_oracle/` e `small_case_oracles/`.
PF executou em 24,790 s; os outros três em 2,148 s, 79,993 s e 0,155 s,
sem novos fits xhdfe. Protocolos, selftests, inputs, resultados e recibos
estão congelados. A decisão é terminar a cobertura com qualificações, sem
mais rondas de precisão nesta fase.

AKM secondreg C++ CPU Fast 2.26.2 excedeu primeiro os 420 s de fit. A única
ampliação para 1800 s terminou em 983,397 s/543 iterações, mas a baseline
declarou `converged=false` e `precision_certified=false`; rho foi 4,53e-13.
O retorno não é uma estimação válida nem permite inferir sozinho erro real
ou recusa indevida. **Missing coverage** de uma baseline válida. Tentativa,
warning, amostra e resultados finitos ficam preservados; nenhuma tolerância
foi mudada para obter PASS. A execução prossegue nas outras 19 especificações;
o manifesto mestre conserva uma falha AKM2 observada e 47 slots pendentes.

## Fecho delimitado G2/G3

O [relatório final](../build-worktrees/rebase_2262_v48_20260911/provenance/g2_g3_20260912/REPORT.md)
regista 1104 execuções válidas, 552 pares, 530 PASS (três por adjudicação
independente) e 22 REVIEW, em 23/24 especificações. Os oito contratos foram
executados três vezes por papel em cada especificação observada. A ausência
AKM2 e as qualificações numéricas impedem um GO global. O fecho da campanha
não autoriza publicar nem retomar loops de precisão.

Os custos adicionais ficaram explícitos: Marta C++ CPU Fast 3,675→12,301 s;
simulated_panel Stata CUDA Comparable +19,789 s/+39,36%. Comparações sob
carga e dispersão de três pares não são testes de significância. Todos os
793 fontes, sete artefactos finais, 1897 inputs e 22 dados foram conferidos
no fecho; ativos/worktrees conservam o estado anterior.

**Evaluation-harness error de agregação.** Os seis recibos ready32 CUDA
reutilizados chamam `gpu_status` ao campo que os novos chamam
`gpu_status_code`. A primeira tabela marcava False no agregado status1 dessa
célula, apesar dos seis valores observados serem 1. A v2 lê os dois nomes:
a única diferença substantiva do CSV é esse False→True, com tempos e
contagens inalterados. Script/reprodutor `aggregate_core24_v2.py`; v1 e v2
preservados. Confirmaram-se 184/184 células com 32 workers nativos e 92/92
CUDA com uso real/status1. Nenhum fit foi repetido por este erro de metadata.

## AKM2 — recusa de aceitação distinguida da última convergência

O [trace privado](../build-worktrees/rebase_2262_v48_20260911/provenance/g2_g3_20260912/akm2_slope_trace/REPORT.md)
foi autorizado e executado uma vez. Só se acrescentou registo a uma cópia
de fe_absorption.cpp; remover os oito blocos restitui os bytes originais.
A última etapa convergiu e passou o certificado de bloco (3,68349e-8 ≤ 8e-8),
mas falhou o alvo adicional 1e-8. A saída por esgotamento da sequência mudou
os flags para falsos sem trabalho numérico novo. O máximo era em idtrab/
exp_akm, grupo interno 2217218, diagonal 16. O resíduo global 4,53485e-13
mostrado no erro não era o indicador decisivo.

O diagnóstico deve preservar estes estágios de decisão: convergência da
última etapa, certificado de bloco e aceitação final. Corrigir a mensagem
e publicar os dois limites não equivale a remover o alvo adicional, já
presente na 2.26.2. O trace não demonstrou erro de estimação ou recusa indevida;
a legitimidade numérica desse alvo permanece missing coverage. Não aceitar
o fit só porque um dos certificados passou, nem procurar nova precisão.

A execução preservou inputs, 543 iterações e erro da candidata congelada,
observando 32 workers. Tempo 944,409 s exclusivamente diagnóstico. As quatro
continuações com alvos internos menores já pertenciam ao código congelado;
não foram introduzidas pela instrumentação. Nenhuma correção de produto ou
novo oracle foi aplicado. Fontes, sete artefactos finais, dados e checkouts
permanecem intactos. Adjudicação e fonte privada ficam na mesma pasta.

## Âmbito das outras recusas e tentativa de referência AKM2

A [avaliação delimitada](../build-worktrees/rebase_2262_v48_20260911/provenance/g2_g3_20260912/refusal_scope_review/REPORT.md)
separou recusas runtime, negativos esperados e alertas de comparação. Os
22 REVIEW G3 são 44 fits retornados e não usam slopes/group; não percorrem
o gate AKM2. Os 417 negativos lógicos main/R não são todos os controlos G1:
SAFE_ERROR/SAFE_RANGE_ERROR, fault injection e suplementos foram tratados
na adenda, sem somar possíveis sobreposições. Não apareceu outra recusa
natural diretamente observada do mesmo gate; a instrumentação disponível
não permite uma conclusão universal. Resets em N1, group, cache e opt-ins
têm condições próprias e não autorizam remover guardas em conjunto.

A pesquisa não encontrou referência independente anterior comparável do
AKM2 completo. Uma execução reghdfe com o modelo literal, tolerance1e-12
fixa e dependências privadas passou o preflight, mas atingiu o limite externo
de 1800 s durante o fit. Não houve b/V, bitmap ou convergência final; nenhuma
comparação numérica foi feita. Classificação: missing coverage, sem inferir
erro de estimação ou recusa indevida. A contagem de singletons coincidiu,
mas contagem não comprova identidade de amostra. Não foi repetido o fit.

A candidata AKM2 recusou devolver resultados; a 2.26.2 Python/C++ devolveu
valores com aviso. Esse facto não demonstra que os valores fossem errados.
As fontes, sete artefactos finais, dados e 28 dependências ficaram intactos.
Corrigir o diagnóstico continua sustentado; retirar critérios de aceitação
continua sem fundamento numérico novo nesta avaliação.

## 13 setembro — saída CUDA localizada e mensagem de recusa corrigida

Dois replays directors, com os mesmos inputs, amostra e opções, confirmaram
o erro herdado CUDA Comparable. A saída nominal honest ocorreu em 206
iterações; Fast usou a reparação CG já existente (86+234). A revisão não
encontrou troca de buffers, tolerância mal passada ou estado antigo. O erro
numérico demonstrado permanece; o trace não autoriza alterar o alvo nominal
nem ativar provas suficientes por defeito. A proposta de completar a última
extrapolação não sustentava uma mudança de decisão e foi encerrada antes de
build. Reprodutor e adjudicação em
`build-worktrees/rebase_2262_v48_20260911/provenance/g2_g3_20260912/directors_cuda_20260913/REPORT_v2.md`.

A mensagem da recusa AKM2 foi corrigida na fonte privada, sem mudar o
estimador ou a condição de aceitação. Passou a distinguir sequência de
continuações esgotada de limite de iterações, mostrando medida de bloco e
os dois limites existentes. Não atribui aos flags finais o estado apagado
da última etapa. Reprodutor permanente:
`build-worktrees/rebase_2262_v48_20260911/tests/analytic_precision/absorption_failure_diagnostic_contract.py`.
A função real compilada falha o diagnóstico na fonte antiga e passa a
fixture AKM2, 96 estados de aceitação/recusa e os controlos específicos na
fonte corrigida, SHA `281168ab5b920133dae7daba09a4a7ce229f15aac85ff9b3e22195458587b550`.
A correção está integrada na fonte privada; os binários finais ainda não
foram reconstruídos. A legitimidade numérica da recusa permanece missing
coverage, à espera do reghdfe em background sem limite de tempo.

No host, o supervisor e o Stata continuam vivos; o sandbox pode ocultar os
PIDs. Não concluir término nem lançar duplicados a partir de ps restrito.
Verificar PID e start_ticks no contexto host de lançamento. A espera deve
ficar no processo, sem vigilância contínua do modelo.

## 13 setembro — OpenMP na distribuição e limites da evidência

A presença de duas arquiteturas num Mach-O e uma ligação OpenMP não prova
trabalho paralelo. A infraestrutura privada passou a exigir testes SPI dos
dois plugins, com pedidos 1/2, workers observados 1/2 e acordo numa fixture
analítica, sobre os universais finais e com negativos seriais separados.
O host do teste não liga o core nem OpenMP; carrega pginit/stata_call dos
plugins. O smoke Linux real passou com erro zero; os controlos macOS
simulados passaram, mas a validação macOS nativa permanece missing coverage.

O runtime é entregue como xhdfe_libomp.dylib para partilhar a pasta x dos
plugins; G maiúsculo garante a instalação da biblioteca pelos pacotes Stata.
ID/RPATH, arquiteturas, mínimos de OS e hashes são verificados. A biblioteca,
licença, recibos e fonte LLVM fixada acompanham os caminhos de distribuição.
Os testes não dão GO aos REVIEW numéricos nem certificam performance.

A revisão encontrou dois erros de infraestrutura: backticks não escapados
num heredoc executariam texto do guia offline; a flag de release do workflow
não atravessava docker run. Ambos foram corrigidos e verificados. Comandos
em documentos gerados exigem teste de renderização, além de bash -n.

Relatório e reprodutores:
`build-worktrees/rebase_2262_v48_20260911/provenance/openmp_release_20260913/REPORT.md`.
Testes permanentes: `tests/test_macos_openmp.py` e os probes em tools/ na
mesma árvore privada. Estimador, tolerâncias e sete binários finais ficaram
intactos nesta fase; CI remoto, alinhamento e publicação continuam pendentes.

## 13 setembro — revisão finita dos 22 REVIEW e impacto numérico

REVIEW numa comparação externa não significa recusa do fit: os 44 fits
correspondentes devolveram b/V com convergência e rc=0. A revisão reutilizou
exclusivamente resultados e referências existentes. Fechou um caso como PASS
funcional — workers Stata CUDA Comparable p2 — porque b/V satisfazem os
limites originais perante ambas as referências, e resíduos Stata não eram
pedidos. A resolução insuficiente de uma quantidade não invalida outras
quantidades cuja resolução independente foi demonstrada. Os recibos brutos
foram preservados; a adjudicação derivada soma 531 PASS e 21 qualificações.

Permanecem 13 desvios formais b/V, cinco qualificações de resíduos e três
arredondamentos zigzag. Não houve mudanças de sinal ou decisões individuais
a 1/5/10%; maior erro de contraste 0,192223% do seu SE de referência. Estes
diagnósticos descrevem pequena magnitude, não estabelecem novos limites,
não provam ausência universal de impacto e não apagam incumprimentos formais.
Uma exceção de release é uma decisão explícita, separada de um PASS numérico.
A investigação foi encerrada; a proposta de exceção ainda aguarda decisão.

Erros do harness corrigidos nesta revisão, sem tocar no estimador: exigir
simetria bit a bit de V para formas quadráticas; exigir hashes de arrays que
o worker Stata não exporta; voltar a arredondar a referência racional zigzag.
Usou-se a parte simétrica apenas nos diagnósticos conjuntos, conservaram-se
os erros formais sobre V bruta, a identidade Stata previamente autenticada
e os erros racionais já guardados. Reprodutor offline analyze.py e controlos
de diagonal, rotação, escala e antissimetria em
`build-worktrees/rebase_2262_v48_20260911/provenance/g2_g3_20260912/g3_release_review_20260913/`.
Não acrescentar este diagnóstico ao fit nem transformá-lo numa calibração H2.
Zero novos fits, oracles, builds ou mudanças de precisão; AKM2 permanece
separado, em background sem limite de tempo.

## 14 setembro — recusa AKM2 corrigida sem mudar o cálculo

A referência reghdfe concluiu normalmente e confirmou os b/V guardados da
2.26.2. O trace da candidata mostrava último solve convergido/certificado,
com medida de bloco 3,6835e-8 abaixo do certificado 8e-8. Um alvo adicional
1e-8 reprovava e o esgotamento da sequência apagava os flags sem novo
trabalho. A guard fatal herdada de v47/v48 interrompia antes do OLS.

Quatro linhas passaram a preservar last quando ambos os flags já são
verdadeiros, mantendo etapas, tolerâncias, certificados, guard global e
N1. A candidata corrigida devolveu os mesmos cinco b, V, RSS e hash do vetor
completo de resíduos da baseline, com 543 iterações e 32 workers. Os quatro
slopes identificados passam contra reghdfe: máximo |Δb|2,9458e-11,
|ΔV|3,2670e-14. Classificação fechada: **undue verifier refusal** corrigida
no âmbito C++ CPU Fast; nenhum polimento adicional foi necessário.

Reprodutor permanente:
`build-worktrees/rebase_2262_v48_20260911/tests/analytic_precision/slope_continuation_acceptance_contract.py`.
O wrapper e a guard reais passam 17 cenários; a fonte anterior falha os
positivos que apagavam estados válidos. A bateria nativa acrescenta
17 estimações PASS e seis recusas esperadas. Os controlos GPU do wrapper
são de metadados, não uma execução CUDA. Fonte corrigida SHA
41977fc7cfddd21529219cd965d958b8abc4713dc9c439d33a87aca8095bb2c4.

Na revisão do comparador, ligar um artefacto apenas ao hash declarado pelo
próprio RESULT era insuficiente. RESULT passou a ser ligado ao JOB, caminho
e hash da build, case, worker e célula esperados; um positivo e três
negativos de identidade passaram. A correção ocorreu antes da adjudicação,
sem repetir o fit. Quatro hashes antigos do subconjunto G1 correspondiam
a correções de harness já validadas e foram reconciliados pelos recibos,
sem alterar limites ou reabrir casos encerrados.

A auditoria de outras recusas encontrou padrões semelhantes em Fast/CUDA,
mas nenhuma outra recusa natural indevida demonstrada. Não retirar guards
em bloco. Os 21 REVIEW aceites permanecem encerrados. Relatório, patch,
referência reutilizada e custódia em
`build-worktrees/rebase_2262_v48_20260911/provenance/g2_g3_20260912/slope_acceptance_fix_20260914/`.
Os sete artefactos N05 antigos ficaram intactos; restantes células AKM2,
artefactos finais, alinhamento e CI continuam por validar antes de release.

## 14 setembro — integração C0 e hipótese SIMD não promovida

O snapshot atual congelou 809 ficheiros, incluindo as duas correções recentes
de slopes e a infraestrutura OpenMP. A integração completa tem 3057 ficheiros
e 196 espelhos de fontes reconciliados. O CPU corrigido foi reutilizado pelos
seus bytes; CUDA, plugins e R foram reconstruídos, preservando os sete N05.
Fontes atuais e binários antigos nunca foram apresentados como um único C0.

AKM2 C++ CPU Comparable e CUDA Fast/Comparable passaram a referência já
concluída, sem repetir CPU Fast ou reghdfe. Tempos diagnósticos: 1624,232 s
(846 iterações), 247,868 s (1934) e 248,884 s (1004), todos com 32 workers
e os backends pedidos. Células Stata e restantes marcos têm recibos próprios.

Uma hipótese B1 vetorizou quatro entradas independentes do Gram preservando
o grafo OlsWide. Operações/Gram, 240 máscaras, 4864 controlos adicionais de
rank/bases/erros e fallback passaram. Nos módulos, 153 estimações e cinco
recusas previstas passaram os contratos existentes, incluindo 64 estimações
CUDA. Isso prova o âmbito exercitado; não é uma aprovação de performance.

Os três pares C0/candidata simulated_panel CUDA Fast passaram a comparação
numérica, mas deram ganhos de fit de 4,768%, 4,739% e 12,782%. A mediana
emparelhada pré-declarada, 4,768%, não atingiu a hipótese de 10%. A razão
entre medianas marginais seria outra estatística e não pode substituir a
métrica depois de observar os resultados. A candidata ficou preservada e
não promovida, sem repetir a mesma carga ou produzir outra revisão nesse dia.

Evidência e reprodução:
`build-worktrees/rebase_2262_v48_20260911/provenance/performance_2270_20260914/`:
`C0_SOURCE_MANIFEST.json`, `C0_STAGE_MANIFEST.json`, `PROFILE_REPORT.md`,
`rank_simd/{PROPOSAL.md,run_probes.py,TRIAL_PLAN.md,TRIAL_MANIFEST.json,DECISION.json}`.
Não houve erro numérico demonstrado, relaxamento ou aperto de contrato nesta
hipótese. Os 21 casos qualificados e as remoções de recusas indevidas continuam
preservados. Os originais Marta não são uma pendência.

## 14 setembro — B2 e o significado do erro residual

O warm-start CPU sem pesos/retenção preserva os inputs originais, referências
de paragem, estágios, orçamento e aceitação de C0. Os ensaios AKM2 C++ guardados
passaram b/V/amostra/DoF/RSS perante a referência independente: Fast378,544 s
contra980,595 s e Comparable965,471 s contra1624,232 s. São observações
exploratórias, não uma promoção ou adjudicação global de performance.

O controlo A/B copiado do recibo anterior marcou uma diferença amostral de
u de1,6285e-7 perante C0Fast contra1e-7. A classificação foi **erro de
interpretação do harness**: evaluate.py mede erro perante `truth.exact_residual`,
enquanto essa distância A/B comparava dois resultados aproximados. Por exemplo,
dois números a distância2^-24 de zero satisfazem individualmente1e-7 e podem
distar2^-23 entre si. Esse exemplo de métrica não é uma prova sobre o vetor
AKM2 nem um novo limite.

A falha bruta, a referência e1e-7 foram preservados. Não se tomou Comparable
como oracle de resíduos, não se concedeu uma nova exceção e não se modificou
o solver para satisfazer o controlo. A consequência foi prosseguir a validação
independente autorizada. O teste científico continua contra a verdade das
fixtures, nos limites originais; a definição está permanentemente em
tests/analytic_precision/evaluate.py e precision_contract.py.

Proveniência e reprodução delimitada em
`build-worktrees/rebase_2262_v48_20260911/provenance/performance_2270_20260914/slopes_warm_start/`:
`RESIDUAL_CLASSIFICATION.md`, `ROOT_ADJUDICATION.json`, `TRIAL_DECISION.json`,
`AKM2_VS_REFERENCE.json`, `AKM2_CPP_REMAINING_REFERENCE_COMPARISON.json` e
`permanent_tests/`. Os controlos novos conservam os17estados de continuação e
o probe de referências originais, ponto inicial e domínios excluídos.

## 15 setembro — fechar a matriz com evidência já adjudicada

As 1128 execuções finais B2 terminaram. Dez diagnósticos do comparador exigiram
classificação, sem autorizar alterações ao estimador: seis PF já pertenciam
aos 21 casos aceites, três Marta reproduziam a solução anteriormente validada
e um workers tinha o desvio na perna B2262. O comparador bruto conservou os
seus resultados; a adjudicação final foi gravada num ficheiro separado.

Nos seis PF, tratar o desvio de RSS como estrutural ocultava a qualificação
já aceite: **erro de interpretação do harness**. Nas três Marta, o hash do
vetor completo de resíduos B2 coincidia com o replay N05 e o erro perante os
oracles LSQR/LSMR guardados satisfazia o limite original. Em workers, B2 passou
ambos os oracles b/V guardados e B2262 falhou o limite V: o erro demonstrado
na referência histórica não exige polir a candidata. Não foi feita uma nova
alegação sobre os resíduos workers.

Reprodução exclusivamente sobre os resultados guardados em
`build-worktrees/rebase_2262_v48_20260911/provenance/performance_2270_20260914/slopes_warm_start/`:
`FINAL_REVIEW_MARTA_QUALIFIERS.md`, `FINAL_REVIEW_WORKERS.md` e
`finalize_review.py`. As definições científicas e testes offline existentes
ficaram intactos. Zero novos fits, limiares, exceções ou mudanças de código
do estimador foram necessários para esta adjudicação.

B2 reduz o tempo CPU AKM2 perante C0 em 61,74%/40,26% no C++ e
57,42%/37,75% no Stata, Fast/Comparable, em três pares intercalados. A meta
global B2262 continua por atingir. A entrega foi classificada como candidata
para colegas, com as 21 qualificações explícitas, sem converter ganhos locais
em aprovação global de performance. Relatório, decisão e hashes finais em
`performance_report/FINAL_REPORT.md`, `FINAL_DECISION.json` e
`ROOT_PACKAGE_READBACK.json` da mesma pasta.
