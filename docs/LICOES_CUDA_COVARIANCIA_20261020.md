# Covariância CUDA e orçamento de reparação — 20out2026

Este registo acompanha uma candidata em validação. Não é uma declaração de
publicação nem de certificação de todas as plataformas.

## Projeção incompleta em CUDA Comparable

**Classe:** erro de estimador demonstrado.

Na ready, a versão publicada devolve uma covariância com erro máximo
normalizado de cerca de `3,2e-8`, acima do contrato de `1e-8`. A referência
independente usa os efeitos fixos explícitos, refinamento em precisão alargada
e inferência MP80; as implementações CPU e CUDA dessa referência concordam.
O erro já está presente na 2.29.1.

Uma pequena deslocação entre varrimentos IT pode deixar componentes fracas dos
efeitos fixos incompletamente projetadas. A referência de engenharia com
correção por grafo resolve o problema, mas a sua fatorização acrescenta cerca
de 15 segundos na ready. Compactar o armazenamento preservou os resultados,
mas não demonstrou ganho nos pares intercalados, pelo que essa tentativa não
foi integrada.

A candidata conclui a projeção restante com o LSMR ordinário existente. O
profiler identificou que as somas por nível serializavam grupos muito grandes;
a implementação cooperativa usa FP64 e ordem de redução fixa por warp/bloco.
Não altera tolerâncias, critérios de convergência ou a definição do estimador.
A nova conclusão fica limitada à entrada principal Comparable, sem pesos,
com três efeitos fixos, sem declives e sem recuperação de efeitos fixos.
Reinícios internos de Fast não podem ativá-la por tradução interna das opções.

Os testes permanentes são
`tests/analytic_precision/cuda_three_fe_covariance_contract.py` e
`tests/stata/cuda_three_fe_covariance.do`. Cobrem a covariância completa,
constante, escalas, permutações e pedidos de trabalhadores. O oráculo é
independente; os resultados de xhdfe e reghdfe não são oráculos um do outro.
O erro observado próximo de `1e-11` não estabelece um novo alvo de precisão.

O controlo adicional de `xfepout` confirma que a correção também melhora a
inferência calculada a partir das variáveis residualizadas. Na versão
publicada, o caso sintético original com um trabalhador tinha erro normalizado
de `3,55e-7` nos coeficientes e `1,11e-4` na covariância; a candidata passa os
limites existentes nas três variantes. O cálculo de inferência do teste é
independente da rotina de regressão do package.

Com 16 trabalhadores, estas três variantes de `xfepout` em CUDA recusam tanto
na versão publicada como na candidata. Os resíduos do controlo necessário
continuam acima de `8*min(tol,1e-9)` após uma continuação no mesmo alvo; as
iterações e os resíduos coincidem entre as versões. Não ficam estimativas.
Esta limitação é registada como `EXPECTED_REFUSAL`, separadamente dos nove
casos numéricos aprovados. O teste permanente
`tests/stata/xfepout_three_fe_covariance.do` e o avaliador
`tests/core/xfepout_covariance.py` exigem o diagnóstico específico, a ausência
de estimativas e o respeito pelo orçamento; não aceitam erros genéricos.

A nova geração de cache `0x22624841443a` impede a reutilização de
transformações anteriores à correção. O teste
`tests/analytic_precision/covariance_cache_generation_contract.py` verifica
a rejeição da geração anterior e os acertos exatos da nova geração. O
controlo negativo com a candidata sem o incremento foi rejeitado nas duas
modalidades, como esperado.

## Orçamento perdido num reinício após uma tentativa GPU

**Classe:** erro de estimador demonstrado.

No fixture sintético de três efeitos fixos, um pedido de 150 iterações
consumia 143 na fase inicial e sete no LSMR, sem convergir. A reparação geral
reiniciava então com um orçamento novo e devolvia uma solução após mais
45 iterações, reportando apenas 45. Os coeficientes eram corretos, mas o
orçamento e o contador públicos eram incorretos.

A causa é distinguir trabalho GPU efetuado apenas por `gpu_used`: uma tentativa
GPU que falhou tem `gpu_attempted=true` e `gpu_used=false`. O tempo de cálculo
e as iterações não desaparecem por a tentativa não ter produzido estimativas.

A reparação passa a descontar o trabalho de todas as tentativas GPU, a
acumular os contadores em todos os resultados e a passar apenas o orçamento
restante à reparação estrita ponderada. Um orçamento esgotado devolve uma
recusa informativa e não deixa estimativas.

O reproducer permanente é
`tests/analytic_precision/cuda_warm_budget_contract.py`, com equivalente em
`tests/stata/cuda_warm_budget.do`. O controlo de 150 iterações deve ser
`EXPECTED_REFUSAL`; o de 1000 deve convergir, usar CUDA e passar o oráculo.
Uma recusa esperada não conta como sucesso numérico.

As correções encontram-se em `src/fe_absorption_cuda.cu` e
`src/hdfe_regressor_v11.cpp`, com espelhos Stata, R e distribuição. Estes
testes integram o estágio analítico de `tests/core/run.py`. Os tempos de
aceitação são medidos separadamente, sem instrumentação e com a 2.29.1 como
referência histórica. A matriz completa core24 permanece adiada pelo utilizador.

## Nome do log Stata com um argumento terminado em `.do`

**Classe:** erro do harness de avaliação.

Ao passar o ficheiro do oráculo como argumento final de `stata-mp -b do`,
o Stata escolheu `three_fe_oracle.log` para o log do teste. Os 18 ajustes CPU
tinham passado, mas o driver procurava o nome do teste e falhava ao abrir o
ficheiro esperado.

O runner passa a executar um `driver.do` local com o comando completo dentro
do ficheiro. O log é sempre `driver.log`; o runner exige tanto o marcador
do teste como `DRIVER_RC 0`. O erro e o log inicial são preservados, sem
alteração do estimador ou dos limiares numéricos.

## Âmbito dos testes de frontend e diretórios de saída

**Classe:** erro do harness de avaliação.

A suite geral de 97 testes Python inclui ajustes sem efeitos fixos, apenas
CPU, e comparações de igualdade exata próprias desse backend. Forçar
`XHDFE_GPU_BACKEND=cuda` nessa suite reproduz os mesmos 18 erros de domínio
e sete diferenças de arredondamento na versão publicada e na candidata.
A validação correta mantém os 97 testes CPU e usa o contrato permanente
`formula_cuda_analytic_contract.py`, com 16 casos de referência analítica,
para os caminhos CUDA. Não se alteram tolerâncias nem se classificam os
resultados do harness mal configurado como sucesso numérico.

O runner Stata completo inclui um contrato DK IV que exige outputs dentro
da raiz da candidata selecionada. Uma pasta irmã, embora ainda dentro do
repositório privado principal, foi corretamente rejeitada por esse controlo.
A localização foi corrigida para `Verifications/` da própria candidata;
os logs anteriores ficaram preservados e o estimador não mudou.
