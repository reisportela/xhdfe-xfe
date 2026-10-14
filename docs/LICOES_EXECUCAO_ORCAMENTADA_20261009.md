# Lições — decisão e execução MLSMR com orçamento comum

Revisão privada da 2.29.0.20261006 após o bloqueio Claude. O mantenedor
pediu implementar a resolução e aceitou explicitamente o excesso medido
de +15,45% de memória em AKM Comparable CPU T1. A margem global de 15%
e os contratos numéricos das restantes células não mudaram.

## Decidir com a partição histórica e limitar a execução

O predicado anterior usava o preenchimento máximo admissível de B para
uma decisão que, na máquina validada, executa 48 chunks. Isso era um
pior caso de capacidade, não o pedido de memória efetivo do host.
Não basta substituir o termo por 48 se a execução futura puder crescer
sem respeitar o limite conjunto.

O novo helper mantém R=8*n*L, B=max(1 GiB,2R), J=8*L*(U+Dc+1),
Q=56*F*L. Calcula H=3R-Q e Cjoint=min(n,B/J,H/J) com operações
verificadas. A decisão usa C48=min(48,Cjoint), nunca a equipa/capacidade.
A execução usa min(target geral,Cjoint), garantindo para as famílias
modeladas Cexec*J<=B e Cexec*J+Q<=3R. Não há clamp a um chunk quando
nenhum cabe; o chamador tem um erro interno informativo se receber zero.

São limites de pedidos selecionados de scratch, não de RSS, arena,
inputs ou de toda a workspace. A grelha física continua obrigatória.
O trace distingue decisão, limite máximo e partição realmente executada.

## A partição faz parte da identidade aritmética

Até à capacidade histórica de 48, lockstep só se mantém quando C48
coincide com a partição anterior calculada nos mesmos dados. O caso
n266000/L250/FE10000+11184 que passava de 48 para 47 chunks vai agora
para lanes. A alteração de via é explícita e adjudicada independentemente.

Em capacidades maiores, o orçamento pode limitar a partição. Esse caso
é declarado/diagnosticado como alteração aritmética; não se promete
igualdade de bits apenas pelo booleano lockstep. Simulações de 64/96
verificam decisão e pedidos, não hardware nem precisão nesses anfitriões.

## Oráculos e testes permanentes

O contrato C++ verifica 48/64/96, limites conjuntos, overflow, E1,
ready/AKM e o caso de partição. O contador permanente extrai as expressões
reais de allocate/assign/row_partials e regista pedidos de elementos.
Interceta os blocos grandes e não executa o solver; omitir carries tem
de falhar. Isto não certifica RSS ou OOM.

O novo oráculo permanente do caso de partição usa dummies esparsas e
MQO explícito, componentes/rank/df independentes, intercepto normalizado,
covariância convencional completa e resíduos. O fit da candidata usa lanes,
19 iterações e passou os limites predefinidos; uma perturbação de b de
1e-4 é rejeitada. O oráculo é offline e não entra no estimador.

Os seis ensaios E1 (31 RHS, T1/T2, sem quota/1,3/1,6 milhões KiB)
passaram com 14 iterações; o seu MQO independente também passou.
Os módulos CPU/CUDA sm90 e o contrato CTest foram reconstruídos em
pastas novas. Os restantes gates e a revisão do diff completo são
registados nos recibos, sem converter pendências em PASS.

## Âmbito e recuperação

Não se alteraram métodos públicos, tolerâncias, pisos, paragem,
normalização, interfaces ou output por omissão. O TLS tardio e H2
mantêm o corpo e ownership já validados. Github é previsto voltar à
partição histórica de 48; as vias protegidas têm de ficar exatas.

Pasta de evidência: Verifications/perf_max_20260929/
implementacao_bloqueios_20261007/tranche_joint_budget_20261009/.
P0.json contém hashes/cópias anteriores, a exceção D-A do mantenedor e
os 15 caminhos. DESENHO.md e ROUTE_PREDICTIONS.json precedem o código.
Patch próprio, manifesto e contexto de revisão ficam nessa pasta.

Revisão, matriz de desempenho, B1/B2/B3, R completo e plataformas nativas
são gates separados. Nada nesta lição autoriza commit, push, ativação,
publicação ou afirmação de compatibilidade Windows/macOS não executada.

## Seguimento da revisão independente

O Claude devolveu **APROVADO COM RESERVAS** para o pacote autorizado
`REVIEW_JOINT_BUDGET_PACKET.txt`, SHA-256 b60c9bfa98018dc8a4607a0d206e0bed638c22546b16bf2d6f4c43d3231ac26c.
Esse pacote ficou imutável. As correções posteriores são testes offline e
documentação; o núcleo e os módulos revistos conservaram os bytes.

- A fórmula histórica foi extraída literalmente do commit privado selado
  `0114ce79cff4d52c390096672f2dccd7d663d5d2`, núcleo SHA-256
  `8f7b0a816746576837cd0fcaceb0b7e3ca041daf3fe620bfef24176552d13351`.
  O loop antigo coincide com `legacy_execution_chunks` em 165 888
  comparações de 3 456 geometrias. Nas 2 088 geometrias elegíveis,
  100 224 comparações confirmam a partição conservada até 48. O caso
  n300000/L250/F20000/U19000/Dc1 conserva os 31 chunks históricos.
  É evidência de fonte e fórmula, não execução do binário histórico nem
  alegação de que a candidata privada foi publicada.
- O oráculo de fronteira exige agora presença e valores do dispatch,
  batch e fase com workers úteis. Retirar qualquer dessas linhas tem de
  falhar. O falso PASS por ausência era um **erro do harness de avaliação**;
  não autorizava alterar o estimador. O teste corrigido passou com 16
  workers úteis, além do MQO independente.
- O teste permanente conserva a geometria N9 original Dc0 e a geometria
  física Dc1. Ambas passam intencionalmente de lanes para lockstep48
  quando se deixa de tratar o preenchimento máximo de B como a decisão.
  `N9_ROUTE_CHANGE_RECORD.json` regista as duas mudanças; a previsão
  anterior é preservada como evidência histórica.
- O contador rejeita agora também a omissão dos row partials. O seu âmbito
  são apenas as famílias persistentes selecionadas. O vetor `partial`
  transitório do arranque, que coexiste com `row_partials`, outras
  workspaces e overhead do alocador ficam fora deste modelo. Só os ensaios
  físicos podem estabelecer RSS ou disponibilidade sob uma quota.
- Existe uma definição e um único call-site de `mlsmr_solve_lockstep`,
  protegido por `if (lockstep)`. A decisão elegível garante uma partição
  executável; a exceção interna para zero chunks não é atingida nessa via.

### Semântica do trace opt-in

Não comparar `private_bytes` entre revisões sem identificar o esquema:

| Revisão | Significado de `private_bytes` no dispatch |
|---|---|
| Fonte selada de 06out | Scatter privado da partição histórica; orçamento só sobre os níveis não contíguos. |
| Helper bloqueado anterior | Limite `max_chunks*J`, incluindo carries e row partials. |
| Helper conjunto atual | Pedido da decisão fixa `decision_chunks*J`; o limite fica em `max_private_bytes`. |

Na execução atual, `row_scratch_bytes` descreve a partição executada e
`selected_working_bytes` acrescenta os sete blocos de níveis. São pedidos
selecionados, não RSS total. O trace depende de `XHDFE_MLSMR_TRACE=1`;
o output por omissão mantém-se. As evidências locais que respondem às
reservas não foram reenviadas ao Claude sem autorização específica.
