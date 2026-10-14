# Reserva comum TLS — implementação e limites

Base: candidata privada 2.29.0.20261006, tranche 1 V2 + H2. O ensaio começou
em 08out e continuou em 09out, sem mudar a data do pacote de referência.

## Problema e alteração

A reserva comum `tls_zt` era criada antes de saber se seria consumida.
Contadores da experiência anterior observaram 53 668 224 bytes sem uso nas
quatro classes batch críticas: as lanes usavam reservas próprias por RHS.
O controlo serial usava a reserva comum. Classe original: cobertura em falta /
uso de recursos; sem erro demonstrado do estimador.

O setup conserva chunks, owner traversal e CSR. `prepare_shared_zt` cria os
mesmos vetores e zeros no primeiro consumidor, marcando ready só no fim.
`compute_Zt` prepara antes do seu loop; `apply_lsmr_At_impl` prepara apenas
quando não recebe override, antes da sua região OpenMP. Cada fit tem o seu
estado local. As lanes conservam as reservas privadas existentes. Nenhuma
operação de soma, partição, método, tolerância ou critério de paragem mudou.

O teste permanente `shared_scatter_storage_contract.py` verifica o corpo
exato, a publicação após sucesso, a falha de alocação e dois controlos
negativos. As campanhas reais do estimador e seus oráculos são verificações
separadas. O primeiro extractor exigiu um espaço depois de `=` onde o núcleo
tem uma mudança de linha: erro do harness corrigido, sem mudar o estimador.

## Evidência e cautelas

Os ficheiros/recibos estão em
`Verifications/perf_max_20260929/implementacao_bloqueios_20261007/`
`tranche_tls_storage_20261008/`. `P0.json` guarda hashes/cópias do estado
anterior; os diffs próprios permitem recuperar só esta alteração.

Os quatro casos críticos em lanes forçadas passaram o limite 1,15 face à
2.28.1, com nove campos substantivos exatos face à via equivalente H2.
A reserva tardia reduz o pico nesses casos; não é apenas uma soma teórica.
Os interruptores forçados são instrumentos de diagnóstico: o despacho
rejeitado N1/N6/N9 não foi retomado e os problemas por omissão são reportados
separadamente.

Ready T1 caiu para 1,1244143 vezes o RSS da referência no ensaio. AKM T1
ficou em 1,1548745: ainda fora de D-A. Os vinte campos numéricos/amostra/
iterações/via mantiveram-se exatos nas bases protegidas medidas. Estes são
diagnósticos de memória, não adjudicação de tempo. Não se altera a margem.

Os testes CPU de 71 linhas/arm deram 69 PASS e duas recusas esperadas;
21 linhas audit/arm mantiveram os raws exatos. Cache, pesos e recuperação
incluem oráculos independentes. Os testes CUDA de equipa/b/V preservam as
falhas adicionais de resíduos já classificadas contra a 2.26.2; não chamar
esses resultados certificação global. A ponte R direta não é instalação
nem validação da interface de fórmula. Windows/macOS nativos ficam NOT_RUN.

## Estado

Alteração de alocação implementada localmente, com validação no âmbito dos
recibos. A regra N1/N6/N9 e D-A permanecem pendentes; os tempos/core24×8,
revisão externa, validação multiplataforma e decisão de release são gates
separados. Os artefactos novos são candidatos isolados, não entradas ativas.
Zero instalação, commit, push, ativação ou publicação.
