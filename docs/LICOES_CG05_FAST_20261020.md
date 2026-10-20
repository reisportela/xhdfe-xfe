# CG05: preservar o critério publicado do Fast

Classificação: recusa indevida do verificador. No core24, `github` CUDA/Fast
devolveu estimativas na 2.29.1 em 336 iterações, mas a candidata recusou três
vezes após 10 658–15 185 iterações. O diagnóstico identificou o controlo novo
de norma do passo, introduzido para o erro CG02 em Comparable. Esse controlo
estava a impor trabalho adicional ao Fast, contrariando o seu contrato.

A correção limita apenas esse controlo adicional aos modos distintos de
Fast. Mantém os controlos de valores finitos, denominadores, resíduos e
quebras reais do CG. Comparable conserva a proteção de CG02; os testes MP120
OLS/IV nos três modos continuam a passar. Não se mudou qualquer tolerância.
A geração do cache passa a 4439.

O Fast voltou a concluir `github` em 336 iterações. Uma referência independente
com dummies esparsas, LSQR/LSMR, OLS reduzido MP80 e sandwich long-double deu
erro absoluto de b inferior a 4,4e-11. As duas referências concordam em b à
precisão reportada e em V a 3,93e-12 na escala diagonal.

A comparação geral de V por elemento usada em fixtures sintéticos falhou
tanto na 2.29.1 como no Fast restaurado. Classificação desta comparação:
erro de aplicação do harness, por tratá-la como contrato universal do Fast.
Os FAIL brutos ficam preservados; as diferenças diagonais escaladas de V
foram 2,92e-8 e 4,45e-8. Não se ajustou o solver nem se adotou um novo limite
Fast para as eliminar. Duas tentativas com `areg` nativo excederam o limite
de 420 segundos e não são evidência de aprovação.

Testes permanentes: `tests/analytic_precision/fast_cg_mode_contract.py`, a
referência agregada `fixtures/github_fast_aggregate_reference.json`, os
controlos MP120 existentes de CG02 e a linha `github` do core24 completo.
Nenhum registo individual do benchmark integra os novos ficheiros.
