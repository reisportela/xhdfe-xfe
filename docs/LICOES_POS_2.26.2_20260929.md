# Lições da interface Python: Polars e fórmulas — 29set2026

## Transformações inteiras anteriores à conversão para float64

**Classe:** erro demonstrado do estimador na construção da matriz do modelo.

A versão 2.28.0 promovia as colunas numéricas de pandas antes da avaliação por
Formulaic, mas devolvia outros DataFrames diretamente. Uma coluna Polars
`int32` com `z=50000` produzia `I(z**2)=-1794967296` antes de o resultado ser
convertido para float64. Num desenho explícito, o coeficiente de referência
era aproximadamente 0,699961 e a entrada Polars devolvia 0,305013, com
convergência declarada. O PR público #12, commit
`5273fe206954ddfe9a97634a073672dd936ed815`, preservava este defeito anterior.

**Correção:** a revisão local de `xhdfe/_formula.py` aplica a mesma análise de
papéis numéricos/categóricos e a mesma promoção antes das operações aos
DataFrames reconhecidos pelo Narwhals que Formulaic já utiliza. São
substituídas apenas as colunas numéricas da fórmula; IDs e metadados de
categorias mantêm-se, e o input não é alterado. O caminho numérico direto e o
solver não recebem novos alvos de precisão nem novas tolerâncias.
As colunas já float64 mantêm as verificações de omissos e de papéis, mas são
reutilizadas sem substituição no DataFrame Polars.

**Teste permanente:** `tests/analytic_precision/polars_formula_contract.py`.
O desenho gera as potências em float64 independentemente da candidata e
compara b, a matriz V completa, resíduos, amostra e graus de liberdade com
OLS explícitos e HC1, em `int16`, `int32` e `int64`, com pandas e Polars.
Os testes de `tests/test_formula_frontend.py` cobrem também interações,
resposta transformada, nomes citados, blocos de endógenas/instrumentos,
IDs acima de 2^53, categorias Enum, papéis incompatíveis e preservação do input.

Um teste inicial confundia a ordem dos contrastes de `C(g)` em pandas e
Narwhals. A diferença também ocorre no código anterior à correção, e os
valores/tipo Enum permanecem intactos. Classificação: erro do harness. O teste
definitivo verifica a categoria Enum diretamente e os níveis explicitamente
indicados em `C(...)`; o diagnóstico anterior foi preservado na evidência.

## Verificação da interface na integração e na release

**Classe:** erro do harness e cobertura em falta.

O CI ordinário não executava a suite de fórmulas. O portão da release exigia
83 testes e não instalava Polars; a proposta do PR #12 acrescentava seis
testes, cinco dependentes dessa biblioteca, sem atualizar esse portão.

Os dois workflows passam a instalar o extra de testes e a usar
`tools/validate_formula_frontend.py`. O verificador exige as contagens
declaradas, ausência de falhas e de testes ignorados e integridade dos inputs.
As suites são copiadas para fora do checkout para testar o pacote instalado;
o verificador rejeita uma importação que provenha do checkout de origem.

## Dependências e âmbito

O ganho do PR #12 é conservado: a inferência do dtype evita a validação Python
elemento a elemento apenas nas colunas de rótulos, mantendo a validação de
colunas mistas. Polars e PyArrow são dependências opcionais de teste/input;
não são ligados ao núcleo C++ ou ao plugin Stata. A documentação explica
que algumas categorias Polars são convertidas internamente para pandas e
requerem PyArrow. Preparação de dados e motores SQL permanecem externos.

A correção integra as fontes da release 2.28.1.20260929, sobre a base
2.28.0.20260924. Os hashes, as execuções e a cópia reversível dos ficheiros
anteriores são guardados em `Verifications/polars_frontend_20260929/`.
A validação dos artefactos da release é registada separadamente em
`Verifications/release_2_28_1_20260929/` e na respetiva nota de validação pública.
