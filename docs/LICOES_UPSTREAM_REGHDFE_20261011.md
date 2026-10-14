# Lições dos testes upstream reghdfe — 11 de outubro de 2026

Campanha privada: `Verifications/perf_max_20260929/implementacao_fecho_revisao_20261010/reghdfe_upstream_20261011/`.
Base preservada: 2.29.0.20261006 com PF3FE corrigido. A correção permanece num
candidato isolado; este documento não declara uma release nem certificação global.

## Erros demonstrados e respetivas correções

1. **Interface:** abreviaturas reghdfe válidas, `constant`, `noabsorb` e
   `absorb(..., save)` eram rejeitadas. Completar o parser sem alterar o solver.
2. **Interface FV/TS:** materializar a resposta para o plugin e preservar a
   expressão pública; aceitar operadores temporais nos regressores e nos
   declives absorvidos. Previsões comparadas ao areg/regress nativo.
3. **Estatísticas:** R² ajustado em modelo saturado e F conjunto com covariância
   singular devem ficar missing no estilo reghdfe. Não inventar uma estatística
   definida quando o denominador ou o posto necessário não existe.
4. **Recuperação:** preservar os nomes `__hdfe1__Slope1`, `Slope2`, etc. de um
   bloco de declives; reutilizar apenas outputs automáticos pertencentes ao
   package, protegendo variáveis de input e nomes criados pelo utilizador.
5. **Modelo sem coeficientes livres:** não tentar criar uma matriz Stata 1×0.
   Transportar o resultado sem publicar coeficientes fictícios; validar também
   previsões e reexecução de savefe.
6. **Ausência de observações:** depois da remoção de todos os grupos singleton,
   devolver r(2001) antes de construir o solver de indivíduos.
7. **Mobilidade:** os IDs nativos já começam em 1; não somar novamente 1 no plugin.
8. **R² agrupado:** reconhecer uma constante explícita nos regressores para
   a escala do TSS, sem a confundir com uma constante no espaço dos FE.
9. **Apresentação:** preservar r(level) produzido pela tabela de resultados.
10. **Recusa indevida:** no modelo puro de FE sem pesos, normalizar os momentos
    necessários pela escala original. A variante float do upstream tinha RSS
    correto dentro de 1e-8, provado por projeção racional exata, mas era recusada
    pelo quociente relativo a um resíduo muito pequeno. Nenhum alvo ou método
    do solver foi alterado. Regressores e pesos mantêm o verificador anterior.

Os reprodutores permanentes estão em
`tests/analytic_precision/supplemental/upstream_reghdfe_20261011/`. O diff final,
os hashes e os logs da campanha identificam a revisão efetivamente testada.

## Erros do executor e limites das referências

* O storedresults nativo concatena nomes que podem exceder 32 caracteres e
  deixa escalares antigos depois de `drop`; isso chegou a ocultar um teste
  incorreto de metadados. O adaptador foi validado por perturbações deliberadas.
* Algumas asserções genéricas comparam diagnósticos exclusivos do xhdfe ou
  contadores de um algoritmo diferente. A execução literal é preservada;
  a adaptação compara a superfície pública partilhada e não modifica b/V.
* A referência reghdfe também falha seis ficheiros no Stata 19.5 instalado.
  Há deriva no areg com fvset, novos metadados do regress, uma constante na
  convenção do log-likelihood e testes antigos com dimensões incompatíveis.
* O benchmark reghdfe v3 de alphas tem erro superior ao seu próprio limiar
  contra uma regressão nativa com dummies explícitas. Paridade com esse valor
  não pode ser o oracle matemático do xhdfe.
* `fetolerance()` é um controlo separado e absoluto; copiar automaticamente
  um tolerance relativo não preserva o contrato. Uma experiência com essa
  transposição foi classificada inválida e conservada, sem alterar o produto.
* Igualdade bit a bit em GPU e contagens estritamente crescentes de iterações
  não decorrem do contrato FP64. Os resultados literais e as verificações do
  contrato publicado são reportados separadamente.

Driscoll–Kraay foi explicitamente adiado pelo utilizador. Testes de classes
Mata privadas do reghdfe ou de ivreghdfe são controlos de referência. Os sete
ficheiros de instalação, geração de fixtures e desenvolvimento permanecem
inventariados, sem serem contados como testes numéricos executados.
