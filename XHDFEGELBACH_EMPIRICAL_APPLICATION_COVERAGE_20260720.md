# `xhdfegelbach`: cobertura das aplicações empíricas de Gelbach

**Data:** 20 de julho de 2026

**Estado:** candidato `2.19.0.20260720`, autorizado para release
**Âmbito:** releitura do corpus local, matriz paper-a-paper e camada aplicada
cross-frontend

## Veredicto

O `xhdfegelbach` é agora uma ferramenta particularmente forte para o domínio
em que a identidade de Gelbach é válida: regressões lineares na mesma amostra,
com uma ou várias focais, controlos comuns, blocos observados simultâneos e
efeitos fixos acrescentados de alta dimensão. Nesse domínio combina capacidades
que não encontrei reunidas noutro produto auditado: HDFE nativo, target
absorvido explicitamente restringido, pesos, covariância conjunta analítica,
CPU/CUDA e o mesmo núcleo em Stata, Python e R.

Não é — e não deve ser apresentado como — um comando universal para qualquer
objeto a que um paper chame “decomposition”. IV/2SLS, correções SPJ de painéis
dinâmicos, MM-quantile regression, curvas kernel, KHB/GLM, decomposições
distribucionais e causal mediation têm estimandos e inferência próprios. A
decisão state of the art é manter essas fronteiras visíveis, em vez de lhes
aplicar silenciosamente a álgebra OLS.

## Evidência relida

Foram relidos no texto integral o paper metodológico de Gelbach (2016), o paper
âncora de Guimarães, Portugal e Reis (2026) e os dez usos empíricos verificados
no [manifesto local](literature/gelbach_applications/MANIFEST.md). A admissão no
corpus exige uso efetivo da decomposição; uma citação, uma sequência de modelos
ou uma decomposição da variância não contam.

As práticas recorrentes observadas são:

- uma focal e controlos que pertencem simultaneamente aos modelos base e full;
- muitos controlos agrupados por conteúdo substantivo;
- worker, firm, occupation, match, attorney-year ou outros HDFE como blocos;
- targets invariáveis dentro de worker FE e, por isso, absorvidos no full;
- contribuições em níveis, shares assinadas e residual full;
- contribuições negativas e shares acima de 100%, sem truncagem;
- SE, intervalos e somas de blocos a partir da covariância conjunta;
- polinómios, bins, indicadores e interações tratados como colunas do bloco;
- repetição por outcome, subamostra, cidade ou especificação;
- clustering alinhado com a variação identificadora — incluindo um caso de
  two-way clustering que excede a VCE atual.

## O que foi acrescentado

### 1. Focal separada dos controlos comuns

`focal`/`focal()` é apenas seleção de output. Todas as colunas de `x1` continuam
nos dois modelos. Isto implementa sem ambiguidade o desenho frequente

\[
X_1=[\text{focal},\text{controlos comuns},\text{FE comuns explicitados}],
\]

sem reclassificar controlos comuns como blocos “explicativos”. Essa
reclassificação mudaria o modelo base e o estimando.

Um novo oracle determinístico em `VALIDATE_GELBACH.py` inclui uma focal, um
controlo contínuo e cinco indicadores de year FE comuns em `x1`, mais um bloco
observado e firm HDFE acrescentados. O resultado reproduz LSDV/FWL externo com
erros máximos de `2.22e-16` no coeficiente base, `8.88e-16` no coeficiente full
e `2.09e-15` nas parcelas. Com e sem `focal`, `cov` e `total` são
bit-idênticos.

Esta solução cobre FE comuns de dimensão moderada por indicadores explícitos.
Não afirma absorver nativamente um conjunto HDFE comum a base e full; essa é
uma extensão futura do núcleo.

### 2. Tabelas tidy e nomes estáveis

Python expõe `gelbach.tidy()` e R expõe `xhdfe_gelbach_tidy()`. Ambos produzem
uma linha por focal e componente, com estimativa, SE, intervalo normal,
`component_kind`, `se_type` e metadados de share. Não acrescentam dependências
de data frames ao wrapper Python: a saída é uma lista de dicionários pronta
para pandas, Polars ou CSV.

Os nomes e seletores são coerentes:

- Stata: `r(x1_names)`, `r(focal_names)` e índices zero-based em
  `r(focal_indices)`;
- Python: `x1_names`, `focal_names` e `focal_indices` zero-based;
- R: os mesmos campos e a mesma base zero, embora a entrada R continue a usar
  índices one-based por convenção da linguagem.

### 3. Shares assinadas com denominador e inferência explícitos

Há três modos opt-in:

- `movement`: \(\Delta_g/\sum_h\Delta_h\), com delta method usando toda a
  covariância conjunta dos componentes;
- `base`: \(\Delta_g/\widehat\beta^{base}\), com pontos mas SE deliberadamente
  missing, porque o contrato público ainda não contém
  \(\operatorname{Cov}(\Delta,\widehat\beta^{base})\);
- `base_fixed`: a convenção descritiva de escalar o SE do componente mantendo
  o coeficiente base reportado fixo, inequivocamente rotulada
  `fixed_base_denominator_scaling`.

Nenhum modo trunca sinais ou renormaliza. O exemplo standard produz, de forma
intencional, `+107.31%`, `+2.26%` e `-9.57%`, somando 100%. Um threshold
`share_tol`/`sharetol()` torna missing um rácio cujo denominador é quase zero,
em vez de publicar percentagens explosivas sem aviso.

### 4. Somas e contrastes com a matriz conjunta

Python expõe `gelbach.contrast()` e R `xhdfe_gelbach_contrast()`. Uma soma ou
contraste linear usa todos os termos de covariância; não combina SE como se os
blocos fossem independentes. O teste em que o contraste inclui todos os blocos
reproduz exatamente o total e o seu SE. Em Stata, `r(cov)` continua a ser o
objeto programático completo e `r(fe_total)` fornece o subtotal FE certificado.

### 5. Exemplos e ajuda

O exemplo standard nas três linguagens contém agora uma focal, um controlo
comum, blocos observados, HDFE, shares assinadas e — em Python/R — um contraste.
O exemplo absorbed-target continua separado e diz `0 (imposed)`. A ajuda Stata
documenta ainda a geração explícita de polinómios, splines, fatores, bins e
interações antes de os agrupar, tornando a matriz de desenho auditável.

## Matriz paper-a-paper

“Direto” significa que o estimando pontual do exercício cabe no comando atual.
“Parcial” identifica exatamente a parte que ainda não deve ser reproduzida ou
certificada como equivalente.

| Paper / aplicação | Pontos | Inferência e reporting | Fronteira honesta |
|---|---|---|---|
| Gelbach (2016), OLS | **Direto**: X1 vetorial, blocos simultâneos, pesos | **Direto**: unadjusted, robust, one-way cluster, `gamma0`/`cov0`, covariância conjunta | IV é um estimador futuro separado |
| Cardoso–Guimarães–Portugal (2016), gender gap | **Direto**: target absorvido + worker/firm/job-title HDFE | Total do target é a VCE base; parcelas FE permanecem `conditional_gamma0` | A VCE deve clusterizar no FE que absorve o target; clustering apenas na firma gera warning |
| Addison–Portugal–Vilares (2023), union density | **Direto** para a decomposição linear e três HDFE | One-way firm cluster e shares disponíveis | Curva com 1,000 bins e MM-QR não são a identidade linear genérica |
| Carneiro–Portugal–Raposo–Rodrigues (2023), persistence | **Direto** para o exercício OLS não corrigido | One-way cluster disponível | A decomposição SPJ/corrigida exige transformar coeficiente e parcelas conjuntamente |
| Portugal–Reis–Guimarães–Cardoso (2024), schooling | **Direto** para OLS/absorbed-target e HDFE | Metadados distinguem zero imposto e inferência mista | 2SLS, LATE e exact/overidentification ainda não estão implementados |
| Stantcheva (2021), beliefs and partisan gap | **Direto**: focal + controlos comuns + fatores agrupados | Shares sobre base e residual full; níveis e SE disponíveis | As shares são accounting correlacional; o paper não mostra IC por parcela |
| Cook et al. (2021), gig-economy gender gap | **Direto**: bins/indicators como colunas explícitas, quatro blocos, city loops | One-way driver cluster, IC dos componentes, shares base/movement | IC de uma share sobre base requer a covariance ainda não exposta; não se inventa |
| Agan–Freedman–Owens (2021), assigned counsel | **Direto** para pontos, blocos e attorney-year FE | IC analíticos existem com one-way cluster | O paper usa two-way cluster; essa inferência não é ainda reproduzível |
| Carruthers–Wanamaker (2017), Jim Crow wage gap | **Direto** após gerar cúbicos e interações e agrupá-los | One-way county cluster, SE/IC e contrastes | Stata não aceita ainda factor syntax diretamente dentro de `x2groups()` |
| Carlana–La Ferrara–Pinotti (2022), channels | **Direto** para a tabela Gelbach linear | One-way school cluster analítico e p/IC deriváveis | Bootstrap e análise causal dos mediadores são procedimentos próprios |
| Chinoy–Nunn–Sequeira–Stantcheva (2026), zero-sum beliefs | **Direto**: muitos índices e versões US/international | Tidy/shares facilitam loops e exportação | A decomposição não identifica a direção causal entre beliefs e política |

## Os cinco casos do paper-âncora

| Caso | Cobertura atual | O que não se deve fingir que está coberto |
|---|---|---|
| Gender | Absorbed-target allocation, worker/firm/occupation HDFE, worker-clustered total | Inferência não condicional completa de cada parcela FE |
| Schooling | Pontos OLS/absorbed-target e HDFE | A decomposição IV/LATE do paper |
| Cyclicality | Exercício OLS com blocos match/occupation | Split hierárquico de match e wild bootstrap com poucos clusters |
| Unionization | Decomposição linear, shares com ambos os denominadores | Curva kernel e extensão MM-QR |
| Persistence | Decomposição do coeficiente OLS | Coeficiente e parcelas corrigidos por SPJ/Nickell |

O paper-âncora continua a ser um requirements generator, não um oracle
numérico: contém especificações e números internamente incompatíveis já
registados no levantamento de 19 de julho.

## Padrões de utilização

### Stata

```stata
xhdfegelbach y, x1(target common_control i_year_2-i_year_10) ///
    focal(target) ///
    x2groups("human_capital = educ educ2 : job = tenure exper") ///
    fes(firm_id occupation_id) ///
    vce(cluster) cluster(worker_id) shares(movement)

matrix list r(delta)
matrix list r(cov)
matrix list r(share)
```

### Python

```python
res = gelbach.decompose(
    y, np.column_stack([target, common_controls]),
    x2_groups={"human_capital": human_capital, "job": job_controls},
    fes={"firm": firm_id, "occupation": occupation_id},
    x1_names=["target", *common_names], focal="target",
    vce="cluster", cluster=worker_id,
)
table = gelbach.tidy(res, share="movement")
joint = gelbach.contrast(res, "target", ["human_capital", "job"])
```

### R

```r
res <- xhdfe_gelbach(
  y, x1 = cbind(target = target, common_controls),
  x2_groups = list(human_capital = human_capital, job = job_controls),
  fes = list(firm = firm_id, occupation = occupation_id),
  focal = "target", vce = "cluster", cluster = worker_id
)
table <- xhdfe_gelbach_tidy(res, share = "movement")
joint <- xhdfe_gelbach_contrast(res, "target", c("human_capital", "job"))
```

Para um target invariável no trabalhador, acrescenta-se explicitamente
`absorbedtargets(target)` em Stata ou `absorbed_targets` em Python/R e
clusteriza-se no worker FE que o absorve. O zero full é imposto, nunca estimado.

## Limites que permanecem prioritários

1. **Inferência FE completa.** Os componentes HDFE têm pontos exatos, mas SE
   `conditional_gamma0`; totais mistos continuam rotulados como mistos.
2. **Covariance com o coeficiente base.** Necessária para inferência completa
   de `share=base`; até lá os SE são missing por desenho.
3. **Multiway cluster e poucos clusters.** Devem operar sobre o score empilhado;
   não se pode combinar SE finais. Esta lacuna impede reproduzir integralmente
   Agan et al.
4. **HDFE comuns aos dois modelos.** Hoje são possíveis apenas via indicadores
   explícitos de dimensão moderada em `x1`; um `basefes()` nativo requer
   derivação e alterações no núcleo.
5. **Factor expansion ergonómica em Stata.** A matriz explícita funciona, mas
   uma interface segura para `i.`, `c.#c.` e regex ainda acrescentaria valor.
6. **IV, dinâmica e não linearidade.** Devem ser módulos separados com oracles
   próprios, nunca switches cosméticos da decomposição OLS.
7. **Manifesto de amostra e connectedness.** Já há contagens de input, usado,
   N efetivo e singletons; faltam hash/máscara de amostra e resumo das
   componentes conexas.

## Validação executada nesta revisão

- `VALIDATE_GELBACH.py`: **PASS**, incluindo focal inerte, shares, contraste,
  common-controls/LSDV e oracle externo absorbed-target;
- `VALIDATE_GELBACH_FRONTENDS.py`: **PASS**, paridade standard e absorbed entre
  Python, Stata e R, shares incluídas, e os seis exemplos reais;
- suite Stata: **PASS**, 28 ficheiros;
- suite R completa: **PASS**, um skip CUDA esperado na build CPU-only e três
  warnings AKM já documentados;
- `tools/check_cpp_core_alignment.sh`: **PASS**;
- Python compile e R parse: **PASS**.

As diferenças máximas das shares foram `1.39e-17` entre Python e Stata e
`4.44e-16` entre Python e R. Os hashes dos `.so` e plugins de produção são
iguais aos da recertificação anterior; nenhum binário, solver, tolerância,
estimando ou backend foi alterado por esta camada. Por isso não há uma nova
alegação de benchmark: o custo e o caminho standard compilado são os mesmos.

## Conclusão científica

O contributo distintivo e defensável é uma **plataforma de coefficient-movement
accounting linear, HDFE e cross-language**, suficientemente flexível para a
grande maioria dos usos OLS encontrados na literatura e agora muito mais fácil
de levar a tabelas empíricas reais. A melhor forma de o tornar “enorme” não é
chamar Gelbach a todos os problemas: é manter este núcleo extremamente preciso,
adicionar as extensões de inferência em falta e construir IV/dinâmica/não
linearidade como estimadores cientificamente separados.
