# Pastas locais ativas do xhdfe

## Candidata 2.31.1.20261112 — issue #18

Worktree privada `build-worktrees/issue18-optin-20261111` e staging público
`build-worktrees/issue18-public-20261112`, ambos dentro da raiz privada.
`xhdfeakm` 1.9.0 acrescenta `sigmacentering(personyear)` e inferência por
projeção com clusters de trabalhador, preservando o comportamento por omissão.
O âmbito estatístico e as provas estão em
`docs/releases/VALIDATION_2.31.1.20261112.md`.

O utilizador autorizou commit, push, integração nos dois `main` e release
após validação. As provas locais finais CPU/CUDA, Stata e R passaram;
o preflight remoto é `43153369775`. A publicação exige os artefactos exatos
do tag, validação H100, reconciliação dos pontos de entrada canónicos e
readback da instalação pública. Uma worktree validada não atualiza por si
só os dois checkouts canónicos. O manifesto é selado apenas no fim.

## Histórico: candidata 2.31.0.20261111 — integrada na 2.31.1

Construção final `cand_2310_lean_r2` (worktree `build-worktrees/perf-fereg-gap-20261106`; manifesto
`Verifications/perf_fereg_gap_20261106/p8/MANIFEST_cand_2310_lean_r2.json`). O que muda:
- correcção da t de Student no Windows (guarda x87);
- `residuals()` em Stata guarda o resíduo do próprio ajuste em Fast e Comparable;
- nova tentativa MLSMR ponderada em CPU, ligada por omissão (o CUDA ponderado com 3 FE fica como limitação conhecida).

`xfepout` passa a 1.14.0, `xhdfegelbach` a 1.6.2, `xhdfegelbachbootstrap` a 1.0.2 e `xhdfeakm` a 1.8.2.
Validação em `docs/releases/VALIDATION_2.31.0.20261111.md`. O tag 2.31.0
permanece draft na verificação de 12nov2026; a última release publicada
continua a ser a 2.30.1 até ao fecho dos gates da 2.31.1.

## Histórico: release 2.30.1.20261020 — publicada

Publicada a 20out2026, às 22:36:48 UTC, em
https://github.com/reisportela/xhdfe-xfe/releases/tag/v2.30.1.20261020.
O corte público é `6118b955b72ca259482cb24966f23ee880e2d6fe`.
`xfepout` é 1.13.5, de 20out2026. A correção de covariância CUDA foi
validada por referência independente e a nova geração de cache é
`0x22624841443a`; a diferença herdada de covariância já não está em REVIEW.

Os dois checkouts canónicos estão em `main`, com fontes comuns idênticos,
builds CPU/CUDA `sm_90` reconciliadas e carregamento Stata/Python verificado
em processos novos. As quatro builds passam 16/16 CTests, e pedidos de
1, 2, 8, 16, 24 e 48 trabalhadores foram exercitados. Os plugins locais
H100 são distintos dos artefactos portáteis, validados separadamente.

Os 13 anexos publicados estão autenticados. A instalação pública nova
confirmou os 49 ficheiros do site e os 37 ficheiros instalados/descarregados.
O core analítico tem 664 passes xhdfe e 223 casos não suportados esperados;
os 48 ajustes funcionais CUDA do core24 terminaram. Os 48 ajustes de
desempenho na ready contra a 2.29.1 documentam o custo aceite pelo
utilizador: +54,63% em CUDA Comparable nativo e +22,31% em Stata/CUDA
Comparable, por medianas de rácios emparelhados. O desempenho core24
completo continua adiado.

Mantém-se o limite de cobertura anteriormente autorizado: Stata/MP nos
binários finais em Mac ARM/Intel nativos está em NOT_RUN. O runtime OpenMP
é byte a byte idêntico ao da 2.30.0; os novos plugins passaram as provas
nativas de plataforma. Isto não equivale a um novo teste dentro do Stata.
O issue #14 foi encerrado após a publicação das correções; não há issues
abertos nos dois repositórios na verificação de fecho.

Consultar `docs/releases/VALIDATION_2.30.1.20261020.md`. A evidência e as
cópias de recuperação estão em
`Verifications/ready_covariance_review_20261020/release_2_30_1_20261020/`.
O manifesto local é selado após os commits documentais finais.

## Histórico: release 2.30.0.20261020 — publicada

Publicada a 20out2026, às 10:53:51 UTC, em
https://github.com/reisportela/xhdfe-xfe/releases/tag/v2.30.0.20261020.
O corte público é `1dd3addded1350b656bfda0544e961606b7d0a85`; os commits de
fecho documental em `main` preservam os mesmos fontes executáveis e binários.
Integra os PR #15 e #17 do Tiago, as correções privadas CG/CUDA e o core de
testes com as adaptações do reghdfe. `xfepout` é 1.13.4, de 20out2026.

Os pontos de entrada locais são `/home/mangelo/Documents/GitHub/xhdfe` e
`/home/mangelo/Documents/GitHub/xhdfe-xfe`, ambos em `main`. As builds CPU e
CUDA `sm_90`, os imports Python e os plugins Stata OpenMP/nativos foram
validados em processos novos. As quatro builds passam 16/16 CTests; os
pedidos de 1, 2, 8, 16, 24 e 48 trabalhadores foram exercitados. O backend
por omissão continua a ser CPU. Os binários portáteis da release e os
binários locais otimizados para este servidor têm alvos de compilação distintos.

Os 13 anexos estão autenticados. A instalação pública nova confirmou os 49
ficheiros do site e os 37 ficheiros instalados/descarregados. A comparação
de desempenho completou 48 ajustes em ready contra a 2.29.1; o core24
completo fica adiado por decisão do utilizador. A publicação foi autorizada
com os testes dos binários finais em Stata/MP Mac ARM/Intel nativos em
NOT_RUN e a diferença de covariância CPU/CUDA herdada em REVIEW.

Consultar `docs/releases/VALIDATION_2.30.0.20261020.md` para o âmbito exato.
A evidência e as cópias de recuperação estão em
`Verifications/integracao_pr15_pr17_20261019/release_2_30_0_20261020/`.
O manifesto local é selado após os commits finais e identifica os ficheiros,
os artefactos validados e as alterações locais preexistentes preservadas.

## Histórico: candidata 2.30.0.20261016 — substituída pela release de 20out2026

Worktree `build-worktrees/pr15-integration-20261016`. Integra o PR #15 do Tiago
(issue #14: isolamento do runtime OpenMP macOS, `e(version)`, interacções
`a#b#c` em `absorb()`) mais a generalização de `cluster()` e do `xfepout`
(1.13.4), gates macOS adicionais e o recarimbo 2.30.0.20261016 (data provisória,
recarimbada no corte público). Fontes C++ e plugins Linux inalterados. Estado e
pendências em `docs/releases/VALIDATION_2.30.0.20261016.md`; nessa investigação
a release publicada era a 2.29.1.20261014 abaixo.

## Histórico: release 2.29.1.20261014 — substituída pela 2.30.0

Publicada a 14out2026, às 18:17:43 UTC, em
https://github.com/reisportela/xhdfe-xfe/releases/tag/v2.29.1.20261014.
Os dois checkouts contêm as correcções e a ajuda actual, com builds CPU/CUDA
sm_90 e plugins Stata OpenMP/nativos recompilados e verificados em processos
novos. O corte público é `8ce9b96cd2b1deca6ff052cead8bfbffbc7a67ef`.

Os 13 anexos foram autenticados e os binários exactos testados na H100. Uma
instalação Stata nova confirmou os 49 ficheiros do site e os 37 ficheiros
instalados/descarregados. O âmbito e os limites estão em
`docs/releases/VALIDATION_2.29.1.20261014.md`. A evidência e a recuperação estão
em `Verifications/release_2_29_1_20261014/` e no checkpoint anterior
`Verifications/remediacao_main_2291_20261014/`. O manifesto local é selado
depois dos commits finais; não deve ser actualizado apenas para ocultar desvios.

## Histórico: release 2.29.0.20261014 (substituída pela 2.29.1)

Ramo `integrate/pr13-tiago-2.29-20261012` (worktree
`build-worktrees/pr13-integration-20261012`): candidata 2.29.0.20261006 + PR #13
reconciliados; versão **2.29.0.20261014** (corte de 14out2026). Guia `TESTAR_TIAGO_2290_INTEGRADA_20261013.md`; manifesto
`TIAGO_ARTIFACTS.sha256`. **Publicada a 14out2026**: release `v2.29.0.20261014` em
`xhdfe-xfe` (13 assets autenticados, site net-install em `gh-pages`); `main` privado e
público no estado da release; registo em `docs/releases/VALIDATION_2.29.0.20261014.md`
e evidência em `Verifications/pr13_integracao_2290_20261012/release_2_29_0_20261014/`.

## Branch privada de testes de 12out2026 (histórico; substituída pela candidata integrada)

No ramo `test/tiago-2.29.0-20261012`, a candidata é **2.29.0.20261006**. Os pontos de entrada são
`stata/` e os módulos em `native/cpu/` e `native/cuda/`; consultar
`TESTAR_TIAGO_2290_20261012.md` e verificar `TIAGO_ARTIFACTS.sha256`.
O checkout de testes não atualiza as pastas canónicas descritas abaixo.

## Estado canónico publicado em 20out2026

Versão local validada: 2.30.1.20261020. As pastas principais xhdfe e xhdfe-xfe são os pontos de entrada locais.

- Stata: usar stata/ da respetiva pasta principal. O comando de partial-out atual é xfepout.
- Python: importar xhdfe a partir da respetiva pasta principal.
- C++/benchmarks: build/ para CPU; build_cuda/ para CUDA sm_90.
- Os binários locais permitem CPU e H100; o backend por omissão continua a ser CPU.

Antes de uma auditoria, benchmark ou conclusão de publicação, executar:

    python3 /home/mangelo/Documents/GitHub/xhdfe/tools/check_active_checkouts.py --online

O verificador compara commits, ficheiros, binários e configuração das builds. Uma falha deve ser investigada; não se atualizam hashes apenas para silenciar o aviso.

Worktrees de desenvolvimento e evidência histórica permanecem separadas. Uma publicação feita numa worktree só fica encerrada depois de reconciliar, validar e identificar estas duas pastas principais. Preservar alterações locais antes dessa reconciliação; nunca usar limpeza global ou reset --hard.

O arquivo reversível desta organização está em Verifications/local_alignment_20260905/ do repositório privado; PLAN.json contém os caminhos originais e os destinos. O arquivo é histórico e não deve ser acrescentado ao adopath ou ao sys.path ativo.

Sessões Stata/Python já abertas podem manter código antigo em memória. Os testes de identidade devem usar processos novos.

A validação de 29sep2026 e as cópias de recuperação desta atualização estão em
`Verifications/release_2_28_1_20260929/`. Incluem as builds canónicas CPU/CUDA
sm_90, os imports Python ativos, os plugins Stata e os artefactos exatos do
GitHub. O manifesto é selado depois do readback público e dos commits finais;
o seu estado local identifica os HEAD e os binários efetivamente validados.
