# Pastas locais ativas do xhdfe

## Release activa 2.29.1.20261014 — publicada

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

## Estado canónico publicado em 14out2026

Versão local validada: 2.29.1.20261014. As pastas principais xhdfe e xhdfe-xfe são os pontos de entrada locais.

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
