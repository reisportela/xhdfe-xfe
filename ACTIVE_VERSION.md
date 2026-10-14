# Pastas locais ativas do xhdfe

## Correcções locais 2.29.1.20261014 — não publicadas

Os dois checkouts contêm as correcções da auditoria de 14out2026 e a ajuda
correspondente, com builds CPU/CUDA e plugins Stata recompilados e validados
localmente. O âmbito está em `docs/releases/VALIDATION_2.29.1.20261014.md`;
a evidência e as cópias dos originais estão em
`Verifications/remediacao_main_2291_20261014/`. A release publicada e o manifesto selado abaixo continuam a ser
2.29.0.20261014. Uma divergência do verificador durante este desenvolvimento
não deve ser ocultada actualizando os hashes da release anterior.

## Release 2.29.0.20261014 (corte de 14out2026)

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

Versão local validada: 2.29.0.20261014. As pastas principais xhdfe e xhdfe-xfe são os pontos de entrada locais.

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
