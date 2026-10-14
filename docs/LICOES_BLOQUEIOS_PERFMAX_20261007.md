# Lições da tranche 1 dos bloqueios perf-max — 07out2026

Estas correções pertencem à candidata privada 2.29.0.20261006. Permanecem
pendentes a revisão independente, a CI nativa e as tranches seguintes; este
ficheiro não certifica merge nem release.

| Achado e classe | Reprodutor anterior | Correção | Teste permanente e limite |
|---|---|---|---|
| N3, cobertura em falta | probe_aarch64.cpp e fma_liveness.cpp com clang, -ffp-contract=off, -D__FP_FAST_FMA=1: 8/64 entradas divergentes | XHDFE_FP_CONTRACT_OFF acompanha a flag clang em CMake, Stata e R; target_mul_add segue a política da unidade de tradução | fma_liveness.cpp; assembly AArch64 e emulação x86 na evidência N3. Hardware arm64 ainda NOT_RUN. |
| N4, cobertura em falta | fake-mingw-g++ com XHDFE_RELEASE_BUILD=1: ambos os scripts terminavam no NOT_RUN | O cross-build regista NOT_RUN; as sondas Windows ligam estaticamente, são verificadas como PE sem DLLs externas e correm no job Windows native-loader | Emulação na evidência N4 e análise de F3 na adjudicação da revisão. Windows nativo e CI ainda NOT_RUN. |
| N5, erro do harness | fakedarwin/ com sonda que devolve 1: FAIL era rotulado NOT_RUN e o build avançava | Os scripts distinguem arquitetura indisponível de sonda executada e falhada; esta termina o build em qualquer modo. Uma lista `lipo -archs` vazia dá NOT_RUN e impede uma release. | Emulações de FAIL e NOT_RUN na evidência N5; `lipo` vazio em ambos os scripts deu rc 1 na cópia isolada de seguimento. macOS nativo ainda NOT_RUN. |
| N7, erro do harness | run_e5_parser.sh: CSR_DOMAIN="0 ", PARALLEL_APPLY="off " e DISJOINT_DIRECT_APPLY="1 " eram ignorados | Os três interruptores entram no snapshot validado por fit. Quando ausente, CSR_DOMAIN conserva o valor de omissão calculado pelo chamador | audit_environment_contract.py passou 144/144; E5 passou. Valor vazio, incluindo XHDFE_MLSMR_LOCKSTEP=, é erro explícito; variável ausente usa a omissão. |
| N8, erro do harness | Portão Makevars/Makevars.win deixava um executável temporário e imprimia uma barra seguida de n | trap remove o executável e a pasta conhecidos; a mensagem usa mudança de linha real | Emulação positiva e negativa de ambos os Makevars, zero resíduos. R CMD INSTALL nativo ainda pendente. |
| N2, cobertura em falta | Módulo selado: g2=300 000, 21 935 MiB com arena contra 7 906 sem arena; mesmos b/V/resíduos e 12 iterações | A hipótese de limitar blocos por pico virtual de capacidade da referência foi rejeitada: desviou ready da arena. O patch foi retirado e a tranche 1 V2 ficou intacta. | Reprodutor e recibos em `implementacao_bloqueios_20261007/n2/`; ready mantém 20 campos idênticos, mas viola o gate de caminho. Fix e teste permanente ainda pendentes. |

Continuam no parser antigo de mlsmr_env_enabled:
XHDFE_MLSMR_AC_BUCKET_ORDERING, XHDFE_MLSMR_AC_DEDUP_DEGREE_UPDATES,
XHDFE_MLSMR_COMPONENT_TRACE, XHDFE_MLSMR_PARALLEL_SCHUR,
XHDFE_MLSMR_BLOCK_ELIM, XHDFE_MLSMR_BATCH_RHS,
XHDFE_KRYLOV_ATOMIC_SCATTER e XHDFE_MLSMR_COLUMN_SCALE.
XHDFE_MLSMR_TRACE usa ainda um parser próprio.
