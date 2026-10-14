# Lições da auditoria e implementação de performance — 30set2026

Referência: [auditoria](../Verifications/perf_max_20260929/AUDITORIA_CODEX_20260930.md)
e [plano de execução](../Verifications/perf_max_20260929/PLANO_ENCOMPASSING_CODEX_20260930.md).
Estas lições descrevem evidência local; não certificam release ou ótimo global.

## Pesos declarados têm de ser passados ao fit

Classificação: erro do harness de avaliação. O worker antigo gerava pesos mas
as branches group/slopes não os passavam. none/aw/fw podiam representar o
mesmo fit com rótulos diferentes. Foi corrigida a construção comum de kwargs;
os pesos sintéticos dos grupos são constantes dentro de cada grupo.

Reprodução preservada em `auditoria_codex_20260930/weighted_worker/`;
correção aplicada a `Verifications/perf_max_20260929/perf_worker.py` após
autorização específica. Validação no ficheiro alterado: seis fits, estimativas
com pesos efetivamente distintas e resultados iguais aos recibos corrigidos.
Recuperação e hashes em `implementacao_codex_20260930/C0_PREVIEW.json`.

## A sonda de compilação deve observar o domínio aritmético do kernel

Classificação: erro do harness de avaliação. Em Clang baseline, uma chamada
std::fma no main sob fast-math podia ser transformada fora do âmbito FP exato
dos kernels, produzindo falsa recusa da toolchain. A sonda passou a chamar uma
função noinline com os mesmos atributo/pragmas FP exatos do produto.

Regressão permanente: `tests/fma_liveness.cpp`. GCC/Clang × native/baseline:
quatro compilações e execuções PASS sobre o ficheiro já corrigido. A correção
não altera kernels, tolerâncias ou a definição das estimativas.

## A matriz esperada é declarada antes de ler os resultados

Classificação: erro do harness e falta de cobertura. Um glob só encontra as
linhas presentes; omitir ambos os braços de uma célula pode produzir falso
PASS. Core24 inclui literalmente AKM2 e Marta. Cada run tem dataset, modo,
backend, worker/avaliador/artefactos, pesos, threads e ambiente declarados.

Regressão offline: `implementacao_codex_20260930/gate_controls.py`.
Treze controlos, incluindo uma célula inteira ausente e etiquetas falsas nos
dois braços, passaram. NOT_RUN, falha, recusa e PASS permanecem distintos.
Uma mudança do código de avaliação exige um novo manifest.

## Stata precisa de resultados completos e um sentinel executado

Classificação: erro do harness de avaliação. O exit code 0 do batch não prova
sucesso do comando. O worker novo exige uma linha terminal exatamente igual
ao sentinel; o eco de `display "sentinel"` não serve. Guarda b e V FP64,
metadados, omissões, amostra e resíduos por fit, numa sessão nova por plugin.

Dois erros encontrados no desenvolvimento do próprio worker foram mantidos
nos recibos falhados: `datasignature varlist` foi substituído por
`_datasignature varlist`, após inspeção da implementação Stata instalada;
diagnósticos GPU ausentes em CPU são preservados como ausentes. Não foram
substituídos por zero para aparentar cobertura. Primeiro par válido credit
CPU Comparable: b/SE/V/RSS bit a bit, sample e inputs iguais.

## Um microbenchmark rápido não implica um fit rápido

U1 removeu scans redundantes de edges: 54 comparações conservaram bytes,
com ganhos locais grandes. Seis pares de fits completos ready/AKM não
demonstraram ganho material (−0,586% e +0,028%). A proposta foi conservada
como experiência e não foi integrada.

## Projeção e recuperação devem ser avaliadas juntas

U2 iniciou a projeção sem alpha capture, mas perdeu o warm start da recuperação.
No ready savefe, MAP passou de 17 para 349 iterações e o fit ficou 59,75% mais
lento. A diferença V também ultrapassou o diagnóstico Comparable. A proposta
foi rejeitada; não se afrouxou o stop nem se apresentou apenas o tempo da
projeção. Identidade reconstruída, por si só, não certifica a normalização de
cada efeito fixo.

## As threads pedidas têm de ser usadas e o seu custo deve ser visível

U8 conserva as somas dos operadores group/individual mas partilha a região
OpenMP entre segmentos. Nos três pares por modo, a mediana com 16 threads
melhorou menos de 2%; com 48, cerca de 25%. O tempo absoluto com 48 continua
muito pior do que com 16 neste host. Os dois resultados são reportados;
não se impõe um cap oculto nem se confunde este ensaio com a matriz completa.
O protótipo está isolado e desligado por omissão; validação adicional em curso.
