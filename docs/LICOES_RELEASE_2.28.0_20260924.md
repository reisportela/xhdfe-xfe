# Lições da preparação da release 2.28.0

## Windows: recusa indevida por empacotamento

O issue público [#11](https://github.com/reisportela/xhdfe-xfe/issues/11)
reporta `r(9999)` ao carregar o plugin em Stata 19.5. As entradas `G WIN64`
instalam os runtimes em pastas por letra do Stata, que não são automaticamente
pastas de pesquisa de dependências do carregador Windows. Testar com as DLLs
ao lado do executável não reproduzia essa instalação.

A correcção liga estaticamente os runtimes GNU/OpenMP dos dois plugins Stata.
O teste `windows-stata-loader.yml` usa `LoadLibraryW`, executável separado,
plugins em `plus/x`, runtimes do controlo negativo em `plus/l` e PATH limpo.
O controlo dinâmico reproduz erro 126; os plugins corrigidos executam um e dois
workers, com concordância numérica. A verificação PE exige apenas DLLs do
sistema. A roda Python mantém o seu mecanismo distinto de dependências.

O comportamento de pesquisa está documentado pela
[Microsoft](https://learn.microsoft.com/en-us/windows/win32/dlls/dynamic-link-library-search-order).
As opções de ligação são documentadas pelo
[GCC](https://gcc.gnu.org/onlinedocs/gcc/Link-Options.html).
O teste nativo não é uma licença Stata nem uma execução na máquina do autor
do issue; esse teste continua a ser solicitado após a publicação.

## Verificador PE: erro de harness

O verificador antigo exigia uma importação de `libgomp-1.dll` e pelo menos
uma DLL redistribuída. Essa condição é adequada ao contrato dinâmico da roda
Python, mas rejeita um plugin com OpenMP estático. O modo explícito
`--system-only` exige zero dependências externas; a prova de OpenMP é dada
pelos arquivos efectivamente ligados e pelos workers observados.

Os testes permanentes `test_windows_stata_static.py` rejeitam dependências
escondidas, substituição de plugins e proveniência incompleta. Os controlos
dinâmicos anteriores continuam activos. Não houve alteração ao estimador.

## macOS Intel: cobertura em falta no alvo mínimo

O preflight nativo demonstrou que a escrita atómica da cache usava
`std::filesystem`, disponível na libc++ apenas desde macOS 10.15, enquanto
o alvo Intel anunciado é 10.12. O ramo Apple passa a resolver os caminhos
com `realpath` e a publicar com `rename`; a criação exclusiva do temporário,
as permissões e a limpeza continuam no mesmo caminho POSIX.

`tests/analytic_precision/cache_macos_compat_contract.py` compara as duas
implementações em nove casos, incluindo aliases e interrupção sem commit.
O workflow macOS executa este controlo e compila o alvo mínimo real. Os ramos
pré-processados Linux/Windows da classe permanecem idênticos. Não se mudou
a definição do estimador, uma tolerância ou um critério de convergência.

O teste de controlo macOS com `--out` conserva os seus recibos. Uma descoberta
genérica de testes não inicializa esse contexto; passa a indicar explicitamente
que é necessário usar a entrada documentada, em vez de falhar com `OUT=None`.

## Outros erros de harness encontrados pelo CI

O gate CUDA de disassembly parava por ausência de `TMPDIR`, antes de concluir
a verificação SASS. A contagem auxiliar passa a ser recolhida em memória,
sem ficheiro temporário. Mantêm-se as regras PTX/SASS e os casos de expansão
de divisão já permitidos. Vinte controlos positivos/negativos cobrem os dois
scripts; o plugin CUDA local também passou a análise real de disassembly.

O build Windows dentro do job Linux encontrou ainda um directório temporário
criado como root pelo contentor manylinux. Passa a correr no job isolado já
validado, com as mesmas flags e gates, sem alterar permissões do workspace.

Dois testes R tinham expectativas anteriores às correcções auditadas:

- Instrumentos exactamente duplicados não tornam subidentificado um modelo
  cujo espaço de instrumentos continua suficiente. A candidata concordou
  com 2SLS por projecção QR independente a 2,44e-15; duplicar o instrumento
  mudou b em 1,33e-15 e V em 6,51e-19. O teste conserva as recusas para
  instrumento nulo e subidentificação verdadeira.
- A correcção KSS pode produzir estimativas negativas de componentes de
  variância. No fixture com controlos, ambas são negativas; a correlação é
  indefinida e deve ser NA. Exigir que todos os campos, incluindo essa
  correlação, fossem finitos premiava a apresentação de um valor inválido.
  O teste exige componentes finitas e verifica explicitamente esse NA.

Os reprodutores conservaram os dados e o estimador. As duas baterias R
afectadas passaram depois de corrigir as expectativas. As advertências sobre
inferência AKM não identificada e graus de liberdade aproximados permanecem.
