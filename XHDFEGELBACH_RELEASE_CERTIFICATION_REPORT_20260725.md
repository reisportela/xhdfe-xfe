# Relatório de certificação do release xhdfe 2.21.0.20260725

## Veredicto

**GO local para criação do tag e do draft release.**

A implementação revista do Gelbach passou os gates funcionais, numéricos,
documentais, de empacotamento e de desempenho aplicáveis no checkout privado.
O release só pode ser promovido de draft a público depois de os binários CUDA
exatos produzidos pelo CI dos dois repositórios serem descarregados e
executados no H100 com prova de uso real da GPU. Esta separação é deliberada:
um tag cria o candidato; o marcador `publish-v2.21.0.20260725` certifica e
publica apenas o candidato que passou a segunda fase.

Versões certificadas localmente:

- pacote C++/Python/R: `2.21.0.20260725`;
- Stata `xhdfe`: `2.21.0`;
- Stata `xhdfegelbach`: `1.5.0`;
- companheiros Stata de bootstrap/tabela/gráfico: `1.0.0`;
- data comum dos ficheiros Stata de produção: `25jul2026`.

## Âmbito auditado

A certificação cobriu:

- o núcleo C++ partilhado e as suas cópias vendorizadas;
- Python, R e Stata, incluindo ajuda e exemplos;
- CPU e CUDA, com CUDA compilado especificamente para `sm_90` no checkout
  privado;
- decomposições com vários coeficientes-alvo, blocos observados
  multicoluna, FEs adicionados, FEs comuns, targets absorvidos e conectividade;
- inferência analítica, bootstrap por pares, shares, contrasts e agregação de
  componentes filtrados;
- proveniência da amostra retida;
- artefactos instaláveis Python e R;
- plugins Stata CPU/OpenMP e CUDA/OpenMP;
- o caminho normal de estimação fora de Gelbach através da matriz
  `core23 × 8`.

Não foram alterados a definição do estimador, os critérios de convergência, as
tolerâncias, a normalização dos efeitos fixos ou o comportamento por omissão
dos comandos não Gelbach.

## Funcionalidades fechadas neste release

O release deixa de tratar `b1x2` como limite da aplicabilidade. O oracle
continua a ser usado no seu domínio clássico, mas a API pública passa a
suportar:

- vários targets e vários blocos observados com uma ou mais colunas;
- qualquer número de dimensões FE adicionadas;
- FEs comuns às especificações base e completa;
- diagnóstico de conectividade para um par de FEs selecionável e modo
  `connected(require)` no domínio efetivamente identificado;
- targets explicitamente absorvidos, com falha fechada fora do contrato;
- contrasts lineares múltiplos usando a covariância conjunta;
- diagnósticos de regularidade e de colinearidade FE;
- bootstrap iid-pairs e cluster-pairs com refit integral;
- intervalos para componentes, total, coeficientes base/completo e shares;
- tabelas Markdown, LaTeX, HTML e CSV, além de dados para gráficos;
- proveniência opt-in da amostra, com posições retidas e identificador
  `fnv1a64-le-v1`.

As fronteiras científicas permanecem explícitas: trata-se de decomposição
exata do movimento de coeficientes em modelos lineares, não de mediação causal
automática nem de descoberta de narrativas.

## Evidência funcional e numérica

| Gate | Resultado |
|---|---|
| `VALIDATE_GELBACH.py` | PASS, incluindo identidades e oracles |
| `VALIDATE_GELBACH_ADVERSARIAL.py` | PASS, incluindo guards de threads, rank, pesos e regularidade |
| `VALIDATE_GELBACH_PYFIXEST_FEATURES.py` | PASS |
| `VALIDATE_GELBACH_SAMPLE_PROVENANCE.py` | PASS, 36 checks |
| `VALIDATE_GELBACH_REMEDIATION_COVERAGE.py` | PASS |
| `VALIDATE_GELBACH_FRONTENDS.py`, CPU | PASS |
| `VALIDATE_GELBACH_FRONTENDS.py`, CUDA | PASS, Python/R/Stata reportaram `cuda` e `used` |
| `VALIDATE_GELBACH_HELP.py` | PASS, 111 checks |
| suite Stata completa | PASS, 28 grupos de testes |
| smoke Stata de bootstrap | PASS |
| R `testthat`, CPU | PASS; apenas warnings/skip esperados |
| R `testthat`, CUDA | PASS sem falhas, com CUDA efetivamente usada |
| compilação real de tabelas LaTeX | PASS em Stata e R |

Na paridade Python CPU/CUDA:

- ambas as execuções convergiram em 7 iterações;
- diferença máxima dos coeficientes: `5.773e-15`;
- diferença máxima dos erros-padrão: `3.941e-16`;
- o backend CUDA foi selecionado e efetivamente usado.

O estudo de cobertura da remediação confirmou que os novos gates não são
meramente decorativos. No cenário de share fraco, a cobertura bootstrap foi
`0.916`, contra `0.644` para o intervalo analítico. Nos cenários dominados por
FE, as coberturas bootstrap foram `0.995`, `0.980` e `0.975`, contra `0.870`,
`0.665` e `0.350` para os intervalos analíticos condicionais.

## Matriz de não regressão `core23 × 8`

Foram executados os 23 datasets/especificações nas oito células obrigatórias:

1. C++ CPU fast;
2. C++ CUDA fast;
3. C++ CPU comparable;
4. C++ CUDA comparable;
5. Stata/plugin CPU fast;
6. Stata/plugin CUDA fast;
7. Stata/plugin CPU comparable;
8. Stata/plugin CUDA comparable.

Resultado:

- `184/184` execuções concluídas com sucesso;
- `92/92` execuções CUDA reportaram uso real da GPU;
- nenhuma não convergência;
- diferença máxima absoluta dos coeficientes contra a referência de
  23jul2026: `4.028e-7`;
- diferença relativa máxima: `6.355e-7`;
- diferença máxima dos erros-padrão: `7.99e-7`;
- razão mediana global de runtime atual/referência: `0.883`.

Exemplos das especificações de maior duração:

| Dataset | C++ CPU fast | C++ CUDA fast | Stata CPU fast | Stata CUDA fast |
|---|---:|---:|---:|---:|
| `main_95_21_ready` | 39.906 s | 6.397 s | 43.037 s | 13.540 s |
| `akm_v02_secondreg` | 131.850 s | 50.581 s | 140.944 s | 71.430 s |

Uma observação isolada sugeriu ruído em duas células Stata CUDA de 10 milhões
de linhas. Foram repetidas três vezes, juntamente com duas especificações de
controlo. As medianas foram:

- `pf_simple_10m_2fe`: 6.10 s;
- `pf_simple_10m_3fe`: 6.07 s;
- `pf_difficult_10m_2fe`: 5.94 s;
- `pf_difficult_10m_3fe`: 7.43 s.

Todas as 12 repetições tiveram `rc=0`, convergiram e reportaram
`gpu_used=1`. A anomalia não se confirmou. O resumidor do harness foi também
corrigido para aceitar subconjuntos com apenas CPU ou apenas CUDA.

## Builds e instalação isolada

### C++ e Stata

- builds CMake CPU e CUDA: PASS;
- CUDA local: apenas `sm_90`;
- plugins de produção: OpenMP confirmado por ligação a `libgomp`;
- grande smoke Stata CPU em `main_95_21_ready`: 64.833 s, 20 iterações,
  dentro da referência de 60–68 s;
- bundle Stata CPU: PASS;
- bundle Stata CUDA: PASS para `xhdfe` e `xfe`, com
  `e(gpu_used)==1`, backend `cuda` e estado `used`;
- diagnóstico `xfe`: CUDA usada, Metal rejeitado no domínio ainda não
  implementado e zigzag concluído.

Hashes dos plugins privados locais:

```text
d99aecd0399a5df5bd3d1871c9782459fa50fc6a1855612d9ebcc181ac904ab4  stata/xhdfe.plugin
075216354223853e0033cc154cbf25d2938195829c433dc4593ea1e7bc1dfb86  stata/xfe.plugin
```

### Python

Foram construídos e instalados em ambientes limpos tanto a wheel como o
sdist:

```text
7bccc3f2fff7727bd9525e8855fb86fb220ef1f40d43b2f5ca8002913ec13c39  xhdfe-2.21.0.20260725-cp312-cp312-linux_x86_64.whl
9ac4bc5de7cf2d7cfc231346badad602731757764aca12d859d7c3b4818f2868  xhdfe-2.21.0.20260725.tar.gz
```

O teste isolado do sdist revelou e permitiu corrigir um defeito real de
empacotamento: o CMake procurava apenas o mirror Eigen do pacote R, que não
pertence ao sdist Python. O resolver offline usa agora esse mirror no checkout
completo e o vendored canónico `third_party/eigen-3.4.0` no sdist. Não existe
fallback para a rede nem para uma instalação de sistema.

### R

O pacote source foi construído e instalado em CPU e CUDA:

```text
219e64cef46d17b6a398d2d4bc5f7faa3377546e68474e33ae85c4c7781ffdf8  xhdfe_2.21.0.20260725.tar.gz
```

`R CMD check` concluiu todos os checks de código, documentação, exemplos e
testes: `1068` testes passaram e não houve falhas. O estado `1 WARNING,
1 NOTE` corresponde a limitações já conhecidas e declaradas:

- extensões CUDA/Metal adicionais dentro de `src`;
- referências a stdout/stderr existentes no backend compilado.

Os pacotes sugeridos `nanoparquet` e `gt` não estavam instalados e foram
tratados como `Suggests`, não como dependências de execução.

## Alinhamento e documentação

- `tools/check_cpp_core_alignment.sh`: PASS para os 20 mirrors R/share/Stata;
- `tools/check_default_builds.sh`: PASS, incluindo build a partir do sdist;
- versão partilhada alinhada em CMake, Python, R e metadados Stata;
- todos os ficheiros Stata de produção usam `25jul2026`;
- os seis novos ficheiros companheiros Stata estão em `xhdfe.pkg`;
- ajuda Python, R e Stata cobre defaults, estados de inferência, provenance,
  RNG entre frontends e vida útil de `r()` após gráficos;
- as tabelas LaTeX produzidas por Stata e R compilaram com `pdflatex`.

## Gates da publicação remota

Depois do commit e da sincronização byte-a-byte com o repositório público:

1. criar e enviar `v2.21.0.20260725` em ambos os repositórios;
2. deixar os workflows criarem apenas drafts;
3. descarregar os artefactos `plugins-linux` exatos dos dois workflows;
4. executar os plugins CUDA fatbin no H100 e exigir uso real da GPU;
5. verificar checksums, conteúdo do bundle offline e pacote net-install;
6. só então enviar `publish-v2.21.0.20260725`;
7. verificar releases públicos, `gh-pages` e uma instalação Stata isolada.

Qualquer falha nesta segunda fase converte o veredicto em **NO-GO remoto** sem
invalidar a evidência local; o draft deve permanecer não publicado até
remediação.
