# Pastas locais ativas do xhdfe

Versão local validada: 2.28.0.20260924. As pastas principais xhdfe e xhdfe-xfe são os pontos de entrada locais.

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
