// Deterministic patent-chain fixture; long/wide design follows reghdfe tests.
version 16

capture program drop toy_dificil
program define toy_dificil
    args nos reps
    if "`nos'"  == "" local nos  128
    if "`reps'" == "" local reps 2

    clear
    set seed 1234
    set obs `= 4 * `reps' * `nos''
    generate long patent_id = _n
    generate int  posicao = mod(floor((_n - 1) / (4 * `reps')), `nos')
    generate byte tipo    = mod(floor((_n - 1) /  `reps'), 4)

    // Ano de concessao: as patentes de cada posicao da cadeia concentram-se no
    // tempo, o que impede o FE de ano de reconstruir o modo lento sozinho.
    generate int year = posicao + 1 + `nos' * (tipo >= 2)

    // Cada equipa tem duas colaboracoes locais (tipos 0 e 1) ou um so inventor
    // com o seu grupo (tipos 2 e 3), mais cinco inventores seniores presentes
    // em todas as patentes.
    generate byte equipa = cond(tipo < 2, 7, 6)
    expand equipa
    bysort patent_id: generate byte lugar = _n
    generate long inventor_id = .
    replace inventor_id = mod(posicao + `nos' - 1, `nos') + 1 if tipo == 0 & lugar == 1
    replace inventor_id = mod(posicao + 1, `nos') + 1         if tipo == 0 & lugar == 2
    replace inventor_id = posicao + 1                         if tipo == 1 & lugar == 1
    replace inventor_id = posicao + 1 + `nos'                 if tipo == 1 & lugar == 2
    replace inventor_id = posicao + 1                         if tipo == 2 & lugar == 1
    replace inventor_id = posicao + 1 + `nos'                 if tipo == 3 & lugar == 1
    replace inventor_id = 2 * `nos' + lugar - 2               if tipo <  2 & lugar >= 3
    replace inventor_id = 2 * `nos' + lugar - 1               if tipo >= 2 & lugar >= 2
    assert !missing(inventor_id)
    isid patent_id inventor_id, sort

    // Capacidade suave ao longo da cadeia. O sinal de cada patente e a soma das
    // capacidades dos seus membros, como no ficheiro do Sergio.
    generate double ability = cos(2 * _pi * (inventor_id - 1) / `nos') if inventor_id <= 2 * `nos'
    replace ability = 0 if missing(ability)
    by patent_id: egen double sum_ability = total(ability)

    // Ciclo de financiamento que opoe colaboracao externa (tipo 0) a colaboracao
    // interna (tipo 1) e varia lentamente ao longo da cadeia. E a direccao em que
    // os FE individuais quase nao tem poder: para a acompanhar, as capacidades
    // estimadas tem de crescer muito. Como o financiamento tambem a segue, um
    // solver que pare cedo transfere o erro dos FE para o coeficiente.
    generate double ciclo = cond(tipo == 0, -1, cond(tipo == 1, 1, 0)) * cos(2 * _pi * posicao / `nos')
    generate double funding  = 3 + .4 * ciclo + rnormal(0, .2)
    generate double lab_size = 2 + .5 * ciclo + rnormal(0, .8)
    generate double citations = 1.5 + 3 * funding - 1 * lab_size ///
        + .1 * sum_ability + .35 * ciclo + rnormal(0, .113)
    by patent_id: replace funding   = funding[1]
    by patent_id: replace lab_size  = lab_size[1]
    by patent_id: replace citations = citations[1]

    generate byte c = 1
    generate double w1 = 10 * runiform()
    by patent_id: replace w1 = w1[1]
    generate double w2 = ceil(w1)

    keep patent_id inventor_id year citations funding lab_size c w1 w2
    order patent_id inventor_id year citations funding lab_size c w1 w2
    label data "toy-patents dificil: rede de inventores em cadeia"

    quietly tabulate patent_id
    local np = r(r)
    quietly tabulate inventor_id
    local ni = r(r)
    di as text _n "toy-patents dificil: " as res _N as text " linhas, " ///
       as res `np' as text " patentes, " as res `ni' as text " inventores, " ///
       as res `= 2 * `nos'' as text " anos"
end
