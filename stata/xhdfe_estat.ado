*! version 2.18.2 15jul2026
program xhdfe_estat, rclass
    version `=cond(c(version)<14, c(version), 13)'
    if ("`e(cmd)'" != "xhdfe") {
        error 301
    }

    gettoken key 0 : 0, parse(", ")
    local lkey = length(`"`key'"')

    if `"`key'"' == substr("summarize", 1, max(2, `lkey')) {
        syntax [anything] , [*] [noheader]
        estat_summ `anything', `options' noheader
    }
    else if `"`key'"' == "vce" {
        vce `0'
    }
    else if `"`key'"' == "ic" {
        syntax, [*]
        estat_default ic, df(`=e(df_m)+1') `options'
    }
    else {
        di as error `"invalid subcommand `key'"'
        exit 321
    }
    return add
end
