version 16
clear all
set more off

args adodir
adopath ++ "`adodir'"

which xhdfegelbach
findfile xhdfegelbach.sthlp
di as txt "HELP_FILE=" r(fn)

exit 0
