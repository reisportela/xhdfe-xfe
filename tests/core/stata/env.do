* Only the isolated dependency copy and selected candidate are non-system paths.
local task_root : env XHDFE_UPSTREAM_ROOT
local task_tests : env XHDFE_UPSTREAM_TESTROOT
local task_dependencies : env XHDFE_CORE_DEPENDENCIES
local task_package : env XHDFE_CORE_PACKAGE
sysdir set PLUS "`task_dependencies'/ado/plus/"
sysdir set PERSONAL "`task_dependencies'/ado/personal/"
sysdir set OLDPLACE "`task_dependencies'/ado/"
global S_ADO "BASE;SITE;`task_root';`task_package';`task_tests';`task_tests'/part1;`task_tests'/part2;`task_dependencies'/aux;PERSONAL;PLUS;OLDPLACE"
set more off
set linesize 255
set varabbrev on
