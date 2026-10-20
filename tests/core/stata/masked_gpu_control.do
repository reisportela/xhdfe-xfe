version 16
clear all
sysuse auto, clear
capture noisily _up_xhdfe price weight length, absorb(turn)
assert _rc != 0
display "CORE_MASKED_GPU_PASS"
