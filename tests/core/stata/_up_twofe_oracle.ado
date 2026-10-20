program define _up_twofe_oracle, eclass
    version 16
    // The legacy v3 call supplies its declared sample. Native explicit dummy
    // regression supplies the two identified, mean-zero FE components.
    tempvar sample turn_effect linear trunk_effect
    quietly gen byte `sample' = e(sample)
    quietly areg price weight length i.trunk if `sample', absorb(turn)
    quietly predict double `turn_effect' if `sample', d
    quietly predict double `linear' if `sample', xb
    quietly gen double `trunk_effect' = `linear' - _b[weight]*weight - _b[length]*length - _b[_cons] if `sample'
    quietly summarize `trunk_effect' if `sample', meanonly
    quietly replace bench2 = `trunk_effect' - r(mean) if `sample'
    quietly replace bench1 = `turn_effect' if `sample'
end
