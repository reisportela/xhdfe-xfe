cmake_minimum_required(VERSION 3.18)

get_filename_component(repo_root "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(sources
    src/akm_kss.cpp
    src/fe_absorption.cpp
    src/hdfe_regressor_v11.cpp)
set(pattern
    "std::is(finite|nan|inf)[ \t]*\\(|\\.allFinite[ \t]*\\(|\\.hasNaN[ \t]*\\(")
set(violations "")

foreach(source IN LISTS sources)
    file(STRINGS "${repo_root}/${source}" lines)
    set(line_number 0)
    foreach(line IN LISTS lines)
        math(EXPR line_number "${line_number} + 1")
        if(line MATCHES "${pattern}")
            list(APPEND violations "${source}:${line_number}: ${line}")
        endif()
    endforeach()
endforeach()

if(violations)
    list(JOIN violations "\n" details)
    message(FATAL_ERROR
        "Raw non-finite guard found in a fast-math translation unit:\n"
        "${details}\nUse include/hdfe/ieee_bits.hpp instead.")
endif()

message(STATUS
    "Fast-math translation units use only IEEE bit-level non-finite guards")
