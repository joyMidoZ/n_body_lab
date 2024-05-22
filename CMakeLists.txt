cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_COMPILER "icx-cl.exe")
set(CMAKE_C_COMPILER "icx-cl.exe")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Debug)

add_compile_options(-O3 -Qopt-report=max -debug -Qopenmp /Qprec-sqrt- /Qprec-div-)

#add_compile_options(-O3 -Qopt-report=max -debug -Qopenmp /arch:SSE2 /Qprec-sqrt- /Qprec-div-)

# /arch:CORE-AVX2 -Qopenmp /QxCORE-AVX2 /Qprec-sqrt- /Qprec-div-

project(getor)

add_subdirectory(getor)