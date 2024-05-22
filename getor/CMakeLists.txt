# file(GLOB SOURCE_V0 *.cpp)
file(GLOB SOURCE_V0 getor.cpp)
add_executable(getor ${SOURCE_V0})
message(STATUS ${CMAKE_CURRENT_SOURCE_DIR})

target_include_directories(getor PUBLIC "C:\\Program Files (x86)\\Intel\\oneAPI\\advisor\\2024.0\\include")
#target_include_directories(getor PUBLIC "D:\\Program Files\\oneAPI\\advisor\\2023.2.0\\include") 