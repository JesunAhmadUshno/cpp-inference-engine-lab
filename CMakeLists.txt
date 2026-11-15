cmake_minimum_required(VERSION 3.20)
project(cpp-inference-engine-lab CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Threads REQUIRED)

add_library(engine INTERFACE)
target_include_directories(engine INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(engine INTERFACE -O3 -Wall -Wextra)
target_link_libraries(engine INTERFACE Threads::Threads)

add_executable(engine_bench src/bench.cpp)
target_link_libraries(engine_bench PRIVATE engine)

add_executable(engine_generate src/generate.cpp)
target_link_libraries(engine_generate PRIVATE engine)

install(TARGETS engine_generate engine_bench RUNTIME DESTINATION bin)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/engine DESTINATION include)

enable_testing()
add_executable(engine_tests tests/test_engine.cpp)
target_link_libraries(engine_tests PRIVATE engine)
add_test(NAME engine_tests
         COMMAND engine_tests ${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures)
