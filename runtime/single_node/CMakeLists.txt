compile_as_cuda(run.cpp)

add_library(runtime run.cpp)
target_link_libraries(runtime common_runtime)