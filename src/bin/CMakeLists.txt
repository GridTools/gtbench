compile_as_cuda(benchmark.cpp convergence_tests.cpp)
add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE gtbench)
target_compile_definitions(benchmark PRIVATE
  GTBENCH_RUNTIME=${GTBENCH_RUNTIME}
)
add_executable(convergence_tests convergence_tests.cpp)
target_link_libraries(convergence_tests PRIVATE gtbench)
target_compile_definitions(convergence_tests PRIVATE
  GTBENCH_RUNTIME=${GTBENCH_RUNTIME}
)
