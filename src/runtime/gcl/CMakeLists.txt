if(GTBENCH_BACKEND STREQUAL "gpu")
  set(GTBENCH_GCL_ARCH "gpu")
else()
  set(GTBENCH_GCL_ARCH "cpu")
endif()

compile_as_cuda(run.cpp TARGET_DIRECTORY runtime)
target_sources(runtime PRIVATE run.cpp)
target_link_libraries(runtime PUBLIC GridTools::gcl_${GTBENCH_GCL_ARCH})

# Workaround for GT export strategy
install(TARGETS gcl_${GTBENCH_GCL_ARCH} EXPORT GTBenchTargets)
