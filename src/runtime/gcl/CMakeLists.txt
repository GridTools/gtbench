if(GTBENCH_BACKEND STREQUAL "gpu")
    set(GTBENCH_GCL_ARCH "gpu")
else()
    set(GTBENCH_GCL_ARCH "cpu")
endif()

compile_as_cuda(run.cpp)
target_sources(gtbench PRIVATE run.cpp)
target_link_libraries(gtbench PUBLIC GridTools::gcl_${GTBENCH_GCL_ARCH})
