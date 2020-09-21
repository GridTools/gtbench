if(GTBENCH_BACKEND STREQUAL "gpu")
    set(GTBENCH_GCL_ARCH "gpu")
else()
    set(GTBENCH_GCL_ARCH "cpu")
endif()

compile_as_cuda(run.cpp)

add_library(runtime run.cpp)
target_link_libraries(runtime common_runtime GridTools::gcl_${GTBENCH_GCL_ARCH})
