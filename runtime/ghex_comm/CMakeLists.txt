compile_as_cuda(run.cpp)

find_package(GHEX REQUIRED)

add_library(runtime run.cpp)
target_link_libraries(runtime common GHEX::ghexlib)

if(GHEX_USE_UCP)
    target_compile_definitions(runtime INTERFACE GTBENCH_USE_GHEX_UCP)
    if(GHEX_USE_PMIX)
        target_compile_definitions(runtime INTERFACE GTBENCH_USE_GHEX_PMIX)
    endif()
endif()