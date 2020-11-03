compile_as_cuda(run.cpp)

if(NOT _ghex_already_fetched)
    find_package(GHEX QUIET)
endif()
if(NOT GHEX_FOUND)
    set(_ghex_repository "https://github.com/GridTools/GHEX.git")
    set(_ghex_tag        "dedd9765a40ae950f7cdaa2808000a2e3952d6fb")
    if(NOT _ghex_already_fetched)
        message(STATUS "Fetching GHEX ${_ghex_tag} from ${_ghex_repository}")
    endif()
    include(FetchContent)
    FetchContent_Declare(
        ghex
        GIT_REPOSITORY ${_ghex_repository}
        GIT_TAG        ${_ghex_tag}
    )
    FetchContent_MakeAvailable(ghex)
    set(_ghex_already_fetched ON CACHE INTERNAL "")
endif()

add_library(runtime run.cpp)
target_link_libraries(runtime common_runtime GHEX::ghexlib)

if(GHEX_USE_UCP)
    target_compile_definitions(runtime PRIVATE GTBENCH_USE_GHEX_UCP)
    if(GHEX_USE_PMIX)
        target_compile_definitions(runtime PRIVATE GTBENCH_USE_GHEX_PMIX)
    endif()
endif()
if(GHEX_USE_XPMEM)
    target_compile_definitions(runtime PRIVATE GHEX_USE_XPMEM)
endif()
