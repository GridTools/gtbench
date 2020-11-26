if(NOT _ghex_already_fetched)
  find_package(GHEX QUIET)
endif()
if(NOT GHEX_FOUND)
  set(_ghex_repository "https://github.com/GridTools/GHEX.git")
  set(_ghex_tag        "7606d6be813288f2f1efe67c6c8aef8d66e37bd1")
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

compile_as_cuda(run.cpp TARGET_DIRECTORY gtbench)
target_sources(gtbench PRIVATE run.cpp)
target_link_libraries(gtbench PUBLIC GHEX::ghexlib)
target_compile_options(gtbench PUBLIC "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--default-stream per-thread>")

if(GHEX_USE_UCP)
  target_compile_definitions(runtime PRIVATE GTBENCH_USE_GHEX_UCP)
  if(GHEX_USE_PMIX)
    target_compile_definitions(runtime PRIVATE GTBENCH_USE_GHEX_PMIX)
  endif()
endif()
