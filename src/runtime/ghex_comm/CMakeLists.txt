if(NOT _ghex_already_fetched)
  find_package(GHEX QUIET)
endif()
if(NOT GHEX_FOUND)
  set(_ghex_repository "https://github.com/GridTools/GHEX.git")
  set(_ghex_tag        "4d48f3145349064942941f191021b970db3cb36e")
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

compile_as_cuda(run.cpp TARGET_DIRECTORY runtime)
target_sources(runtime PRIVATE run.cpp)
target_link_libraries(runtime PUBLIC GHEX::ghexlib)
target_compile_options(runtime PUBLIC "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--default-stream per-thread>")

if(GHEX_USE_UCP)
  target_compile_definitions(runtime PRIVATE GTBENCH_USE_GHEX_UCP)
  if(GHEX_USE_PMIX)
    target_compile_definitions(runtime PRIVATE GTBENCH_USE_GHEX_PMIX)
  endif()
endif()
if(GHEX_USE_XPMEM)
    target_compile_definitions(runtime PRIVATE GHEX_USE_XPMEM)
endif()
