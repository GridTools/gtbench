if(NOT _ghex_already_fetched)
  find_package(GHEX QUIET)
endif()
if(NOT GHEX_FOUND)
  set(_ghex_repository "https://github.com/ghex-org/GHEX.git")
  set(_ghex_tag        "v0.2.0")
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
target_link_libraries(runtime PUBLIC GHEX::lib)
target_link_libraries(runtime PUBLIC oomph::mpi)
target_compile_options(runtime PUBLIC "$<$<COMPILE_LANGUAGE:CUDA>:SHELL:--default-stream per-thread>")

if(GHEX_USE_XPMEM)
    target_compile_definitions(runtime PRIVATE GHEX_USE_XPMEM)
endif()
