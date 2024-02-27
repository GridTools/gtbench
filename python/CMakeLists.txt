set(GTBENCH_PYTHON_MODULE_NAME "gtbench" CACHE STRING "Name of the generated Python module")

find_package(Python COMPONENTS Interpreter Development)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY "https://github.com/pybind/pybind11.git"
  GIT_TAG        "v2.11.1"
)
FetchContent_MakeAvailable(pybind11)

compile_as_cuda(gtbench.cpp)
Python_add_library(gtbenchpy gtbench.cpp)
target_link_libraries(gtbenchpy PRIVATE pybind11::headers gtbench)
target_compile_definitions(gtbenchpy PRIVATE GTBENCH_PYTHON_MODULE_NAME=${GTBENCH_PYTHON_MODULE_NAME})
set_target_properties(
  gtbenchpy
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    OUTPUT_NAME ${GTBENCH_PYTHON_MODULE_NAME}
)
install(TARGETS gtbenchpy DESTINATION ${Python_SITEARCH})
