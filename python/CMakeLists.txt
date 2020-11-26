find_package(Python COMPONENTS Interpreter Development)

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY "https://github.com/pybind/pybind11.git"
  GIT_TAG        "v2.6"
)
FetchContent_MakeAvailable(pybind11)

Python_add_library(gtbenchpy gtbench.cpp)
target_link_libraries(gtbenchpy PRIVATE pybind11::headers gtbench)
set_target_properties(gtbenchpy PROPERTIES POSITION_INDEPENDENT_CODE ON)
