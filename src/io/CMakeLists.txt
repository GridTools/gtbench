compile_as_cuda(io.cpp numpy.cpp util.cpp vtk.cpp)
add_library(io io.cpp numpy.cpp util.cpp vtk.cpp)
target_link_libraries(io PUBLIC gtbench_common)
target_link_libraries(gtbench PUBLIC io)
