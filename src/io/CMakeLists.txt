compile_as_cuda(io.cpp numpy.cpp util.cpp vtk.cpp TARGET_DIRECTORY gtbench)
target_sources(gtbench PRIVATE io.cpp numpy.cpp util.cpp vtk.cpp)
