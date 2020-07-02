add_library(io io.cpp numpy.cpp util.cpp vtk.cpp)

find_package(ZLIB)
if(ZLIB_FOUND)
  target_link_libraries(io PRIVATE ZLIB::ZLIB)
  target_compile_definitions(io PRIVATE GTBENCH_USE_ZLIB)
endif()

target_link_libraries(io PRIVATE common)
