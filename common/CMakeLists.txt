add_library(options options.cpp)
target_link_libraries(common INTERFACE options)