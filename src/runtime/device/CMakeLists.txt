compile_as_cuda(set_device.cpp TARGET_DIRECTORY runtime)
target_sources(runtime PRIVATE set_device.cpp)
