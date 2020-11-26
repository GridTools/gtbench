compile_as_cuda(set_device.cpp TARGET_DIRECTORY gtbench)
target_sources(gtbench PRIVATE set_device.cpp)
