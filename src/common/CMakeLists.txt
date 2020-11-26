compile_as_cuda(options.cpp TARGET_DIRECTORY gtbench)
target_sources(gtbench PRIVATE options.cpp)
