compile_as_cuda(run.cpp TARGET_DIRECTORY gtbench)
target_sources(gtbench PRIVATE run.cpp)
