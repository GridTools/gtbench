add_library(runtime)
target_link_libraries(runtime PUBLIC gtbench_common)
target_link_libraries(gtbench PUBLIC runtime)

add_subdirectory(${GTBENCH_RUNTIME})
add_subdirectory(device)
