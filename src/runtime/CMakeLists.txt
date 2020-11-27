add_library(runtime)
target_link_libraries(runtime PUBLIC gtbench_common)
set_target_properties(runtime PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(gtbench PUBLIC runtime)

add_subdirectory(${GTBENCH_RUNTIME})
add_subdirectory(device)
