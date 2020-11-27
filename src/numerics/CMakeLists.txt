compile_as_cuda(advection.cpp diffusion.cpp solver.cpp)
add_library(numerics advection.cpp diffusion.cpp solver.cpp)
set_target_properties(numerics PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_definitions(numerics PRIVATE
  GTBENCH_BPARAMS_HADV=${GTBENCH_BPARAMS_HADV}
  GTBENCH_BPARAMS_VADV=${GTBENCH_BPARAMS_VADV}
  GTBENCH_BPARAMS_HDIFF=${GTBENCH_BPARAMS_HDIFF}
  GTBENCH_BPARAMS_VDIFF=${GTBENCH_BPARAMS_VDIFF}
)
target_link_libraries(numerics PUBLIC gtbench_common)
target_link_libraries(gtbench PUBLIC numerics)
