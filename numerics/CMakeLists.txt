compile_as_cuda(advection.cpp diffusion.cpp)

add_library(advection advection.cpp)
target_compile_definitions(advection PRIVATE
  GTBENCH_BPARAMS_HADV=${GTBENCH_BPARAMS_HADV}
  GTBENCH_BPARAMS_VADV=${GTBENCH_BPARAMS_VADV}
)
target_link_libraries(advection common)

add_library(diffusion diffusion.cpp)
target_compile_definitions(diffusion PRIVATE
  GTBENCH_BPARAMS_HDIFF=${GTBENCH_BPARAMS_HDIFF}
  GTBENCH_BPARAMS_VDIFF=${GTBENCH_BPARAMS_VDIFF}
)
target_link_libraries(diffusion common)
