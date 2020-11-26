compile_as_cuda(advection.cpp diffusion.cpp solver.cpp TARGET_DIRECTORY gtbench)
target_sources(gtbench PRIVATE advection.cpp diffusion.cpp solver.cpp)
target_compile_definitions(gtbench PRIVATE
  GTBENCH_BPARAMS_HADV=${GTBENCH_BPARAMS_HADV}
  GTBENCH_BPARAMS_VADV=${GTBENCH_BPARAMS_VADV}
  GTBENCH_BPARAMS_HDIFF=${GTBENCH_BPARAMS_HDIFF}
  GTBENCH_BPARAMS_VDIFF=${GTBENCH_BPARAMS_VDIFF}
)
