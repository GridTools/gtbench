compile_as_cuda(advection.cpp diffusion.cpp)

add_library(advection advection.cpp)
target_compile_definitions(advection PRIVATE
  GTBENCH_BPARAMS_RKADV1=${GTBENCH_BPARAMS_RKADV1}
  GTBENCH_BPARAMS_RKADV2=${GTBENCH_BPARAMS_RKADV2}
  )
target_link_libraries(advection common)

add_library(diffusion diffusion.cpp)
target_compile_definitions(diffusion PRIVATE
  GTBENCH_BPARAMS_HDIFF=${GTBENCH_BPARAMS_HDIFF}
  GTBENCH_BPARAMS_VDIFF1=${GTBENCH_BPARAMS_VDIFF1}
  GTBENCH_BPARAMS_VDIFF2=${GTBENCH_BPARAMS_VDIFF2}
)
target_link_libraries(diffusion common)
