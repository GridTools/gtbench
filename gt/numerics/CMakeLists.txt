compile_as_cuda(advection.cpp diffusion.cpp)

add_library(advection advection.cpp)
target_link_libraries(advection common GridTools::gridtools)

add_library(diffusion diffusion.cpp)
target_link_libraries(diffusion common GridTools::gridtools)
