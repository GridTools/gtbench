add_library(diffusion diffusion.cpp)
target_link_libraries(diffusion GridTools::gridtools)

add_library(advection advection.cpp)
target_link_libraries(advection GridTools::gridtools)