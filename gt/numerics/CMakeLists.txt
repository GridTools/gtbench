get_property(languages GLOBAL PROPERTY ENABLED_LANGUAGES)
if("CUDA" IN_LIST languages)
    set_source_files_properties(advection.cpp diffusion.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_library(diffusion diffusion.cpp)
target_link_libraries(diffusion GridTools::gridtools)

add_library(advection advection.cpp)
target_link_libraries(advection GridTools::gridtools)