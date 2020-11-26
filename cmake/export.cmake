set(GTBENCH_MODULE_PATH lib/cmake/GTBench)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
configure_package_config_file(
  cmake/GTBenchConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/install/GTBenchConfig.cmake"
  PATH_VARS GTBENCH_MODULE_PATH
  INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/GTBench"
)
write_basic_package_version_file(
  "${CMAKE_CURRENT_BINARY_DIR}/install/GTBenchConfigVersion.cmake"
  COMPATIBILITY SameMajorVersion
)

install(
  TARGETS gtbench
  EXPORT GTBenchTargets
  INCLUDES DESTINATION include
)
install(TARGETS benchmark convergence_tests)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Workaround for GT export strategy
install(
  TARGETS stencil_${GTBENCH_BACKEND} storage_${GTBENCH_BACKEND}
  EXPORT GTBenchTargets
)

include(GNUInstallDirs)
install(
  FILES
    "${CMAKE_CURRENT_BINARY_DIR}/install/GTBenchConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/install/GTBenchConfigVersion.cmake"
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
)

install(
  EXPORT GTBenchTargets
  FILE GTBenchTargets.cmake
  NAMESPACE GTBench::
  DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/${PROJECT_NAME}"
)
