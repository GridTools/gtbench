#pragma once

#include <utility>

#include "../common.hpp"

namespace communication {

template <class CommTag> CommTag comm_world(CommTag, int &, char **&) {
  return {};
}

template <class CommGrid>
std::size_t comm_global_resolution_x(CommGrid const &grid) {
  return grid.global_resolution_x;
}

template <class CommGrid>
std::size_t comm_global_resolution_y(CommGrid const &grid) {
  return grid.global_resolution_y;
}

template <class CommGrid> std::size_t comm_resolution_x(CommGrid const &grid) {
  return grid.resolution_x;
}

template <class CommGrid> std::size_t comm_resolution_y(CommGrid const &grid) {
  return grid.resolution_y;
}

template <class CommGrid> std::size_t comm_resolution_z(CommGrid const &grid) {
  return grid.resolution_z;
}

template <class CommGrid> std::size_t comm_offset_x(CommGrid const &grid) {
  return grid.offset_x;
}

template <class CommGrid> std::size_t comm_offset_y(CommGrid const &grid) {
  return grid.offset_y;
}

template <class CommTag> auto world(CommTag &&tag, int &argc, char **&argv) {
  return comm_world(std::forward<CommTag>(tag), argc, argv);
}

template <class CommWorld>
auto grid(CommWorld &&world, std::size_t global_resolution_x,
          std::size_t global_resolution_y, std::size_t resolution_z) {
  return comm_grid(std::forward<CommWorld>(world), global_resolution_x,
                   global_resolution_y, resolution_z);
}

template <class CommGrid> std::size_t global_resolution_x(CommGrid &&grid) {
  using namespace communication;
  return comm_global_resolution_x(std::forward<CommGrid>(grid));
}

template <class CommGrid> std::size_t global_resolution_y(CommGrid &&grid) {
  using namespace communication;
  return comm_global_resolution_y(std::forward<CommGrid>(grid));
}

template <class CommGrid> std::size_t resolution_x(CommGrid &&grid) {
  using namespace communication;
  return comm_resolution_x(std::forward<CommGrid>(grid));
}

template <class CommGrid> std::size_t resolution_y(CommGrid &&grid) {
  using namespace communication;
  return comm_resolution_y(std::forward<CommGrid>(grid));
}

template <class CommGrid> std::size_t resolution_z(CommGrid &&grid) {
  using namespace communication;
  return comm_resolution_z(std::forward<CommGrid>(grid));
}

template <class CommGrid> std::size_t offset_x(CommGrid &&grid) {
  using namespace communication;
  return comm_offset_x(std::forward<CommGrid>(grid));
}

template <class CommGrid> std::size_t offset_y(CommGrid &&grid) {
  using namespace communication;
  return comm_offset_y(std::forward<CommGrid>(grid));
}

template <class CommGrid>
auto halo_exchanger(CommGrid &&grid, storage_t::storage_info_t const &sinfo) {
  using namespace communication;
  return comm_halo_exchanger(std::forward<CommGrid>(grid), sinfo);
}

template <class CommGrid, class T> T global_sum(CommGrid &&grid, T const &t) {
  return comm_global_sum(std::forward<CommGrid>(grid), t);
}

} // namespace communication
