#pragma once

#include <utility>

#include "../common/types.hpp"

namespace communication {

template <class CommGrid>
vec<std::size_t, 3> comm_global_resolution(CommGrid const &grid) {
  return {grid.global_resolution.x, grid.global_resolution.y,
          grid.resolution.z};
}

template <class CommGrid>
vec<std::size_t, 3> comm_resolution(CommGrid const &grid) {
  return grid.resolution;
}

template <class CommGrid>
vec<std::size_t, 2> comm_offset(CommGrid const &grid) {
  return grid.offset;
}

template <class CommWorld>
auto grid(CommWorld &&world, vec<std::size_t, 3> const &global_resolution) {
  return comm_grid(std::forward<CommWorld>(world), global_resolution);
}

template <class CommGrid>
vec<std::size_t, 3> global_resolution(CommGrid &&grid) {
  return comm_global_resolution(std::forward<CommGrid>(grid));
}

template <class CommGrid> vec<std::size_t, 3> resolution(CommGrid &&grid) {
  return comm_resolution(std::forward<CommGrid>(grid));
}

template <class CommGrid> vec<std::size_t, 2> offset(CommGrid &&grid) {
  return comm_offset(std::forward<CommGrid>(grid));
}

template <class CommGrid>
auto halo_exchanger(CommGrid &&grid, storage_t::storage_info_t const &sinfo) {
  return comm_halo_exchanger(std::forward<CommGrid>(grid), sinfo);
}

template <class CommGrid, class T> T global_max(CommGrid &&grid, T const &t) {
  return comm_global_max(std::forward<CommGrid>(grid), t);
}

} // namespace communication
