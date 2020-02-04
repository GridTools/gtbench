#pragma once

#include "./communication.hpp"

namespace communication {

namespace single_node {

struct world {
  world(int, char **, bool) {}
};

struct grid {
  vec<std::size_t, 3> resolution;
  grid& sub_grid(int id = 0) { return *this; }
};

inline grid comm_grid(world, vec<std::size_t, 3> const &resolution, int) {
  return {resolution};
}

inline vec<std::size_t, 3> comm_global_resolution(grid const &grid) {
  return grid.resolution;
}

constexpr vec<std::size_t, 2> comm_offset(grid) { return {0, 0}; }

std::function<void(storage_t &)>
comm_halo_exchanger(grid const &grid, storage_t::storage_info_t const &sinfo);

template <class T> constexpr T comm_global_max(grid, T const &t) { return t; }

} // namespace single_node

} // namespace communication
