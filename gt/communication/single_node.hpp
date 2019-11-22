#pragma once

#include "communication.hpp"

namespace communication {

namespace single_node {

struct tag {};

struct grid {
  std::size_t resolution_x, resolution_y, resolution_z;
};

inline grid comm_grid(tag, std::size_t resolution_x, std::size_t resolution_y,
                      std::size_t resolution_z) {
  return {resolution_x, resolution_y, resolution_z};
}

inline std::size_t comm_global_resolution_x(grid const &grid) {
  return grid.resolution_x;
}

inline std::size_t comm_global_resolution_y(grid const &grid) {
  return grid.resolution_y;
}

constexpr std::size_t comm_offset_x(grid) { return 0; }

constexpr std::size_t comm_offset_y(grid) { return 0; }

std::function<void(storage_t &)>
comm_halo_exchanger(grid const &grid, storage_t::storage_info_t const &sinfo);

template <class T> constexpr T comm_global_sum(grid, T const &t) { return t; }
} // namespace single_node

} // namespace communication
