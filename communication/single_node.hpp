/*
 * gtbench
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "./communication.hpp"

namespace communication {

namespace single_node {

struct world {
  world(int, char **, bool) {}
};

struct grid {
  vec<std::size_t, 3> resolution;
};

inline grid comm_grid(world, vec<std::size_t, 3> const &resolution, int) {
  return {resolution};
}

inline grid &comm_sub_grid(grid &g, int) { return g; }

inline vec<std::size_t, 3> comm_global_resolution(grid const &grid) {
  return grid.resolution;
}

constexpr vec<std::size_t, 2> comm_offset(grid) { return {0, 0}; }

std::function<void(storage_t &)>
comm_halo_exchanger(grid const &grid, storage_t::storage_info_t const &sinfo);

template <class T> constexpr T comm_global_max(grid, T const &t) { return t; }

void comm_barrier(grid &);

} // namespace single_node

} // namespace communication
