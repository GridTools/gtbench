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

#include <mpi.h>
#include <gridtools/communication/GCL.hpp>
#include <gridtools/communication/halo_exchange.hpp>

#include "./communication.hpp"

namespace communication {

namespace gcl {
  
  using pattern_type = gt::halo_exchange_dynamic_ut<
    typename storage_info_ijk_t::layout_t,
    gt::layout_map<0, 1, 2>,
    real_t,
#ifdef __CUDACC__
    gridtools::gcl_gpu
#else
    gridtools::gcl_cpu
#endif
  >;

struct world {
  world(int &argc, char **&argv, bool);
  world(world const &) = delete;
  world(world &&);
  world &operator=(world const &) = delete;
  world &operator=(world &&);
  ~world();

  bool active = true;
};

struct grid {
  grid(vec<std::size_t, 3> const &global_resolution);
  grid(grid const &) = delete;
  grid(grid &&);
  grid &operator=(grid const &) = delete;
  grid &operator=(grid &&);
  ~grid();

  vec<std::size_t, 3> resolution;
  vec<std::size_t, 2> global_resolution, offset;
  MPI_Comm comm_cart = MPI_COMM_NULL;
  std::unique_ptr<pattern_type> pattern;
};

inline grid comm_grid(world &, vec<std::size_t, 3> const &global_resolution,
                      int) {
  return {global_resolution};
}

inline grid &comm_sub_grid(grid &g, int) { return g; }

std::function<void(storage_t &)>
comm_halo_exchanger(grid &grid, storage_t::storage_info_t const &sinfo);

double comm_global_max(grid const &grid, double t);

void comm_barrier(grid &);

} // namespace gcl

} // namespace communication
