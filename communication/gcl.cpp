/*
 * gtbench
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "./gcl.hpp"

#include <array>
#include <iostream>

namespace communication {

namespace gcl {

world::world(int &argc, char **&argv, bool) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
  gt::GCL_Init(argc,argv);
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (size > 1 && rank != 0)
    std::cout.setstate(std::ios_base::failbit);
  
  // setup halos
}

world::world(world &&other) : active(std::exchange(other.active, false)) {}

world &world::operator=(world &&other) {
  active = std::exchange(other.active, false);
  return *this;
}

world::~world() {
  if (active)
    gt::GCL_Finalize();
}

grid::grid(vec<std::size_t, 3> const &global_resolution)
    : global_resolution{global_resolution.x, global_resolution.y}
{
  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::array<int, 3> dims = {0, 0, 1}, periods = {1, 1, 0};
  MPI_Dims_create(size, 3, dims.data());
  MPI_Cart_create(MPI_COMM_WORLD, 3, dims.data(), periods.data(), 0,
                  &comm_cart);
  std::array<int, 3> coords;
  MPI_Cart_coords(comm_cart, rank, 3, coords.data());

  offset.x = global_resolution.x * coords[0] / dims[0];
  offset.y = global_resolution.y * coords[1] / dims[1];

  vec<std::size_t, 2> next_offset = {
      global_resolution.x * (coords[0] + 1) / dims[0],
      global_resolution.y * (coords[1] + 1) / dims[1]};

  resolution.x = next_offset.x - offset.x;
  resolution.y = next_offset.y - offset.y;
  resolution.z = global_resolution.z;

  if (resolution.x < halo || resolution.y < halo)
    throw std::runtime_error("local resolution is smaller than halo size!");
}

grid::grid(grid &&other)
    : resolution(std::move(other.resolution)),
      global_resolution(std::move(other.global_resolution)),
      offset(std::move(other.offset)),
      comm_cart(std::exchange(other.comm_cart, MPI_COMM_NULL)) {}

grid &grid::operator=(grid &&other) {
  resolution = std::move(other.resolution);
  global_resolution = std::move(other.global_resolution);
  offset = std::move(other.offset);
  comm_cart = std::exchange(other.comm_cart, MPI_COMM_NULL);
  return *this;
}

grid::~grid() {
  if (comm_cart != MPI_COMM_NULL)
    MPI_Comm_free(&comm_cart);
}

std::function<void(storage_t &)>
comm_halo_exchanger(grid &comm_grid, storage_t::storage_info_t const &sinfo) {
  if (!comm_grid.pattern) {
    comm_grid.pattern = std::make_unique<pattern_type>(
        pattern_type::grid_type::period_type{true, true, false}, comm_grid.comm_cart);

    comm_grid.pattern->template add_halo<0>(
        halo, halo, halo, comm_grid.resolution.x + halo - 1, sinfo.padded_length<0>());
    comm_grid.pattern->template add_halo<1>(
        halo, halo, halo, comm_grid.resolution.y + halo - 1, sinfo.padded_length<1>());
    comm_grid.pattern->template add_halo<2>(0, 0, 0, comm_grid.resolution.z -1,
                                            sinfo.padded_length<2>());
    comm_grid.pattern->setup(1);
  }
  auto pattern_ptr = comm_grid.pattern.get();

  return [pattern_ptr](const storage_t &storage) mutable {
    auto ptr = storage.get_storage_ptr()->get_target_ptr();
    pattern_ptr->pack(ptr);
    pattern_ptr->exchange();
    pattern_ptr->unpack(ptr);
  };
}

double comm_global_max(grid const &grid, double t) {
  double max;
  MPI_Allreduce(&t, &max, 1, MPI_DOUBLE, MPI_MAX, grid.comm_cart);
  return max;
}

void comm_barrier(grid &) {
  MPI_Barrier(MPI_COMM_WORLD);
}
        

} // namespace gcl

} // namespace communication
