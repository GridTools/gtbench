#pragma once

#include <mpi.h>

#include "communication.hpp"

namespace communication {

namespace simple_mpi {

struct tag {};

struct world {
  world(int &argc, char **&argv);
  world(world const &) = delete;
  world(world &&) = default;
  world &operator=(world const &) = delete;
  world &operator=(world &&) = default;
  ~world();
};

inline world comm_world(tag, int &argc, char **&argv) {
  return world(argc, argv);
}

struct grid {
  grid(std::size_t global_resolution_x, std::size_t global_resolution_y,
       std::size_t resolution_z);
  grid(grid const &) = delete;
  grid(grid &&) = default;
  grid &operator=(grid const &) = delete;
  grid &operator=(grid &&) = default;
  ~grid();

  std::size_t resolution_x, resolution_y, resolution_z;
  std::size_t global_resolution_x, global_resolution_y;
  std::size_t offset_x, offset_y;
  MPI_Comm comm_cart;
};

inline grid comm_grid(world &, std::size_t global_resolution_x,
                      std::size_t global_resolution_y,
                      std::size_t resolution_z) {
  return {global_resolution_x, global_resolution_y, resolution_z};
}

std::function<void(storage_t &)>
comm_halo_exchanger(grid const &grid, storage_t::storage_info_t const &sinfo);

double comm_global_sum(grid const &grid, double t);

} // namespace simple_mpi

} // namespace communication