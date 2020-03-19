/*
 * gtbench
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "./run.hpp"

#include <ghex/glue/gridtools/field.hpp>

namespace runtime {

namespace ghex_comm {

std::function<void(storage_t &)>
comm_halo_exchanger(grid::sub_grid &g, storage_t::storage_info_t const &sinfo) {
  auto co_ptr = g.m_comm_obj.get();
  auto patterns_ptr = g.m_patterns;
  const auto domain_id = g.m_domain_id;
  auto context_ptr = g.m_context;
  auto token = g.m_token;
  return [co_ptr, patterns_ptr, domain_id, context_ptr,
          token](const storage_t &storage) mutable {
    auto &co = *co_ptr;
    auto &patterns = *patterns_ptr;
    auto field = ::gridtools::ghex::wrap_gt_field(domain_id, storage);

#ifdef __CUDACC__
    cudaStreamSynchronize(0);
#endif

    co.exchange(patterns(field)).wait();
  };
}

double comm_global_max(grid::sub_grid const &g, double t) {
  double max_v = t;
  if (g.m_rank == 0) {
    for (int i = 1; i < g.m_size; ++i) {
      double max_i;
      MPI_Recv(&max_i, 1, MPI_DOUBLE, i, g.m_token.id(), MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);
      max_v = std::max(max_v, max_i);
    }
    for (int i = 1; i < g.m_size; ++i) {
      MPI_Send(&max_v, 1, MPI_DOUBLE, i, g.m_token.id(), MPI_COMM_WORLD);
    }
  } else {
    MPI_Send(&t, 1, MPI_DOUBLE, 0, g.m_token.id(), MPI_COMM_WORLD);
    MPI_Recv(&max_v, 1, MPI_DOUBLE, 0, g.m_token.id(), MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  }
  return max_v;
}

void comm_barrier(grid::sub_grid &g) { g.m_comm.barrier(); }

world::world(int &argc, char **&argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

  int size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef __CUDACC__
  int device_count = 1;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess)
    throw std::runtime_error("cudaGetDeviceCount failed");
  MPI_Comm shmem_comm;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                      &shmem_comm);
  int node_rank = 0;
  MPI_Comm_rank(shmem_comm, &node_rank);
  MPI_Comm_free(&shmem_comm);
  const int device_id = node_rank % device_count;
  if (cudaSetDevice(device_id) != cudaSuccess)
    throw std::runtime_error("cudaSetDevice failed");
  if (device_count > 1) {
    for (int i = 0; i < device_count; ++i) {
      if (i != device_id) {
        int flag;
        if (cudaDeviceCanAccessPeer(&flag, device_id, i) != cudaSuccess)
          throw std::runtime_error("cudaDeviceAccessPeer failed");
        if (flag) {
          cudaDeviceEnablePeerAccess(i, 0);
        }
      }
    }
  }
#endif

  if (size > 1 && rank != 0)
    std::cout.setstate(std::ios_base::failbit);
}

world::~world() { MPI_Finalize(); }

void runtime_register_options(world const &, options &options) {
  options("sub-domains", "number of sub-domains", "S", {1});
}

runtime runtime_init(world const &world, options_values const &options) {
  return {world, options.get<int>("sub-domains")};
}

} // namespace ghex_comm

} // namespace runtime
