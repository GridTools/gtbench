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

namespace runtime {

namespace ghex_comm {

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

sub_grid grid::operator[](unsigned int i) {
  const auto &dom = m_domains[i];
  if (!m_tokens[i])
    m_tokens[i] = std::make_unique<thread_token>(m_context->get_token());
  auto comm = m_context->get_communicator(*m_tokens[i]);
  comm.barrier();
  return {
      m_rank,
      m_size,
      dom.domain_id(),
      m_context.get(),
      m_patterns.get(),
      *m_tokens[i],
      comm_obj_ptr_t{new comm_obj_type{
          ::gridtools::ghex::make_communication_object<patterns_type>(comm)}},
      comm,
      m_global_resolution,
      {(std::size_t)dom.first()[0], (std::size_t)dom.first()[1]},
      {(std::size_t)(dom.last()[0] - dom.first()[0] + 1),
       (std::size_t)(dom.last()[1] - dom.first()[1] + 1),
       (std::size_t)(dom.last()[2] - dom.first()[2] + 1)}};
}

void runtime_register_options(world const &, options &options) {
  options("sub-domains", "number of sub-domains", "S", {1});
}

runtime runtime_init(world const &world, options_values const &options) {
  return {world, options.get<int>("sub-domains")};
}

} // namespace ghex_comm

} // namespace runtime
