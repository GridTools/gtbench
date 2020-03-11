/*
 * gtbench
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "device.hpp"
#include <mpi.h>

namespace communication {

#ifdef __CUDACC__

int set_device() {

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
  return device_id;
}

#else

int set_device() { return 0; }

#endif

} // namespace communication
