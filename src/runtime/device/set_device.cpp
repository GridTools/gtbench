/*
 * gtbench
 *
 * Copyright (c) 2014-2022, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <stdexcept>

#include <gridtools/common/defs.hpp>
#include <gtbench/runtime/device/set_device.hpp>

#ifdef GT_CUDACC
#include <gridtools/common/cuda_runtime.hpp>
#if defined(__HIP__)
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#endif
#endif

namespace gtbench {
namespace runtime {

#ifdef GT_CUDACC
int set_device(int device_id) {
  int device_count = 1;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess)
    throw std::runtime_error("cudaGetDeviceCount failed");
  device_id %= device_count;
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
int set_device(int) { return 0; }
#endif

} // namespace runtime
} // namespace gtbench
