/*
 * gtbench
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "set_device.hpp"
#include <stdexcept>

namespace runtime {

#ifdef __CUDACC__
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
