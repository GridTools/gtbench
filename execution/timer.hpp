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

#include <chrono>

namespace execution {
namespace timer {

template <class Backend>
inline std::chrono::high_resolution_clock::time_point timer_now(Backend) {
  return std::chrono::high_resolution_clock::now();
}

#ifdef __CUDACC__
inline std::chrono::high_resolution_clock::time_point
timer_now(gt::backend::cuda) {
  if (cudaDeviceSynchronize() != cudaSuccess)
    throw std::runtime_error("cudaDeviceSynchronize() failed");
  return std::chrono::high_resolution_clock::now();
}
#endif

inline double
timer_duration(std::chrono::high_resolution_clock::time_point const &start,
               std::chrono::high_resolution_clock::time_point const &stop) {
  return std::chrono::duration<double>(stop - start).count();
}

template <class Backend> inline auto now(Backend &&backend) {
  return timer_now(std::forward<Backend>(backend));
}

template <class TimePoint>
inline double duration(TimePoint &&start, TimePoint &&stop) {
  return timer_duration(std::forward<TimePoint>(start),
                        std::forward<TimePoint>(stop));
}

} // namespace timer
} // namespace execution
