/*
 * gtbench
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "./util.hpp"

namespace io {
std::size_t rank(vec<std::size_t, 3> const &global_resolution,
                 vec<std::size_t, 3> const &local_resolution,
                 vec<std::size_t, 3> const &local_offset) {
  auto i = local_offset.x / local_resolution.x;
  auto j = local_offset.y / local_resolution.y;
  auto k = local_offset.z / local_resolution.z;
  auto imax = global_resolution.x / local_resolution.x;
  auto jmax = global_resolution.y / local_resolution.y;
  return i + imax * (j + jmax * k);
}

std::size_t ranks(vec<std::size_t, 3> const &global_resolution,
                  vec<std::size_t, 3> const &local_resolution) {
  auto imax = global_resolution.x / local_resolution.x;
  auto jmax = global_resolution.y / local_resolution.y;
  auto kmax = global_resolution.z / local_resolution.z;
  return imax * jmax * kmax;
}
} // namespace io
