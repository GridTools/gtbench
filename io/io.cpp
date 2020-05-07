/*
 * gtbench
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cassert>

#include "./io.hpp"

#include "./vtk.hpp"

namespace io {

namespace {
inline bool ends_with(std::string const &str, std::string const &suffix) {
  return str.size() >= suffix.size() &&
         str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}
} // namespace

time_series::time_series(std::string const &filename,
                         vec<std::size_t, 3> const &global_resolution,
                         vec<std::size_t, 3> const &local_resolution,
                         vec<std::size_t, 3> const &local_offset)
    : m_filename(filename), m_global_resolution(global_resolution),
      m_local_resolution(local_resolution), m_local_offset(local_offset) {
  assert(global_resolution.x % local_resolution.x == 0);
  assert(global_resolution.y % local_resolution.y == 0);
  assert(global_resolution.z % local_resolution.z == 0);
  assert(local_offset.x % local_resolution.x == 0);
  assert(local_offset.y % local_resolution.y == 0);
  assert(local_offset.z % local_resolution.z == 0);
}

std::size_t time_series::rank(vec<std::size_t, 3> const &local_offset) const {
  std::size_t i = local_offset.x / m_local_resolution.x;
  std::size_t j = local_offset.y / m_local_resolution.y;
  std::size_t k = local_offset.z / m_local_resolution.z;
  std::size_t imax = m_global_resolution.x / m_local_resolution.x;
  std::size_t jmax = m_global_resolution.y / m_local_resolution.y;
  return i + imax * (j + jmax * k);
}

std::size_t time_series::rank() const { return rank(m_local_offset); }

std::size_t time_series::ranks() const {
  std::size_t imax = m_global_resolution.x / m_local_resolution.x;
  std::size_t jmax = m_global_resolution.y / m_local_resolution.y;
  std::size_t kmax = m_global_resolution.z / m_local_resolution.z;
  return imax * jmax * kmax;
}

std::shared_ptr<time_series>
time_series_from_filename(std::string const &filename,
                          vec<std::size_t, 3> const &global_resolution,
                          vec<std::size_t, 3> const &local_resolution,
                          vec<std::size_t, 3> const &local_offset) {
  if (filename.empty())
    return nullptr;

  if (ends_with(filename, ".pvd")) {
    return std::make_shared<vtk::time_series>(filename, global_resolution,
                                              local_resolution, local_offset);
  }

  throw std::runtime_error("can not determine format for \"" + filename + "\"");
}

} // namespace io
