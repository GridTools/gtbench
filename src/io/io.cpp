/*
 * gtbench
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gtbench/io/io.hpp>
#include <gtbench/io/numpy.hpp>
#include <gtbench/io/vtk.hpp>

namespace gtbench {
namespace io {

namespace {
inline bool ends_with(std::string const &str, std::string const &suffix) {
  return str.size() >= suffix.size() &&
         str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}
} // namespace

std::function<void(real_t, numerics::solver_state const &state)>
write_time_series(std::string const &filename,
                  vec<std::size_t, 3> const &global_resolution,
                  vec<std::size_t, 3> const &local_resolution,
                  vec<std::size_t, 3> const &local_offset) {
  if (filename.empty())
    return {};

  if (ends_with(filename, ".npy")) {
    return numpy::write_time_series(filename, global_resolution,
                                    local_resolution, local_offset);
  }
  if (ends_with(filename, ".pvd")) {
    return vtk::write_time_series(filename, global_resolution, local_resolution,
                                  local_offset);
  }

  throw std::runtime_error("file format for \"" + filename +
                           "\" not supported");
}

} // namespace io
} // namespace gtbench
