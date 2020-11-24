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

#include <vector>

#include "./io.hpp"

namespace io {
namespace vtk {

std::function<void(real_t, numerics::solver_state const &state)>
write_time_series(std::string const &filename,
                  vec<std::size_t, 3> const &global_resolution,
                  vec<std::size_t, 3> const &local_resolution,
                  vec<std::size_t, 3> const &local_offset);

} // namespace vtk

} // namespace io
