/*
 * gtbench
 *
 * Copyright (c) 2014-2022, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <functional>
#include <string>

#include "../common/types.hpp"
#include "../numerics/solver.hpp"

namespace gtbench {
namespace io {

std::function<void(real_t, numerics::solver_state const &state)>
write_time_series(std::string const &filename,
                  vec<std::size_t, 3> const &global_resolution,
                  vec<std::size_t, 3> const &local_resolution,
                  vec<std::size_t, 3> const &local_offset);

} // namespace io
} // namespace gtbench
