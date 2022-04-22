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

#include "../common/types.hpp"

namespace gtbench {
namespace io {
std::size_t rank(vec<std::size_t, 3> const &global_resolution,
                 vec<std::size_t, 3> const &local_resolution,
                 vec<std::size_t, 3> const &local_offset);

std::size_t ranks(vec<std::size_t, 3> const &global_resolution,
                  vec<std::size_t, 3> const &local_resolution);
} // namespace io
} // namespace gtbench
