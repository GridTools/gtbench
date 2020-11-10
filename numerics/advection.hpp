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

#include "./computation.hpp"

namespace numerics {
namespace advection {

std::function<void(storage_t, storage_t, storage_t, storage_t, storage_t, real_t)>
horizontal(vec<std::size_t, 3> const &resolution, vec<real_t, 3> const &delta);

std::function<void(storage_t, storage_t, storage_t, storage_t, real_t)>
vertical(vec<std::size_t, 3> const &resolution, vec<real_t, 3> const &delta);

} // namespace advection
} // namespace numerics
