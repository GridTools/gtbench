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

#include "./computation.hpp"

namespace gtbench {
namespace numerics {
namespace diffusion {

std::function<void(storage_t, storage_t, real_t dt)>
horizontal(vec<std::size_t, 3> const &resolution, vec<real_t, 3> const &delta,
           real_t coeff);

std::function<void(storage_t, storage_t, real_t dt)>
vertical(vec<std::size_t, 3> const &resolution, vec<real_t, 3> const &delta,
         real_t coeff);

} // namespace diffusion
} // namespace numerics
} // namespace gtbench
