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

#include <gridtools/stencil/cartesian.hpp>
#include <gridtools/stencil/frontend/axis.hpp>
#include <gridtools/stencil/global_parameter.hpp>

#include "../common/types.hpp"

namespace gtbench {
namespace numerics {

using axis_t = gt::stencil::axis<1, gt::stencil::axis_config::offset_limit<3>>;
using full_t = axis_t::full_interval;
using grid_t = decltype(gt::stencil::make_grid(
    std::declval<gt::halo_descriptor>(), std::declval<gt::halo_descriptor>(),
    std::declval<axis_t>()));

using global_parameter_t = gt::stencil::global_parameter<real_t>;
using global_parameter_int_t = gt::stencil::global_parameter<gt::int_t>;

inline grid_t computation_grid(gt::uint_t resolution_x, gt::uint_t resolution_y,
                               gt::uint_t resolution_z) {
  return gt::stencil::make_grid(
      {halo, halo, halo, halo + resolution_x - 1, resolution_x + 2 * halo},
      {halo, halo, halo, halo + resolution_y - 1, resolution_y + 2 * halo},
      axis_t{resolution_z});
}

// when we access data across vertical periodic boundaries, we need an infinite
// extent. Be aware that infinite extents must not be used for cached or
// temporary data.
constexpr int infinite_extent = 99999;

} // namespace numerics
} // namespace gtbench
