#pragma once

#include <gridtools/stencil_composition/stencil_composition.hpp>

#include "../common/types.hpp"

namespace numerics {

using axis_t = gt::axis<1, gt::axis_config::offset_limit<3>>;
using full_t = axis_t::full_interval;
using grid_t = gt::grid<axis_t::axis_interval_t>;

using global_parameter_t = gt::global_parameter<real_t>;
using global_parameter_int_t = gt::global_parameter<gt::int_t>;

inline grid_t computation_grid(gt::uint_t resolution_x, gt::uint_t resolution_y,
                               gt::uint_t resolution_z) {
  return gt::make_grid(
      {halo, halo, halo, halo + resolution_x - 1, resolution_x + 2 * halo},
      {halo, halo, halo, halo + resolution_y - 1, resolution_y + 2 * halo},
      axis_t{resolution_z});
}

}
