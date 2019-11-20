#pragma once

#include <gridtools/common/gt_math.hpp>
#include <gridtools/stencil_composition/expressions/expressions.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/stencil_composition/stencil_functions.hpp>

#include <gridtools/boundary_conditions/boundary.hpp>

namespace gt = gridtools;

using real_t = double;

static constexpr gt::int_t halo = 3;
static constexpr gt::int_t huge_offset = 10000;

using axis_t = gt::axis<1, gt::axis_config::offset_limit<3>>;
using full_t = axis_t::full_interval;
using grid_t = gt::grid<axis_t::axis_interval_t>;

using backend_t = gt::backend::x86;
using storage_tr = gt::storage_traits<backend_t>;
using storage_info_ijk_t =
    storage_tr::storage_info_t<0, 3, gt::halo<halo, halo, 0>>;
using storage_info_ij_t =
    storage_tr::special_storage_info_t<3, gt::selector<1, 1, 0>,
                                       gt::halo<halo, halo, 0>>;
using storage_t = storage_tr::data_store_t<real_t, storage_info_ijk_t>;
using storage_ij_t = storage_tr::data_store_t<real_t, storage_info_ij_t>;
using global_parameter_t = gt::global_parameter<backend_t, real_t>;
using global_parameter_int_t = gt::global_parameter<backend_t, gt::int_t>;

using halos_t = gt::array<gt::halo_descriptor, 3>;