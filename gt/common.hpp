#pragma once

#include <gridtools/common/gt_math.hpp>
#include <gridtools/stencil_composition/expressions/expressions.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/stencil_composition/stencil_functions.hpp>

#include <gridtools/boundary_conditions/boundary.hpp>

static constexpr int halo_i = 3;
static constexpr int halo_j = 3;
static constexpr int halo_k = 1;

using real_t = double;

namespace gt = gridtools;

using axis_t = gt::axis<1, gt::axis_config::offset_limit<3>>;
using full_t = axis_t::full_interval::modify<halo_k, -halo_k>;
using grid_t = gt::grid<axis_t::axis_interval_t>;

using backend_t = gt::backend::x86;
using storage_tr = gt::storage_traits<backend_t>;
using storage_info_ijk_t =
    storage_tr::storage_info_t<0, 3, gt::halo<halo_i, halo_j, halo_k>>;
using storage_info_ij_t =
    storage_tr::special_storage_info_t<3, gt::selector<1, 1, 0>,
                                       gt::halo<halo_i, halo_j, 0>>;
using storage_t = storage_tr::data_store_t<real_t, storage_info_ijk_t>;
using storage_ij_t = storage_tr::data_store_t<real_t, storage_info_ij_t>;
using global_parameter_t = gt::global_parameter<backend_t, real_t>;

template <class T> struct vec { T x, y, z; };