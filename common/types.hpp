/*
 * gtbench
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <gridtools/storage/storage_facility.hpp>

namespace gt = gridtools;

using real_t = GTBENCH_FLOAT;

constexpr real_t operator"" _r(long double value) { return real_t(value); }
constexpr real_t operator"" _r(unsigned long long value) {
  return real_t(value);
}

static constexpr gt::int_t halo = 3;

using backend_t = gt::backend::GTBENCH_BACKEND;
using storage_tr = gt::storage_traits<backend_t>;
using storage_info_ijk_t =
    storage_tr::storage_info_t<0, 3, gt::halo<halo, halo, 0>>;
using storage_info_ij_t =
    storage_tr::special_storage_info_t<3, gt::selector<1, 1, 0>,
                                       gt::halo<halo, halo, 0>>;
using storage_t = storage_tr::data_store_t<real_t, storage_info_ijk_t>;
using storage_ij_t = storage_tr::data_store_t<real_t, storage_info_ij_t>;

template <class T, std::size_t N> struct vec;
template <class T> struct vec<T, 3> { T x, y, z; };
template <class T> struct vec<T, 2> { T x, y; };
