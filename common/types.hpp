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

#include <gridtools/stencil/cpu_ifirst.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>
#include <gridtools/storage/cpu_ifirst.hpp>

namespace gt = gridtools;

using real_t = GTBENCH_FLOAT;

constexpr real_t operator"" _r(long double value) { return real_t(value); }
constexpr real_t operator"" _r(unsigned long long value) {
  return real_t(value);
}

static constexpr gt::int_t halo = 3;

using backend_t = gt::stencil::GTBENCH_BACKEND<>;
using storage_tr = gt::storage::GTBENCH_BACKEND;

template <class T, std::size_t N> struct vec;
template <class T> struct vec<T, 3> { T x, y, z; };
template <class T> struct vec<T, 2> { T x, y; };

inline auto storage_builder(vec<std::size_t, 3> const &resolution) {
  return gt::storage::builder<storage_tr>
  .type<real_t>()
  .id<0>()
  .halos(halo, halo, 0)
  .dimensions(resolution.x + 2 * halo, resolution.y + 2 * halo, resolution.z + 1);
}

using storage_t =
    decltype(storage_builder(std::declval<vec<std::size_t, 3>>())());
