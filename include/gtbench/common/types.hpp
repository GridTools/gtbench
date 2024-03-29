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

#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/sid.hpp>

#ifdef GTBENCH_BACKEND_CPU_IFIRST
#include <gridtools/stencil/cpu_ifirst.hpp>
#include <gridtools/storage/cpu_ifirst.hpp>
#endif

#ifdef GTBENCH_BACKEND_CPU_KFIRST
#include <gridtools/stencil/cpu_kfirst.hpp>
#include <gridtools/storage/cpu_kfirst.hpp>
#endif

#ifdef GTBENCH_BACKEND_GPU
#include <gridtools/stencil/gpu.hpp>
#include <gridtools/storage/gpu.hpp>
#endif

namespace gtbench {

namespace gt = gridtools;

using real_t = GTBENCH_FLOAT;

constexpr real_t operator"" _r(long double value) { return real_t(value); }
constexpr real_t operator"" _r(unsigned long long value) {
  return real_t(value);
}

static constexpr gt::int_t halo = 3;

template <class... Params>
using backend_t = gt::stencil::GTBENCH_BACKEND<Params...>;
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

} // namespace gtbench
