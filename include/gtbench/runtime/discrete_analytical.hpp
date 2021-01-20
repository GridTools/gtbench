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

#include "../verification/analytical.hpp"

namespace gtbench {
namespace runtime {
namespace discrete_analytical {

namespace impl {

template <class Analytical> struct discrete {
  Analytical analytical;

  vec<std::size_t, 3> global_resolution, local_resolution, local_offset;
};

template <class Analytical>
inline vec<real_t, 3> delta(discrete<Analytical> const &disc) {
  const vec<real_t, 3> domain =
      verification::analytical::domain(disc.analytical);
  return {domain.x / disc.global_resolution.x,
          domain.y / disc.global_resolution.y,
          domain.z / disc.global_resolution.z};
}

template <class Analytical, class F>
inline auto remap(discrete<Analytical> const &d, F f,
                  bool staggered_z = false) {
  const real_t staggered_offset = staggered_z ? -0.5_r : 0_r;
  return
      [f = std::move(f), delta = delta(d), r = d.global_resolution,
       o = d.local_offset, staggered_offset](vec<long, 3> const &p, real_t t) {
        return f({(p.x - halo + o.x) * delta.x, (p.y - halo + o.y) * delta.y,
                  (p.z + o.z + staggered_offset) * delta.z},
                 t);
      };
}

} // namespace impl

template <class Analytical>
impl::discrete<Analytical>
discretize(Analytical analytical, vec<std::size_t, 3> const &global_resolution,
           vec<std::size_t, 3> const &local_resolution,
           vec<std::size_t, 3> const &local_offset) {
  return {std::move(analytical), global_resolution, local_resolution,
          local_offset};
}

template <class Discrete> inline auto data(Discrete const &discrete) {
  return impl::remap(discrete,
                     verification::analytical::data(discrete.analytical));
}

template <class Discrete> inline auto u(Discrete const &discrete) {
  return impl::remap(discrete,
                     verification::analytical::u(discrete.analytical));
}

template <class Discrete> inline auto v(Discrete const &discrete) {
  return impl::remap(discrete,
                     verification::analytical::v(discrete.analytical));
}

template <class Discrete> inline auto w(Discrete const &discrete) {
  return impl::remap(discrete, verification::analytical::w(discrete.analytical),
                     true);
}

template <class Discrete>
inline vec<std::size_t, 3> global_resolution(Discrete const &discrete) {
  return discrete.global_resolution;
}

template <class Discrete>
inline vec<std::size_t, 3> local_resolution(Discrete const &discrete) {
  return discrete.local_resolution;
}

template <class Discrete>
inline vec<std::size_t, 3> local_offset(Discrete const &discrete) {
  return discrete.local_offset;
}

template <class Discrete>
inline vec<real_t, 3> delta(Discrete const &discrete) {
  return impl::delta(discrete);
}

} // namespace discrete_analytical
} // namespace runtime
} // namespace gtbench
