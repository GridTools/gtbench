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

#include <cmath>

#include "../common/types.hpp"

namespace gtbench {
namespace verification {
namespace analytical {

template <class Analytical>
inline auto analytical_velocity(Analytical const &) {
  return [](vec<real_t, 3> const &, real_t) { return vec<real_t, 3>{0, 0, 0}; };
}

template <class Analytical>
inline vec<real_t, 3> analytical_domain(Analytical const &) {
  static constexpr real_t four_pies = real_t(4 * M_PI);
  return {four_pies, four_pies, four_pies};
}

template <class Analytical> inline auto data(Analytical const &analytical) {
  return analytical_data(analytical);
}

template <class Analytical> inline auto velocity(Analytical const &analytical) {
  return analytical_velocity(analytical);
}

template <class Analytical> inline auto u(Analytical const &analytical) {
  return [f = velocity(analytical)](vec<real_t, 3> const &p, real_t t) {
    return f(p, t).x;
  };
}

template <class Analytical> inline auto v(Analytical const &analytical) {
  return [f = velocity(analytical)](vec<real_t, 3> const &p, real_t t) {
    return f(p, t).y;
  };
}

template <class Analytical> inline auto w(Analytical const &analytical) {
  return [f = velocity(analytical)](vec<real_t, 3> const &p, real_t t) {
    return f(p, t).z;
  };
}

template <class Analytical>
inline vec<real_t, 3> domain(Analytical const &analytical) {
  return analytical_domain(analytical);
}

struct horizontal_diffusion {
  real_t diffusion_coeff;
};

inline auto analytical_data(horizontal_diffusion const &hdiff) {
  return
      [d = hdiff.diffusion_coeff](vec<real_t, 3> const &p, real_t t) -> real_t {
        return std::sin(p.x) * std::cos(p.y) * std::exp(-2 * d * t);
      };
}

struct vertical_diffusion {
  real_t diffusion_coeff;
};

inline auto analytical_data(vertical_diffusion const &vdiff) {
  return
      [d = vdiff.diffusion_coeff](vec<real_t, 3> const &p, real_t t) -> real_t {
        return std::cos(p.z) * std::exp(-d * t);
      };
}

struct full_diffusion {
  real_t diffusion_coeff;
};

inline auto analytical_data(full_diffusion const &vdiff) {
  return [d = vdiff.diffusion_coeff](vec<real_t, 3> const &p,
                                     real_t t) -> real_t {
    return std::sin(p.x) * std::cos(p.y) * std::cos(p.z) * std::exp(-3 * d * t);
  };
}

struct horizontal_advection {};

inline auto analytical_data(horizontal_advection const &) {
  return [](vec<real_t, 3> const &p, real_t t) -> real_t {
    return std::sin(p.x - 0.2_r * t) * std::cos(p.y + 0.3_r * t) *
           std::cos(p.z);
  };
}

inline auto analytical_velocity(horizontal_advection const &) {
  return [](vec<real_t, 3> const &p, real_t t) -> vec<real_t, 3> {
    return {0.2_r, -0.3_r, 0};
  };
}

struct vertical_advection {};

inline auto analytical_data(vertical_advection const &) {
  return [](vec<real_t, 3> const &p, real_t t) -> real_t {
    return std::sin(p.x) * std::cos(p.y) * std::cos(p.z - 0.6_r * t);
  };
}

inline auto analytical_velocity(vertical_advection const &) {
  return [](vec<real_t, 3> const &p, real_t t) -> vec<real_t, 3> {
    return {0, 0, 0.6_r};
  };
}

struct full_advection {};

inline auto analytical_data(full_advection const &) {
  return [](vec<real_t, 3> const &p, real_t t) -> real_t {
    return std::sin(p.x - 0.1_r * t) * std::cos(p.y + 0.2_r * t) *
           std::cos(p.z - 0.3_r * t);
  };
}

inline auto analytical_velocity(full_advection const &) {
  return [](vec<real_t, 3> const &p, real_t t) -> vec<real_t, 3> {
    return {0.1_r, -0.2_r, 0.3_r};
  };
}

struct advection_diffusion {
  real_t diffusion_coeff;
};

inline auto analytical_data(advection_diffusion const &advdiff) {
  return [d = advdiff.diffusion_coeff](vec<real_t, 3> const &p,
                                       real_t t) -> real_t {
    const real_t a = std::sqrt(2_r) / 2;
    return -std::sin(p.x) * std::sin(a * (p.y - p.z)) * std::exp(-2 * d * t);
  };
}

inline auto analytical_velocity(advection_diffusion const &) {
  return [](vec<real_t, 3> const &p, real_t t) -> vec<real_t, 3> {
    const real_t a = std::sqrt(2_r) / 2;
    return {-std::sin(p.x) * std::cos(a * (p.y - p.z)) * 0.1_r,
            a * std::cos(p.x) * std::sin(a * (p.y - p.z)) * 0.1_r,
            -a * std::cos(p.x) * std::sin(a * (p.y - p.z)) * 0.1_r};
  };
}

inline vec<real_t, 3> analytical_domain(advection_diffusion const &) {
  static constexpr real_t two_pi = real_t(2 * M_PI);
  return {two_pi, two_pi * std::sqrt(2_r), two_pi * std::sqrt(2_r)};
}

template <class Analytical> struct repeated {
  Analytical wrapped;
  vec<std::size_t, 3> repeats;
};

template <class Analytical>
inline auto analytical_data(repeated<Analytical> const &repeated) {
  return [f = data(repeated.wrapped), d = domain(repeated.wrapped)](
             vec<real_t, 3> const &p, real_t t) -> real_t {
    return f({std::fmod(p.x, d.x), std::fmod(p.y, d.y), std::fmod(p.z, d.z)},
             t);
  };
}

template <class Analytical>
inline auto analytical_velocity(repeated<Analytical> const &repeated) {
  return [f = velocity(repeated.wrapped), d = domain(repeated.wrapped)](
             vec<real_t, 3> const &p, real_t t) -> vec<real_t, 3> {
    return f({std::fmod(p.x, d.x), std::fmod(p.y, d.y), std::fmod(p.z, d.z)},
             t);
  };
}

template <class Analytical>
inline vec<real_t, 3> analytical_domain(repeated<Analytical> const &repeated) {
  auto d = domain(repeated.wrapped);
  return {d.x * repeated.repeats.x, d.y * repeated.repeats.y,
          d.z * repeated.repeats.z};
}

template <class Analytical>
inline repeated<std::decay_t<Analytical>>
repeat(Analytical &&analytical, vec<std::size_t, 3> const &repeats) {
  return {std::forward<Analytical>(analytical), repeats};
}

} // namespace analytical
} // namespace verification
} // namespace gtbench
