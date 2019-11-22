#pragma once

#include <cmath>

#include "../common/types.hpp"

namespace analytical {

template <class Analytical>
inline auto analytical_velocity(Analytical const &) {
  return [](vec<real_t, 3> const &, real_t) { return vec<real_t, 3>{0, 0, 0}; };
}

template <class Analytical>
inline vec<real_t, 3> analytical_domain(Analytical const &) {
  return {2 * M_PI, 2 * M_PI, 2 * M_PI};
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
    return std::sin(p.x - t) * std::cos(p.y + 2 * t) * std::cos(p.z);
  };
}

inline auto analytical_velocity(horizontal_advection const &) {
  return [](vec<real_t, 3> const &p, real_t t) -> vec<real_t, 3> {
    return {1, -2, 0};
  };
}

struct vertical_advection {};

inline auto analytical_data(vertical_advection const &) {
  return [](vec<real_t, 3> const &p, real_t t) -> real_t {
    return std::sin(p.x) * std::cos(p.y) * std::cos(p.z - 0.5 * t);
  };
}

inline auto analytical_velocity(vertical_advection const &) {
  return [](vec<real_t, 3> const &p, real_t t) -> vec<real_t, 3> {
    return {0, 0, 0.5};
  };
}

struct full_advection {};

inline auto analytical_data(full_advection const &) {
  return [](vec<real_t, 3> const &p, real_t t) -> real_t {
    return std::sin(p.x - t) * std::cos(p.y + 2 * t) * std::cos(p.z - 0.5 * t);
  };
}

inline auto analytical_velocity(full_advection const &) {
  return [](vec<real_t, 3> const &p, real_t t) -> vec<real_t, 3> {
    return {1, -2, 0.5};
  };
}

struct advection_diffusion {
  real_t diffusion_coeff;
};

static constexpr real_t phi = M_PI / 4;

inline auto analytical_data(advection_diffusion const &advdiff) {
  return [d = advdiff.diffusion_coeff](vec<real_t, 3> const &p,
                                       real_t t) -> real_t {
    return -std::sin(p.x) *
           std::sin(p.y * std::sin(phi) - p.z * std::cos(phi)) *
           std::exp(-2 * d * t);
  };
}

inline auto analytical_velocity(advection_diffusion const &) {
  return [](vec<real_t, 3> const &p, real_t t) -> vec<real_t, 3> {
    return {-std::sin(p.x) *
                std::cos(p.y * std::sin(phi) - p.z * std::cos(phi)),
            std::sin(phi) * std::cos(p.x) *
                std::sin(p.y * std::sin(phi) - p.z * std::cos(phi)),
            -std::cos(phi) * std::cos(p.x) *
                std::sin(p.y * std::sin(phi) - p.z * std::cos(phi))};
  };
}

inline vec<real_t, 3> analytical_domain(advection_diffusion const &) {
  return {2 * M_PI, 2 * M_PI * std::sqrt(2), 2 * M_PI * std::sqrt(2)};
}

} // namespace analytical