#pragma once

#include <cmath>

#include "../common/types.hpp"

namespace analytical {

struct horizontal_diffusion {
  auto data() const {
    return [d = diffusion_coeff](vec<real_t, 3> const &p, real_t t) -> real_t {
      return std::sin(p.x) * std::cos(p.y) * std::exp(-2 * d * t);
    };
  }

  auto u() const {
    return [](vec<real_t, 3> const &p, real_t t) -> real_t { return 0; };
  }

  auto v() const {
    return [](vec<real_t, 3> const &p, real_t t) -> real_t { return 0; };
  }
  auto w() const {
    return [](vec<real_t, 3> const &p, real_t t) -> real_t { return 0; };
  }

  constexpr vec<real_t, 3> domain() const {
    return {2 * M_PI, 2 * M_PI, 2 * M_PI};
  }

  real_t diffusion_coeff;
};

struct vertical_diffusion {
  auto data() const {
    return [d = diffusion_coeff](vec<real_t, 3> const &p, real_t t) -> real_t {
      return std::cos(p.z) * std::exp(-d * t);
    };
  }

  auto u() const {
    return [](vec<real_t, 3> const &p, real_t t) -> real_t { return 0; };
  }

  auto v() const {
    return [](vec<real_t, 3> const &p, real_t t) -> real_t { return 0; };
  }
  auto w() const {
    return [](vec<real_t, 3> const &p, real_t t) -> real_t { return 0; };
  }

  constexpr vec<real_t, 3> domain() const {
    return {2 * M_PI, 2 * M_PI, 2 * M_PI};
  }

  real_t diffusion_coeff;
};

struct full_diffusion {
  auto data() const {
    return [d = diffusion_coeff](vec<real_t, 3> const &p, real_t t) -> real_t {
      return std::sin(p.x) * std::cos(p.y) * std::cos(p.z) *
             std::exp(-3 * d * t);
    };
  }

  auto u() const {
    return [](vec<real_t, 3> const &p, real_t t) -> real_t { return 0; };
  }

  auto v() const {
    return [](vec<real_t, 3> const &p, real_t t) -> real_t { return 0; };
  }
  auto w() const {
    return [](vec<real_t, 3> const &p, real_t t) -> real_t { return 0; };
  }

  constexpr vec<real_t, 3> domain() const {
    return {2 * M_PI, 2 * M_PI, 2 * M_PI};
  }

  real_t diffusion_coeff;
};

struct horizontal_advection {
  auto data() const {
    return [](vec<real_t, 3> const &p, real_t t) -> real_t {
      return std::sin(p.x - t) * std::cos(p.y + 2 * t) * std::cos(z);
    };
  }

  auto u() const {
    return [](vec<real_t, 3> const &p, real_t t) -> real_t { return 1; };
  }

  auto v() const {
    return [](vec<real_t, 3> const &p, real_t t) -> real_t { return -2; };
  }
  auto w() const {
    return [](vec<real_t, 3> const &p, real_t t) -> real_t { return 0; };
  }

  constexpr vec<real_t, 3> domain() const {
    return {2 * M_PI, 2 * M_PI, 2 * M_PI};
  }
};

struct vertical_advection {
  auto data() const {
    return [](vec<real_t, 3> const &p, real_t t) -> real_t {
      return std::sin(p.x) * std::cos(p.y) * std::cos(z - 0.5 * t);
    };
  }

  auto u() const {
    return [](vec<real_t, 3> const &p, real_t t) -> real_t { return 0; };
  }

  auto v() const {
    return [](vec<real_t, 3> const &p, real_t t) -> real_t { return 0; };
  }
  auto w() const {
    return [](vec<real_t, 3> const &p, real_t t) -> real_t { return 0.5; };
  }

  constexpr vec<real_t, 3> domain() const {
    return {2 * M_PI, 2 * M_PI, 2 * M_PI};
  }
};

struct full_advection {
  auto data() const {
    return [](vec<real_t, 3> const &p, real_t t) -> real_t {
      return std::sin(p.x - t) * std::cos(p.y + 2 * t) * std::cos(z - 0.5 * t);
    };
  }

  auto u() const {
    return [](vec<real_t, 3> const &p, real_t t) -> real_t { return 1; };
  }

  auto v() const {
    return [](vec<real_t, 3> const &p, real_t t) -> real_t { return -2; };
  }
  auto w() const {
    return [](vec<real_t, 3> const &p, real_t t) -> real_t { return 0.5; };
  }

  constexpr vec<real_t, 3> domain() const {
    return {2 * M_PI, 2 * M_PI, 2 * M_PI};
  }
};

struct advection_diffusion {
  static constexpr real_t phi = M_PI / 4;

  auto data() const {
    return [d = diffusion_coeff](vec<real_t, 3> const &p, real_t t) -> real_t {
      return -std::sin(p.x) *
             std::sin(p.y * std::sin(phi) - p.z * std::cos(phi)) *
             std::exp(-2 * d * t);
    };
  }

  auto u() const {
    return [](vec<real_t, 3> const &p, real_t t) -> real_t {
      return -std::sin(p.x) *
             std::cos(p.y * std::sin(phi) - p.z * std::cos(phi));
    };
  }

  auto v() const {
    return [](vec<real_t, 3> const &p, real_t t) -> real_t {
      return std::sin(phi) * std::cos(p.x) *
             std::sin(p.y * std::sin(phi) - p.z * std::cos(phi));
    };
  }
  auto w() const {
    return [](vec<real_t, 3> const &p, real_t t) -> real_t {
      return -std::cos(phi) * std::cos(p.x) *
             std::sin(p.y * std::sin(phi) - p.z * std::cos(phi));
    };
  }

  constexpr vec<real_t, 3> domain() const {
    return {2 * M_PI, 2 * M_PI * std::sqrt(2), 2 * M_PI * std::sqrt(2)};
  }

  real_t diffusion_coeff;
};

template <class Analytical> struct at_time_wrapper {
  template <class F> auto fix_time(F &&f) {
    return [f = std::forward<F>, t = t](vec<real_t, 3> const &p) {
      return f(p, t);
    };
  }

  auto data() const { return fix_time(analytical.data()); }
  auto u() const { return fix_time(analytical.u()); }
  auto v() const { return fix_time(analytical.v()); }
  auto w() const { return fix_time(analytical.w()); }

  Analytical analytical;
  real_t t;
};

template <class Analytical>
Analytical at_time(Analytical &&analytical, real_t t) {
  return {std::forward<Analytical>(analytical), t};
}

} // namespace analytical