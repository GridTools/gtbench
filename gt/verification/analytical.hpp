#pragma once

#include "../common.hpp"

namespace analytical {

struct horizontal_diffusion {
  auto data() const {
    return [d = diffusion_coeff](real_t x, real_t y, real_t z,
                                 real_t t) -> real_t {
      using namespace gt::math;
      return sin(x) * cos(y) * exp(-2 * d * t);
    };
  }

  auto u() const {
    return [](real_t x, real_t y, real_t z, real_t t) -> real_t { return 0; };
  }

  auto v() const {
    return [](real_t x, real_t y, real_t z, real_t t) -> real_t { return 0; };
  }
  auto w() const {
    return [](real_t x, real_t y, real_t z, real_t t) -> real_t { return 0; };
  }

  constexpr real_t domain_x() const { return 2 * M_PI; }
  constexpr real_t domain_y() const { return 2 * M_PI; }
  constexpr real_t domain_z() const { return 2 * M_PI; }

  real_t diffusion_coeff;
};

struct vertical_diffusion {
  auto data() const {
    return [d = diffusion_coeff](real_t x, real_t y, real_t z,
                                 real_t t) -> real_t {
      using namespace gt::math;
      return cos(z) * exp(-d * t);
    };
  }

  auto u() const {
    return [](real_t x, real_t y, real_t z, real_t t) -> real_t { return 0; };
  }

  auto v() const {
    return [](real_t x, real_t y, real_t z, real_t t) -> real_t { return 0; };
  }
  auto w() const {
    return [](real_t x, real_t y, real_t z, real_t t) -> real_t { return 0; };
  }

  constexpr real_t domain_x() const { return 2 * M_PI; }
  constexpr real_t domain_y() const { return 2 * M_PI; }
  constexpr real_t domain_z() const { return 2 * M_PI; }

  real_t diffusion_coeff;
};

struct full_diffusion {
  auto data() const {
    return [d = diffusion_coeff](real_t x, real_t y, real_t z,
                                 real_t t) -> real_t {
      using namespace gt::math;
      return sin(x) * cos(y) * cos(z) * exp(-3 * d * t);
    };
  }

  auto u() const {
    return [](real_t x, real_t y, real_t z, real_t t) -> real_t { return 0; };
  }

  auto v() const {
    return [](real_t x, real_t y, real_t z, real_t t) -> real_t { return 0; };
  }
  auto w() const {
    return [](real_t x, real_t y, real_t z, real_t t) -> real_t { return 0; };
  }

  constexpr real_t domain_x() const { return 2 * M_PI; }
  constexpr real_t domain_y() const { return 2 * M_PI; }
  constexpr real_t domain_z() const { return 2 * M_PI; }

  real_t diffusion_coeff;
};

struct horizontal_advection {
  auto data() const {
    return [](real_t x, real_t y, real_t z, real_t t) -> real_t {
      using namespace gt::math;
      return sin(x - t) * cos(y + 2 * t) * cos(z);
    };
  }

  auto u() const {
    return [](real_t x, real_t y, real_t z, real_t t) -> real_t { return 1; };
  }

  auto v() const {
    return [](real_t x, real_t y, real_t z, real_t t) -> real_t { return -2; };
  }
  auto w() const {
    return [](real_t x, real_t y, real_t z, real_t t) -> real_t { return 0; };
  }

  constexpr real_t domain_x() const { return 2 * M_PI; }
  constexpr real_t domain_y() const { return 2 * M_PI; }
  constexpr real_t domain_z() const { return 2 * M_PI; }
};

struct vertical_advection {
  auto data() const {
    return [](real_t x, real_t y, real_t z, real_t t) -> real_t {
      using namespace gt::math;
      return sin(x) * cos(y) * cos(z - 0.5 * t);
    };
  }

  auto u() const {
    return [](real_t x, real_t y, real_t z, real_t t) -> real_t { return 0; };
  }

  auto v() const {
    return [](real_t x, real_t y, real_t z, real_t t) -> real_t { return 0; };
  }
  auto w() const {
    return [](real_t x, real_t y, real_t z, real_t t) -> real_t { return 0.5; };
  }

  constexpr real_t domain_x() const { return 2 * M_PI; }
  constexpr real_t domain_y() const { return 2 * M_PI; }
  constexpr real_t domain_z() const { return 2 * M_PI; }
};

struct full_advection {
  auto data() const {
    return [](real_t x, real_t y, real_t z, real_t t) -> real_t {
      using namespace gt::math;
      return sin(x - t) * cos(y + 2 * t) * cos(z - 0.5 * t);
    };
  }

  auto u() const {
    return [](real_t x, real_t y, real_t z, real_t t) -> real_t { return 1; };
  }

  auto v() const {
    return [](real_t x, real_t y, real_t z, real_t t) -> real_t { return -2; };
  }
  auto w() const {
    return [](real_t x, real_t y, real_t z, real_t t) -> real_t { return 0.5; };
  }

  constexpr real_t domain_x() const { return 2 * M_PI; }
  constexpr real_t domain_y() const { return 2 * M_PI; }
  constexpr real_t domain_z() const { return 2 * M_PI; }
};

struct advection_diffusion {
  static constexpr real_t phi = M_PI / 4;

  auto data() const {
    return [d = diffusion_coeff](real_t x, real_t y, real_t z,
                                 real_t t) -> real_t {
      using namespace gt::math;
      return -sin(x) * sin(y * sin(phi) - z * cos(phi)) * exp(-2 * d * t);
    };
  }

  auto u() const {
    return [](real_t x, real_t y, real_t z, real_t t) -> real_t {
      using namespace gt::math;
      return -sin(x) * cos(y * sin(phi) - z * cos(phi));
    };
  }

  auto v() const {
    return [](real_t x, real_t y, real_t z, real_t t) -> real_t {
      using namespace gt::math;
      return sin(phi) * cos(x) * sin(y * sin(phi) - z * cos(phi));
    };
  }
  auto w() const {
    return [](real_t x, real_t y, real_t z, real_t t) -> real_t {
      using namespace gt::math;
      return -cos(phi) * cos(x) * sin(y * sin(phi) - z * cos(phi));
    };
  }

  constexpr real_t domain_x() const { return 2 * M_PI; }
  constexpr real_t domain_y() const { return 2 * M_PI * gt::math::sqrt(2.0); }
  constexpr real_t domain_z() const { return 2 * M_PI * gt::math::sqrt(2.0); }

  real_t diffusion_coeff;
};

template <class Analytical> struct to_domain_wrapper {
  template <class F> auto remap(F &&f) const {
    return [f = std::forward<F>(f), dx = dx, dy = dy, dz = dz,
            t = t, offset_i = offset_i, offset_j = offset_j](gt::int_t i, gt::int_t j, gt::int_t k) {
      return f((i - halo + offset_i) * dx, (j - halo + offset_j) * dy, k * dz,
               t);
    };
  }

  template <class F> auto remap_staggered_z(F &&f) const {
    return [f = std::forward<F>(f), dx = dx, dy = dy, dz = dz,
            t = t, offset_i = offset_i, offset_j = offset_j](gt::int_t i, gt::int_t j, gt::int_t k) {
      return f((i - halo + offset_i) * dx, (j - halo + offset_j) * dy,
               k * dz - real_t(0.5) * dz, t);
    };
  }

  auto data() const { return remap(analytical.data()); }
  auto u() const { return remap(analytical.u()); }
  auto v() const { return remap(analytical.v()); }
  auto w() const { return remap_staggered_z(analytical.w()); }

  Analytical analytical;
  real_t dx, dy, dz, t;
  gt::int_t offset_i, offset_j;
};

template <class Analytical>
to_domain_wrapper<Analytical>
to_domain(Analytical &&analytical, std::size_t resolution_x,
          std::size_t resolution_y, std::size_t resolution_z, real_t t,
          gt::int_t offset_i = 0, gt::int_t offset_j = 0) {
  return {std::forward<Analytical>(analytical),
          analytical.domain_x() / resolution_x,
          analytical.domain_y() / resolution_y,
          analytical.domain_z() / resolution_z,
          t,
          offset_i,
          offset_j};
}

} // namespace analytical