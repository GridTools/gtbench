#pragma once

#include "../communication/communication.hpp"
#include "../numerics/solver.hpp"
#include "analytical.hpp"

template <class Analytical> struct on_domain_wrapper {
  template <class F> auto remap(F &&f) const {
    return [f = std::forward<F>(f), delta = delta,
            offset = offset](gt::int_t i, gt::int_t j, gt::int_t k) {
      return f((i - halo + offset.x) * delta.x, (j - halo + offset.y) * delta.y,
               k * delta.z);
    };
  }

  template <class F> auto remap_staggered_z(F &&f) const {
    return remap(
        [f = std::forward<F>(f), delta = delta](real_t x, real_t y, real_t z) {
          return f(x, y, z - 0.5 * delta.z);
        });
  }

  auto data() const { return remap(analytical.data()); }
  auto u() const { return remap(analytical.u()); }
  auto v() const { return remap(analytical.v()); }
  auto w() const { return remap_staggered_z(analytical.w()); }

  Analytical analytical;
  vec<real_t, 3> delta;
  vec<gt::int_t, 2> offset;
};

template <class Analytical>
on_domain_wrapper<Analytical>
on_domain(Analytical &&analytical, vec<std::size_t, 3> const &resolution,
          vec<std::size_t, 2> const &offset, real_t t) {
  return {std::forward<Analytical>(analytical),
          {analytical.domain().x / resolution.x,
           analytical.domain().y / resolution.y,
           analytical.domain().z / resolution.z},
          {gt::int_t(offset.x), gt::int_t(offset.y)},
          t};
}

template <class CommGrid, class Stepper, class Analytical>
double run(CommGrid &&comm_grid, Stepper &&stepper, real_t tmax, real_t dt,
           Analytical &&exact) {
  const auto initial = on_domain(analytical::at_time(exact, 0),
                                 communication::global_resolution(comm_grid),
                                 communication::offset(comm_grid));

  const auto n = communication::resolution(comm_grid);

  solver_state state{n, initial.data(), initial.u(), initial.v(), initial.w()};

  const vec<real_t, 3> delta = initial.delta;

  auto exchange = communication::halo_exchanger(comm_grid, state.sinfo);

  auto step = stepper(n, delta, exchange);

  real_t t;
  for (t = 0; t < tmax; t += dt)
    step(state, dt);

  auto view = gt::make_host_view(state.data);

  auto expected = on_domain(analytical::at_time(exact, t),
                            communication::global_resolution(comm_grid),
                            communication::offset(comm_grid))
                      .data();
  double error = 0.0;
#pragma omp parallel for reduction(+ : error)
  for (std::size_t i = halo; i < halo + n.x; ++i) {
    for (std::size_t j = halo; j < halo + n.y; ++j) {
      for (std::size_t k = 0; k < n.z; ++k) {
        double diff = view(i, j, k) - expected(i, j, k);
        error += diff * diff;
      }
    }
  }
  error *= delta.x * delta.y * delta.z;

  return std::sqrt(communication::global_sum(comm_grid, error));
}