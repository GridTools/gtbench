#pragma once

#include "../communication/communication.hpp"
#include "../numerics/solver.hpp"
#include "../verification/analytical.hpp"
#include "./timer.hpp"

namespace execution {

template <class Analytical> struct on_domain_wrapper {
  template <class F> auto remap(F &&f) const {
    return [f = std::forward<F>(f), delta = delta, offset = offset,
            t = t](gt::int_t i, gt::int_t j, gt::int_t k) {
      return f({(i - halo + offset.x) * delta.x,
                (j - halo + offset.y) * delta.y, k * delta.z},
               t);
    };
  }

  template <class F> auto remap_staggered_z(F &&f) const {
    return remap([f = std::forward<F>(f),
                  delta = delta](vec<real_t, 3> const &p, real_t t) {
      return f({p.x, p.y, p.z - 0.5_r * delta.z}, t);
    });
  }

  auto data() const { return remap(verification::analytical::data(exact)); }
  auto u() const { return remap(verification::analytical::u(exact)); }
  auto v() const { return remap(verification::analytical::v(exact)); }
  auto w() const {
    return remap_staggered_z(verification::analytical::w(exact));
  }

  Analytical exact;
  vec<real_t, 3> delta;
  vec<gt::int_t, 2> offset;
  real_t t;
};

template <class Analytical>
on_domain_wrapper<Analytical>
on_domain(Analytical const &exact, vec<std::size_t, 3> const &resolution,
          vec<std::size_t, 2> const &offset, real_t t) {
  return {exact,
          {verification::analytical::domain(exact).x / resolution.x,
           verification::analytical::domain(exact).y / resolution.y,
           verification::analytical::domain(exact).z / resolution.z},
          {gt::int_t(offset.x), gt::int_t(offset.y)},
          t};
}

struct result {
  double error;
  double time;
};

template <class CommGrid, class Stepper, class Analytical>
result run_(CommGrid &&comm_grid, Stepper &&stepper, real_t tmax, real_t dt,
           Analytical &&exact) {
  const auto initial =
      on_domain(exact, communication::global_resolution(comm_grid),
                communication::offset(comm_grid), 0);

  const auto n = communication::resolution(comm_grid);

  numerics::solver_state state{n, initial.data(), initial.u(), initial.v(),
                               initial.w()};

  const vec<real_t, 3> delta = initial.delta;

  auto exchange = communication::halo_exchanger(comm_grid, state.sinfo);

  auto step = stepper(n, delta, exchange);

  auto start = timer::now(backend_t{});
  real_t t;
  for (t = 0; t < tmax; t += dt)
    step(state, dt);
  auto stop = timer::now(backend_t{});
  double time = timer::duration(start, stop);

  state.data.sync();
  auto view = gt::make_host_view(state.data);

  auto expected = on_domain(exact, communication::global_resolution(comm_grid),
                            communication::offset(comm_grid), t)
                      .data();
  double error = 0.0;
#pragma omp parallel for reduction(max : error)
  for (std::size_t i = halo; i < halo + n.x; ++i)
    for (std::size_t j = halo; j < halo + n.y; ++j)
      for (std::size_t k = 0; k < n.z; ++k)
        error = std::max(error,
                         double(std::abs(view(i, j, k) - expected(i, j, k))));

  return { error, time};
}

template <class CommGrid, class Stepper, class Analytical>
result run(CommGrid &&comm_grid, Stepper &&stepper, real_t tmax, real_t dt, Analytical &&exact)
{
    auto result = run_(std::forward<CommGrid>(comm_grid), std::forward<Stepper>(stepper), tmax, dt, std::forward<Analytical>(exact));
  
    return {communication::global_max(comm_grid, result.error),
            communication::global_max(comm_grid, result.time)};
}

} // namespace execution
