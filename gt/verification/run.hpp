#pragma once

#include "../communication/communication.hpp"
#include "../numerics/solver.hpp"
#include "analytical.hpp"

template <class CommGrid, class Stepper, class Analytical>
double run(CommGrid &&comm_grid, Stepper &&stepper, real_t tmax, real_t dt,
           Analytical &&exact) {
  const auto initial =
      analytical::to_domain(exact, communication::global_resolution(comm_grid),
                            communication::offset(comm_grid), 0);

  const auto n = communication::resolution(comm_grid);

  solver_state state{n, initial.data(), initial.u(), initial.v(), initial.w()};

  const vec<real_t, 3> delta = {initial.dx, initial.dy, initial.dz};

  auto exchange = communication::halo_exchanger(comm_grid, state.sinfo);

  auto step = stepper(n, delta, exchange);

  real_t t;
  for (t = 0; t < tmax; t += dt)
    step(state, dt);

  auto view = gt::make_host_view(state.data);

  auto expected =
      analytical::to_domain(exact, communication::global_resolution(comm_grid),
                            communication::offset(comm_grid), t)
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