#pragma once

#include "../communication/communication.hpp"
#include "../numerics/solver.hpp"
#include "analytical.hpp"

template <class CommGrid, class Stepper, class Analytical>
double run(CommGrid &&comm_grid, Stepper &&stepper, real_t tmax, real_t dt,
           Analytical &&exact) {
  const auto initial = analytical::to_domain(
      exact, communication::global_resolution(comm_grid).x,
      communication::global_resolution(comm_grid).y,
      communication::global_resolution(comm_grid).z, 0,
      communication::offset(comm_grid).x, communication::offset(comm_grid).y);

  const auto n = communication::resolution(comm_grid);

  solver_state state{n.x,         n.y,         n.z,        initial.data(),
                     initial.u(), initial.v(), initial.w()};

  const real_t dx = initial.dx, dy = initial.dy, dz = initial.dz;

  auto exchange = communication::halo_exchanger(comm_grid, state.sinfo);

  auto step = stepper(n.x, n.y, n.z, dx, dy, dz, exchange);

  real_t t;
  for (t = 0; t < tmax; t += dt)
    step(state, dt);

  auto view = gt::make_host_view(state.data);

  auto expected = analytical::to_domain(
                      exact, communication::global_resolution(comm_grid).x,
                      communication::global_resolution(comm_grid).y, n.z, t,
                      communication::offset(comm_grid).x,
                      communication::offset(comm_grid).y)
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
  error *= dx * dy * dz;

  return std::sqrt(communication::global_sum(comm_grid, error));
}