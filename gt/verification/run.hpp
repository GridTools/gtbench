#pragma once

#include "../communication/communication.hpp"
#include "../numerics/solver.hpp"
#include "analytical.hpp"

template <class CommGrid, class Stepper, class Analytical>
double run(CommGrid &&comm_grid, Stepper &&stepper, real_t tmax, real_t dt,
           Analytical &&exact) {
  const auto initial = analytical::to_domain(
      exact, communication::global_resolution_x(comm_grid),
      communication::global_resolution_y(comm_grid),
      communication::resolution_z(comm_grid), 0,
      communication::offset_x(comm_grid), communication::offset_y(comm_grid));

  const std::size_t nx = communication::resolution_x(comm_grid);
  const std::size_t ny = communication::resolution_y(comm_grid);
  const std::size_t nz = communication::resolution_z(comm_grid);

  solver_state state{nx,          ny,          nz,         initial.data(),
                     initial.u(), initial.v(), initial.w()};

  const real_t dx = initial.dx, dy = initial.dy, dz = initial.dz;

  auto exchange = communication::halo_exchanger(comm_grid, state.sinfo);

  auto step = stepper(nx, ny, nz, dx, dy, dz, exchange);

  real_t t;
  for (t = 0; t < tmax; t += dt)
    step(state, dt);

  auto view = gt::make_host_view(state.data);

  auto expected = analytical::to_domain(
                      exact, communication::global_resolution_x(comm_grid),
                      communication::global_resolution_y(comm_grid), nz, t,
                      communication::offset_x(comm_grid),
                      communication::offset_y(comm_grid))
                      .data();
  double error = 0.0;
#pragma omp parallel for reduction(+ : error)
  for (std::size_t i = halo; i < halo + nx; ++i) {
    for (std::size_t j = halo; j < halo + ny; ++j) {
      for (std::size_t k = 0; k < nz; ++k) {
        double diff = view(i, j, k) - expected(i, j, k);
        error += diff * diff;
      }
    }
  }
  error *= dx * dy * dz;

  return std::sqrt(communication::global_sum(comm_grid, error));
}