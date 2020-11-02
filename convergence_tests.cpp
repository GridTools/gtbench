/*
 * gtbench
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <iostream>

#include "./numerics/solver.hpp"
#include "./runtime/run.hpp"
#include "./verification/analytical.hpp"
#include "./verification/convergence.hpp"

int main(int argc, char **argv) {
  constexpr auto rt_tag = runtime::GTBENCH_RUNTIME();

  options opts;
  runtime::register_options(rt_tag, opts);

  auto args = opts.parse(argc, argv);

  auto rt = runtime::init(rt_tag, args);

  auto run_tests = [&rt](std::string const &title, auto const &exact,
                         auto const &stepper) {
    std::cout << "=== " << title << " ===" << std::endl;
    std::cout << "Spatial convergence:" << std::endl;
    auto spatial_error_f = [&](std::size_t n) {
      real_t tmax = std::is_same<real_t, float>() ? 2e-2 : 1e-3;
      return runtime::solve(rt, exact, stepper, {n, n, n}, tmax, tmax / 100)
          .error;
    };
    std::size_t n = std::is_same<real_t, float>() ? 16 : 32;
    verification::print_order_verification_result(
        verification::order_verification(spatial_error_f, n / 2, n));

    std::cout << "Temporal convergence:" << std::endl;
    auto spacetime_error_f = [&](std::size_t n) {
      real_t tmax = std::is_same<real_t, float>() ? 1e-1 : 1e-2;
      return runtime::solve(rt, exact, stepper, {128, 128, 128}, tmax, tmax / n)
          .error;
    };
    verification::print_order_verification_result(
        verification::order_verification(spacetime_error_f, 8, 16));
  };

  const real_t diffusion_coeff = 0.05;

  run_tests("HORIZONTAL DIFFUSION",
            verification::analytical::horizontal_diffusion{diffusion_coeff},
            numerics::hdiff_stepper(diffusion_coeff));
  run_tests("VERTICAL DIFFUSION",
            verification::analytical::vertical_diffusion{diffusion_coeff},
            numerics::vdiff_stepper(diffusion_coeff));
  run_tests("FULL DIFFUSION",
            verification::analytical::full_diffusion{diffusion_coeff},
            numerics::diff_stepper(diffusion_coeff));
  run_tests("HORIZONTAL ADVECTION",
            verification::analytical::horizontal_advection{},
            numerics::hadv_stepper());
  run_tests("VERTICAL ADVECTION",
            verification::analytical::vertical_advection{},
            numerics::vadv_stepper());
  run_tests("RUNGE-KUTTA ADVECTION", verification::analytical::full_advection{},
            numerics::rkadv_stepper());
  run_tests("ADVECTION-DIFFUSION",
            verification::analytical::advection_diffusion{diffusion_coeff},
            numerics::advdiff_stepper(diffusion_coeff));

  return 0;
}
