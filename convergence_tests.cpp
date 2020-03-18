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
  runtime::GTBENCH_RUNTIME::world rtw(argc, argv);

  cxxopts::Options options(argv[0], "GTBench convergence tests.");
  options.add_options()("h,help", "Print this help message and exit.");

  runtime::register_options(rtw, options);

  auto args = options.parse(argc, argv);

  if (args.count("help")) {
    std::cout << options.help() << std::endl;
    return 0;
  }

  auto rt = runtime::init(rtw, args);

  auto run_tests = [&rt](std::string const &title, auto const &exact,
                         auto const &stepper) {
    std::size_t max_resolution = std::is_same<real_t, float>() ? 16 : 32;

    std::cout << "=== " << title << " ===" << std::endl;
    std::cout << "Spatial convergence:" << std::endl;
    auto spatial_error_f = [&](std::size_t n) {
      return runtime::solve(rt, exact, stepper, {n, n, n}, 1e-2,
                            std::is_same<real_t, float>() ? 1e-3 : 1e-5)
          .error;
    };
    verification::print_order_verification_result(
        verification::order_verification(spatial_error_f, 8, max_resolution));

    std::cout << "Temporal convergence:" << std::endl;
    auto spacetime_error_f = [&](std::size_t n) {
      return runtime::solve(rt, exact, stepper, {128, 128, 128}, 1e-1, 1e-1 / n)
          .error;
    };
    verification::print_order_verification_result(
        verification::order_verification(spacetime_error_f, 8, max_resolution));
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
