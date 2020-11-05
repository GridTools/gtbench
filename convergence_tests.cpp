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
  opts("mode", "fast (two grid sizes) or full-range run (many grid sizes)",
       "{fast,full-range}", {"fast"});
  runtime::register_options(rt_tag, opts);

  auto args = opts.parse(argc, argv);
  std::string mode = args.get<std::string>("mode");
  if (mode != "fast" && mode != "full-range") {
    std::cerr << "invalid value '" << mode << "' passed to --mode" << std::endl;
    return 1;
  }
  bool full_range = mode == "full-range";

  auto rt = runtime::init(rt_tag, args);

  auto run_tests = [&rt,
                    full_range](std::string const &title, auto const &exact,
                                auto const &stepper, real_t tmax_spat_conv,
                                std::size_t n_spat_conv, real_t tmax_temp_conv,
                                std::size_t n_temp_conv) {
    std::cout << "=== " << title << " ===" << std::endl;
    std::cout << "Spatial convergence:" << std::endl;
    auto spatial_error_f = [&](std::size_t n) {
      return runtime::solve(rt, exact, stepper, {n, n, n}, tmax_spat_conv,
                            tmax_spat_conv / 100)
          .error;
    };
    std::size_t n_min = full_range ? 2 : n_spat_conv / 2;
    std::size_t n_max = full_range ? 128 : n_spat_conv;
    verification::print_order_verification_result(
        verification::order_verification(spatial_error_f, n_min, n_max));

    std::cout << "Temporal convergence:" << std::endl;
    auto spacetime_error_f = [&](std::size_t n) {
      return runtime::solve(rt, exact, stepper, {32, 32, 1024}, tmax_temp_conv,
                            tmax_temp_conv / n)
          .error;
    };
    n_min = full_range ? 2 : n_temp_conv / 2;
    n_max = full_range ? 128 : n_temp_conv;
    verification::print_order_verification_result(
        verification::order_verification(spacetime_error_f, n_min, n_max));
  };

  constexpr real_t diffusion_coeff = 0.05;
  constexpr bool is_float = std::is_same<real_t, float>();

  run_tests("HORIZONTAL DIFFUSION",
            verification::analytical::horizontal_diffusion{diffusion_coeff},
            numerics::hdiff_stepper(diffusion_coeff), is_float ? 1e-1 : 1e-3,
            is_float ? 16 : 32, 5e-1, 16);
  run_tests("VERTICAL DIFFUSION",
            verification::analytical::vertical_diffusion{diffusion_coeff},
            numerics::vdiff_stepper(diffusion_coeff), 5, 64, 100, 32);
  run_tests("FULL DIFFUSION",
            verification::analytical::full_diffusion{diffusion_coeff},
            numerics::diff_stepper(diffusion_coeff), is_float ? 1e-1 : 1e-3, 32,
            5e-1, 16);
  run_tests("HORIZONTAL ADVECTION",
            verification::analytical::horizontal_advection{},
            numerics::hadv_stepper(), is_float ? 1e-1 : 1e-4,
            is_float ? 32 : 64, 1e-1, 16);
  run_tests("VERTICAL ADVECTION", // needs even smaller vertical grid spacing
                                  // for perfect 2nd order temporal convergence
            verification::analytical::vertical_advection{},
            numerics::vadv_stepper(), 1e-1, 128, 10, 32);
  run_tests("RUNGE-KUTTA ADVECTION", verification::analytical::full_advection{},
            numerics::rkadv_stepper(), 1e-2, 64, 1, 8);
  run_tests("ADVECTION-DIFFUSION",
            verification::analytical::advection_diffusion{diffusion_coeff},
            numerics::advdiff_stepper(diffusion_coeff), is_float ? 1e-1 : 1e-3,
            64, 1, is_float ? 16 : 64);

  return 0;
}
