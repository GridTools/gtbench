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

#include <gtbench/numerics/solver.hpp>
#include <gtbench/runtime/run.hpp>
#include <gtbench/verification/analytical.hpp>
#include <gtbench/verification/convergence.hpp>

int main(int argc, char **argv) {
  using namespace gtbench;

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

  constexpr bool is_float = std::is_same<real_t, float>();
  constexpr real_t atol_spatial = is_float ? 0.02 : 0.0;
  constexpr real_t rtol_spatial = is_float ? 0.03 : 0.01;
  constexpr real_t atol_temporal = is_float ? 0.02 : 0.0;
  constexpr real_t rtol_temporal = is_float ? 0.03 : 0.02;

  auto run_tests = [=, &rt](std::string const &title, auto const &exact,
                            auto const &stepper, real_t order_spatial,
                            real_t tmax_spatial, std::size_t n_spatial,
                            real_t order_temporal, real_t tmax_temporal,
                            std::size_t n_temporal) {
    std::cout << "=== " << title << " ===" << std::endl;
    std::cout << "Spatial convergence:" << std::endl;
    auto spatial_error_f = [&](std::size_t n) {
      return runtime::solve(rt, exact, stepper, {n, n, n}, tmax_spatial,
                            tmax_spatial / 100)
          .error;
    };
    std::size_t n_min = full_range ? 4 : n_spatial / 2;
    std::size_t n_max = full_range ? 128 : n_spatial;
    auto result_spatial =
        verification::order_verification(spatial_error_f, n_min, n_max);
    verification::print_order_verification_result(result_spatial);

    std::cout << "Temporal convergence:" << std::endl;
    auto spacetime_error_f = [&](std::size_t n) {
      return runtime::solve(rt, exact, stepper, {32, 32, 1024}, tmax_temporal,
                            tmax_temporal / n)
          .error;
    };
    n_min = full_range ? 2 : n_temporal / 2;
    n_max = full_range ? 128 : n_temporal;
    auto result_temporal =
        verification::order_verification(spacetime_error_f, n_min, n_max);
    verification::print_order_verification_result(result_temporal);

    return verification::check_order(result_spatial, order_spatial,
                                     atol_spatial, rtol_spatial) &&
           verification::check_order(result_temporal, order_temporal,
                                     atol_temporal, rtol_temporal);
  };

  constexpr real_t diffusion_coeff = 0.05;

  bool passed =
      run_tests("HORIZONTAL DIFFUSION",
                verification::analytical::horizontal_diffusion{diffusion_coeff},
                numerics::hdiff_stepper(diffusion_coeff), 6,
                is_float ? 1e-1 : 1e-3, is_float ? 16 : 32, 1, 5e-1, 16) &&
      run_tests("VERTICAL DIFFUSION",
                verification::analytical::vertical_diffusion{diffusion_coeff},
                numerics::vdiff_stepper(diffusion_coeff), 2, 5, 64, 2, 50,
                is_float ? 8 : 16) &&
      run_tests("FULL DIFFUSION",
                verification::analytical::full_diffusion{diffusion_coeff},
                numerics::diff_stepper(diffusion_coeff), 2,
                is_float ? 1e-1 : 1e-3, 32, 1, 5e-1, 16) &&
      run_tests("HORIZONTAL ADVECTION",
                verification::analytical::horizontal_advection{},
                numerics::hadv_stepper(), 5, is_float ? 1e-1 : 1e-4,
                is_float ? 32 : 64, 1, 1e-1, 16) &&
      run_tests(
          "VERTICAL ADVECTION", // needs even smaller vertical grid spacing
                                // for perfect 2nd order temporal convergence
          verification::analytical::vertical_advection{},
          numerics::vadv_stepper(), 2, 1e-1, 128, 2, 10, 32) &&
      run_tests("RUNGE-KUTTA ADVECTION",
                verification::analytical::full_advection{},
                numerics::rkadv_stepper(), 2, 1e-2, 64, 1, 1, 8) &&
      run_tests("ADVECTION-DIFFUSION",
                verification::analytical::advection_diffusion{diffusion_coeff},
                numerics::advdiff_stepper(diffusion_coeff), 2,
                is_float ? 1e-1 : 1e-3, 64, 1, 1, is_float ? 16 : 64);

  if (!passed) {
    std::cerr << "ERROR: some convergence tests failed" << std::endl;
    return 1;
  }
  return 0;
}
