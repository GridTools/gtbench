/*
 * gtbench
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <vector>

#include "./numerics/solver.hpp"
#include "./runtime/run.hpp"
#include "./verification/analytical.hpp"
#include "./verification/convergence.hpp"

int main(int argc, char **argv) {
  // user input handling
  std::size_t runs = 101;
  const std::size_t nz = 60;

  if (argc < 2 || argc > 4) {
    std::cerr << "Usage: " << argv[0] << " N [RUNS] [THREADS]" << std::endl;
    std::cerr << "N: global domain size, full domain will be NxNx" << nz
              << std::endl;
    std::cerr << "RUNS: number of runs, reported is the median and "
              << "95% confidence intervals (for RUNS > 100, "
              << "default value: " << runs << std::endl;
    return 1;
  }

  const std::size_t n = std::atoll(argv[1]);
  if (argc > 2)
    runs = std::atoll(argv[2]);

  // communication setup
  auto rt = runtime::GTBENCH_RUNTIME(argc, argv);

  auto fmt = [&]() -> std::ostream & {
    return std::cout << std::endl << std::setw(26) << std::left;
  };

  std::cout << "Running GTBENCH";
  fmt() << "Domain size:" << n << "x" << n << "x" << nz;

#define GTBENCH_STR2(var) #var
#define GTBENCH_STR(var) GTBENCH_STR2(var)
  fmt() << "Floating-point type:" << GTBENCH_STR(GTBENCH_FLOAT);
  fmt() << "GridTools backend:" << GTBENCH_STR(GTBENCH_BACKEND);
  fmt() << "GTBench runtime:"
        << GTBENCH_STR(GTBENCH_RUNTIME);
#undef GTBENCH_STR
#undef GTBENCH_STR2

  // analytical solution
  const real_t diffusion_coeff = 0.05;
  auto exact = verification::analytical::repeat(
      verification::analytical::advection_diffusion{diffusion_coeff},
      {(n + nz - 1) / nz, (n + nz - 1) / nz, 1});
  auto stepper = numerics::advdiff_stepper(diffusion_coeff);

  // benchmark executions
  std::vector<runtime::result> results;
  for (std::size_t r = 0; r < runs; ++r) {
    results.push_back(runtime::solve(rt, exact, stepper, {n, n, nz}, 0.1, 1e-3));
  }

  // computation and reporting of median and confidence interval times
  std::sort(std::begin(results), std::end(results),
            [](runtime::result const &a, runtime::result const &b) {
              return a.time < b.time;
            });
  const double median_time = results[results.size() / 2].time;
  const double lower_time = results[results.size() * 25 / 1000].time;
  const double upper_time = results[results.size() * 975 / 1000].time;

  fmt() << "Median time: " << median_time << "s";
  if (runs > 100) {
    std::cout << " (95% confidence: " << lower_time << "s - " << upper_time
              << "s)";
  }

  fmt() << "Columns per second: " << (n * n / median_time);
  if (runs > 100) {
    std::cout << " (95% confidence: " << (n * n / upper_time) << " - "
              << (n * n / lower_time) << ")";
  }
  std::cout << std::endl;

  // just for safety: check if errors of all runs are (almost) the same
  for (runtime::result const &r : results) {
    if (std::abs(r.error - results.front().error) > 1e-9) {
      std::cerr << "Detected error differences between the runs, "
                << "there must be something wrong!" << std::endl;
      return 1;
    }
  }

  return 0;
}
