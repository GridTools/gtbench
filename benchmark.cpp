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
  runtime::GTBENCH_RUNTIME::world rtw(argc, argv);

  options opts;
  opts("domain-size", "size of domain along horizontal axes", "NX NY", 2);
  opts("runs", "number of runs, reported is the median result", "RUNS", {101});
  runtime::register_options(rtw, opts);

  auto args = opts.parse(argc, argv);

  if (!args.has("domain-size")) {
    std::cerr << "value for --domain-size must be provided" << std::endl;
    return 1;
  }

  auto rt = runtime::init(rtw, args);

  const auto domain_size = args.get<std::array<std::size_t, 2>>("domain-size");
  const std::size_t nx = domain_size[0];
  const std::size_t ny = domain_size[1];
  const std::size_t nz = 60;
  const std::size_t runs = args.get<std::size_t>("runs");

  auto fmt = [&]() -> std::ostream & {
    return std::cout << std::endl << std::setw(26) << std::left;
  };

  std::cout << "Running GTBENCH";
  fmt() << "Domain size:" << nx << "x" << ny << "x" << nz;

#define GTBENCH_STR2(var) #var
#define GTBENCH_STR(var) GTBENCH_STR2(var)
  fmt() << "Floating-point type:" << GTBENCH_STR(GTBENCH_FLOAT);
  fmt() << "GridTools backend:" << GTBENCH_STR(GTBENCH_BACKEND);
  fmt() << "GTBench runtime:" << GTBENCH_STR(GTBENCH_RUNTIME);
#undef GTBENCH_STR
#undef GTBENCH_STR2

  // analytical solution
  const real_t diffusion_coeff = 0.05;
  auto exact = verification::analytical::repeat(
      verification::analytical::advection_diffusion{diffusion_coeff},
      {(nx + nz - 1) / nz, (ny + nz - 1) / nz, 1});
  auto stepper = numerics::advdiff_stepper(diffusion_coeff);

  // benchmark executions
  std::vector<runtime::result> results;
  for (std::size_t r = 0; r < runs; ++r) {
    results.push_back(
        runtime::solve(rt, exact, stepper, {nx, ny, nz}, 0.1, 1e-3));
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

  fmt() << "Columns per second: " << (nx * ny / median_time);
  if (runs > 100) {
    std::cout << " (95% confidence: " << (nx * ny / upper_time) << " - "
              << (nx * ny / lower_time) << ")";
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
