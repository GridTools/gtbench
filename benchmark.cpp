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
#include <thread>
#include <vector>

#include "./communication/backends.hpp"
#include "./execution/run.hpp"
#include "./numerics/solver.hpp"

int main(int argc, char **argv) {
  // user input handling
  std::size_t runs = 101;
  const std::size_t nz = 60;
  int num_threads = 1;

  if (argc < 2 || argc > 4) {
    std::cerr << "Usage: " << argv[0] << " N [RUNS] [THREADS]" << std::endl;
    std::cerr << "N: global domain size, full domain will be NxNx" << nz
              << std::endl;
    std::cerr << "RUNS: number of runs, reported is the median and "
              << "95% confidence intervals (for RUNS > 100, "
              << "default value: " << runs << ")\n"
              << "THREADS: number of threads (sub-domains) per rank, default "
                 "value: 1\n"
              << "(this option affects the GHEX backend only)" << std::endl;
    return 1;
  }

  const std::size_t n = std::atoll(argv[1]);
  if (argc > 2)
    runs = std::atoll(argv[2]);
  if (argc > 3)
    num_threads = std::atoi(argv[3]);

    // override number of threads for simple backends
#if !defined(GTBENCH_USE_GHEX)
  if (num_threads != 1) {
    std::cerr << "number of threads cannot be larger than 1 for this backend."
              << std::endl;
    return 1;
  }
#endif
  const bool multi_threaded = num_threads > 1;

  // communication setup
  auto comm_world = communication::GTBENCH_COMMUNICATION_BACKEND::world(
      argc, argv, multi_threaded);

  auto fmt = [&]() -> std::ostream & {
    return std::cout << std::endl << std::setw(26) << std::left;
  };

  std::cout << "Running GTBENCH";
  fmt() << "Domain size:" << n << "x" << n << "x" << nz;
  fmt() << "Number of domain threads:" << num_threads;

#ifdef _OPENMP
  // get number of openmp threads
  fmt() << "Number of OpenMP threads:" << omp_get_max_threads();
#endif

#define GTBENCH_STR2(var) #var
#define GTBENCH_STR(var) GTBENCH_STR2(var)
  fmt() << "Floating-point type:" << GTBENCH_STR(GTBENCH_FLOAT);
  fmt() << "GridTools backend:" << GTBENCH_STR(GTBENCH_BACKEND);
  fmt() << "Communication backend:"
        << GTBENCH_STR(GTBENCH_COMMUNICATION_BACKEND);
#undef GTBENCH_STR
#undef GTBENCH_STR2

  // more communication setup
  auto comm_grid = communication::grid(comm_world, {n, n, nz}, num_threads);

  // analytical solution
  const real_t diffusion_coeff = 0.05;
  auto exact = verification::analytical::repeat(
      verification::analytical::advection_diffusion{diffusion_coeff},
      {(n + nz - 1) / nz, (n + nz - 1) / nz, 1});

  // benchmark executions
  std::vector<std::vector<execution::result>> all_results(num_threads);
#ifdef __CUDACC__
  auto execution_func = [&](int id = 0, int cuda_device = 0) {
    cudaSetDevice(cuda_device);
#else
  auto execution_func = [&](int id = 0) {
#endif
    for (std::size_t r = 0; r < runs; ++r) {
      all_results[id].push_back(execution::run(
          communication::sub_grid(comm_grid, id),
          numerics::advdiff_stepper(diffusion_coeff), 0.1, 1e-3, exact));
    }
  };

#if defined(GTBENCH_USE_GHEX)
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
#ifdef __CUDACC__
  int cuda_device;
  cudaGetDevice(&cuda_device);
#endif
  for (int i = 0; i < num_threads; ++i) {
#ifdef __CUDACC__
    threads.push_back(std::thread{execution_func, i, cuda_device});
#else
    threads.push_back(std::thread{execution_func, i});
#endif
  }
  for (auto &t : threads)
    t.join();
  for (std::size_t i = 0; i < all_results[0].size(); ++i)
    for (std::size_t j = 1; j < all_results.size(); ++j) {
      all_results[0][i].time =
          std::max(all_results[0][i].time, all_results[j][i].time);
      all_results[0][i].error =
          std::max(all_results[0][i].error, all_results[j][i].error);
    }
#else
  execution_func();
#endif

  auto &results = all_results[0];

  // computation and reporting of median and confidence interval times
  std::sort(std::begin(results), std::end(results),
            [](execution::result const &a, execution::result const &b) {
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
  for (execution::result const &r : results) {
    if (std::abs(r.error - results.front().error) > 1e-9) {
      std::cerr << "Detected error differences between the runs, "
                << "there must be something wrong!" << std::endl;
      return 1;
    }
  }

  return 0;
}
