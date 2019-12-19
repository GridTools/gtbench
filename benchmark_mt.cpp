#include <algorithm>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>

#include "./communication/ghex_comm_mt.hpp"
#include "./execution/run.hpp"
#include "./numerics/solver.hpp"

int main(int argc, char **argv) {
  // communication setup
  auto comm_world = communication::ghex_comm::world(argc, argv, true);

  // user input handling
  std::size_t runs = 101;
  int num_threads = 1;
  const std::size_t nz = 60;

  if (argc < 2 || argc > 4) {
    std::cerr << "Usage: " << argv[0] << " N [RUNS] [THREADS]" << std::endl;
    std::cerr << "N: global domain size, full domain will be NxNx" << nz
              << std::endl;
    std::cerr
        << "RUNS: number of runs, reported is the median and "
        << "95% confidence intervals (for RUNS > 100, "
        << "default value: " << runs << ")\n"
        << "THREADS: number of threads (sub-domains) per rank, default value: 1"
        << std::endl;
    return 1;
  }

  const std::size_t n = std::atoll(argv[1]);
  if (argc > 2)
    runs = std::atoll(argv[2]);
  if (argc > 3)
    num_threads = std::atoi(argv[3]);

  auto fmt = [&]() -> std::ostream & {
    return std::cout << std::endl << std::setw(25) << std::left;
  };

  std::cout << "Running GTBENCH-MT";
  fmt() << "Domain size:" << n << "x" << n << "x" << nz;
  fmt() << "Number of threads:" << num_threads;

#define GTBENCH_STR2(var) #var
#define GTBENCH_STR(var) GTBENCH_STR2(var)
  fmt() << "Floating-point type:" << GTBENCH_STR(GTBENCH_FLOAT);
  fmt() << "GridTools backend:" << GTBENCH_STR(GTBENCH_BACKEND);
  fmt() << "Communication backend:" << GTBENCH_STR(ghex_comm_mt);
#undef GTBENCH_STR
#undef GTBENCH_STR2

  // more communication setup
  auto comm_grid = communication::ghex_comm::grid_mt({n, n, nz}, num_threads);

  // analytical solution
  const real_t diffusion_coeff = 0.05;
  auto exact = verification::analytical::repeat(
      verification::analytical::advection_diffusion{diffusion_coeff},
      {(n + nz - 1) / nz, (n + nz - 1) / nz, 1});

  // benchmark executions
  std::vector<std::vector<execution::result>> all_results(num_threads);
  auto execution_func = [&](unsigned int id) {
    auto s_grid = comm_grid[id];
    for (std::size_t r = 0; r < runs; ++r) {
      all_results[id].push_back(
          execution::run(s_grid, numerics::advdiff_stepper(diffusion_coeff),
                         0.1, 1e-3, exact));
    }
  };
  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  for (int i = 0; i < num_threads; ++i)
    threads.push_back(std::thread{execution_func, i});

  for (auto &t : threads)
    t.join();

  for (std::size_t i = 0; i < all_results[0].size(); ++i)
    for (std::size_t j = 1; j < all_results.size(); ++j) {
      all_results[0][i].time =
          std::max(all_results[0][i].time, all_results[j][i].time);
      all_results[0][i].error =
          std::max(all_results[0][i].error, all_results[j][i].error);
    }

  fmt() << "error:" << all_results[0][0].error;

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
