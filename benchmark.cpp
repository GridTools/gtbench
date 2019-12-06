#include <algorithm>
#include <iostream>
#include <vector>

#include "./communication/backends.hpp"
#include "./execution/run.hpp"
#include "./numerics/solver.hpp"

int main(int argc, char **argv) {
  // communication setup
  auto comm_world =
      communication::GTBENCH_COMMUNICATION_BACKEND::world(argc, argv);

  // user input handling
  std::size_t runs = 101;
  const std::size_t nz = 100;

  if (argc < 2 || argc > 3) {
    std::cerr << "Usage: " << argv[0] << " N [RUNS]" << std::endl;
    std::cerr << "N: global domain size, full domain will be NxNx" << nz
              << std::endl;
    std::cerr << "RUNS: number of runs, reported is the median and "
              << "95% confidence intervals (for RUNS > 100, "
              << "default value: " << runs << ")" << std::endl;
    return 1;
  }

  const std::size_t n = std::atoll(argv[1]);
  if (argc > 2)
    runs = std::atoll(argv[2]);

  std::cout << "Running benchmark on a " << n << "x" << n << "x" << nz
            << " domain" << std::endl;

  // more communication setup
  auto comm_grid = communication::grid(comm_world, {n, n, nz});

  // analytical solution
  const real_t diffusion_coeff = 0.05;
  auto exact = verification::analytical::repeat(
      verification::analytical::advection_diffusion{diffusion_coeff},
      {(n + nz - 1) / nz, (n + nz - 1) / nz, 1});

  // benchmark executions
  std::vector<execution::result> results;
  for (std::size_t r = 0; r < runs; ++r) {
    results.push_back(execution::run(comm_grid,
                                     numerics::advdiff_stepper(diffusion_coeff),
                                     0.1, 1e-3, exact));
  }

  // computation and reporting of median and confidence interval times
  std::sort(std::begin(results), std::end(results),
            [](execution::result const &a, execution::result const &b) {
              return a.time < b.time;
            });
  const double median_time = results[results.size() / 2].time;
  const double lower_time = results[results.size() * 25 / 1000].time;
  const double upper_time = results[results.size() * 975 / 1000].time;

  std::cout << "Median time: " << median_time << "s" << std::endl;
  if (runs > 100) {
    std::cout << "95% confidence interval: " << lower_time << "s - "
              << upper_time << "s" << std::endl;
  }

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
