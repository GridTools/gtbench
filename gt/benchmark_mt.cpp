#include <iostream>

#include "./communication/ghex_comm_mt.hpp"
#include "./execution/run.hpp"
#include "./numerics/solver.hpp"

#include <thread>

int main(int argc, char **argv) {
  auto comm_world = communication::ghex_comm::world(argc, argv);

  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " N [num_threads]" << std::endl;
    return 1;
  }

  omp_set_num_threads(1);
  const std::size_t n = std::atoll(argv[1]);
  const std::size_t nz = 100;
  int num_threads = 1;
  if (argc >=3)
      num_threads = std::atoi(argv[2]);

  std::cout << "running benchmark with " << n << "x" << n << " columns (" << num_threads << " threads)"
            << std::endl;


  auto comm_grid = communication::ghex_comm::grid_mt({n,n,100}, num_threads);
  
  real_t diffusion_coeff = 0.05;
  
  auto exact = verification::analytical::repeat(
      verification::analytical::advection_diffusion{diffusion_coeff},
      {(n + nz - 1) / nz, (n + nz - 1) / nz, 1});

  auto execution_func = [diffusion_coeff, &exact](communication::ghex_comm::grid_mt::sub_grid& g, execution::result& res)
  {
    res = execution::run_( g, numerics::advdiff_stepper(diffusion_coeff), 1, 1e-3, exact);
  };

  std::vector<communication::ghex_comm::grid_mt::sub_grid> sub_grids;
  sub_grids.reserve(num_threads);

  for (int i=0; i<num_threads; ++i)
    sub_grids.push_back( comm_grid[i] );

  std::vector<std::thread> threads;
  threads.reserve(num_threads);
  std::vector<execution::result> results(num_threads);
  for (int i=0; i<num_threads; ++i)
    threads.push_back( std::thread{ execution_func, std::ref(sub_grids[i]), std::ref(results[i]) } );
  for (auto& t : threads)
    t.join();

  double error = results[0].error;
  double time  = results[0].time;
  for (int i=1; i<num_threads; ++i)
  {
    error = std::max(results[i].error,error);
    time = std::max(results[i].time,time);
  }
  error = communication::global_max(comm_world, error);
  time = communication::global_max(comm_world, time);
  std::cout << "error: " << error << std::endl
            << "time: " << time << "s" << std::endl;

  return 0;
}
