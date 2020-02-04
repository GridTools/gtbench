#include <iostream>

#include "./communication/backends.hpp"
#include "./execution/run.hpp"
#include "./numerics/solver.hpp"
#include "./verification/analytical.hpp"
#include "./verification/convergence.hpp"

int main(int argc, char **argv) {
  auto comm_world =
      communication::GTBENCH_COMMUNICATION_BACKEND::world(argc, argv, false);

  auto run_tests = [&comm_world](std::string const &title, auto &&exact,
                                 auto &&stepper) {
    std::size_t max_resolution = std::is_same<real_t, float>() ? 16 : 32;

    std::cout << "=== " << title << " ===" << std::endl;
    std::cout << "Spatial convergence:" << std::endl;
    auto spatial_error_f = [&comm_world, exact = std::move(exact),
                            stepper =
                                std::move(stepper)](std::size_t resolution) {
      auto grid =
          communication::grid(comm_world, {resolution, resolution, resolution});
      return execution::run(grid.sub_grid(), stepper, 1e-2,
                            std::is_same<real_t, float>() ? 1e-3 : 1e-5, exact)
          .error;
    };
    verification::print_order_verification_result(
        verification::order_verification(spatial_error_f, 8, max_resolution));

    std::cout << "Temporal convergence:" << std::endl;
    auto spacetime_error_f = [&comm_world, exact = std::move(exact),
                              stepper =
                                  std::move(stepper)](std::size_t resolution) {
      auto grid = communication::grid(comm_world, {128, 128, 128});
      return execution::run(grid.sub_grid(), stepper, 1e-1,
                            1e-1 / resolution, exact)
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
