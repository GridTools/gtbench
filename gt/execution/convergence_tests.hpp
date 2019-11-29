#pragma once

#include <iostream>

#include "../communication/communication.hpp"
#include "../numerics/solver.hpp"
#include "../verification/analytical.hpp"
#include "../verification/convergence.hpp"
#include "./run.hpp"

namespace execution {

template <class CommWorld> void run_convergence_tests(CommWorld &&comm_world) {
  auto run_tests = [&comm_world](auto &&exact, auto &&stepper) {
    std::cout << "Spatial convergence:" << std::endl;
    auto spatial_error_f = [&comm_world, exact = std::move(exact),
                            stepper =
                                std::move(stepper)](std::size_t resolution) {
      return run(communication::grid(comm_world,
                                     {resolution, resolution, resolution}),
                 stepper, 1e-5, 1e-6, exact)
          .error;
    };
    verification::print_order_verification_result(
        verification::order_verification(spatial_error_f, 8, 128));

    std::cout << "Space-Time convergence:" << std::endl;
    auto spacetime_error_f = [&comm_world, exact = std::move(exact),
                              stepper =
                                  std::move(stepper)](std::size_t resolution) {
      return run(communication::grid(comm_world,
                                     {resolution, resolution, resolution}),
                 stepper, 1e-4, 1e-5 / resolution, exact)
          .error;
    };
    verification::print_order_verification_result(
        verification::order_verification(spacetime_error_f, 8, 64));
  };

  const real_t diffusion_coeff = 0.05;

  {
    std::cout << "HORIZONTAL DIFFUSION" << std::endl;
    run_tests(verification::analytical::horizontal_diffusion{diffusion_coeff},
              numerics::hdiff_stepper(diffusion_coeff));
  }

  {
    std::cout << "VERTICAL DIFFUSION" << std::endl;
    run_tests(verification::analytical::vertical_diffusion{diffusion_coeff},
              numerics::vdiff_stepper(diffusion_coeff));
  }

  {
    std::cout << "FULL DIFFUSION" << std::endl;
    run_tests(verification::analytical::full_diffusion{diffusion_coeff},
              numerics::diff_stepper(diffusion_coeff));
  }

  {
    std::cout << "HORIZONTAL ADVECTION" << std::endl;
    run_tests(verification::analytical::horizontal_advection{},
              numerics::hadv_stepper());
  }

  {
    std::cout << "VERTICAL ADVECTION" << std::endl;
    run_tests(verification::analytical::vertical_advection{},
              numerics::vadv_stepper());
  }

  {
    std::cout << "RUNGE-KUTTA ADVECTION" << std::endl;
    run_tests(verification::analytical::full_advection{},
              numerics::rkadv_stepper());
  }

  {
    std::cout << "ADVECTION-DIFFUSION: Spatial Convergence" << std::endl;
    run_tests(verification::analytical::advection_diffusion{diffusion_coeff},
              numerics::advdiff_stepper(diffusion_coeff));
  }
}

} // namespace execution