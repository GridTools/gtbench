#pragma once

#include <iostream>

#include "../communication/communication.hpp"
#include "../verification/analytical.hpp"
#include "../verification/convergence.hpp"
#include "../verification/run.hpp"
#include "./solver.hpp"

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
    print_order_verification_result(
        order_verification(spatial_error_f, 8, 128));

    std::cout << "Space-Time convergence:" << std::endl;
    auto spacetime_error_f = [&comm_world, exact = std::move(exact),
                              stepper =
                                  std::move(stepper)](std::size_t resolution) {
      return run(communication::grid(comm_world,
                                     {resolution, resolution, resolution}),
                 stepper, 1e-4, 1e-5 / resolution, exact)
          .error;
    };
    print_order_verification_result(
        order_verification(spacetime_error_f, 8, 64));
  };

  const real_t diffusion_coeff = 0.05;

  {
    std::cout << "HORIZONTAL DIFFUSION" << std::endl;
    run_tests(analytical::horizontal_diffusion{diffusion_coeff},
              hdiff_stepper(diffusion_coeff));
  }

  {
    std::cout << "VERTICAL DIFFUSION" << std::endl;
    run_tests(analytical::vertical_diffusion{diffusion_coeff},
              vdiff_stepper(diffusion_coeff));
  }

  {
    std::cout << "FULL DIFFUSION" << std::endl;
    run_tests(analytical::full_diffusion{diffusion_coeff},
              diff_stepper(diffusion_coeff));
  }

  {
    std::cout << "HORIZONTAL ADVECTION" << std::endl;
    run_tests(analytical::horizontal_advection{}, hadv_stepper());
  }

  {
    std::cout << "VERTICAL ADVECTION" << std::endl;
    run_tests(analytical::vertical_advection{}, vadv_stepper());
  }

  {
    std::cout << "RUNGE-KUTTA ADVECTION" << std::endl;
    run_tests(analytical::full_advection{}, rkadv_stepper());
  }

  {
    std::cout << "ADVECTION-DIFFUSION: Spatial Convergence" << std::endl;
    run_tests(analytical::advection_diffusion{diffusion_coeff},
              full_stepper(diffusion_coeff));
  }
}