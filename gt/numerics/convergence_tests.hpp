#pragma once

#include <iostream>

#include "../communication/communication.hpp"
#include "../verification/analytical.hpp"
#include "../verification/convergence.hpp"
#include "../verification/run.hpp"
#include "./solver.hpp"

template <class CommWorld> void run_convergence_tests(CommWorld &&comm_world) {
  auto run_spatial = [&comm_world](auto &&exact, auto &&stepper) {
    auto error_f = [&comm_world, exact = std::move(exact),
                    stepper = std::move(stepper)](std::size_t resolution) {
      return run(communication::grid(comm_world,
                                     {resolution, resolution, resolution}),
                 stepper, 1e-5, 1e-6, exact)
          .error;
    };
    print_order_verification_result(order_verification(error_f, 8, 128));
  };

  auto run_temporal = [&comm_world](auto &&exact, auto &&stepper) {
    auto error_f = [&comm_world, exact = std::move(exact),
                    stepper = std::move(stepper)](std::size_t resolution) {
      return run(communication::grid(comm_world,
                                     {resolution, resolution, resolution}),
                 stepper, 1e-4, 1e-5 / resolution, exact)
          .error;
    };
    print_order_verification_result(order_verification(error_f, 8, 64));
  };

  {
    std::cout << "HORIZONTAL DIFFUSION: Spatial Convergence" << std::endl;
    analytical::horizontal_diffusion exact{0.05};
    run_spatial(exact, hdiff_stepper(exact.diffusion_coeff));
  }
  {
    std::cout << "HORIZONTAL DIFFUSION: Space-Time Convergence" << std::endl;
    analytical::horizontal_diffusion exact{0.05};
    run_temporal(exact, hdiff_stepper(exact.diffusion_coeff));
  }

  {
    std::cout << "VERTICAL DIFFUSION: Spatial Convergence" << std::endl;
    analytical::vertical_diffusion exact{0.05};
    run_spatial(exact, vdiff_stepper(exact.diffusion_coeff));
  }
  {
    std::cout << "VERTICAL DIFFUSION: Space-Time Convergence" << std::endl;
    analytical::vertical_diffusion exact{0.05};
    run_temporal(exact, vdiff_stepper(exact.diffusion_coeff));
  }

  {
    std::cout << "FULL DIFFUSION: Spatial Convergence" << std::endl;
    analytical::full_diffusion exact{0.05};
    run_spatial(exact, diff_stepper(exact.diffusion_coeff));
  }
  {
    std::cout << "FULL DIFFUSION: Space-Time Convergence" << std::endl;
    analytical::full_diffusion exact{0.05};
    run_temporal(exact, diff_stepper(exact.diffusion_coeff));
  }

  {
    std::cout << "HORIZONTAL ADVECTION: Spatial Convergence" << std::endl;
    analytical::horizontal_advection exact;
    run_spatial(exact, hadv_stepper());
  }
  {
    std::cout << "HORIZONTAL ADVECTION: Space-Time Convergence" << std::endl;
    analytical::horizontal_advection exact;
    run_temporal(exact, hadv_stepper());
  }

  {
    std::cout << "VERTICAL ADVECTION: Spatial Convergence" << std::endl;
    analytical::vertical_advection exact;
    run_spatial(exact, vadv_stepper());
  }
  {
    std::cout << "VERTICAL ADVECTION: Space-Time Convergence" << std::endl;
    analytical::vertical_advection exact;
    run_temporal(exact, vadv_stepper());
  }

  {
    std::cout << "RUNGE-KUTTA ADVECTION: Spatial Convergence" << std::endl;
    analytical::full_advection exact;
    run_spatial(exact, rkadv_stepper());
  }
  {
    std::cout << "RUNGE-KUTTA ADVECTION: Space-Time Convergence" << std::endl;
    analytical::full_advection exact;
    run_temporal(exact, rkadv_stepper());
  }

  {
    std::cout << "ADVECTION-DIFFUSION: Spatial Convergence" << std::endl;
    analytical::advection_diffusion exact{0.05};
    run_spatial(exact, full_stepper(exact.diffusion_coeff));
  }
  {
    std::cout << "ADVECTION-DIFFUSION: Space-Time Convergence" << std::endl;
    analytical::advection_diffusion exact{0.05};
    run_temporal(exact, full_stepper(exact.diffusion_coeff));
  }
}