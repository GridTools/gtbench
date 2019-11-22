#pragma once

#include <iostream>

#include "../communication/communication.hpp"
#include "../verification/analytical.hpp"
#include "../verification/convergence.hpp"
#include "../verification/run.hpp"
#include "solver.hpp"

template <class CommWorld> void run_convergence_tests(CommWorld &&comm_world) {
  {
    std::cout << "HORIZONTAL DIFFUSION: Spatial Convergence" << std::endl;
    analytical::horizontal_diffusion exact{0.05};
    auto error_f = [&comm_world, exact](std::size_t resolution) {
      auto comm_grid =
          communication::grid(comm_world, {resolution, resolution, resolution});
      return run(std::move(comm_grid), hdiff_stepper(exact.diffusion_coeff),
                 1e-4, 1e-5, exact);
    };
    print_order_verification_result(order_verification(error_f, 8, 128));
  }
  {
    std::cout << "HORIZONTAL DIFFUSION: Space-Time Convergence" << std::endl;
    analytical::horizontal_diffusion exact{0.05};
    auto error_f = [&comm_world, exact](std::size_t resolution) {
      auto comm_grid =
          communication::grid(comm_world, {resolution, resolution, resolution});
      return run(std::move(comm_grid), hdiff_stepper(exact.diffusion_coeff),
                 1e-2, 1e-3 / resolution, exact);
    };
    print_order_verification_result(order_verification(error_f, 8, 64));
  }

  {
    std::cout << "VERTICAL DIFFUSION: Spatial Convergence" << std::endl;
    analytical::vertical_diffusion exact{0.05};
    auto error_f = [&comm_world, exact](std::size_t resolution) {
      auto comm_grid =
          communication::grid(comm_world, {resolution, resolution, resolution});
      return run(std::move(comm_grid), vdiff_stepper(exact.diffusion_coeff),
                 1e-4, 1e-5, exact);
    };
    print_order_verification_result(order_verification(error_f, 8, 128));
  }
  {
    std::cout << "VERTICAL DIFFUSION: Space-Time Convergence" << std::endl;
    analytical::vertical_diffusion exact{0.05};
    auto error_f = [&comm_world, exact](std::size_t resolution) {
      auto comm_grid =
          communication::grid(comm_world, {resolution, resolution, resolution});
      return run(std::move(comm_grid), vdiff_stepper(exact.diffusion_coeff),
                 1e-2, 1e-3 / resolution, exact);
    };
    print_order_verification_result(order_verification(error_f, 8, 64));
  }

  {
    std::cout << "FULL DIFFUSION: Spatial Convergence" << std::endl;
    analytical::full_diffusion exact{0.05};
    auto error_f = [&comm_world, exact](std::size_t resolution) {
      auto comm_grid =
          communication::grid(comm_world, {resolution, resolution, resolution});
      return run(std::move(comm_grid), diff_stepper(exact.diffusion_coeff),
                 1e-4, 1e-5, exact);
    };
    print_order_verification_result(order_verification(error_f, 8, 128));
  }
  {
    std::cout << "FULL DIFFUSION: Space-Time Convergence" << std::endl;
    analytical::full_diffusion exact{0.05};
    auto error_f = [&comm_world, exact](std::size_t resolution) {
      auto comm_grid =
          communication::grid(comm_world, {resolution, resolution, resolution});
      return run(std::move(comm_grid), diff_stepper(exact.diffusion_coeff),
                 1e-2, 1e-3 / resolution, exact);
    };
    print_order_verification_result(order_verification(error_f, 8, 64));
  }

  {
    std::cout << "HORIZONTAL ADVECTION: Spatial Convergence" << std::endl;
    analytical::horizontal_advection exact;
    auto error_f = [&comm_world, exact](std::size_t resolution) {
      auto comm_grid =
          communication::grid(comm_world, {resolution, resolution, resolution});
      return run(std::move(comm_grid), hadv_stepper(), 1e-5, 1e-6, exact);
    };
    print_order_verification_result(order_verification(error_f, 8, 128));
  }
  {
    std::cout << "HORIZONTAL ADVECTION: Space-Time Convergence" << std::endl;
    analytical::horizontal_advection exact;
    auto error_f = [&comm_world, exact](std::size_t resolution) {
      auto comm_grid =
          communication::grid(comm_world, {resolution, resolution, resolution});
      return run(std::move(comm_grid), hadv_stepper(), 1e-4, 1e-5 / resolution,
                 exact);
    };
    print_order_verification_result(order_verification(error_f, 8, 64));
  }

  {
    std::cout << "VERTICAL ADVECTION: Spatial Convergence" << std::endl;
    analytical::vertical_advection exact;
    auto error_f = [&comm_world, exact](std::size_t resolution) {
      auto comm_grid =
          communication::grid(comm_world, {resolution, resolution, resolution});
      return run(std::move(comm_grid), vadv_stepper(), 1e-5, 1e-6, exact);
    };
    print_order_verification_result(order_verification(error_f, 8, 128));
  }
  {
    std::cout << "VERTICAL ADVECTION: Space-Time Convergence" << std::endl;
    analytical::vertical_advection exact;
    auto error_f = [&comm_world, exact](std::size_t resolution) {
      auto comm_grid =
          communication::grid(comm_world, {resolution, resolution, resolution});
      return run(std::move(comm_grid), vadv_stepper(), 1e-4, 1e-5 / resolution,
                 exact);
    };
    print_order_verification_result(order_verification(error_f, 8, 64));
  }

  {
    std::cout << "RUNGE-KUTTA ADVECTION: Spatial Convergence" << std::endl;
    analytical::full_advection exact;
    auto error_f = [&comm_world, exact](std::size_t resolution) {
      auto comm_grid =
          communication::grid(comm_world, {resolution, resolution, resolution});
      return run(std::move(comm_grid), rkadv_stepper(), 1e-5, 1e-6, exact);
    };
    print_order_verification_result(order_verification(error_f, 8, 128));
  }
  {
    std::cout << "RUNGE-KUTTA ADVECTION: Space-Time Convergence" << std::endl;
    analytical::full_advection exact;
    auto error_f = [&comm_world, exact](std::size_t resolution) {
      auto comm_grid =
          communication::grid(comm_world, {resolution, resolution, resolution});
      return run(std::move(comm_grid), rkadv_stepper(), 1e-4, 1e-5 / resolution,
                 exact);
    };
    print_order_verification_result(order_verification(error_f, 8, 64));
  }

  {
    std::cout << "ADVECTION-DIFFUSION: Spatial Convergence" << std::endl;
    analytical::advection_diffusion exact{0.05};
    auto error_f = [&comm_world, exact](std::size_t resolution) {
      auto comm_grid =
          communication::grid(comm_world, {resolution, resolution, resolution});
      return run(std::move(comm_grid), full_stepper(exact.diffusion_coeff),
                 1e-5, 1e-6, exact);
    };

    print_order_verification_result(order_verification(error_f, 8, 128));
  }
  {
    std::cout << "ADVECTION-DIFFUSION: Space-Time Convergence" << std::endl;
    analytical::advection_diffusion exact{0.05};
    auto error_f = [&comm_world, exact](std::size_t resolution) {
      auto comm_grid =
          communication::grid(comm_world, {resolution, resolution, resolution});
      return run(std::move(comm_grid), full_stepper(exact.diffusion_coeff),
                 1e-4, 1e-5 / resolution, exact);
    };
    print_order_verification_result(order_verification(error_f, 8, 64));
  }
}