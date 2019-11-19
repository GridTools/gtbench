#include <cmath>
#include <fstream>
#include <iostream>

#include "advection.hpp"
#include "boundary.hpp"
#include "diffusion.hpp"
#include "solver_state.hpp"
#include "verification/analytical.hpp"
#include "verification/convergence.hpp"

template <typename T> void dump(std::ostream &os, T const &storage) {
  for (auto const &l : storage.total_lengths())
    os << l << " ";
  os << '\n';

  auto v = gt::make_host_view(storage);
  for (int i0 = 0; i0 < storage.template total_length<2>(); ++i0) {
    for (int i1 = 0; i1 < storage.template total_length<1>(); ++i1) {
      for (int i2 = 0; i2 < storage.template total_length<0>(); ++i2)
        os << v(i2, i1, i0) << " ";
      os << '\n';
    }
    os << '\n';
  }
}

template <class Stepper, class Analytical>
double run(Stepper &&stepper, std::size_t resolution, real_t tmax, real_t dt,
           Analytical &&exact) {
  const auto initial =
      analytical::to_domain(exact, resolution, resolution, resolution, 0);
  solver_state state{resolution,  resolution,  resolution, initial.data(),
                     initial.u(), initial.v(), initial.w()};

  const gt::array<gt::halo_descriptor, 3> halos{
      {{halo, halo, halo, halo + gt::uint_t(resolution) - 1,
        halo + gt::uint_t(resolution) + halo},
       {halo, halo, halo, halo + gt::uint_t(resolution) - 1,
        halo + gt::uint_t(resolution) + halo},
       {0, 0, 0, gt::uint_t(resolution) - 1, gt::uint_t(resolution)}}};

  const auto grid = gt::make_grid(halos[0], halos[1], axis_t{resolution});
  const real_t dx = initial.dx;
  const real_t dy = initial.dy;
  const real_t dz = initial.dz;
  auto step = stepper(grid, halos, dx, dy, dz, dt);

  real_t t;
  for (t = 0; t < tmax; t += dt)
    step(state);

  auto view = gt::make_host_view(state.data);

  const auto expected =
      analytical::to_domain(exact, resolution, resolution, resolution, t)
          .data();
  double error = 0.0;
#pragma omp parallel for reduction(+ : error)
  for (std::size_t i = halo; i < halo + resolution; ++i) {
    for (std::size_t j = halo; j < halo + resolution; ++j) {
      for (std::size_t k = 0; k < resolution; ++k) {
        double diff = view(i, j, k) - expected(i, j, k);
        error += diff * diff;
      }
    }
  }

  return std::sqrt(error * dx * dy * dz);
}

struct hdiff_stepper_f {
  void operator()(solver_state &state) {
    boundary.apply(state.data);
    hdiff(state);
  }

  diffusion::horizontal hdiff;
  gt::boundary<periodic_boundary, backend_t> boundary;
};

auto hdiff_stepper(real_t diffusion_coeff) {
  return [diffusion_coeff](auto grid, auto halos, real_t dx, real_t dy,
                           real_t dz, real_t dt) {
    return hdiff_stepper_f{{grid, dx, dy, dt, diffusion_coeff},
                           {halos, periodic_boundary{}}};
  };
}

auto vdiff_stepper(real_t diffusion_coeff) {
  return [diffusion_coeff](auto grid, auto halos, real_t dx, real_t dy,
                           real_t dz, real_t dt) {
    return diffusion::vertical{grid, dz, dt, diffusion_coeff};
  };
}

struct diff_stepper_f {
  void operator()(solver_state &state) {
    boundary.apply(state.data);
    hdiff(state);
    vdiff(state);
  }

  diffusion::horizontal hdiff;
  diffusion::vertical vdiff;
  gt::boundary<periodic_boundary, backend_t> boundary;
};

auto diff_stepper(real_t diffusion_coeff) {
  return [diffusion_coeff](auto grid, auto halos, real_t dx, real_t dy,
                           real_t dz, real_t dt) {
    return diff_stepper_f{{grid, dx, dy, dt, diffusion_coeff},
                          {grid, dz, dt, diffusion_coeff},
                          {halos, periodic_boundary{}}};
  };
}

struct hadv_stepper_f {
  void operator()(solver_state &state) {
    boundary.apply(state.data);
    hadv(state);
  }

  advection::horizontal hadv;
  gt::boundary<periodic_boundary, backend_t> boundary;
};

auto hadv_stepper() {
  return [](auto grid, auto halos, real_t dx, real_t dy, real_t dz, real_t dt) {
    return hadv_stepper_f{{grid, dx, dy, dt}, {halos, periodic_boundary{}}};
  };
}

int main() {
  {
    std::cout << "HORIZONTAL DIFFUSION: Spatial Convergence" << std::endl;
    analytical::horizontal_diffusion exact{0.05};
    auto error_f = [exact](std::size_t resolution) {
      return run(hdiff_stepper(exact.diffusion_coeff), resolution, 1e-4, 1e-5,
                 exact);
    };
    print_order_verification_result(order_verification(error_f, 8, 128));
  }
  {
    std::cout << "HORIZONTAL DIFFUSION: Space-Time Convergence" << std::endl;
    analytical::horizontal_diffusion exact{0.05};
    auto error_f = [exact](std::size_t resolution) {
      return run(hdiff_stepper(exact.diffusion_coeff), resolution, 1e-2,
                 1e-3 / resolution, exact);
    };
    print_order_verification_result(order_verification(error_f, 8, 64));
  }

  {
    std::cout << "VERTICAL DIFFUSION: Spatial Convergence" << std::endl;
    analytical::vertical_diffusion exact{0.05};
    auto error_f = [exact](std::size_t resolution) {
      return run(vdiff_stepper(exact.diffusion_coeff), resolution, 1e-4, 1e-5,
                 exact);
    };
    print_order_verification_result(order_verification(error_f, 8, 128));
  }
  {
    std::cout << "VERTICAL DIFFUSION: Space-Time Convergence" << std::endl;
    analytical::vertical_diffusion exact{0.05};
    auto error_f = [exact](std::size_t resolution) {
      return run(vdiff_stepper(exact.diffusion_coeff), resolution, 1e-2,
                 1e-3 / resolution, exact);
    };
    print_order_verification_result(order_verification(error_f, 8, 64));
  }

  {
    std::cout << "FULL DIFFUSION: Spatial Convergence" << std::endl;
    analytical::full_diffusion exact{0.05};
    auto error_f = [exact](std::size_t resolution) {
      return run(diff_stepper(exact.diffusion_coeff), resolution, 1e-4, 1e-5,
                 exact);
    };
    print_order_verification_result(order_verification(error_f, 8, 128));
  }
  {
    std::cout << "FULL DIFFUSION: Space-Time Convergence" << std::endl;
    analytical::full_diffusion exact{0.05};
    auto error_f = [exact](std::size_t resolution) {
      return run(diff_stepper(exact.diffusion_coeff), resolution, 1e-2,
                 1e-3 / resolution, exact);
    };
    print_order_verification_result(order_verification(error_f, 8, 64));
  }

  {
    std::cout << "HORIZONTAL ADVECTION: Spatial Convergence" << std::endl;
    analytical::horizontal_advection exact;
    auto error_f = [exact](std::size_t resolution) {
      return run(hadv_stepper(), resolution, 1e-5, 1e-6, exact);
    };
    print_order_verification_result(order_verification(error_f, 4, 128));
  }
  {
    std::cout << "HORIZONTAL ADVECTION: Space-Time Convergence" << std::endl;
    analytical::horizontal_advection exact;
    auto error_f = [exact](std::size_t resolution) {
      return run(hadv_stepper(), resolution, 1e-4, 1e-5 / resolution, exact);
    };
    print_order_verification_result(order_verification(error_f, 8, 64));
  }
}
