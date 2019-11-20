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

  const halos_t halos{
      {{halo, halo, halo, halo + gt::uint_t(resolution) - 1,
        halo + gt::uint_t(resolution) + halo},
       {halo, halo, halo, halo + gt::uint_t(resolution) - 1,
        halo + gt::uint_t(resolution) + halo},
       {0, 0, 0, gt::uint_t(resolution) - 1, gt::uint_t(resolution)}}};

  const auto grid = gt::make_grid(halos[0], halos[1], axis_t{resolution});
  const real_t dx = initial.dx;
  const real_t dy = initial.dy;
  const real_t dz = initial.dz;
  auto step = stepper(grid, halos, dx, dy, dz);

  real_t t;
  for (t = 0; t < tmax; t += dt)
    step(state, dt);

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
  void operator()(solver_state &state, real_t dt) {
    boundary.apply(state.data);
    hdiff(state.data1, state.data, dt);
    std::swap(state.data1, state.data);
  }

  diffusion::horizontal hdiff;
  gt::boundary<periodic_boundary, backend_t> boundary;
};

auto hdiff_stepper(real_t diffusion_coeff) {
  return [diffusion_coeff](auto grid, auto halos, real_t dx, real_t dy,
                           real_t dz) {
    return hdiff_stepper_f{{grid, dx, dy, diffusion_coeff},
                           {halos, periodic_boundary{}}};
  };
}

struct vdiff_stepper_f {
  void operator()(solver_state &state, real_t dt) {
    vdiff(state.data1, state.data, dt);
    std::swap(state.data1, state.data);
  }

  diffusion::vertical vdiff;
};

auto vdiff_stepper(real_t diffusion_coeff) {
  return [diffusion_coeff](auto grid, auto halos, real_t dx, real_t dy,
                           real_t dz) {
    return vdiff_stepper_f{{grid, dz, diffusion_coeff}};
  };
}

struct diff_stepper_f {
  void operator()(solver_state &state, real_t dt) {
    boundary.apply(state.data);
    hdiff(state.data1, state.data, dt);
    vdiff(state.data, state.data1, dt);
  }

  diffusion::horizontal hdiff;
  diffusion::vertical vdiff;
  gt::boundary<periodic_boundary, backend_t> boundary;
};

auto diff_stepper(real_t diffusion_coeff) {
  return [diffusion_coeff](auto grid, auto halos, real_t dx, real_t dy,
                           real_t dz) {
    return diff_stepper_f{{grid, dx, dy, diffusion_coeff},
                          {grid, dz, diffusion_coeff},
                          {halos, periodic_boundary{}}};
  };
}

struct hadv_stepper_f {
  void operator()(solver_state &state, real_t dt) {
    boundary.apply(state.data);
    hadv(state.data1, state.data, state.u, state.v, dt);
    std::swap(state.data1, state.data);
  }

  advection::horizontal hadv;
  gt::boundary<periodic_boundary, backend_t> boundary;
};

auto hadv_stepper() {
  return [](auto grid, auto halos, real_t dx, real_t dy, real_t dz) {
    return hadv_stepper_f{{grid, dx, dy}, {halos, periodic_boundary{}}};
  };
}

struct vadv_stepper_f {
  void operator()(solver_state &state, real_t dt) {
    vadv(state.data1, state.data, state.w, dt);
    std::swap(state.data1, state.data);
  }

  advection::vertical vadv;
};

auto vadv_stepper() {
  return [](auto grid, auto halos, real_t dx, real_t dy, real_t dz) {
    return vadv_stepper_f{{grid, dz}};
  };
}

struct rkadv_stepper_f {
  void operator()(solver_state &state, real_t dt) {
    boundary.apply(state.data);
    rk_step(state.data1, state.data, state.data, state.u, state.v, state.w,
            dt / 3);
    boundary.apply(state.data1);
    rk_step(state.data2, state.data1, state.data, state.u, state.v, state.w,
            dt / 2);
    boundary.apply(state.data2);
    rk_step(state.data, state.data2, state.data, state.u, state.v, state.w, dt);
  }

  advection::runge_kutta_step rk_step;
  gt::boundary<periodic_boundary, backend_t> boundary;
};

auto rkadv_stepper() {
  return [](auto grid, auto halos, real_t dx, real_t dy, real_t dz) {
    return rkadv_stepper_f{{grid, dx, dy, dz}, {halos, periodic_boundary{}}};
  };
}

struct full_stepper_f {
  void operator()(solver_state &state, real_t dt) {
    // VDIFF
    vdiff(state.data1, state.data, dt);
    std::swap(state.data1, state.data);

    // ADV
    boundary.apply(state.data);
    rk_step(state.data1, state.data, state.data, state.u, state.v, state.w,
            dt / 3);
    boundary.apply(state.data1);
    rk_step(state.data2, state.data1, state.data, state.u, state.v, state.w,
            dt / 2);
    boundary.apply(state.data2);
    rk_step(state.data, state.data2, state.data, state.u, state.v, state.w, dt);

    // HDIFF
    boundary.apply(state.data);
    hdiff(state.data1, state.data, dt);
    std::swap(state.data1, state.data);
  }

  diffusion::horizontal hdiff;
  diffusion::vertical vdiff;
  advection::runge_kutta_step rk_step;
  gt::boundary<periodic_boundary, backend_t> boundary;
};

auto full_stepper(real_t diffusion_coeff) {
  return [diffusion_coeff](auto grid, auto halos, real_t dx, real_t dy,
                           real_t dz) {
    return full_stepper_f{{grid, dx, dy, diffusion_coeff},
                          {grid, dz, diffusion_coeff},
                          {grid, dx, dy, dz},
                          {halos, periodic_boundary{}}};
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

  {
    std::cout << "VERTICAL ADVECTION: Spatial Convergence" << std::endl;
    analytical::vertical_advection exact;
    auto error_f = [exact](std::size_t resolution) {
      return run(vadv_stepper(), resolution, 1e-5, 1e-6, exact);
    };
    print_order_verification_result(order_verification(error_f, 4, 128));
  }
  {
    std::cout << "VERTICAL ADVECTION: Space-Time Convergence" << std::endl;
    analytical::vertical_advection exact;
    auto error_f = [exact](std::size_t resolution) {
      return run(vadv_stepper(), resolution, 1e-4, 1e-5 / resolution, exact);
    };
    print_order_verification_result(order_verification(error_f, 8, 64));
  }

  {
    std::cout << "RUNGE-KUTTA ADVECTION: Spatial Convergence" << std::endl;
    analytical::full_advection exact;
    auto error_f = [exact](std::size_t resolution) {
      return run(rkadv_stepper(), resolution, 1e-5, 1e-6, exact);
    };
    print_order_verification_result(order_verification(error_f, 4, 128));
  }
  {
    std::cout << "RUNGE-KUTTA ADVECTION: Space-Time Convergence" << std::endl;
    analytical::full_advection exact;
    auto error_f = [exact](std::size_t resolution) {
      return run(rkadv_stepper(), resolution, 1e-4, 1e-5 / resolution, exact);
    };
    print_order_verification_result(order_verification(error_f, 8, 64));
  }

  {
    std::cout << "ADVECTION-DIFFUSION: Spatial Convergence" << std::endl;
    analytical::advection_diffusion exact{0.05};
    auto error_f = [exact](std::size_t resolution) {
      return run(full_stepper(exact.diffusion_coeff), resolution, 1e-5, 1e-6,
                 exact);
    };
    print_order_verification_result(order_verification(error_f, 4, 128));
  }
  {
    std::cout << "ADVECTION-DIFFUSION: Space-Time Convergence" << std::endl;
    analytical::advection_diffusion exact{0.05};
    auto error_f = [exact](std::size_t resolution) {
      return run(full_stepper(exact.diffusion_coeff), resolution, 1e-4,
                 1e-5 / resolution, exact);
    };
    print_order_verification_result(order_verification(error_f, 8, 64));
  }
}
