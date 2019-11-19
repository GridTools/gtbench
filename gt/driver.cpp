#include <cmath>
#include <fstream>
#include <iostream>

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
      {{halo_i, halo_i, halo_i, halo_i + gt::uint_t(resolution) - 1,
        halo_i + gt::uint_t(resolution) + halo_i},
       {halo_j, halo_j, halo_j, halo_j + gt::uint_t(resolution) - 1,
        halo_j + gt::uint_t(resolution) + halo_j},
       {halo_k, halo_k, halo_k, halo_k + gt::uint_t(resolution) - 1,
        halo_k + gt::uint_t(resolution) + halo_k}}};

  const auto grid =
      gt::make_grid(halos[0], halos[1], axis_t{resolution + 2 * halo_k});
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
  for (std::size_t i = halo_i; i < halo_i + resolution; ++i) {
    for (std::size_t j = halo_j; j < halo_j + resolution; ++j) {
      for (std::size_t k = halo_k; k < halo_k + resolution; ++k) {
        double diff = view(i, j, k) - expected(i, j, k);
        error += diff * diff * dx * dy * dz;
      }
    }
  }
  return std::sqrt(error);
}

struct hdiff_stepper_f {
  void operator()(solver_state &state) {
    hdiff(state);
    boundary.apply(state.data);
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

struct vdiff_stepper_f {
  void operator()(solver_state &state) {
    vdiff(state);
    boundary.apply(state.data);
  }

  diffusion::vertical vdiff;
  gt::boundary<periodic_boundary, backend_t> boundary;
};

auto vdiff_stepper(real_t diffusion_coeff) {
  return [diffusion_coeff](auto grid, auto halos, real_t dx, real_t dy,
                           real_t dz, real_t dt) {
    return vdiff_stepper_f{{grid, dz, dt, diffusion_coeff},
                           {halos, periodic_boundary{}}};
  };
}

int main() {
  std::cout << "HORIZONTAL DIFFUSION" << std::endl;

  analytical::horizontal_diffusion exact{0.01};
  auto error_f = [exact](std::size_t resolution) {
    return run(hdiff_stepper(exact.diffusion_coeff), resolution, 1e-3, 1e-4,
               exact);
  };

  std::vector<std::size_t> ns;
  std::vector<double> errors, orders;
  print_order_verification_result(order_verification(error_f, 4, 128));
}
