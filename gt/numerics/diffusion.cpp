#include "./diffusion.hpp"

#include <gridtools/stencil_composition/expressions/expressions.hpp>
#include <gridtools/stencil_composition/stencil_functions.hpp>

#include "./computation.hpp"
#include "./tridiagonal.hpp"

namespace diffusion {

namespace {
using gt::extent;
using gt::in_accessor;
using gt::inout_accessor;
using gt::make_param_list;
using namespace gt::expressions;

struct stage_horizontal {
  using out = inout_accessor<0>;
  using in = in_accessor<1, extent<-3, 3, -3, 3>>;

  using dx = in_accessor<2>;
  using dy = in_accessor<3>;
  using dt = in_accessor<4>;
  using coeff = in_accessor<5>;

  using param_list = make_param_list<out, in, dx, dy, dt, coeff>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t) {
    constexpr static real_t weights[] = {real_t(-1) / 90,  real_t(5) / 36,
                                         real_t(-49) / 36, real_t(49) / 36,
                                         real_t(-5) / 36,  real_t(1) / 90};

    auto flx_x0 = eval((weights[0] * in(-3, 0) + weights[1] * in(-2, 0) +
                        weights[2] * in(-1, 0) + weights[3] * in(0, 0) +
                        weights[4] * in(1, 0) + weights[5] * in(2, 0)) /
                       dx());
    auto flx_x1 = eval((weights[0] * in(-2, 0) + weights[1] * in(-1, 0) +
                        weights[2] * in(0, 0) + weights[3] * in(1, 0) +
                        weights[4] * in(2, 0) + weights[5] * in(3, 0)) /
                       dx());
    auto flx_y0 = eval((weights[0] * in(0, -3) + weights[1] * in(0, -2) +
                        weights[2] * in(0, -1) + weights[3] * in(0, 0) +
                        weights[4] * in(0, 1) + weights[5] * in(0, 2)) /
                       dy());
    auto flx_y1 = eval((weights[0] * in(0, -2) + weights[1] * in(0, -1) +
                        weights[2] * in(0, 0) + weights[3] * in(0, 1) +
                        weights[4] * in(0, 2) + weights[5] * in(0, 3)) /
                       dy());

    flx_x0 = flx_x0 * eval(in() - in(-1, 0)) < real_t(0) ? real_t(0) : flx_x0;
    flx_x1 = flx_x1 * eval(in(1, 0) - in()) < real_t(0) ? real_t(0) : flx_x1;
    flx_y0 = flx_y0 * eval(in() - in(0, -1)) < real_t(0) ? real_t(0) : flx_y0;
    flx_y1 = flx_y1 * eval(in(0, 1) - in()) < real_t(0) ? real_t(0) : flx_y1;

    eval(out()) =
        eval(in() + coeff() * dt() *
                        ((flx_x1 - flx_x0) / dx() + (flx_y1 - flx_y0) / dy()));
  }
};

struct stage_diffusion_w_forward1 {
  using alpha = inout_accessor<0>;
  using beta = inout_accessor<1>;
  using gamma = inout_accessor<2>;
  using a = inout_accessor<3>;
  using b = inout_accessor<4>;
  using c = inout_accessor<5, extent<0, 0, 0, 0, -1, 0>>;
  using d = inout_accessor<6, extent<0, 0, 0, 0, -1, 0>>;

  using data = in_accessor<7, extent<0, 0, 0, 0, -huge_offset, huge_offset>>;

  using dz = in_accessor<8>;
  using dt = in_accessor<9>;
  using coeff = in_accessor<10>;

  using k_size = in_accessor<11>;

  using param_list = make_param_list<alpha, beta, gamma, a, b, c, d, data, dz,
                                     dt, coeff, k_size>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::first_level) {
    const gt::int_t k_offset = eval(k_size() - 1);

    eval(a()) = eval(c()) = eval(-coeff() / (real_t(2) * dz() * dz()));
    eval(b()) = eval(real_t(1) / dt() - a() - c());
    eval(d()) =
        eval(real_t(1) / dt() * data() +
             real_t(0.5) * coeff() *
                 (data(0, 0, k_offset) - real_t(2) * data() + data(0, 0, 1)) /
                 (dz() * dz()));

    eval(alpha()) = eval(beta()) = eval(-coeff() / (real_t(2) * dz() * dz()));
    eval(gamma()) = eval(-b());

    gridtools::call_proc<tridiagonal::periodic_forward1,
                         full_t::first_level>::with(eval, a(), b(), c(), d(),
                                                    alpha(), beta(), gamma());
  }

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, -1>) {
    eval(a()) = eval(c()) = eval(-coeff() / (real_t(2) * dz() * dz()));
    eval(b()) = eval(real_t(1) / dt() - a() - c());
    eval(d()) = eval(real_t(1) / dt() * data() +
                     real_t(0.5) * coeff() *
                         (data(0, 0, -1) - real_t(2) * data() + data(0, 0, 1)) /
                         (dz() * dz()));

    gridtools::call_proc<tridiagonal::periodic_forward1,
                         full_t::modify<1, -1>>::with(eval, a(), b(), c(), d(),
                                                      alpha(), beta(), gamma());
  }
  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
    const gt::int_t k_offset = eval(k_size() - 1);

    eval(a()) = eval(c()) = eval(-coeff() / (real_t(2) * dz() * dz()));
    eval(b()) = eval(real_t(1) / dt() - a() - c());
    eval(d()) =
        eval(real_t(1) / dt() * data() +
             real_t(0.5) * coeff() *
                 (data(0, 0, -1) - real_t(2) * data() + data(0, 0, -k_offset)) /
                 (dz() * dz()));
    gridtools::call_proc<tridiagonal::periodic_forward1,
                         full_t::last_level>::with(eval, a(), b(), c(), d(),
                                                   alpha(), beta(), gamma());
  }
};

using stage_diffusion_w_backward1 = tridiagonal::periodic_backward1;
using stage_diffusion_w_forward2 = tridiagonal::periodic_forward2;
using stage_diffusion_w_backward2 = tridiagonal::periodic_backward2;

struct stage_diffusion_w3 {
  using out = inout_accessor<0>;
  using x = in_accessor<1>;
  using z = in_accessor<2>;
  using fact = in_accessor<3>;
  using in = in_accessor<4>;

  using dt = in_accessor<5>;

  using param_list = make_param_list<out, x, z, fact, in, dt>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t) {
    gridtools::call_proc<tridiagonal::periodic3, full_t>::with(eval, out(), x(),
                                                               z(), fact());
    // eval(out()) = eval((out() - in()) / dt());
  }
};

} // namespace

horizontal::horizontal(vec<std::size_t, 3> const &resolution,
                       vec<real_t, 3> const &delta, real_t coeff)
    : comp_(gt::make_computation<backend_t>(
          computation_grid(resolution.x, resolution.y, resolution.z),
          p_dx() = gt::make_global_parameter(delta.x),
          p_dy() = gt::make_global_parameter(delta.y),
          p_coeff() = gt::make_global_parameter(coeff),
          gt::make_multistage(
              gt::execute::parallel(),
              gt::make_stage<stage_horizontal>(p_out(), p_in(), p_dx(), p_dy(),
                                               p_dt(), p_coeff())))) {}

void horizontal::operator()(storage_t &out, storage_t const &in, real_t dt) {
  comp_.run(p_out() = out, p_in() = in, p_dt() = gt::make_global_parameter(dt));
}

vertical::vertical(vec<std::size_t, 3> const &resolution,
                   vec<real_t, 3> const &delta, real_t coeff)
    : sinfo_ij_(resolution.x + 2 * halo, resolution.y + 2 * halo, 1),
      alpha_(sinfo_ij_, "alpha"), beta_(sinfo_ij_, "beta"),
      gamma_(sinfo_ij_, "gamma"), fact_(sinfo_ij_, "fact"),
      comp_(gt::make_computation<backend_t>(
          computation_grid(resolution.x, resolution.y, resolution.z),
          p_dz() = gt::make_global_parameter(delta.z),
          p_coeff() = gt::make_global_parameter(coeff), p_alpha() = alpha_,
          p_beta() = beta_, p_gamma() = gamma_, p_fact() = fact_,
          p_k_size() = gt::make_global_parameter(gt::int_t(resolution.z)),
          gt::make_multistage(
              gt::execute::forward(),
              gt::define_caches(
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_a()),
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_b()),
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_c()),
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_d())),
              gt::make_stage<stage_diffusion_w_forward1>(
                  p_alpha(), p_beta(), p_gamma(), p_a(), p_b(), p_c(), p_d(),
                  p_data_in(), p_dz(), p_dt(), p_coeff(), p_k_size())),
          gt::make_multistage(
              gt::execute::backward(),
              gt::define_caches(
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_x())),
              gt::make_stage<stage_diffusion_w_backward1>(p_x(), p_c(), p_d())),
          gt::make_multistage(
              gt::execute::forward(),
              gt::define_caches(
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_c()),
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_d())),
              gt::make_stage<stage_diffusion_w_forward2>(
                  p_a(), p_b(), p_c(), p_d(), p_alpha(), p_gamma())),
          gt::make_multistage(gt::execute::backward(),
                              gt::make_stage<stage_diffusion_w_backward2>(
                                  p_z(), p_c(), p_d(), p_x(), p_beta(),
                                  p_gamma(), p_fact(), p_k_size())),
          gt::make_multistage(gt::execute::parallel(),
                              gt::make_stage<stage_diffusion_w3>(
                                  p_data_out(), p_x(), p_z(), p_fact(),
                                  p_data_in(), p_dt())))) {}

void vertical::operator()(storage_t &out, storage_t const &in, real_t dt) {
  comp_.run(p_data_out() = out, p_data_in() = in,
            p_dt() = gt::make_global_parameter(dt));
}

} // namespace diffusion