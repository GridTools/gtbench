#include "./advection.hpp"

#include <gridtools/stencil_composition/expressions/expressions.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/stencil_composition/stencil_functions.hpp>

#include "./computation.hpp"
#include "./tridiagonal.hpp"

namespace advection {
namespace {
using gt::extent;
using gt::in_accessor;
using gt::inout_accessor;
using gt::make_param_list;
using namespace gt::expressions;

struct stage_u {
  using flux = inout_accessor<0>;
  using u = in_accessor<1>;
  using in = in_accessor<2, extent<-3, 3, 0, 0>>;
  using dx = in_accessor<3>;
  using param_list = make_param_list<flux, u, in, dx>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t) {
    static constexpr real_t weights[] = {real_t(1) / 30, -real_t(1) / 4,
                                         real_t(1),      -real_t(1) / 3,
                                         -real_t(1) / 2, real_t(1) / 20};

    if (eval(u()) < 0) {
      eval(flux()) =
          eval(u() *
               -(weights[0] * in(-3, 0, 0) + weights[1] * in(-2, 0, 0) +
                 weights[2] * in(-1, 0, 0) + weights[3] * in() +
                 weights[4] * in(1, 0, 0) + weights[5] * in(2, 0, 0)) /
               dx());
    } else if (eval(u()) > 0) {
      eval(flux()) =
          eval(u() *
               (weights[5] * in(-2, 0, 0) + weights[4] * in(-1, 0, 0) +
                weights[3] * in() + weights[2] * in(1, 0, 0) +
                weights[1] * in(2, 0, 0) + weights[0] * in(3, 0, 0)) /
               dx());
    } else {
      eval(flux()) = real_t(0);
    }
  }
};
struct stage_v {
  using flux = inout_accessor<0>;
  using v = in_accessor<1>;
  using in = in_accessor<2, extent<0, 0, -3, 3>>;
  using dy = in_accessor<3>;

  using param_list = make_param_list<flux, v, in, dy>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t) {
    static constexpr real_t weights[] = {real_t(1) / 30, -real_t(1) / 4,
                                         real_t(1),      -real_t(1) / 3,
                                         -real_t(1) / 2, real_t(1) / 20};

    if (eval(v()) < 0) {
      eval(flux()) =
          eval(v() *
               -(weights[0] * in(0, -3, 0) + weights[1] * in(0, -2, 0) +
                 weights[2] * in(0, -1, 0) + weights[3] * in() +
                 weights[4] * in(0, 1, 0) + weights[5] * in(0, 2, 0)) /
               dy());
    } else if (eval(v()) > 0) {
      eval(flux()) =
          eval(v() *
               (weights[5] * in(0, -2, 0) + weights[4] * in(0, -1, 0) +
                weights[3] * in() + weights[2] * in(0, 1, 0) +
                weights[1] * in(0, 2, 0) + weights[0] * in(0, 3, 0)) /
               dy());
    } else {
      eval(flux()) = real_t(0);
    }
  }
};

struct stage_horizontal {
  using out = inout_accessor<0>;
  using in = in_accessor<1, extent<-3, 3, -3, 3>>;
  using u = in_accessor<2>;
  using v = in_accessor<3>;

  using dx = in_accessor<4>;
  using dy = in_accessor<5>;
  using dt = in_accessor<6>;

  using param_list = make_param_list<out, in, u, v, dx, dy, dt>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t) {
    auto flx = gridtools::call<stage_u, full_t>::with(eval, u(), in(), dx());
    auto fly = gridtools::call<stage_v, full_t>::with(eval, v(), in(), dy());

    // auto flx = eval(u()) >= 0 ? eval((u() * in() - u(-1, 0) * in(-1, 0)) /
    // dx())
    //                           : eval((u(1, 0) * in(1, 0) - u() * in()) /
    //                           dx());
    // auto fly = eval(v()) >= 0 ? eval((v() * in() - v(0, -1) * in(0, -1)) /
    // dy())
    //                           : eval((v(0, 1) * in(0, 1) - v() * in()) /
    //                           dy());

    eval(out()) = eval(in() - dt() * (flx + fly));
  }
};

struct stage_advection_w_forward1 {
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
  using w = in_accessor<10, extent<0, 0, 0, 0, -1, 1>>;

  using k_size = in_accessor<11>;

  using param_list =
      make_param_list<alpha, beta, gamma, a, b, c, d, data, dz, dt, w, k_size>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::first_level) {
    const gt::int_t k_offset = eval(k_size() - 1);

    eval(a()) = eval(real_t(-0.25) * w() / dz());
    eval(c()) = eval(real_t(0.25) * w(0, 0, 1) / dz());
    eval(b()) = eval(real_t(1) / dt() - a() - c());
    eval(d()) =
        eval(real_t(1) / dt() * data() -
             real_t(0.25) * w(0, 0, 1) * (data(0, 0, 1) - data()) / dz() -
             real_t(0.25) * w() * (data() - data(0, 0, k_offset)) / dz());

    eval(alpha()) = eval(-a());
    eval(beta()) = eval(a());
    eval(gamma()) = eval(-b());

    gridtools::call_proc<tridiagonal::periodic_forward1,
                         full_t::first_level>::with(eval, a(), b(), c(), d(),
                                                    alpha(), beta(), gamma());
  }

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, -1>) {
    eval(a()) = eval(real_t(-0.25) * w() / dz());
    eval(c()) = eval(real_t(0.25) * w(0, 0, 1) / dz());
    eval(b()) = eval(real_t(1) / dt() - a() - c());
    eval(d()) =
        eval(real_t(1) / dt() * data() -
             real_t(0.25) * w(0, 0, 1) * (data(0, 0, 1) - data()) / dz() -
             real_t(0.25) * w() * (data() - data(0, 0, -1)) / dz());

    gridtools::call_proc<tridiagonal::periodic_forward1,
                         full_t::modify<1, -1>>::with(eval, a(), b(), c(), d(),
                                                      alpha(), beta(), gamma());
  }
  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
    const gt::int_t k_offset = eval(k_size() - 1);

    eval(a()) = eval(real_t(-0.25) * w() / dz());
    eval(c()) = eval(real_t(0.25) * w(0, 0, 1) / dz());
    eval(b()) = eval(real_t(1) / dt() - a() - c());
    eval(d()) = eval(real_t(1) / dt() * data() -
                     real_t(0.25) * w(0, 0, 1) *
                         (data(0, 0, -k_offset) - data()) / dz() -
                     real_t(0.25) * w() * (data() - data(0, 0, -1)) / dz());

    gridtools::call_proc<tridiagonal::periodic_forward1,
                         full_t::last_level>::with(eval, a(), b(), c(), d(),
                                                   alpha(), beta(), gamma());
  }
};

using stage_advection_w_backward1 = tridiagonal::periodic_backward1;
using stage_advection_w_forward2 = tridiagonal::periodic_forward2;
using stage_advection_w_backward2 = tridiagonal::periodic_backward2;

struct stage_advection_w3 {
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

struct stage_advection_w3_rk {
  using out = inout_accessor<0>;
  using x = in_accessor<1>;
  using z = in_accessor<2>;
  using fact = in_accessor<3>;
  using in = in_accessor<4, extent<-3, 3, -3, 3>>;
  using in0 = in_accessor<5>;

  using u = in_accessor<6>;
  using v = in_accessor<7>;
  using dx = in_accessor<8>;
  using dy = in_accessor<9>;
  using dt = in_accessor<10>;

  using param_list =
      make_param_list<out, x, z, fact, in, in0, u, v, dx, dy, dt>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t) {
    auto vout = gridtools::call<tridiagonal::periodic3, full_t>::with(
        eval, x(), z(), fact());
    auto flx = gridtools::call<stage_u, full_t>::with(eval, u(), in(), dx());
    auto fly = gridtools::call<stage_v, full_t>::with(eval, v(), in(), dy());
    eval(out()) = eval(in0() - dt() * (flx + fly) + (vout - in()));
  }
};

} // namespace

horizontal::horizontal(vec<std::size_t, 3> const &resolution,
                       vec<real_t, 3> const &delta)
    : comp_(gt::make_computation<backend_t>(
          computation_grid(resolution.x, resolution.y, resolution.z),
          p_dx() = gt::make_global_parameter(delta.x),
          p_dy() = gt::make_global_parameter(delta.y),
          gt::make_multistage(
              gt::execute::parallel(),
              gt::make_stage<stage_horizontal>(p_out(), p_in(), p_u(), p_v(),
                                               p_dx(), p_dy(), p_dt())))) {}

void horizontal::operator()(storage_t &out, storage_t const &in,
                            storage_t const &u, storage_t const &v, real_t dt) {
  comp_.run(p_out() = out, p_in() = in, p_u() = u, p_v() = v,
            p_dt() = gt::make_global_parameter(dt));
}

vertical::vertical(vec<std::size_t, 3> const &resolution,
                   vec<real_t, 3> const &delta)
    : sinfo_ij_(resolution.x + 2 * halo, resolution.y + 2 * halo, 1),
      alpha_(sinfo_ij_, "alpha"), beta_(sinfo_ij_, "beta"),
      gamma_(sinfo_ij_, "gamma"), fact_(sinfo_ij_, "fact"),
      comp_(gt::make_computation<backend_t>(
          computation_grid(resolution.x, resolution.y, resolution.z),
          p_dz() = gt::make_global_parameter(delta.z), p_alpha() = alpha_,
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
              gt::make_stage<stage_advection_w_forward1>(
                  p_alpha(), p_beta(), p_gamma(), p_a(), p_b(), p_c(), p_d(),
                  p_data_in(), p_dz(), p_dt(), p_w(), p_k_size())),
          gt::make_multistage(
              gt::execute::backward(),
              gt::define_caches(
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_x())),
              gt::make_stage<stage_advection_w_backward1>(p_x(), p_c(), p_d())),
          gt::make_multistage(
              gt::execute::forward(),
              gt::define_caches(
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_c()),
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_d())),
              gt::make_stage<stage_advection_w_forward2>(
                  p_a(), p_b(), p_c(), p_d(), p_alpha(), p_gamma())),
          gt::make_multistage(gt::execute::backward(),
                              gt::make_stage<stage_advection_w_backward2>(
                                  p_z(), p_c(), p_d(), p_x(), p_beta(),
                                  p_gamma(), p_fact(), p_k_size())),
          gt::make_multistage(gt::execute::parallel(),
                              gt::make_stage<stage_advection_w3>(
                                  p_data_out(), p_x(), p_z(), p_fact(),
                                  p_data_in(), p_dt())))) {}

void vertical::operator()(storage_t &out, storage_t const &in,
                          storage_t const &w, real_t dt) {
  comp_.run(p_data_out() = out, p_data_in() = in, p_w() = w,
            p_dt() = gt::make_global_parameter(dt));
}

runge_kutta_step::runge_kutta_step(vec<std::size_t, 3> const &resolution,
                                   vec<real_t, 3> const &delta)
    : sinfo_ij_(resolution.x + 2 * halo, resolution.y + 2 * halo, 1),
      alpha_(sinfo_ij_, "alpha"), beta_(sinfo_ij_, "beta"),
      gamma_(sinfo_ij_, "gamma"), fact_(sinfo_ij_, "fact"),
      comp_(gt::make_computation<backend_t>(
          computation_grid(resolution.x, resolution.y, resolution.z),
          p_dx() = gt::make_global_parameter(delta.x),
          p_dy() = gt::make_global_parameter(delta.y),
          p_dz() = gt::make_global_parameter(delta.z), p_alpha() = alpha_,
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
              gt::make_stage<stage_advection_w_forward1>(
                  p_alpha(), p_beta(), p_gamma(), p_a(), p_b(), p_c(), p_d(),
                  p_data_in(), p_dz(), p_dt(), p_w(), p_k_size())),
          gt::make_multistage(
              gt::execute::backward(),
              gt::define_caches(
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_x())),
              gt::make_stage<stage_advection_w_backward1>(p_x(), p_c(), p_d())),
          gt::make_multistage(
              gt::execute::forward(),
              gt::define_caches(
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_c()),
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_d())),
              gt::make_stage<stage_advection_w_forward2>(
                  p_a(), p_b(), p_c(), p_d(), p_alpha(), p_gamma())),
          gt::make_multistage(gt::execute::backward(),
                              gt::make_stage<stage_advection_w_backward2>(
                                  p_z(), p_c(), p_d(), p_x(), p_beta(),
                                  p_gamma(), p_fact(), p_k_size())),
          gt::make_multistage(gt::execute::parallel(),
                              gt::make_stage<stage_advection_w3_rk>(
                                  p_data_out(), p_x(), p_z(), p_fact(),
                                  p_data_in(), p_data_in0(), p_u(), p_v(),
                                  p_dx(), p_dy(), p_dt())))) {}

void runge_kutta_step::operator()(storage_t &out, storage_t const &in,
                                  storage_t const &in0, storage_t const &u,
                                  storage_t const &v, storage_t const &w,
                                  real_t dt) {
  comp_.run(p_data_out() = out, p_data_in() = in, p_data_in0() = in0, p_u() = u,
            p_v() = v, p_w() = w, p_dt() = gt::make_global_parameter(dt));
}

} // namespace advection