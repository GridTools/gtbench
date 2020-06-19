/*
 * gtbench
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * Copyright (c) 2020, NVIDIA
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "./advection.hpp"

#include <gridtools/stencil_composition/expressions/expressions.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/stencil_composition/stencil_functions.hpp>

#include "./computation.hpp"

namespace numerics {
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
    static constexpr real_t weights[] = {1_r / 30, -1_r / 4, 1_r,
                                         -1_r / 3, -1_r / 2, 1_r / 20};

    if (eval(u()) > 0_r) {
      eval(flux()) =
          eval(u() *
               -(weights[0] * in(-3, 0, 0) + weights[1] * in(-2, 0, 0) +
                 weights[2] * in(-1, 0, 0) + weights[3] * in() +
                 weights[4] * in(1, 0, 0) + weights[5] * in(2, 0, 0)) /
               dx());
    } else if (eval(u()) < 0_r) {
      eval(flux()) =
          eval(u() *
               (weights[5] * in(-2, 0, 0) + weights[4] * in(-1, 0, 0) +
                weights[3] * in() + weights[2] * in(1, 0, 0) +
                weights[1] * in(2, 0, 0) + weights[0] * in(3, 0, 0)) /
               dx());
    } else {
      eval(flux()) = 0_r;
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
    static constexpr real_t weights[] = {1_r / 30, -1_r / 4, 1_r,
                                         -1_r / 3, -1_r / 2, 1_r / 20};

    if (eval(v()) > 0_r) {
      eval(flux()) =
          eval(v() *
               -(weights[0] * in(0, -3, 0) + weights[1] * in(0, -2, 0) +
                 weights[2] * in(0, -1, 0) + weights[3] * in() +
                 weights[4] * in(0, 1, 0) + weights[5] * in(0, 2, 0)) /
               dy());
    } else if (eval(v()) < 0_r) {
      eval(flux()) =
          eval(v() *
               (weights[5] * in(0, -2, 0) + weights[4] * in(0, -1, 0) +
                weights[3] * in() + weights[2] * in(0, 1, 0) +
                weights[1] * in(0, 2, 0) + weights[0] * in(0, 3, 0)) /
               dy());
    } else {
      eval(flux()) = 0_r;
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

    eval(out()) = eval(in() - dt() * (flx + fly));
  }
};

struct stage_advection_w_forward {
  using alpha = inout_accessor<0>;
  using gamma = inout_accessor<1>;
  using c = inout_accessor<2, extent<0, 0, 0, 0, -1, 0>>;
  using d = inout_accessor<3, extent<0, 0, 0, 0, -1, 0>>;
  using c2 = inout_accessor<4, extent<0, 0, 0, 0, -1, 0>>;
  using d2 = inout_accessor<5, extent<0, 0, 0, 0, -1, 0>>;

  using data = in_accessor<
      6, extent<0, 0, 0, 0, -150 /* RANDOM NUMBER */, 150 /* RANDOM NUMBER */>>;

  using dz = in_accessor<7>;
  using dt = in_accessor<8>;
  using w = in_accessor<9, extent<0, 0, 0, 0, 0, 1>>;
  using data_p1_k_cache = inout_accessor<10, extent<0, 0, 0, 0, -2, 0>>;
  using a_c_cache = inout_accessor<11>;

  using k_size = in_accessor<12>;

  using param_list = make_param_list<alpha, gamma, c, d, c2, d2, data, dz, dt,
                                     w, data_p1_k_cache, a_c_cache, k_size>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::first_level) {
    auto k_offset = eval(k_size()) - 1;

    eval(data_p1_k_cache()) = eval(data(0, 0, 1));

    eval(a_c_cache()) = eval(0.25_r * w(0, 0, 1) / dz());

    auto a = eval(-0.25_r * w() / dz());
    eval(c()) = eval(a_c_cache());
    auto b = eval(1_r / dt() - a - c());
    eval(d()) = eval(1_r / dt() * data() - c() * (data_p1_k_cache() - data()) +
                     a * (data() - data(0, 0, k_offset)));

    eval(alpha()) = -a;
    eval(gamma()) = -b;

    b *= 2;
    eval(c()) = eval(c() / b);
    eval(d()) = eval(d() / b);

    eval(c2()) = eval(c() / b);
    eval(d2()) = eval(gamma() / b);
  }

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval,
                                full_t::first_level::shift<1>) {
    eval(data_p1_k_cache()) = eval(data(0, 0, 1));

    auto old_a_c_cache = eval(a_c_cache());
    eval(a_c_cache()) = eval(0.25_r * w(0, 0, 1) / dz());

    auto a = -old_a_c_cache;
    eval(c()) = eval(a_c_cache());
    auto b = eval(1_r / dt() - a - c());
    eval(d()) = eval(1_r / dt() * data_p1_k_cache(0, 0, -1) -
                     c() * (data_p1_k_cache() - data_p1_k_cache(0, 0, -1)) +
                     a * (data_p1_k_cache(0, 0, -1) - data(0, 0, -1)));

    eval(c()) = eval(c() / (b - c(0, 0, -1) * a));
    eval(d()) = eval((d() - a * d(0, 0, -1)) / (b - c(0, 0, -1) * a));

    eval(c2()) = eval(c() / (b - c2(0, 0, -1) * a));
    eval(d2()) = eval((-a * d2(0, 0, -1)) / (b - c2(0, 0, -1) * a));
  }
  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::modify<2, -1>) {
    eval(data_p1_k_cache()) = eval(data(0, 0, 1));

    auto old_a_c_cache = eval(a_c_cache());
    eval(a_c_cache()) = eval(0.25_r * w(0, 0, 1) / dz());

    auto a = -old_a_c_cache;
    eval(c()) = eval(a_c_cache());
    auto b = eval(1_r / dt() - a - c());
    eval(d()) =
        eval(1_r / dt() * data_p1_k_cache(0, 0, -1) -
             c() * (data_p1_k_cache() - data_p1_k_cache(0, 0, -1)) +
             a * (data_p1_k_cache(0, 0, -1) - data_p1_k_cache(0, 0, -2)));

    eval(c()) = eval(c() / (b - c(0, 0, -1) * a));
    eval(d()) = eval((d() - a * d(0, 0, -1)) / (b - c(0, 0, -1) * a));

    eval(c2()) = eval(c() / (b - c2(0, 0, -1) * a));
    eval(d2()) = eval((-a * d2(0, 0, -1)) / (b - c2(0, 0, -1) * a));
  }
  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
    auto k_offset = eval(k_size()) - 1;

    auto a = eval(-a_c_cache());
    eval(c()) = eval(0.25_r * w(0, 0, 1) / dz());
    auto b = eval(1_r / dt() - a - c());
    eval(d()) =
        eval(1_r / dt() * data_p1_k_cache(0, 0, -1) -
             c() * (data(0, 0, -k_offset) - data_p1_k_cache(0, 0, -1)) +
             a * (data_p1_k_cache(0, 0, -1) - data_p1_k_cache(0, 0, -2)));

    b += eval(alpha() * alpha() / gamma());
    eval(c()) = eval(c() / (b - c(0, 0, -1) * a));
    eval(d()) = eval((d() - a * d(0, 0, -1)) / (b - c(0, 0, -1) * a));

    eval(c2()) = eval(c() / (b - c2(0, 0, -1) * a));
    eval(d2()) = eval((alpha() - a * d2(0, 0, -1)) / (b - c2(0, 0, -1) * a));
  }
};

struct stage_advection_w_backward {
  using c = in_accessor<0>;
  using d = inout_accessor<1, extent<0, 0, 0, 0, 0, 1>>;
  using c2 = in_accessor<2>;
  using d2 = inout_accessor<3, extent<0, 0, 0, 0, 0, 1>>;

  using alpha = in_accessor<4>;
  using gamma = in_accessor<5>;

  using fact = inout_accessor<6>;

  using d_uncached =
      in_accessor<7, extent<0, 0, 0, 0, 0, 150 /* RANDOM NUMBER */>>;
  using d2_uncached =
      in_accessor<8, extent<0, 0, 0, 0, 0, 150 /* RANDOM NUMBER */>>;

  using k_size = in_accessor<9>;

  using param_list = make_param_list<c, d, c2, d2, alpha, gamma, fact,
                                     d_uncached, d2_uncached, k_size>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::first_level) {
    auto k_offset = eval(k_size() - 1);

    eval(d()) = eval(d() - c() * d(0, 0, 1));

    eval(d2()) = eval(d2() - c2() * d2(0, 0, 1));
    eval(fact()) =
        eval((d() - alpha() * d_uncached(0, 0, k_offset) / gamma()) /
             (1_r + d2() - alpha() * d2_uncached(0, 0, k_offset) / gamma()));
  }

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, -1>) {
    eval(d()) = eval(d() - c() * d(0, 0, 1));
    eval(d2()) = eval(d2() - c2() * d2(0, 0, 1));
  }
};

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
    eval(out()) = eval(x() - fact() * z());
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
    auto vout = eval(x() - fact() * z());
    auto flx = gridtools::call<stage_u, full_t>::with(eval, u(), in(), dx());
    auto fly = gridtools::call<stage_v, full_t>::with(eval, v(), in(), dy());
    eval(out()) = eval(in0() - dt() * (flx + fly) + (vout - in()));
  }
};

} // namespace

horizontal::horizontal(vec<std::size_t, 3> const &resolution,
                       vec<real_t, 3> const &delta)
    : comp_(gt::make_computation<backend_t<32, 6>>(
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
      sinfo_(resolution.x + 2 * halo, resolution.y + 2 * halo, resolution.z),
      alpha_(sinfo_ij_, "alpha"), gamma_(sinfo_ij_, "gamma"),
      fact_(sinfo_ij_, "fact"), d_(sinfo_, "d"), d2_(sinfo_, "d2"),
      comp_(gt::make_computation<backend_t<16, 16>>(
          computation_grid(resolution.x, resolution.y, resolution.z),
          p_k_size() = gt::make_global_parameter((int)resolution.z),
          p_dz() = gt::make_global_parameter(delta.z), p_alpha() = alpha_,
          p_gamma() = gamma_, p_fact() = fact_, p_d() = d_, p_d_uncached() = d_,
          p_d2() = d2_, p_d2_uncached() = d2_,
          gt::make_multistage(
              gt::execute::forward(),
              gt::define_caches(
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_c()),
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_d()),
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_c2()),
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_d2()),
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::fill>(
                      p_w()),
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::local>(
                      p_data_p1_k_cache()),
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::local>(
                      p_a_c_cache())),
              gt::make_stage<stage_advection_w_forward>(
                  p_alpha(), p_gamma(), p_c(), p_d(), p_c2(), p_d2(),
                  p_data_in(), p_dz(), p_dt(), p_w(), p_data_p1_k_cache(),
                  p_a_c_cache(), p_k_size())),
          gt::make_multistage(
              gt::execute::backward(),
              gt::define_caches(
                  gt::cache<gt::cache_type::k,
                            gt::cache_io_policy::fill_and_flush>(p_d()),
                  gt::cache<gt::cache_type::k,
                            gt::cache_io_policy::fill_and_flush>(p_d2())),
              gt::make_stage<stage_advection_w_backward>(
                  p_c(), p_d(), p_c2(), p_d2(), p_alpha(), p_gamma(), p_fact(),
                  p_d_uncached(), p_d2_uncached(), p_k_size())),
          gt::make_multistage(gt::execute::parallel(),
                              gt::make_stage<stage_advection_w3>(
                                  p_data_out(), p_d(), p_d2(), p_fact(),
                                  p_data_in(), p_dt())))) {}

void vertical::operator()(storage_t &out, storage_t const &in,
                          storage_t const &w, real_t dt) {
  comp_.run(p_data_out() = out, p_data_in() = in, p_w() = w,
            p_dt() = gt::make_global_parameter(dt));
}

runge_kutta_step::runge_kutta_step(vec<std::size_t, 3> const &resolution,
                                   vec<real_t, 3> const &delta)
    : sinfo_(resolution.x + 2 * halo, resolution.y + 2 * halo, resolution.z),
      sinfo_ij_(resolution.x + 2 * halo, resolution.y + 2 * halo, 1),
      alpha_(sinfo_ij_, "alpha"), gamma_(sinfo_ij_, "gamma"),
      fact_(sinfo_ij_, "fact"), d_(sinfo_, "d"), d2_(sinfo_, "d2"),
      comp1_(gt::make_computation<backend_t<32, 6>>(
          computation_grid(resolution.x, resolution.y, resolution.z),
          p_k_size() = gt::make_global_parameter((int)resolution.z),
          p_dz() = gt::make_global_parameter(delta.z), p_alpha() = alpha_,
          p_d() = d_, p_d2() = d2_, p_gamma() = gamma_, p_fact() = fact_,
          p_d_uncached() = d_, p_d2_uncached() = d2_,
          gt::make_multistage(
              gt::execute::forward(),
              gt::define_caches(
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_c()),
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_d()),
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_c2()),
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_d2()),
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::local>(
                      p_data_p1_k_cache()),
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::local>(
                      p_a_c_cache())),
              gt::make_stage<stage_advection_w_forward>(
                  p_alpha(), p_gamma(), p_c(), p_d(), p_c2(), p_d2(),
                  p_data_in(), p_dz(), p_dt(), p_w(), p_data_p1_k_cache(),
                  p_a_c_cache(), p_k_size())),
          gt::make_multistage(
              gt::execute::backward(),
              gt::define_caches(
                  gt::cache<gt::cache_type::k,
                            gt::cache_io_policy::fill_and_flush>(p_d()),
                  gt::cache<gt::cache_type::k,
                            gt::cache_io_policy::fill_and_flush>(p_d2())),
              gt::make_stage<stage_advection_w_backward>(
                  p_c(), p_d(), p_c2(), p_d2(), p_alpha(), p_gamma(), p_fact(),
                  p_d_uncached(), p_d2_uncached(), p_k_size())))),
      comp2_(gt::make_computation<backend_t<16, 16>>(
          computation_grid(resolution.x, resolution.y, resolution.z),
          p_dx() = gt::make_global_parameter(delta.x),
          p_dy() = gt::make_global_parameter(delta.y), p_d() = d_, p_d2() = d2_,
          p_fact() = fact_,
          gt::make_multistage(gt::execute::parallel(),
                              gt::make_stage<stage_advection_w3_rk>(
                                  p_data_out(), p_d(), p_d2(), p_fact(),
                                  p_data_in(), p_data_in0(), p_u(), p_v(),
                                  p_dx(), p_dy(), p_dt())))) {}

void runge_kutta_step::operator()(storage_t &out, storage_t const &in,
                                  storage_t const &in0, storage_t const &u,
                                  storage_t const &v, storage_t const &w,
                                  real_t dt) {
  comp1_.run(p_data_in() = in, p_w() = w,
             p_dt() = gt::make_global_parameter(dt));
  comp2_.run(p_data_out() = out, p_data_in() = in, p_data_in0() = in0,
             p_u() = u, p_v() = v, p_dt() = gt::make_global_parameter(dt));
}

} // namespace advection
} // namespace numerics
