/*
 * gtbench
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "./diffusion.hpp"

#include <gridtools/stencil_composition/expressions/expressions.hpp>
#include <gridtools/stencil_composition/stencil_functions.hpp>

#include "./computation.hpp"

namespace numerics {
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
    constexpr static real_t weights[] = {-1_r / 90, 5_r / 36,  -49_r / 36,
                                         49_r / 36, -5_r / 36, 1_r / 90};

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

    flx_x0 = flx_x0 * eval(in() - in(-1, 0)) < 0_r ? 0_r : flx_x0;
    flx_x1 = flx_x1 * eval(in(1, 0) - in()) < 0_r ? 0_r : flx_x1;
    flx_y0 = flx_y0 * eval(in() - in(0, -1)) < 0_r ? 0_r : flx_y0;
    flx_y1 = flx_y1 * eval(in(0, 1) - in()) < 0_r ? 0_r : flx_y1;

    eval(out()) =
        eval(in() + coeff() * dt() *
                        ((flx_x1 - flx_x0) / dx() + (flx_y1 - flx_y0) / dy()));
  }
};

struct stage_diffusion_w_forward {
  using c = inout_accessor<0, extent<0, 0, 0, 0, -1, 0>>;
  using d = inout_accessor<1, extent<0, 0, 0, 0, -1, 0>>;
  using c2 = inout_accessor<2, extent<0, 0, 0, 0, -1, 0>>;
  using d2 = inout_accessor<3, extent<0, 0, 0, 0, -1, 0>>;

  using data = in_accessor<4, extent<0, 0, 0, 0, -1, 1>>;
  using data_uncached =
      in_accessor<5, extent<0, 0, 0, 0, -infinite_extent, infinite_extent>>;

  using dz = in_accessor<6>;
  using dt = in_accessor<7>;
  using coeff = in_accessor<8>;

  using k_size = in_accessor<9>;

  using param_list =
      make_param_list<c, d, c2, d2, data, data_uncached, dz, dt, coeff, k_size>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::first_level) {
    auto k_offset = eval(k_size()) - 1;

    auto ac = eval(-coeff() / (2_r * dz() * dz()));
    auto b = eval(1_r / dt() - 2 * ac);

    eval(d()) = eval(1_r / dt() * data() + 0.5_r * coeff() *
                                               (data_uncached(0, 0, k_offset) -
                                                2_r * data() + data(0, 0, 1)) /
                                               (dz() * dz()));

    b *= 2;
    eval(c()) = ac / b;
    eval(d()) = eval(d() / b);

    eval(c2()) = eval(c() / b);
    eval(d2()) = -0.5_r;
  }

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, -1>) {
    auto ac = eval(-coeff() / (2_r * dz() * dz()));
    auto b = eval(1_r / dt() - 2 * ac);

    eval(d()) =
        eval(1_r / dt() * data() +
             0.5_r * coeff() * (data(0, 0, -1) - 2_r * data() + data(0, 0, 1)) /
                 (dz() * dz()));

    eval(c()) = eval(ac / (b - c(0, 0, -1) * ac));
    eval(d()) = eval((d() - ac * d(0, 0, -1)) / (b - c(0, 0, -1) * ac));

    eval(c2()) = eval(c() / (b - c2(0, 0, -1) * ac));
    eval(d2()) = eval((-ac * d2(0, 0, -1)) / (b - c2(0, 0, -1) * ac));
  }
  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
    auto k_offset = eval(k_size()) - 1;

    auto ac = eval(-coeff() / (2_r * dz() * dz()));
    auto b = eval(1_r / dt() - 2 * ac);

    eval(d()) =
        eval(1_r / dt() * data() + 0.5_r * coeff() *
                                       (data(0, 0, -1) - 2_r * data() +
                                        data_uncached(0, 0, -k_offset)) /
                                       (dz() * dz()));

    b += ac * ac / b;
    eval(c()) = eval(ac / (b - c(0, 0, -1) * ac));
    eval(d()) = eval((d() - ac * d(0, 0, -1)) / (b - c(0, 0, -1) * ac));

    eval(c2()) = eval(c() / (b - c2(0, 0, -1) * ac));
    eval(d2()) = eval((ac - ac * d2(0, 0, -1)) / (b - c2(0, 0, -1) * ac));
  }
};
struct stage_diffusion_w_backward {
  using c = inout_accessor<0>;
  using d = inout_accessor<1, extent<0, 0, 0, 0, 0, 1>>;
  using c2 = inout_accessor<2>;
  using d2 = inout_accessor<3, extent<0, 0, 0, 0, 0, 1>>;

  using fact = inout_accessor<4>;

  using d_uncached = in_accessor<5, extent<0, 0, 0, 0, 0, infinite_extent>>;
  using d2_uncached = in_accessor<6, extent<0, 0, 0, 0, 0, infinite_extent>>;

  using dz = in_accessor<7>;
  using dt = in_accessor<8>;
  using coeff = in_accessor<9>;
  using k_size = in_accessor<10>;

  using param_list = make_param_list<c, d, c2, d2, fact, d_uncached,
                                     d2_uncached, dz, dt, coeff, k_size>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::first_level) {
    auto k_offset = eval(k_size() - 1);
    auto beta = eval(-coeff() / (2_r * dz() * dz()));
    auto gamma = -eval(1_r / dt() - 2 * beta);

    eval(d()) = eval(d() - c() * d(0, 0, 1));
    eval(d2()) = eval(d2() - c2() * d2(0, 0, 1));

    eval(fact()) =
        eval((d() + beta * d_uncached(0, 0, k_offset) / gamma) /
             (1_r + d2() + beta * d2_uncached(0, 0, k_offset) / gamma));
  }

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, -1>) {
    eval(d()) = eval(d() - c() * d(0, 0, 1));
    eval(d2()) = eval(d2() - c2() * d2(0, 0, 1));
  }
};

struct stage_diffusion_w3 {
  using out = inout_accessor<0>;
  using x = in_accessor<1>;
  using z = in_accessor<2>;
  using fact = in_accessor<3>;

  using param_list = make_param_list<out, x, z, fact>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t) {
    eval(out()) = eval(x() - fact() * z());
  }
};

} // namespace

horizontal::horizontal(vec<std::size_t, 3> const &resolution,
                       vec<real_t, 3> const &delta, real_t coeff)
    : comp_(gt::make_computation<backend_t<16, 12>>(
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
      sinfo_(resolution.x + 2 * halo, resolution.y + 2 * halo, resolution.z),
      fact_(sinfo_ij_, "fact"), d_(sinfo_, "d"), d2_(sinfo_, "d2"),
      comp1_(gt::make_computation<backend_t<32, 6>>(
          computation_grid(resolution.x, resolution.y, resolution.z),
          p_k_size() = gt::make_global_parameter((int)resolution.z),
          p_dz() = gt::make_global_parameter(delta.z),
          p_coeff() = gt::make_global_parameter(coeff), p_fact() = fact_,
          p_d() = d_, p_d_uncached() = d_, p_d2() = d2_, p_d2_uncached() = d2_,
          gt::make_multistage(
              gt::execute::forward(),
              gt::define_caches(
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::fill>(
                      p_data_in()),
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_c()),
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_d()),
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_c2()),
                  gt::cache<gt::cache_type::k, gt::cache_io_policy::flush>(
                      p_d2())),
              gt::make_stage<stage_diffusion_w_forward>(
                  p_c(), p_d(), p_c2(), p_d2(), p_data_in(),
                  p_data_in_uncached(), p_dz(), p_dt(), p_coeff(), p_k_size())),
          gt::make_multistage(
              gt::execute::backward(),
              gt::define_caches(
                  gt::cache<gt::cache_type::k,
                            gt::cache_io_policy::fill_and_flush>(p_d()),
                  gt::cache<gt::cache_type::k,
                            gt::cache_io_policy::fill_and_flush>(p_d2())),
              gt::make_stage<stage_diffusion_w_backward>(
                  p_c(), p_d(), p_c2(), p_d2(), p_fact(), p_d_uncached(),
                  p_d2_uncached(), p_dz(), p_dt(), p_coeff(), p_k_size())))),
      comp2_(gt::make_computation<backend_t<64, 1>>(
          computation_grid(resolution.x, resolution.y, resolution.z),
          p_fact() = fact_, p_d() = d_, p_d2() = d2_,
          gt::make_multistage(gt::execute::parallel(),
                              gt::make_stage<stage_diffusion_w3>(
                                  p_data_out(), p_d(), p_d2(), p_fact())))) {}

void vertical::operator()(storage_t &out, storage_t const &in, real_t dt) {
  comp1_.run(p_data_in() = in, p_data_in_uncached() = in,
             p_dt() = gt::make_global_parameter(dt));
  comp2_.run(p_data_out() = out);
}

} // namespace diffusion
} // namespace numerics
