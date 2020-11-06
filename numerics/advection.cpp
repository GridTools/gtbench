/*
 * gtbench
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "./advection.hpp"

#include <gridtools/stencil/cartesian.hpp>
#include <gridtools/stencil/frontend/run.hpp>
#include <gridtools/stencil/global_parameter.hpp>

#include "./computation.hpp"

namespace numerics {
namespace advection {
namespace {
using gt::stencil::extent;
using gt::stencil::make_param_list;
using gt::stencil::cartesian::call;
using gt::stencil::cartesian::call_proc;
using gt::stencil::cartesian::in_accessor;
using gt::stencil::cartesian::inout_accessor;
using namespace gt::stencil::cartesian::expressions;

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
  using in0 = in_accessor<2, extent<-3, 3, -3, 3>>;
  using u = in_accessor<3>;
  using v = in_accessor<4>;

  using dx = in_accessor<5>;
  using dy = in_accessor<6>;
  using dt = in_accessor<7>;

  using param_list = make_param_list<out, in, in0, u, v, dx, dy, dt>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t) {
    auto flx = call<stage_u, full_t>::with(eval, u(), in(), dx());
    auto fly = call<stage_v, full_t>::with(eval, v(), in(), dy());

    eval(out()) = eval(in0() - dt() * (flx + fly));
  }
};

struct stage_advection_w_forward {
  using b = inout_accessor<0, extent<0, 0, 0, 0, -1, 0>>;
  using d1 = inout_accessor<1, extent<0, 0, 0, 0, -1, 0>>;
  using d2 = inout_accessor<2, extent<0, 0, 0, 0, -1, 0>>;

  using data = in_accessor<3, extent<0, 0, 0, 0, -1, 1>>;
  using data_uncached = in_accessor<4, extent<0, 0, 0, 0, 0, infinite_extent>>;

  using dz = in_accessor<5>;
  using dt = in_accessor<6>;
  using w = in_accessor<7, extent<0, 0, 0, 0, 0, 1>>;
  using k_size = in_accessor<8>;

  using param_list =
      make_param_list<b, d1, d2, data, data_uncached, dz, dt, w, k_size>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::first_level) {
    auto k_offset = eval(k_size()) - 1;

    real_t av = eval(-0.25_r / dz() * w());
    real_t bv = eval(1_r / dt() + 0.25_r * (w() - w(0, 0, 1)) / dz());
    real_t d1v = eval(1_r / dt() * data() -
                      0.25_r / dz() *
                          (w() * (data() - data_uncached(0, 0, k_offset)) +
                           w(0, 0, 1) * (data(0, 0, 1) - data())));
    real_t d2v = -av;

    eval(b()) = bv;
    eval(d1()) = d1v;
    eval(d2()) = d2v;
  }

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, -2>) {
    real_t av = eval(-0.25_r / dz() * w());
    real_t bv = eval(1_r / dt() + 0.25_r * (w() - w(0, 0, 1)) / dz());
    real_t cv_km1 = -av;
    real_t d1v =
        eval(1_r / dt() * data() - 0.25_r / dz() *
                                       (w() * (data() - data(0, 0, -1)) +
                                        w(0, 0, 1) * (data(0, 0, 1) - data())));
    real_t d2v = 0_r;

    real_t f = eval(av / b(0, 0, -1));
    eval(b()) = bv - f * cv_km1;
    eval(d1()) = eval(d1v - f * d1(0, 0, -1));
    eval(d2()) = eval(d2v - f * d2(0, 0, -1));
  }

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval,
                                full_t::last_level::shift<-1>) {
    real_t av = eval(-0.25_r / dz() * w());
    real_t bv = eval(1_r / dt() + 0.25_r * (w() - w(0, 0, 1)) / dz());
    real_t cv = eval(0.25_r / dz() * w(0, 0, 1));
    real_t cv_km1 = -av;
    real_t d1v =
        eval(1_r / dt() * data() - 0.25_r / dz() *
                                       (w() * (data() - data(0, 0, -1)) +
                                        w(0, 0, 1) * (data(0, 0, 1) - data())));
    real_t d2v = -cv;

    real_t f = eval(av / b(0, 0, -1));
    eval(b()) = bv - f * cv;
    eval(d1()) = eval(d1v - f * d1(0, 0, -1));
    eval(d2()) = eval(d2v - f * d2(0, 0, -1));
  }
};

struct stage_advection_w_backward {
  using b = in_accessor<0>;
  using d1 = inout_accessor<1, extent<0, 0, 0, 0, 0, 1>>;
  using d2 = inout_accessor<2, extent<0, 0, 0, 0, 0, 1>>;

  using dz = in_accessor<3>;
  using w = in_accessor<4, extent<0, 0, 0, 0, 1, 1>>;

  using param_list = make_param_list<b, d1, d2, dz, w>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval,
                                full_t::last_level::shift<-1>) {
    auto f = eval(1_r / b());
    eval(d1()) *= f;
    eval(d2()) *= f;
  }

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::modify<0, -2>) {
    auto cv = eval(0.25_r / dz() * w(0, 0, 1));
    auto f = eval(1_r / b());
    eval(d1()) = eval((d1() - cv * d1(0, 0, 1)) * f);
    eval(d2()) = eval((d2() - cv * d2(0, 0, 1)) * f);
  }
};

struct stage_advection_w3 {
  using out = inout_accessor<0>;
  using out_top = inout_accessor<1, extent<0, 0, 0, 0, 0, 1>>;
  using data = in_accessor<2, extent<0, 0, 0, 0, -infinite_extent, 0>>;
  using data0 = in_accessor<3>;
  using d1 = in_accessor<4, extent<0, 0, 0, 0, -infinite_extent, 0>>;
  using d2 = in_accessor<5, extent<0, 0, 0, 0, -infinite_extent, 0>>;

  using dz = in_accessor<6>;
  using dt = in_accessor<7>;
  using w = in_accessor<8, extent<0, 0, 0, 0, 0, 1>>;
  using k_size = in_accessor<9>;

  using param_list =
      make_param_list<out, out_top, data, data0, d1, d2, dz, dt, w, k_size>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
    auto k_offset = eval(k_size() - 1);

    real_t av = eval(-0.25_r / dz() * w());
    real_t bv = eval(1_r / dt() + 0.25_r * (w() - w(0, 0, 1)) / dz());
    real_t cv = eval(0.25_r / dz() * w(0, 0, 1));

    real_t d1v = eval(1_r / dt() * data() -
                      0.25_r / dz() *
                          (w() * (data() - data(0, 0, -1)) +
                           w(0, 0, 1) * (data(0, 0, -k_offset) - data())));

    eval(out_top()) =
        eval((d1v - cv * d1(0, 0, -k_offset) - av * d1(0, 0, -1)) /
             (bv + cv * d2(0, 0, -k_offset) + av * d2(0, 0, -1)));
    eval(out()) = eval(data0() + (out_top() - data()));
  }

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::modify<0, -1>) {
    eval(out_top()) = eval(out_top(0, 0, 1));
    eval(out()) = eval(data0() + (d1() + d2() * out_top() - data()));
  }
};

} // namespace

std::function<void(storage_t, storage_t, storage_t, storage_t, storage_t,
                   real_t)>
horizontal(vec<std::size_t, 3> const &resolution, vec<real_t, 3> const &delta) {
  auto grid = computation_grid(resolution.x, resolution.y, resolution.z);
  return [grid = std::move(grid), delta](storage_t out, storage_t in,
                                         storage_t in0, storage_t u,
                                         storage_t v, real_t dt) {
    gt::stencil::run_single_stage(stage_horizontal(), backend_t<>(), grid, out,
                                  in, in0, u, v,
                                  gt::stencil::make_global_parameter(delta.x),
                                  gt::stencil::make_global_parameter(delta.y),
                                  gt::stencil::make_global_parameter(dt));
  };
}

std::function<void(storage_t, storage_t, storage_t, storage_t, real_t)>
vertical(vec<std::size_t, 3> const &resolution, vec<real_t, 3> const &delta) {
  auto grid = computation_grid(resolution.x, resolution.y, resolution.z);
  auto const spec = [](auto out, auto in, auto in_uncached, auto in0, auto w,
                       auto d1, auto d2, auto k_size, auto dz, auto dt) {
    using namespace gt::stencil;
    GT_DECLARE_TMP(real_t, b, out_top);
    return multi_pass(
        execute_forward()
            .k_cached(cache_io_policy::fill(), in)
            .k_cached(cache_io_policy::flush(), b, d1, d2)
            .stage(stage_advection_w_forward(), b, d1, d2, in, in_uncached, dz,
                   dt, w, k_size),
        execute_backward()
            .k_cached(cache_io_policy::fill(), cache_io_policy::flush(), d1, d2)
            .stage(stage_advection_w_backward(), b, d1, d2, dz, w),
        execute_backward().k_cached(out_top).stage(stage_advection_w3(), out,
                                                   out_top, in, in0, d1, d2, dz,
                                                   dt, w, k_size));
  };

  auto field = storage_builder(resolution);
  auto d2 = field();

  return [grid = std::move(grid), spec = std::move(spec), d2 = std::move(d2),
          delta, resolution](storage_t out, storage_t in, storage_t in0,
                             storage_t w, real_t dt) {
    gt::stencil::run(spec, backend_t<>(), grid, out, in, in, in0, w,
                     out /* out is used as temporary storage for d1 */, d2,
                     gt::stencil::make_global_parameter(resolution.z),
                     gt::stencil::make_global_parameter(delta.z),
                     gt::stencil::make_global_parameter(dt));
  };
}

} // namespace advection
} // namespace numerics
