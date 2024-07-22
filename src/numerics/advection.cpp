/*
 * gtbench
 *
 * Copyright (c) 2014-2022, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/stencil/cartesian.hpp>
#include <gridtools/stencil/frontend/run.hpp>
#include <gridtools/stencil/global_parameter.hpp>

#include <gtbench/numerics/advection.hpp>
#include <gtbench/numerics/computation.hpp>

namespace gtbench {
namespace numerics {
namespace advection {
namespace {
using gt::stencil::extent;
using gt::stencil::make_param_list;
using gt::stencil::cartesian::in_accessor;
using gt::stencil::cartesian::inout_accessor;

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
    constexpr real_t weight0 = 1_r / 30;
    constexpr real_t weight1 = -1_r / 4;
    constexpr real_t weight2 = 1_r;
    constexpr real_t weight3 = -1_r / 3;
    constexpr real_t weight4 = -1_r / 2;
    constexpr real_t weight5 = 1_r / 20;

    real_t flx_x, flx_y;
    if (eval(u()) > 0_r) {
      flx_x = eval(u()) *
              -(weight0 * eval(in(-3, 0, 0)) + weight1 * eval(in(-2, 0, 0)) +
                weight2 * eval(in(-1, 0, 0)) + weight3 * eval(in()) +
                weight4 * eval(in(1, 0, 0)) + weight5 * eval(in(2, 0, 0))) /
              eval(dx());
    } else if (eval(u()) < 0_r) {
      flx_x = eval(u()) *
              (weight5 * eval(in(-2, 0, 0)) + weight4 * eval(in(-1, 0, 0)) +
               weight3 * eval(in()) + weight2 * eval(in(1, 0, 0)) +
               weight1 * eval(in(2, 0, 0)) + weight0 * eval(in(3, 0, 0))) /
              eval(dx());
    } else {
      flx_x = 0_r;
    }
    if (eval(v()) > 0_r) {
      flx_y = eval(v()) *
              -(weight0 * eval(in(0, -3, 0)) + weight1 * eval(in(0, -2, 0)) +
                weight2 * eval(in(0, -1, 0)) + weight3 * eval(in()) +
                weight4 * eval(in(0, 1, 0)) + weight5 * eval(in(0, 2, 0))) /
              eval(dy());
    } else if (eval(v()) < 0_r) {
      flx_y = eval(v()) *
              (weight5 * eval(in(0, -2, 0)) + weight4 * eval(in(0, -1, 0)) +
               weight3 * eval(in()) + weight2 * eval(in(0, 1, 0)) +
               weight1 * eval(in(0, 2, 0)) + weight0 * eval(in(0, 3, 0))) /
              eval(dy());
    } else {
      flx_y = 0_r;
    }

    eval(out()) = eval(in0()) - eval(dt()) * (flx_x + flx_y);
  }
};

struct stage_advection_w_forward {
  using b = inout_accessor<0, extent<0, 0, 0, 0, -1, 0>>;
  using d1 = inout_accessor<1, extent<0, 0, 0, 0, -1, 0>>;
  using d2 = inout_accessor<2, extent<0, 0, 0, 0, -1, 0>>;

  using in = in_accessor<3, extent<0, 0, 0, 0, -1, 1>>;
  using in_uncached = in_accessor<4, extent<0, 0, 0, 0, 0, infinite_extent>>;

  using dz = in_accessor<5>;
  using dt = in_accessor<6>;
  using w = in_accessor<7, extent<0, 0, 0, 0, 0, 1>>;
  using k_size = in_accessor<8>;

  using param_list =
      make_param_list<b, d1, d2, in, in_uncached, dz, dt, w, k_size>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::first_level) {
    auto k_offset = eval(k_size()) - 1;

    real_t av = -0.25_r / eval(dz()) * eval(w());
    real_t bv =
        1_r / eval(dt()) + 0.25_r * (eval(w()) - eval(w(0, 0, 1))) / eval(dz());
    real_t d1v =
        1_r / eval(dt()) * eval(in()) -
        0.25_r / eval(dz()) *
            (eval(w()) * (eval(in()) - eval(in_uncached(0, 0, k_offset))) +
             eval(w(0, 0, 1)) * (eval(in(0, 0, 1)) - eval(in())));
    real_t d2v = -av;

    eval(b()) = bv;
    eval(d1()) = d1v;
    eval(d2()) = d2v;
  }

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, -2>) {
    real_t av = -0.25_r / eval(dz()) * eval(w());
    real_t bv =
        1_r / eval(dt()) + 0.25_r * (eval(w()) - eval(w(0, 0, 1))) / eval(dz());
    real_t cv_km1 = -av;
    real_t d1v = 1_r / eval(dt()) * eval(in()) -
                 0.25_r / eval(dz()) *
                     (eval(w()) * (eval(in()) - eval(in(0, 0, -1))) +
                      eval(w(0, 0, 1)) * (eval(in(0, 0, 1)) - eval(in())));
    real_t d2v = 0_r;

    real_t f = av / eval(b(0, 0, -1));
    eval(b()) = bv - f * cv_km1;
    eval(d1()) = d1v - f * eval(d1(0, 0, -1));
    eval(d2()) = d2v - f * eval(d2(0, 0, -1));
  }

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval,
                                full_t::last_level::shift<-1>) {
    real_t av = -0.25_r / eval(dz()) * eval(w());
    real_t bv =
        1_r / eval(dt()) + 0.25_r * (eval(w()) - eval(w(0, 0, 1))) / eval(dz());
    real_t cv = 0.25_r / eval(dz()) * eval(w(0, 0, 1));
    real_t cv_km1 = -av;
    real_t d1v = 1_r / eval(dt()) * eval(in()) -
                 0.25_r / eval(dz()) *
                     (eval(w()) * (eval(in()) - eval(in(0, 0, -1))) +
                      eval(w(0, 0, 1)) * (eval(in(0, 0, 1)) - eval(in())));
    real_t d2v = -cv;

    real_t f = av / eval(b(0, 0, -1));
    eval(b()) = bv - f * cv_km1;
    eval(d1()) = d1v - f * eval(d1(0, 0, -1));
    eval(d2()) = d2v - f * eval(d2(0, 0, -1));
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
    auto f = 1_r / eval(b());
    eval(d1()) *= f;
    eval(d2()) *= f;
  }

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::modify<0, -2>) {
    auto cv = 0.25_r / eval(dz()) * eval(w(0, 0, 1));
    auto f = 1_r / eval(b());
    eval(d1()) = (eval(d1()) - cv * eval(d1(0, 0, 1))) * f;
    eval(d2()) = (eval(d2()) - cv * eval(d2(0, 0, 1))) * f;
  }
};

struct stage_advection_w3 {
  using out = inout_accessor<0>;
  using out_top = inout_accessor<1, extent<0, 0, 0, 0, 0, 1>>;
  using in = in_accessor<2, extent<0, 0, 0, 0, -infinite_extent, 0>>;
  using in0 = in_accessor<3>;
  using d1 = in_accessor<4, extent<0, 0, 0, 0, -infinite_extent, 0>>;
  using d2 = in_accessor<5, extent<0, 0, 0, 0, -infinite_extent, 0>>;

  using dz = in_accessor<6>;
  using dt = in_accessor<7>;
  using w = in_accessor<8, extent<0, 0, 0, 0, 0, 1>>;
  using k_size = in_accessor<9>;

  using param_list =
      make_param_list<out, out_top, in, in0, d1, d2, dz, dt, w, k_size>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
    auto k_offset = eval(k_size()) - 1;

    real_t av = -0.25_r / eval(dz()) * eval(w());
    real_t bv =
        1_r / eval(dt()) + 0.25_r * (eval(w()) - eval(w(0, 0, 1))) / eval(dz());
    real_t cv = 0.25_r / eval(dz()) * eval(w(0, 0, 1));

    real_t d1v =
        1_r / eval(dt()) * eval(in()) -
        0.25_r / eval(dz()) *
            (eval(w()) * (eval(in()) - eval(in(0, 0, -1))) +
             eval(w(0, 0, 1)) * (eval(in(0, 0, -k_offset)) - eval(in())));

    eval(out_top()) =
        (d1v - cv * eval(d1(0, 0, -k_offset)) - av * eval(d1(0, 0, -1))) /
        (bv + cv * eval(d2(0, 0, -k_offset)) + av * eval(d2(0, 0, -1)));
    eval(out()) = eval(in0()) + (eval(out_top()) - eval(in()));
  }

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::modify<0, -1>) {
    eval(out_top()) = eval(out_top(0, 0, 1));
    eval(out()) =
        eval(in0()) + (eval(d1()) + eval(d2()) * eval(out_top()) - eval(in()));
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
    gt::stencil::run_single_stage(
        stage_horizontal(), backend_t<GTBENCH_BPARAMS_HADV>(), grid, out, in,
        in0, u, v, gt::stencil::global_parameter{delta.x},
        gt::stencil::global_parameter{delta.y},
        gt::stencil::global_parameter{dt});
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
  auto d1 = field();
  auto d2 = field();

  return [grid = std::move(grid), spec = std::move(spec), d1 = std::move(d1),
          d2 = std::move(d2), delta,
          resolution](storage_t out, storage_t in, storage_t in0, storage_t w,
                      real_t dt) {
    gt::stencil::run(spec, backend_t<GTBENCH_BPARAMS_VADV>(), grid, out, in, in,
                     in0, w, d1, d2,
                     gt::stencil::global_parameter{resolution.z},
                     gt::stencil::global_parameter{delta.z},
                     gt::stencil::global_parameter{dt});
  };
}

} // namespace advection
} // namespace numerics
} // namespace gtbench
