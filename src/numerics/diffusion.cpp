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

#include <gtbench/numerics/computation.hpp>
#include <gtbench/numerics/diffusion.hpp>

namespace gtbench {
namespace numerics {
namespace diffusion {

namespace {
using gt::stencil::extent;
using gt::stencil::make_param_list;
using gt::stencil::cartesian::in_accessor;
using gt::stencil::cartesian::inout_accessor;

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
    constexpr real_t weight0 = -1_r / 90;
    constexpr real_t weight1 = 5_r / 36;
    constexpr real_t weight2 = -49_r / 36;
    constexpr real_t weight3 = 49_r / 36;
    constexpr real_t weight4 = -5_r / 36;
    constexpr real_t weight5 = 1_r / 90;

    auto flx_x0 = (weight0 * eval(in(-3, 0)) + weight1 * eval(in(-2, 0)) +
                   weight2 * eval(in(-1, 0)) + weight3 * eval(in(0, 0)) +
                   weight4 * eval(in(1, 0)) + weight5 * eval(in(2, 0))) /
                  eval(dx());
    auto flx_x1 = (weight0 * eval(in(-2, 0)) + weight1 * eval(in(-1, 0)) +
                   weight2 * eval(in(0, 0)) + weight3 * eval(in(1, 0)) +
                   weight4 * eval(in(2, 0)) + weight5 * eval(in(3, 0))) /
                  eval(dx());
    auto flx_y0 = (weight0 * eval(in(0, -3)) + weight1 * eval(in(0, -2)) +
                   weight2 * eval(in(0, -1)) + weight3 * eval(in(0, 0)) +
                   weight4 * eval(in(0, 1)) + weight5 * eval(in(0, 2))) /
                  eval(dy());
    auto flx_y1 = (weight0 * eval(in(0, -2)) + weight1 * eval(in(0, -1)) +
                   weight2 * eval(in(0, 0)) + weight3 * eval(in(0, 1)) +
                   weight4 * eval(in(0, 2)) + weight5 * eval(in(0, 3))) /
                  eval(dy());

    flx_x0 = flx_x0 * (eval(in()) - eval(in(-1, 0))) < 0_r ? 0_r : flx_x0;
    flx_x1 = flx_x1 * (eval(in(1, 0)) - eval(in())) < 0_r ? 0_r : flx_x1;
    flx_y0 = flx_y0 * (eval(in()) - eval(in(0, -1))) < 0_r ? 0_r : flx_y0;
    flx_y1 = flx_y1 * (eval(in(0, 1)) - eval(in())) < 0_r ? 0_r : flx_y1;

    eval(out()) = eval(in()) + eval(coeff()) * eval(dt()) *
                                   ((flx_x1 - flx_x0) / eval(dx()) +
                                    (flx_y1 - flx_y0) / eval(dy()));
  }
};

struct stage_diffusion_w_forward {
  using b = inout_accessor<0, extent<0, 0, 0, 0, -1, 0>>;
  using d1 = inout_accessor<1, extent<0, 0, 0, 0, -1, 0>>;
  using d2 = inout_accessor<2, extent<0, 0, 0, 0, -1, 0>>;

  using in = in_accessor<3, extent<0, 0, 0, 0, -1, 1>>;
  using in_uncached = in_accessor<4, extent<0, 0, 0, 0, 0, infinite_extent>>;

  using dz = in_accessor<5>;
  using dt = in_accessor<6>;
  using coeff = in_accessor<7>;
  using k_size = in_accessor<8>;

  using param_list =
      make_param_list<b, d1, d2, in, in_uncached, dz, dt, coeff, k_size>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::first_level) {
    auto k_offset = eval(k_size()) - 1;

    real_t av = -eval(coeff()) / (2_r * eval(dz()) * eval(dz()));
    real_t bv = 1_r / eval(dt()) + eval(coeff()) / (eval(dz()) * eval(dz()));
    real_t d1v = 1_r / eval(dt()) * eval(in()) +
                 0.5_r * eval(coeff()) *
                     (eval(in_uncached(0, 0, k_offset)) - 2_r * eval(in()) +
                      eval(in(0, 0, 1))) /
                     (eval(dz()) * eval(dz()));
    real_t d2v = -av;

    eval(b()) = bv;
    eval(d1()) = d1v;
    eval(d2()) = d2v;
  }

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, -2>) {
    real_t av = -eval(coeff()) / (2_r * eval(dz()) * eval(dz()));
    real_t bv = 1_r / eval(dt()) + eval(coeff()) / (eval(dz()) * eval(dz()));
    real_t cv = av;
    real_t d1v =
        1_r / eval(dt()) * eval(in()) +
        0.5_r * eval(coeff()) *
            (eval(in(0, 0, -1)) - 2_r * eval(in()) + eval(in(0, 0, 1))) /
            (eval(dz()) * eval(dz()));
    real_t d2v = 0_r;

    real_t f = av / eval(b(0, 0, -1));
    eval(b()) = bv - f * cv;
    eval(d1()) = d1v - f * eval(d1(0, 0, -1));
    eval(d2()) = d2v - f * eval(d2(0, 0, -1));
  }

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval,
                                full_t::last_level::shift<-1>) {
    real_t av = -eval(coeff()) / (2_r * eval(dz()) * eval(dz()));
    real_t bv = 1_r / eval(dt()) + eval(coeff()) / (eval(dz()) * eval(dz()));
    real_t cv = av;
    real_t d1v =
        1_r / eval(dt()) * eval(in()) +
        0.5_r * eval(coeff()) *
            (eval(in(0, 0, -1)) - 2_r * eval(in()) + eval(in(0, 0, 1))) /
            (eval(dz()) * eval(dz()));
    real_t d2v = -cv;

    real_t f = av / eval(b(0, 0, -1));
    eval(b()) = bv - f * cv;
    eval(d1()) = d1v - f * eval(d1(0, 0, -1));
    eval(d2()) = d2v - f * eval(d2(0, 0, -1));
  }
};

struct stage_diffusion_w_backward {
  using b = in_accessor<0>;
  using d1 = inout_accessor<1, extent<0, 0, 0, 0, 0, 1>>;
  using d2 = inout_accessor<2, extent<0, 0, 0, 0, 0, 1>>;

  using dz = in_accessor<3>;
  using coeff = in_accessor<4>;

  using param_list = make_param_list<b, d1, d2, dz, coeff>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval,
                                full_t::last_level::shift<-1>) {
    real_t f = 1_r / eval(b());
    eval(d1()) *= f;
    eval(d2()) *= f;
  }

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::modify<0, -2>) {
    real_t cv = -eval(coeff()) / (2_r * eval(dz()) * eval(dz()));
    real_t f = 1_r / eval(b());
    eval(d1()) = (eval(d1()) - cv * eval(d1(0, 0, 1))) * f;
    eval(d2()) = (eval(d2()) - cv * eval(d2(0, 0, 1))) * f;
  }
};

struct stage_diffusion_w3 {
  using out = inout_accessor<0>;
  using out_top = inout_accessor<1, extent<0, 0, 0, 0, 0, 1>>;
  using in = in_accessor<2, extent<0, 0, 0, 0, -infinite_extent, 0>>;
  using d1 = in_accessor<3, extent<0, 0, 0, 0, -infinite_extent, 0>>;
  using d2 = in_accessor<4, extent<0, 0, 0, 0, -infinite_extent, 0>>;

  using dz = in_accessor<5>;
  using dt = in_accessor<6>;
  using coeff = in_accessor<7>;
  using k_size = in_accessor<8>;

  using param_list =
      make_param_list<out, out_top, in, d1, d2, dz, dt, coeff, k_size>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
    auto k_offset = eval(k_size()) - 1;

    real_t av = -eval(coeff()) / (2_r * eval(dz()) * eval(dz()));
    real_t bv = 1_r / eval(dt()) + eval(coeff()) / (eval(dz()) * eval(dz()));
    real_t cv = av;
    real_t d1v = 1_r / eval(dt()) * eval(in()) +
                 0.5_r * eval(coeff()) *
                     (eval(in(0, 0, -1)) - 2_r * eval(in()) +
                      eval(in(0, 0, -k_offset))) /
                     (eval(dz()) * eval(dz()));

    eval(out_top()) =
        (d1v - cv * eval(d1(0, 0, -k_offset)) - av * eval(d1(0, 0, -1))) /
        (bv + cv * eval(d2(0, 0, -k_offset)) + av * eval(d2(0, 0, -1)));
    eval(out()) = eval(out_top());
  }

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::modify<0, -1>) {
    eval(out_top()) = eval(out_top(0, 0, 1));
    eval(out()) = eval(d1()) + eval(d2()) * eval(out_top());
  }
};

} // namespace

std::function<void(storage_t, storage_t, real_t dt)>
horizontal(vec<std::size_t, 3> const &resolution, vec<real_t, 3> const &delta,
           real_t coeff) {
  auto grid = computation_grid(resolution.x, resolution.y, resolution.z);
  return [grid = std::move(grid), delta, coeff](storage_t out, storage_t in,
                                                real_t dt) {
    gt::stencil::run_single_stage(
        stage_horizontal(), backend_t<GTBENCH_BPARAMS_HDIFF>(), grid, out, in,
        gt::stencil::global_parameter{delta.x},
        gt::stencil::global_parameter{delta.y},
        gt::stencil::global_parameter{dt},
        gt::stencil::global_parameter{coeff});
  };
}

std::function<void(storage_t, storage_t, real_t dt)>
vertical(vec<std::size_t, 3> const &resolution, vec<real_t, 3> const &delta,
         real_t coeff) {
  auto grid = computation_grid(resolution.x, resolution.y, resolution.z);
  auto const spec = [](auto out, auto in, auto in_uncached, auto d1, auto d2,
                       auto k_size, auto dz, auto dt, auto coeff) {
    using namespace gt::stencil;
    GT_DECLARE_TMP(real_t, b, out_top);
    return multi_pass(
        execute_forward()
            .k_cached(cache_io_policy::fill(), in)
            .k_cached(cache_io_policy::flush(), b, d1, d2)
            .stage(stage_diffusion_w_forward(), b, d1, d2, in, in_uncached, dz,
                   dt, coeff, k_size),
        execute_backward()
            .k_cached(cache_io_policy::fill(), cache_io_policy::flush(), d1, d2)
            .stage(stage_diffusion_w_backward(), b, d1, d2, dz, coeff),
        execute_backward().k_cached(out_top).stage(stage_diffusion_w3(), out,
                                                   out_top, in, d1, d2, dz, dt,
                                                   coeff, k_size));
  };

  auto field = storage_builder(resolution);
  auto d2 = field();

  return [grid = std::move(grid), spec = std::move(spec), d2 = std::move(d2),
          delta, resolution, coeff](storage_t out, storage_t in, real_t dt) {
    gt::stencil::run(spec, backend_t<GTBENCH_BPARAMS_VDIFF>(), grid, out, in,
                     in, out /* out is used as temporary storage d1 */, d2,
                     gt::stencil::global_parameter{resolution.z},
                     gt::stencil::global_parameter{delta.z},
                     gt::stencil::global_parameter{dt},
                     gt::stencil::global_parameter{coeff});
  };
}

} // namespace diffusion
} // namespace numerics
} // namespace gtbench
