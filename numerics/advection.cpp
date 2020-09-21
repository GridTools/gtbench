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
  using u = in_accessor<2>;
  using v = in_accessor<3>;

  using dx = in_accessor<4>;
  using dy = in_accessor<5>;
  using dt = in_accessor<6>;

  using param_list = make_param_list<out, in, u, v, dx, dy, dt>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t) {
    auto flx = call<stage_u, full_t>::with(eval, u(), in(), dx());
    auto fly = call<stage_v, full_t>::with(eval, v(), in(), dy());

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

  using data =
      in_accessor<6, extent<0, 0, 0, 0, -infinite_extent, infinite_extent>>;

  using dz = in_accessor<7>;
  using dt = in_accessor<8>;
  using w = in_accessor<9, extent<0, 0, 0, 0, 0, 1>>;
  using data_p1_k_cache = inout_accessor<10, extent<0, 0, 0, 0, -2, 0>>;
  using a_c_cache = inout_accessor<11, extent<0, 0, 0, 0, -1, 0>>;

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

    auto old_a_c_cache = eval(a_c_cache(0, 0, -1));
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

    auto old_a_c_cache = eval(a_c_cache(0, 0, -1));
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

    auto a = eval(-a_c_cache(0, 0, -1));
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

  using d_uncached = in_accessor<7, extent<0, 0, 0, 0, 0, infinite_extent>>;
  using d2_uncached = in_accessor<8, extent<0, 0, 0, 0, 0, infinite_extent>>;

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
    auto flx = call<stage_u, full_t>::with(eval, u(), in(), dx());
    auto fly = call<stage_v, full_t>::with(eval, v(), in(), dy());
    eval(out()) = eval(in0() - dt() * (flx + fly) + (vout - in()));
  }
};

} // namespace

std::function<void(storage_t, storage_t, storage_t, storage_t, real_t)>
horizontal(vec<std::size_t, 3> const &resolution, vec<real_t, 3> const &delta) {
  auto grid = computation_grid(resolution.x, resolution.y, resolution.z);
  return [grid = std::move(grid), delta](storage_t out, storage_t in,
                                         storage_t u, storage_t v, real_t dt) {
    gt::stencil::run_single_stage(stage_horizontal(), backend_t<>(), grid, out,
                                  in, u, v,
                                  gt::stencil::make_global_parameter(delta.x),
                                  gt::stencil::make_global_parameter(delta.y),
                                  gt::stencil::make_global_parameter(dt));
  };
}

std::function<void(storage_t, storage_t, storage_t, real_t)>
vertical(vec<std::size_t, 3> const &resolution, vec<real_t, 3> const &delta) {
  auto grid = computation_grid(resolution.x, resolution.y, resolution.z);
  auto const spec = [](auto out, auto in, auto w, auto alpha, auto gamma,
                       auto fact, auto d, auto d2, auto d_uncached,
                       auto d2_uncached, auto k_size, auto dz, auto dt) {
    using namespace gt::stencil;
    GT_DECLARE_TMP(real_t, c, c2, p1_k_cache, a_c_cache);
    return multi_pass(
        execute_forward()
            .k_cached(cache_io_policy::flush(), c, d, c2, d2)
            .k_cached(cache_io_policy::fill(), w)
            .k_cached(p1_k_cache, a_c_cache)
            .stage(stage_advection_w_forward(), alpha, gamma, c, d, c2, d2, in,
                   dz, dt, w, p1_k_cache, a_c_cache, k_size),
        execute_backward()
            .k_cached(cache_io_policy::fill(), cache_io_policy::flush(), d, d2)
            .stage(stage_advection_w_backward(), c, d, c2, d2, alpha, gamma,
                   fact, d_uncached, d2_uncached, k_size),
        execute_parallel().stage(stage_advection_w3(), out, d, d2, fact, in,
                                 dt));
  };

  auto field = storage_builder(resolution);

  auto ij_slice = gt::storage::builder<storage_tr>
    .type<real_t>()
    .id<1>()
    .halos(halo, halo)
    .dimensions(resolution.x + 2 * halo, resolution.y + 2 * halo);

  auto alpha = ij_slice();
  auto gamma = ij_slice();
  auto fact = ij_slice();
  auto d = field();
  auto d2 = field();

  return [grid = std::move(grid), spec = std::move(spec),
          alpha = std::move(alpha), gamma = std::move(gamma),
          fact = std::move(fact), d = std::move(d), d2 = std::move(d2), delta,
          resolution](storage_t out, storage_t in, storage_t w, real_t dt) {
    gt::stencil::run(spec, backend_t<>(), grid, out, in, w, alpha, gamma, fact,
                     d, d2, d, d2,
                     gt::stencil::make_global_parameter((int)resolution.z),
                     gt::stencil::make_global_parameter(delta.z),
                     gt::stencil::make_global_parameter(dt));
  };
}

std::function<void(storage_t, storage_t, storage_t, storage_t, storage_t,
                   storage_t, real_t)>
runge_kutta_step(vec<std::size_t, 3> const &resolution,
                 vec<real_t, 3> const &delta) {
  auto grid = computation_grid(resolution.x, resolution.y, resolution.z);
  auto const spec = [](auto in, auto w, auto alpha, auto gamma, auto fact,
                       auto d, auto d2, auto d_uncached, auto d2_uncached,
                       auto k_size, auto dz, auto dt) {
    using namespace gt::stencil;
    GT_DECLARE_TMP(real_t, c, c2, p1_k_cache, a_c_cache);
    return multi_pass(
        execute_forward()
            .k_cached(cache_io_policy::flush(), c, d, c2, d2)
            .k_cached(p1_k_cache, a_c_cache)
            .stage(stage_advection_w_forward(), alpha, gamma, c, d, c2, d2, in,
                   dz, dt, w, p1_k_cache, a_c_cache, k_size),
        execute_backward()
            .k_cached(cache_io_policy::fill(), cache_io_policy::flush(), d, d2)
            .stage(stage_advection_w_backward(), c, d, c2, d2, alpha, gamma,
                   fact, d_uncached, d2_uncached, k_size));
  };

  auto field = storage_builder(resolution);

  auto ij_slice = gt::storage::builder<storage_tr>
    .type<real_t>()
    .id<1>()
    .halos(halo, halo)
    .dimensions(resolution.x + 2 * halo, resolution.y + 2 * halo);

  auto alpha = ij_slice();
  auto gamma = ij_slice();
  auto fact = ij_slice();
  auto d = field();
  auto d2 = field();
  return [grid = std::move(grid), spec = std::move(spec),
          alpha = std::move(alpha), gamma = std::move(gamma),
          fact = std::move(fact), d = std::move(d), d2 = std::move(d2), delta,
          resolution](storage_t out, storage_t in, storage_t in0, storage_t u,
                      storage_t v, storage_t w, real_t dt) {
    gt::stencil::run(spec, backend_t<GTBENCH_BPARAMS_RKADV1>(), grid, in, w,
                     alpha, gamma, fact, d, d2, d, d2,
                     gt::stencil::make_global_parameter(resolution.z),
                     gt::stencil::make_global_parameter(delta.z),
                     gt::stencil::make_global_parameter(dt));
    gt::stencil::run_single_stage(
        stage_advection_w3_rk(), backend_t<GTBENCH_BPARAMS_RKADV2>(), grid, out,
        d, d2, fact, in, in0, u, v, gt::stencil::make_global_parameter(delta.x),
        gt::stencil::make_global_parameter(delta.y),
        gt::stencil::make_global_parameter(dt));
  };
}

} // namespace advection
} // namespace numerics
