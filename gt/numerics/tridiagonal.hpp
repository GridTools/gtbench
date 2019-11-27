#pragma once

#include <gridtools/stencil_composition/expressions/expressions.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/stencil_composition/stencil_functions.hpp>

#include "./computation.hpp"

namespace tridiagonal {
using gt::extent;
using gt::in_accessor;
using gt::inout_accessor;
using gt::make_param_list;
using namespace gt::expressions;

struct forward {
  using a = in_accessor<0>;
  using b = in_accessor<1>;
  using c = inout_accessor<2, extent<0, 0, 0, 0, -1, 0>>;
  using d = inout_accessor<3, extent<0, 0, 0, 0, -1, 0>>;
  using param_list = make_param_list<a, b, c, d>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, 0>) {
    eval(c()) = eval(c() / (b() - c(0, 0, -1) * a()));
    eval(d()) = eval((d() - a() * d(0, 0, -1)) / (b() - c(0, 0, -1) * a()));
  }

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::first_level) {
    eval(c()) = eval(c()) / eval(b());
    eval(d()) = eval(d()) / eval(b());
  }
};

struct backward {
  using out = inout_accessor<0, extent<0, 0, 0, 0, 0, 1>>;
  using c = in_accessor<1>;
  using d = in_accessor<2>;
  using param_list = make_param_list<out, c, d>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::modify<0, -1>) {
    eval(out()) = eval(d() - c() * out(0, 0, 1));
  }

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
    eval(out()) = eval(d());
  }
};

/*
 * tridiagonal_periodic1:
 * b[0] = b[0] - gamma
 * b[-1] = b[-1] - alpha * beta / gamma
 * x = tridiagonal_solve(a, b, c, d)
 */
struct periodic_forward1 {
  using a = in_accessor<0>;
  using b = inout_accessor<1>;
  using c = inout_accessor<2, extent<0, 0, 0, 0, -1, 0>>;
  using d = inout_accessor<3, extent<0, 0, 0, 0, -1, 0>>;

  using alpha = in_accessor<4>;
  using beta = in_accessor<5>;
  using gamma = in_accessor<6>;

  using param_list = make_param_list<a, b, c, d, alpha, beta, gamma>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::first_level) {
    eval(b()) -= eval(gamma());
    gridtools::call_proc<tridiagonal::forward, full_t::first_level>::with(
        eval, a(), b(), c(), d());
  }
  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, -1>) {
    gridtools::call_proc<tridiagonal::forward, full_t::modify<1, 0>>::with(
        eval, a(), b(), c(), d());
  }
  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
    eval(b()) -= eval(alpha() * beta() / gamma());
    gridtools::call_proc<tridiagonal::forward, full_t::modify<1, 0>>::with(
        eval, a(), b(), c(), d());
  }
};

using periodic_backward1 = backward;

/*
 * tridiagonal_periodic2:
 * u = np.zeros_like(a)
 * u[0] = gamma
 * u[-1] = alpha
 * z = tridiagonal_solve(a, b, c, u)
 * fact = (x[0] + beta * x[-1] / gamma) / (1 + z[0] + beta * z[-1] / gamma)
 */
struct periodic_forward2 {
  using a = in_accessor<0>;
  using b = in_accessor<1>;
  using c = inout_accessor<2, extent<0, 0, 0, 0, -1, 0>>;
  using u = inout_accessor<3, extent<0, 0, 0, 0, -1, 0>>;

  using alpha = in_accessor<4>;
  using gamma = in_accessor<5>;

  using param_list = make_param_list<a, b, c, u, alpha, gamma>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::first_level) {
    eval(u()) = eval(gamma());
    gridtools::call_proc<tridiagonal::forward, full_t::first_level>::with(
        eval, a(), b(), c(), u());
  }
  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, -1>) {
    eval(u()) = real_t(0);
    gridtools::call_proc<tridiagonal::forward, full_t::modify<1, 0>>::with(
        eval, a(), b(), c(), u());
  }
  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
    eval(u()) = eval(alpha());
    gridtools::call_proc<tridiagonal::forward, full_t::modify<1, 0>>::with(
        eval, a(), b(), c(), u());
  }
};

struct periodic_backward2 {
  using z = inout_accessor<0, extent<0, 0, 0, 0, 0, huge_offset>>;
  using c = inout_accessor<1>;
  using d = inout_accessor<2>;
  using x = in_accessor<3, extent<0, 0, 0, 0, 0, huge_offset>>;

  using beta = in_accessor<4>;
  using gamma = in_accessor<5>;

  using fact = inout_accessor<6>;

  using k_size = in_accessor<7>;

  using param_list = make_param_list<z, c, d, x, beta, gamma, fact, k_size>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::first_level) {
    gridtools::call_proc<tridiagonal::backward, full_t::modify<0, -1>>::with(
        eval, z(), c(), d());
    const gt::int_t top_offset = eval(k_size() - 1);
    eval(fact()) = eval((x() + beta() * x(0, 0, top_offset) / gamma()) /
                        (1 + z() + beta() * z(0, 0, top_offset) / gamma()));
  }

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, -1>) {
    gridtools::call_proc<tridiagonal::backward, full_t::modify<0, -1>>::with(
        eval, z(), c(), d());
  }

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
    gridtools::call_proc<tridiagonal::backward, full_t::last_level>::with(
        eval, z(), c(), d());
  }
};

/**
 * tridiagonal_periodic3:
 * out = x - fact * z
 */
struct periodic3 {
  using data_out = inout_accessor<0>;
  using x = in_accessor<1>;
  using z = in_accessor<2>;

  using fact = in_accessor<3>;

  using param_list = make_param_list<data_out, x, z, fact>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t) {
    eval(data_out()) = eval(x() - fact() * z());
  }
};

} // namespace tridiagonal