#pragma once

namespace operators {
using namespace gridtools::expressions;

using gridtools::extent;
using gridtools::in_accessor;
using gridtools::inout_accessor;
using gridtools::make_param_list;

struct advection_u {
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
struct advection_v {
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

struct horizontal_advection {
  using flux = inout_accessor<0>;
  using u = in_accessor<1, extent<-3, 3, 0, 0>>;
  using v = in_accessor<2, extent<0, 0, -3, 3>>;
  using in = in_accessor<3>;
  using dx = in_accessor<4>;
  using dy = in_accessor<5>;

  using param_list = make_param_list<flux, u, v, in, dx, dy>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t) {
    eval(flux()) =
        gridtools::call<advection_u, full_t>::with(eval, u(), in(), dx());
    eval(flux()) +=
        gridtools::call<advection_v, full_t>::with(eval, v(), in(), dy());
  }
};

/*
struct advection_w_fwd {
    using out = inout_accessor<0>;
    using data = in_accessor<1, extent<0, 0, 0, 0, -1, 1>>;
    using w = in_accessor<2, extent<0, 0, 0, 0, 0, 1>>;
    using dz = in_accessor<3>;
    using dt = in_accessor<4>;

    using a = inout_accessor<5>;
    using b = inout_accessor<6>;
    using c = inout_accessor<7, extent<0, 0, 0, 0, -1, 0>>;
    using d = inout_accessor<8, extent<0, 0, 0, 0, -1, 0>>;

    using param_list = make_param_list<out, data, w, dz, dt, a, b, c, d>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::first_level) {
        eval(a()) = eval(real_t(-0.25) * w() / dz());
        eval(c()) = eval(real_t(0.25) * w(0, 0, 1) / dz());
        eval(b()) = eval(real_t(1) / dt() - a() - c());
        eval(d()) = eval(
            real_t(1) / dt() * data() - real_t(0.25) * w() * data() / dz() -
            real_t(0.25) * w(0, 0, 1) * (data(0, 0, 1) - data()) / dz());
        gridtools::call_proc<tridiagonal_fwd, full_t::first_level>::with(
            eval, a(), b(), c(), d());
    }
    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, -1>) {
        eval(a()) = eval(real_t(-0.25) * w() / dz());
        eval(c()) = eval(real_t(0.25) * w(0, 0, 1) / dz());
        eval(b()) = eval(real_t(1) / dt() - a() - c());
        eval(d()) =
            eval(real_t(1) / dt() * data() -
                 real_t(0.25) * w() * (data() - data(0, 0, -1)) / dz() -
                 real_t(0.25) * w(0, 0, 1) * (data(0, 0, 1) - data()) / dz());
        gridtools::call_proc<tridiagonal_fwd, full_t::modify<1, 0>>::with(
            eval, a(), b(), c(), d());
    }
    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
        eval(a()) = eval(real_t(-0.25) * w() / dz());
        eval(c()) = eval(real_t(0.25) * w(0, 0, 1) / dz());
        eval(b()) = eval(real_t(1) / dt() - a() - c());
        eval(d()) = eval(real_t(1) / dt() * data() -
                         real_t(0.25) * w() * (data() - data(0, 0, -1)) / dz() -
                         real_t(0.25) * w(0, 0, 1) * -data() / dz());
        gridtools::call_proc<tridiagonal_fwd, full_t::first_level>::with(
            eval, a(), b(), c(), d());
    }
};
struct advection_w_bwd {
    using out = inout_accessor<0>;
    using data = in_accessor<1>;
    using dt = in_accessor<2>;

    using c = inout_accessor<3, extent<0, 0, 0, 0, -1, 0>>;
    using d = inout_accessor<4, extent<0, 0, 0, 0, -1, 0>>;

    using param_list = make_param_list<out, data, dt, c, d>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::modify<0, -1>) {
        gridtools::call_proc<tridiagonal_bwd, full_t::modify<0, -1>>::with(
            eval, out(), c(), d());
        eval(out()) = eval((out() - data()) / dt());
    }
    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
        gridtools::call_proc<tridiagonal_bwd, full_t::last_level>::with(
            eval, out(), c(), d());
        eval(out()) = eval((out() - data()) / dt());
    }
};

struct time_integrator {
    using out = inout_accessor<0>;
    using in = in_accessor<1>;
    using flux = in_accessor<2>;
    using dt = in_accessor<3>;

    using param_list = make_param_list<out, in, flux, dt>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t) {
        eval(out()) = eval(in() + dt() * flux());
    }
};
*/

} // namespace operators

namespace gt = gridtools;

/*
class advection {
    using p_flux = gt::arg<0, storage_t>;
    using p_u = gt::arg<1, storage_t>;
    using p_v = gt::arg<2, storage_t>;
    using p_w = gt::arg<3, storage_t>;
    using p_in = gt::arg<4, storage_t>;
    using p_dx = gt::arg<5, global_parameter_t>;
    using p_dy = gt::arg<6, global_parameter_t>;
    using p_dz = gt::arg<7, global_parameter_t>;
    using p_dt = gt::arg<8, global_parameter_t>;

    using p_a = gt::tmp_arg<9, storage_t>;
    using p_b = gt::tmp_arg<10, storage_t>;
    using p_c = gt::tmp_arg<11, storage_t>;
    using p_d = gt::tmp_arg<12, storage_t>;

   public:
    advection(gridtools::grid<axis_t::axis_interval_t> const& grid)
        : horizontal_comp_(gt::make_computation<backend_t>(
              grid, gt::make_multistage(
                        gt::execute::parallel(),
                        gt::make_stage<operators::horizontal_advection>(
                            p_flux(), p_u(), p_v(), p_in(), p_dx(), p_dy())))),
          vertical_comp_(gt::make_computation<backend_t>(
              grid,
              gt::make_multistage(gt::execute::forward(),
                                  gt::make_stage<operators::advection_w_fwd>(
                                      p_flux(), p_in(), p_w(), p_dz(), p_dt(),
                                      p_a(), p_b(), p_c(), p_d())),
              gt::make_multistage(
                  gt::execute::backward(),
                  gt::make_stage<operators::advection_w_bwd>(
                      p_flux(), p_in(), p_dt(), p_c(), p_d())))) {}

   private:
    gridtools::computation<p_flux, p_u, p_v, p_in, p_dx, p_dy> horizontal_comp_;
    gridtools::computation<p_flux, p_w, p_in, p_dz, p_dt> vertical_comp_;
};
*/