#include <gridtools/stencil_composition/expressions/expressions.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/stencil_composition/stencil_functions.hpp>

#include <gridtools/boundary_conditions/boundary.hpp>

#include <fstream>
#include <iostream>

static constexpr int halo_i = 3;
static constexpr int halo_j = 3;
static constexpr int halo_k = 1;

using real_t = float;

using axis_t = gridtools::axis<1, gridtools::axis_config::offset_limit<3>>;
using full_t = axis_t::full_interval::modify<halo_k, -halo_k>;

using backend_t = gridtools::backend::x86;
using storage_tr = gridtools::storage_traits<backend_t>;
using storage_info_ijk_t =
    storage_tr::storage_info_t<0, 3, gridtools::halo<halo_i, halo_j, halo_k>>;
using storage_info_ij_t =
    storage_tr::special_storage_info_t<3, gridtools::selector<1, 1, 0>,
                                       gridtools::halo<halo_i, halo_j, 0>>;
using storage_t = storage_tr::data_store_t<real_t, storage_info_ijk_t>;
using storage_ij_t = storage_tr::data_store_t<real_t, storage_info_ij_t>;
using global_parameter_t = gridtools::global_parameter<backend_t, real_t>;

namespace operators {
using namespace gridtools::expressions;

using gridtools::extent;
using gridtools::in_accessor;
using gridtools::inout_accessor;
using gridtools::make_param_list;

struct tridiagonal_fwd {
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

struct tridiagonal_bwd {
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
struct tridiagonal_periodic_fwd1 {
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
        gridtools::call_proc<tridiagonal_fwd, full_t::first_level>::with(
            eval, a(), b(), c(), d());
    }
    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, -1>) {
        gridtools::call_proc<tridiagonal_fwd, full_t::modify<1, 0>>::with(
            eval, a(), b(), c(), d());
    }
    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
        eval(b()) -= eval(alpha() * beta() / gamma());
        gridtools::call_proc<tridiagonal_fwd, full_t::modify<1, 0>>::with(
            eval, a(), b(), c(), d());
    }
};
using tridiagonal_periodic_bwd1 = tridiagonal_bwd;
/*
 * tridiagonal_periodic2:
 * u = np.zeros_like(a)
 * u[0] = gamma
 * u[-1] = alpha
 * z = tridiagonal_solve(a, b, c, u)
 * fact = (x[0] + beta * x[-1] / gamma) / (1 + z[0] + beta * z[-1] / gamma)
 */
struct tridiagonal_periodic_fwd2 {
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
        gridtools::call_proc<tridiagonal_fwd, full_t::first_level>::with(
            eval, a(), b(), c(), u());
    }
    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, -1>) {
        eval(u()) = real_t(0);
        gridtools::call_proc<tridiagonal_fwd, full_t::modify<1, 0>>::with(
            eval, a(), b(), c(), u());
    }
    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
        eval(u()) = eval(alpha());
        gridtools::call_proc<tridiagonal_fwd, full_t::modify<1, 0>>::with(
            eval, a(), b(), c(), u());
    }
};
struct tridiagonal_periodic_bwd2 {
    using z = inout_accessor<0>;
    using c = inout_accessor<1>;
    using d = inout_accessor<2>;
    using x = in_accessor<3>;

    using alpha = in_accessor<4>;
    using beta = in_accessor<5>;
    using gamma = in_accessor<6>;

    using x_top = inout_accessor<7>;
    using z_top = inout_accessor<8>;
    using fact = inout_accessor<9>;

    using param_list =
        make_param_list<z, c, d, x, alpha, beta, gamma, x_top, z_top, fact>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::first_level) {
        gridtools::call_proc<tridiagonal_bwd, full_t::modify<0, -1>>::with(
            eval, z(), c(), d());
        eval(fact()) = eval((x() + beta() * x_top() / gamma()) /
                            (1 + z() + beta() * z_top() / gamma()));
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, -1>) {
        gridtools::call_proc<tridiagonal_bwd, full_t::modify<0, -1>>::with(
            eval, z(), c(), d());
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
        gridtools::call_proc<tridiagonal_bwd, full_t::last_level>::with(
            eval, z(), c(), d());
        eval(x_top()) = eval(x());
        eval(z_top()) = eval(z());
    }
};
/**
 * tridiagonal_periodic3:
 * out = x - fact * z
 */
struct tridiagonal_periodic3 {
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

struct horizontal_diffusion {
    using out = inout_accessor<0>;
    using in = in_accessor<1, extent<-1, 1, -1, 1>>;

    using dx = in_accessor<2>;
    using dy = in_accessor<3>;
    using dt = in_accessor<4>;
    using coeff = in_accessor<5>;

    using param_list = make_param_list<out, in, dx, dy, dt, coeff>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t) {
        auto flx_x1 = eval((in(1, 0, 0) - in()) / dx());
        auto flx_x0 = eval((in() - in(-1, 0, 0)) / dx());
        auto flx_y1 = eval((in(0, 1, 0) - in()) / dy());
        auto flx_y0 = eval((in() - in(0, -1, 0)) / dy());

        eval(out()) = eval(
            in() + coeff() * dt() *
                       ((flx_x1 - flx_x0) / dx() + (flx_y1 - flx_y0) / dy()));
    }
};

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

struct diffusion_w0 {
    using data_top = inout_accessor<0>;
    using data = in_accessor<1>;

    using param_list = make_param_list<data_top, data>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
        eval(data_top()) = eval(data());
    }
};
struct diffusion_w_fwd1 {
    using data_bottom = inout_accessor<0>;
    using alpha = inout_accessor<1>;
    using beta = inout_accessor<2>;
    using gamma = inout_accessor<3>;
    using a = inout_accessor<4>;
    using b = inout_accessor<5>;
    using c = inout_accessor<6, extent<0, 0, 0, 0, -1, 0>>;
    using d = inout_accessor<7, extent<0, 0, 0, 0, -1, 0>>;

    using data = in_accessor<8, extent<0, 0, 0, 0, -1, 1>>;
    using data_top = in_accessor<9>;

    using dz = in_accessor<10>;
    using dt = in_accessor<11>;
    using coeff = in_accessor<12>;

    using param_list = make_param_list<data_bottom, alpha, beta, gamma, a, b, c,
                                       d, data, data_top, dz, dt, coeff>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::first_level) {
        eval(data_bottom()) = eval(data());

        eval(a()) = real_t(0);
        eval(c()) = eval(-coeff() / (real_t(2) * dz() * dz()));
        eval(b()) = eval(real_t(1) / dt() - a() - c());
        eval(d()) = eval(real_t(1) / dt() * data() +
                         real_t(0.5) * coeff() *
                             (data_top() - real_t(2) * data() + data(0, 0, 1)) /
                             (dz() * dz()));

        eval(alpha()) = eval(beta()) =
            eval(-coeff() / (real_t(2) * dz() * dz()));
        eval(gamma()) = eval(-b());

        gridtools::call_proc<tridiagonal_periodic_fwd1,
                             full_t::first_level>::with(eval, a(), b(), c(),
                                                        d(), alpha(), beta(),
                                                        gamma());
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, 0>) {
        eval(a()) = eval(c()) = eval(-coeff() / (real_t(2) * dz() * dz()));
        eval(b()) = eval(real_t(1) / dt() - a() - c());
        eval(d()) =
            eval(real_t(1) / dt() * data() +
                 real_t(0.5) * coeff() *
                     (data(0, 0, -1) - real_t(2) * data() + data(0, 0, 1)) /
                     (dz() * dz()));

        gridtools::call_proc<tridiagonal_periodic_fwd1,
                             full_t::modify<1, -1>>::with(eval, a(), b(), c(),
                                                          d(), alpha(), beta(),
                                                          gamma());
    }
    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
        eval(a()) = eval(-coeff() / (real_t(2) * dz() * dz()));
        eval(c()) = real_t(0);
        eval(b()) = eval(real_t(1) / dt() - a() - c());
        eval(d()) =
            eval(real_t(1) / dt() * data() +
                 real_t(0.5) * coeff() *
                     (data(0, 0, -1) - real_t(2) * data() + data_bottom()) /
                     (dz() * dz()));
        gridtools::call_proc<tridiagonal_periodic_fwd1,
                             full_t::last_level>::with(eval, a(), b(), c(), d(),
                                                       alpha(), beta(),
                                                       gamma());
    }
};
using diffusion_w_bwd1 = tridiagonal_periodic_bwd1;
using diffusion_w_fwd2 = tridiagonal_periodic_fwd2;
using diffusion_w_bwd2 = tridiagonal_periodic_bwd2;
struct diffusion_w3 {
    using out = inout_accessor<0>;
    using x = in_accessor<1>;
    using z = in_accessor<2>;
    using fact = in_accessor<3>;
    using in = in_accessor<4>;

    using dt = in_accessor<5>;

    using param_list = make_param_list<out, x, z, fact, in, dt>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t) {
        gridtools::call_proc<tridiagonal_periodic3, full_t>::with(
            eval, out(), x(), z(), fact());
        // eval(out()) = eval((out() - in()) / dt());
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

}  // namespace operators

namespace gt = gridtools;

class horizontal_diffusion {
    using p_out = gt::arg<0, storage_t>;
    using p_in = gt::arg<1, storage_t>;
    using p_dx = gt::arg<2, global_parameter_t>;
    using p_dy = gt::arg<3, global_parameter_t>;
    using p_dt = gt::arg<4, global_parameter_t>;
    using p_coeff = gt::arg<5, global_parameter_t>;

   public:
    horizontal_diffusion(
        gridtools::grid<typename axis_t::axis_interval_t> const& grid,
        real_t dx, real_t dy, real_t dt, real_t coeff)
        : comp_(gt::make_computation<backend_t>(
              grid, p_dx() = gt::make_global_parameter(dx),
              p_dy() = gt::make_global_parameter(dy),
              p_dt() = gt::make_global_parameter(dt),
              p_coeff() = gt::make_global_parameter(coeff),
              gt::make_multistage(
                  gt::execute::parallel(),
                  gt::make_stage<operators::horizontal_diffusion>(
                      p_out(), p_in(), p_dx(), p_dy(), p_dt(), p_coeff())))) {}

    void run(storage_t& out, storage_t const& in) {
        comp_.run(p_out() = out, p_in() = in);
    }

   private:
    gridtools::computation<p_out, p_in> comp_;
};

class vertical_diffusion {
    using p_data_in = gt::arg<0, storage_t>;
    using p_data_out = gt::arg<1, storage_t>;
    using p_data_top = gt::tmp_arg<2, storage_ij_t>;
    using p_data_bottom = gt::tmp_arg<3, storage_ij_t>;

    using p_dz = gt::arg<4, global_parameter_t>;
    using p_dt = gt::arg<5, global_parameter_t>;
    using p_coeff = gt::arg<6, global_parameter_t>;

    using p_a = gt::tmp_arg<7, storage_t>;
    using p_b = gt::tmp_arg<8, storage_t>;
    using p_c = gt::tmp_arg<9, storage_t>;
    using p_d = gt::tmp_arg<10, storage_t>;

    using p_alpha = gt::tmp_arg<11, storage_ij_t>;
    using p_beta = gt::tmp_arg<12, storage_ij_t>;
    using p_gamma = gt::tmp_arg<13, storage_ij_t>;
    using p_fact = gt::tmp_arg<14, storage_ij_t>;

    using p_z = gt::tmp_arg<15, storage_t>;
    using p_z_top = gt::tmp_arg<16, storage_ij_t>;
    using p_x = gt::tmp_arg<17, storage_t>;

   public:
    vertical_diffusion(
        gridtools::grid<typename axis_t::axis_interval_t> const& grid,
        real_t dz, real_t dt, real_t coeff,
        storage_t::storage_info_t const& sinfo,
        storage_ij_t::storage_info_t const& sinfo_ij)
        : comp_(gt::make_computation<backend_t>(
              grid, p_dz() = gt::make_global_parameter(dz),
              p_dt() = gt::make_global_parameter(dt),
              p_coeff() = gt::make_global_parameter(coeff),
              gt::make_multistage(gt::execute::forward(),
                                  gt::make_stage<operators::diffusion_w0>(
                                      p_data_top(), p_data_in())),
              gt::make_multistage(
                  gt::execute::forward(),
                  gt::make_stage<operators::diffusion_w_fwd1>(
                      p_data_bottom(), p_alpha(), p_beta(), p_gamma(), p_a(),
                      p_b(), p_c(), p_d(), p_data_in(), p_data_top(), p_dz(),
                      p_dt(), p_coeff())),
              gt::make_multistage(gt::execute::backward(),
                                  gt::make_stage<operators::diffusion_w_bwd1>(
                                      p_x(), p_c(), p_d())),
              gt::make_multistage(
                  gt::execute::forward(),
                  gt::make_stage<operators::diffusion_w_fwd2>(
                      p_a(), p_b(), p_c(), p_d(), p_alpha(), p_gamma())),
              gt::make_multistage(
                  gt::execute::backward(),
                  gt::make_stage<operators::diffusion_w_bwd2>(
                      p_z(), p_c(), p_d(), p_x(), p_alpha(), p_beta(),
                      p_gamma(), p_data_top(), p_z_top(), p_fact())),
              gt::make_multistage(gt::execute::parallel(),
                                  gt::make_stage<operators::diffusion_w3>(
                                      p_data_out(), p_x(), p_z(), p_fact(),
                                      p_data_in(), p_dt())))) {}

    void run(storage_t& flux, storage_t const& in) {
        comp_.run(p_data_out() = flux, p_data_in() = in);
    }

   private:
    gridtools::computation<p_data_in, p_data_out> comp_;
};

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

template <typename T>
void dump(std::ostream& os, T const& storage) {
    for (auto const& l : storage.total_lengths()) os << l << " ";
    os << '\n';

    auto v = gt::make_host_view(storage);
    for (int i0 = 0; i0 < storage.template total_length<2>(); ++i0) {
        for (int i1 = 0; i1 < storage.template total_length<1>(); ++i1) {
            for (int i2 = 0; i2 < storage.template total_length<0>(); ++i2)
                os << v(i2, i1, i0) << " ";
            os << '\n';
        }
        os << '\n';
    }
}

struct periodic_boundary {
    template <gt::sign I, gt::sign J, gt::sign K, typename DataField>
    GT_FUNCTION void operator()(gt::direction<I, J, K>, DataField& data,
                                gt::uint_t i, gt::uint_t j,
                                gt::uint_t k) const {
        auto const& si = data.storage_info();
        data(i, j, k) = data(
            (i + si.template length<0>() - halo_i) % si.template length<0>() +
                halo_i,
            (j + si.template length<1>() - halo_j) % si.template length<1>() +
                halo_j,
            (k + si.template length<2>() - halo_k) % si.template length<2>() +
                halo_k);
    }
};

int main() {
    static constexpr int isize = 30, jsize = 30, ksize = 30;
    real_t const dx = 1, dy = 1, dz = 1, dt = 1;
    real_t const diffusion_coefficient = 0.2;

    storage_t::storage_info_t sinfo{isize + 2 * halo_i, jsize + 2 * halo_j,
                                    ksize + 2 * halo_k};
    storage_ij_t::storage_info_t sinfo_ij{isize + 2 * halo_i,
                                          jsize + 2 * halo_j, 1};

    storage_t u{sinfo, real_t(1), "u"}, v{sinfo, real_t(1), "v"};
    storage_t w{sinfo, real_t(1), "w"};
    storage_t data_in{sinfo,
                      [](int i, int j, int k) {
                          return i > 5 && i < 8 && j > 5 && j < 8 && k > 1 &&
                                         k < 8
                                     ? real_t(1)
                                     : real_t(0);
                      },
                      "data"};
    storage_t data_out{sinfo, "data2"};
    storage_t flux{sinfo, "flux"};

    gt::array<gt::halo_descriptor, 3> halos{
        {{halo_i, halo_i, halo_i, halo_i + isize - 1, halo_i + isize + halo_i},
         {halo_j, halo_j, halo_j, halo_j + jsize - 1, halo_j + jsize + halo_j},
         {halo_k, halo_k, halo_k, halo_k + ksize - 1,
          halo_k + ksize + halo_k}}};
    auto grid = gt::make_grid(halos[0], halos[1], axis_t{ksize + 2 * halo_k});
    horizontal_diffusion hdiff(grid, dx, dy, dt, diffusion_coefficient);
    vertical_diffusion vdiff(grid, dz, dt, diffusion_coefficient, sinfo,
                             sinfo_ij);

    gt::boundary<periodic_boundary, backend_t> boundary(halos,
                                                        periodic_boundary{});

    boundary.apply(data_in);

    for (int ts = 0; ts < 10; ++ts) {
        std::ofstream of{"out" + std::to_string(ts)};
        dump(of, data_in);

        hdiff.run(data_out, data_in);
        boundary.apply(data_out);
        std::swap(data_out, data_in);

        vdiff.run(data_out, data_in);
        boundary.apply(data_out);
        std::swap(data_out, data_in);
    }
}
