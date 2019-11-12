#include <gridtools/stencil_composition/expressions/expressions.hpp>
#include <gridtools/stencil_composition/stencil_composition.hpp>
#include <gridtools/stencil_composition/stencil_functions.hpp>

#include <iostream>

static constexpr int halo_i = 3;
static constexpr int halo_j = 3;

using real_t = float;

using axis_t = gridtools::axis<1>;
using full_t = axis_t::full_interval;

using backend_t = gridtools::backend::x86;
using storage_tr = gridtools::storage_traits<backend_t>;
using storage_info_ijk_t =
    storage_tr::storage_info_t<0, 3, gridtools::halo<halo_i, halo_j, 0>>;
using storage_info_ijk1_t =
    storage_tr::storage_info_t<0, 3, gridtools::halo<halo_i, halo_j, 1>>;
using storage_type = storage_tr::data_store_t<real_t, storage_info_ijk_t>;
using storage_type_w = storage_tr::data_store_t<real_t, storage_info_ijk1_t>;
using global_parameter_t = gridtools::global_parameter<backend_t, real_t>;

namespace operators {
using namespace gridtools::expressions;

using gridtools::extent;
using gridtools::in_accessor;
using gridtools::inout_accessor;
using gridtools::make_param_list;

struct forward_thomas {
    // four vectors: output, and the 3 diagonals
    using out = inout_accessor<0>;
    using inf = in_accessor<1>;                                // a
    using diag = in_accessor<2>;                               // b
    using sup = inout_accessor<3, extent<0, 0, 0, 0, -1, 0>>;  // c
    using rhs = inout_accessor<4, extent<0, 0, 0, 0, -1, 0>>;  // d
    using param_list = make_param_list<out, inf, diag, sup, rhs>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, 0>) {
        eval(sup()) = eval(sup() / (diag() - sup(0, 0, -1) * inf()));
        eval(rhs()) = eval((rhs() - inf() * rhs(0, 0, -1)) /
                           (diag() - sup(0, 0, -1) * inf()));
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::first_level) {
        eval(sup()) = eval(sup()) / eval(diag());
        eval(rhs()) = eval(rhs()) / eval(diag());
    }
};

struct backward_thomas {
    using out = inout_accessor<0, extent<0, 0, 0, 0, 0, 1>>;
    using sup = inout_accessor<1>;
    using rhs = inout_accessor<2>;
    using param_list = make_param_list<out, sup, rhs>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::modify<0, -1>) {
        eval(out()) = eval(rhs() - sup() * out(0, 0, 1));
    }

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
        eval(out()) = eval(rhs());
    }
};

struct laplacian {
    using out = inout_accessor<0>;
    using in = in_accessor<1, extent<-1, 1, -1, 1>>;
    using param_list = make_param_list<out, in>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t) {
        eval(out()) = eval(-4 * in() + in(-1, 0, 0) + in(1, 0, 0) +
                           in(0, 1, 0) + in(0, -1, 0));
    }
};

struct flux_stage {
    using out = inout_accessor<0>;
    using lap = in_accessor<1, extent<-2, 2, -2, 2>>;
    using in = in_accessor<2, extent<-1, 1, -1, 1>>;
    using param_list = make_param_list<out, lap, in>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t) {
        auto flx_x1 = eval(lap(1, 0, 0) - lap());
        if (eval(flx_x1 * (in(1, 0, 0) - in())) < 0) flx_x1 = 0;

        auto flx_x0 = eval(lap() - lap(-1, 0, 0));
        if (eval(flx_x0 * (in() - in(-1, 0, 0))) < 0) flx_x0 = 0;

        auto flx_y1 = eval(lap(0, 1, 0) - lap());
        if (eval(flx_y1 * (in(0, 1, 0) - in())) < 0) flx_y1 = 0;

        auto flx_y0 = eval(lap() - lap(0, -1, 0));
        if (eval(flx_y0 * (in() - in(0, -1, 0))) < 0) flx_y0 = 0;

        static constexpr real_t diffusion_coefficient = 0.1;
        eval(out()) = eval(in() - diffusion_coefficient *
                                      (flx_x1 - flx_x0 + flx_y1 - flx_y0));
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

struct diffusion_w_fwd {
    using out = inout_accessor<0>;
    using data = in_accessor<1>;
    using dz = in_accessor<2>;
    using dt = in_accessor<3>;

    using a = inout_accessor<4>;
    using b = inout_accessor<5>;
    using c = inout_accessor<6, extent<0, 0, 0, 0, -1, 0>>;
    using d = inout_accessor<7, extent<0, 0, 0, 0, -1, 0>>;

    using param_list = make_param_list<out, data, dz, dt, a, b, c, d>;
    static constexpr real_t diffusion_coefficient = 0.1;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::first_level) {
        eval(a()) = real_t(0);
        eval(c()) = -diffusion_coefficient / real_t(2) * eval(dt());
        eval(b()) = eval(real_t(1) / dt() - a() - c());
        eval(d()) = eval(real_t(1) / dt() * data() +
                         real_t(0.5) * diffusion_coefficient *
                             (data(0, 0, 1) - real_t(2) * (data())) / dz());
        gridtools::call_proc<forward_thomas, full_t::first_level>::with(
            eval, out(), a(), b(), c(), d());
    }
    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::modify<1, -1>) {
        eval(a()) = -diffusion_coefficient / real_t(2) * eval(dt());
        eval(c()) = -diffusion_coefficient / real_t(2) * eval(dt());
        eval(b()) = eval(real_t(1) / dt() - a() - c());
        eval(d()) = eval(
            real_t(1) / dt() * data() +
            real_t(0.5) * diffusion_coefficient *
                (data(0, 0, 1) - real_t(2) * (data() + data(0, 0, -1))) / dz());
        gridtools::call_proc<forward_thomas, full_t::modify<1, 0>>::with(
            eval, out(), a(), b(), c(), d());
    }
    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
        eval(a()) = -diffusion_coefficient / real_t(2) * eval(dt());
        eval(c()) = real_t(0);
        eval(b()) = eval(real_t(1) / dt() - a() - c());
        eval(d()) = eval(real_t(1) / dt() * data() +
                         real_t(0.5) * diffusion_coefficient *
                             (-real_t(2) * (data() + data(0, 0, -1))) / dz());
        gridtools::call_proc<forward_thomas, full_t::modify<1, 0>>::with(
            eval, out(), a(), b(), c(), d());
    }
};
struct diffusion_w_bwd {
    using out = inout_accessor<0>;
    using data = in_accessor<1>;
    using dt = in_accessor<2>;

    using c = inout_accessor<3, extent<0, 0, 0, 0, -1, 0>>;
    using d = inout_accessor<4, extent<0, 0, 0, 0, -1, 0>>;

    using param_list = make_param_list<out, data, dt, c, d>;

    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::modify<0, -1>) {
        gridtools::call_proc<backward_thomas, full_t::modify<0, -1>>::with(
            eval, out(), c(), d());
        eval(out()) = eval((out() - data()) / dt());
    }
    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
        gridtools::call_proc<backward_thomas, full_t::last_level>::with(
            eval, out(), c(), d());
        eval(out()) = eval((out() - data()) / dt());
    }
};

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
        gridtools::call_proc<forward_thomas, full_t::first_level>::with(
            eval, out(), a(), b(), c(), d());
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
        gridtools::call_proc<forward_thomas, full_t::modify<1, 0>>::with(
            eval, out(), a(), b(), c(), d());
    }
    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
        eval(a()) = eval(real_t(-0.25) * w() / dz());
        eval(c()) = eval(real_t(0.25) * w(0, 0, 1) / dz());
        eval(b()) = eval(real_t(1) / dt() - a() - c());
        eval(d()) = eval(real_t(1) / dt() * data() -
                         real_t(0.25) * w() * (data() - data(0, 0, -1)) / dz() -
                         real_t(0.25) * w(0, 0, 1) * -data() / dz());
        gridtools::call_proc<forward_thomas, full_t::first_level>::with(
            eval, out(), a(), b(), c(), d());
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
        gridtools::call_proc<backward_thomas, full_t::modify<0, -1>>::with(
            eval, out(), c(), d());
        eval(out()) = eval((out() - data()) / dt());
    }
    template <typename Evaluation>
    GT_FUNCTION static void apply(Evaluation eval, full_t::last_level) {
        gridtools::call_proc<backward_thomas, full_t::last_level>::with(
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

}  // namespace operators

namespace gt = gridtools;

class horizontal_diffusion {
    using p_out = gt::arg<0, storage_type>;
    using p_in = gt::arg<1, storage_type>;
    using p_lap = gt::tmp_arg<2, storage_type>;

   public:
    horizontal_diffusion(gridtools::grid<axis_t::axis_interval_t> const& grid)
        : comp_(gt::make_computation<backend_t>(
              grid, gt::make_multistage(
                        gt::execute::parallel(),
                        gt::make_stage<operators::laplacian>(p_lap(), p_in()),
                        gt::make_stage<operators::flux_stage>(p_out(), p_lap(),
                                                              p_in())))) {}

   private:
    gridtools::computation<p_out, p_in> comp_;
};

class vertical_diffusion {
    using p_flux = gt::arg<0, storage_type>;
    using p_in = gt::arg<1, storage_type>;

    using p_dz = gt::arg<2, global_parameter_t>;
    using p_dt = gt::arg<3, global_parameter_t>;

    using p_a = gt::tmp_arg<4, storage_type>;
    using p_b = gt::tmp_arg<5, storage_type>;
    using p_c = gt::tmp_arg<6, storage_type>;
    using p_d = gt::tmp_arg<7, storage_type>;

   public:
    vertical_diffusion(gridtools::grid<axis_t::axis_interval_t> const& grid)
        : comp_(gt::make_computation<backend_t>(
              grid,
              gt::make_multistage(gt::execute::forward(),
                                  gt::make_stage<operators::diffusion_w_fwd>(
                                      p_flux(), p_in(), p_dz(), p_dt(), p_a(),
                                      p_b(), p_c(), p_d())),
              gt::make_multistage(
                  gt::execute::backward(),
                  gt::make_stage<operators::diffusion_w_bwd>(
                      p_flux(), p_in(), p_dt(), p_c(), p_d())))) {}

   private:
    gridtools::computation<p_flux, p_in, p_dz, p_dt> comp_;
};

class advection {
    using p_flux = gt::arg<0, storage_type>;
    using p_u = gt::arg<1, storage_type>;
    using p_v = gt::arg<2, storage_type>;
    using p_w = gt::arg<3, storage_type_w>;
    using p_in = gt::arg<4, storage_type>;
    using p_dx = gt::arg<5, global_parameter_t>;
    using p_dy = gt::arg<6, global_parameter_t>;
    using p_dz = gt::arg<7, global_parameter_t>;
    using p_dt = gt::arg<8, global_parameter_t>;

    using p_a = gt::tmp_arg<9, storage_type>;
    using p_b = gt::tmp_arg<10, storage_type>;
    using p_c = gt::tmp_arg<11, storage_type>;
    using p_d = gt::tmp_arg<12, storage_type>;

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

template <typename T>
void dump(std::ostream& os, T const& storage) {
    for (auto const& l : storage.total_lengths()) os << l << " ";
    os << '\n';

    auto v = gt::make_host_view(storage);
    for (int i0 = 0; i0 < storage.template total_length<0>(); ++i0) {
        for (int i1 = 0; i1 < storage.template total_length<1>(); ++i1) {
            for (int i2 = 0; i2 < storage.template total_length<2>(); ++i2)
                os << v(i0, i1, i2) << " ";
            os << '\n';
        }
        os << '\n';
    }
}

int main() {
    static constexpr int isize = 10, jsize = 10, ksize = 10;

    storage_type::storage_info_t sinfo{isize, jsize, ksize};
    storage_type_w::storage_info_t sinfo_w{isize, jsize, ksize};

    storage_type u{sinfo, real_t(1), "u"}, v{sinfo, real_t(1), "v"};
    storage_type_w w{sinfo_w, real_t(1), "w"};
    storage_type data{sinfo,
                      [](int i, int j, int k) {
                          return i > 3 && i < 8 && j > 3 && j < 8 && k > 3 &&
                                         k < 8
                                     ? real_t(1)
                                     : real_t(0);
                      },
                      "data"};

    auto grid = gt::make_grid(
        {halo_i, halo_i, halo_i, halo_i + isize - 1, halo_i + isize + halo_i},
        {halo_j, halo_j, halo_j, halo_j + jsize - 1, halo_j + jsize + halo_j},
        10);
    horizontal_diffusion hdiff(grid);

    dump(std::cout, data);
}
