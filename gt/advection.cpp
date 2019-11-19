#include "advection.hpp"

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
struct stage_v {
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

struct stage_horizontal {
  using out = inout_accessor<0>;
  using in = in_accessor<1, extent<-3, 3, -3, 3>>;
  using u = in_accessor<2, extent<-3, 3, 0, 0>>;
  using v = in_accessor<3, extent<0, 0, -3, 3>>;

  using dx = in_accessor<4>;
  using dy = in_accessor<5>;
  using dt = in_accessor<6>;

  using param_list = make_param_list<out, in, u, v, dx, dy, dt>;

  template <typename Evaluation>
  GT_FUNCTION static void apply(Evaluation eval, full_t) {
    auto flx = gridtools::call<stage_u, full_t>::with(eval, u(), in(), dx());
    auto fly = gridtools::call<stage_v, full_t>::with(eval, v(), in(), dy());

    // auto flx = eval(u()) >= 0 ? eval((u() * in() - u(-1, 0) * in(-1, 0)) /
    // dx())
    //                           : eval((u(1, 0) * in(1, 0) - u() * in()) /
    //                           dx());
    // auto fly = eval(v()) >= 0 ? eval((v() * in() - v(0, -1) * in(0, -1)) /
    // dy())
    //                           : eval((v(0, 1) * in(0, 1) - v() * in()) /
    //                           dy());

    eval(out()) = eval(in() - dt() * (flx + fly));
  }
};

} // namespace

horizontal::horizontal(grid_t const &grid, real_t dx, real_t dy, real_t dt)
    : comp_(gt::make_computation<backend_t>(
          grid, p_dx() = gt::make_global_parameter(dx),
          p_dy() = gt::make_global_parameter(dy),
          p_dt() = gt::make_global_parameter(dt),
          gt::make_multistage(
              gt::execute::parallel(),
              gt::make_stage<stage_horizontal>(p_out(), p_in(), p_u(), p_v(),
                                               p_dx(), p_dy(), p_dt())))) {}

void horizontal::operator()(solver_state &state) {
  comp_.run(p_out() = state.tmp, p_in() = state.data, p_u() = state.u,
            p_v() = state.v);
  std::swap(state.tmp, state.data);
}

} // namespace advection