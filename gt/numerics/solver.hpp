#pragma once

#include "./advection.hpp"
#include "./diffusion.hpp"

struct solver_state {
  template <class DataInit, class UInit, class VInit, class WInit>
  solver_state(vec<std::size_t, 3> const &resolution, DataInit &&data_init,
               UInit &&u_init, VInit &&v_init, WInit &&w_init)
      : sinfo(resolution.x + 2 * halo, resolution.y + 2 * halo,
              resolution.z + 1),
        data(sinfo, std::forward<DataInit>(data_init), "data"),
        u(sinfo, std::forward<UInit>(u_init), "u"),
        v(sinfo, std::forward<VInit>(v_init), "v"),
        w(sinfo, std::forward<WInit>(w_init), "w"), data1(sinfo, "data1"),
        data2(sinfo, "data2") {}

  storage_t::storage_info_t sinfo;
  storage_t data, u, v, w, data1, data2;
};

auto hdiff_stepper(real_t diffusion_coeff) {
  return [diffusion_coeff](vec<std::size_t, 3> const &resolution,
                           vec<real_t, 3> const &delta, auto &&exchange) {
    return [hdiff = diffusion::horizontal(resolution, delta, diffusion_coeff),
            exchange = std::move(exchange)](solver_state &state,
                                            real_t dt) mutable {
      exchange(state.data);
      hdiff(state.data1, state.data, dt);
      std::swap(state.data1, state.data);
    };
  };
}

auto vdiff_stepper(real_t diffusion_coeff) {
  return [diffusion_coeff](vec<std::size_t, 3> const &resolution,
                           vec<real_t, 3> const &delta, auto &&) {
    return [vdiff = diffusion::vertical(resolution, delta, diffusion_coeff)](
               solver_state &state, real_t dt) mutable {
      vdiff(state.data1, state.data, dt);
      std::swap(state.data1, state.data);
    };
  };
}

auto diff_stepper(real_t diffusion_coeff) {
  return [diffusion_coeff](vec<std::size_t, 3> const &resolution,
                           vec<real_t, 3> const &delta, auto &&exchange) {
    return [hdiff = diffusion::horizontal(resolution, delta, diffusion_coeff),
            vdiff = diffusion::vertical(resolution, delta, diffusion_coeff),
            exchange = std::move(exchange)](solver_state &state,
                                            real_t dt) mutable {
      exchange(state.data);
      hdiff(state.data1, state.data, dt);
      vdiff(state.data, state.data1, dt);
    };
  };
}

auto hadv_stepper() {
  return [](vec<std::size_t, 3> const &resolution, vec<real_t, 3> const &delta,
            auto &&exchange) {
    return [hadv = advection::horizontal(resolution, delta),
            exchange = std::move(exchange)](solver_state &state,
                                            real_t dt) mutable {
      exchange(state.data);
      hadv(state.data1, state.data, state.u, state.v, dt);
      std::swap(state.data1, state.data);
    };
  };
}

auto vadv_stepper() {
  return [](vec<std::size_t, 3> const &resolution, vec<real_t, 3> const &delta,
            auto &&) {
    return [vadv = advection::vertical(resolution, delta)](solver_state &state,
                                                           real_t dt) mutable {
      vadv(state.data1, state.data, state.w, dt);
      std::swap(state.data1, state.data);
    };
  };
}

auto rkadv_stepper() {
  return [](vec<std::size_t, 3> const &resolution, vec<real_t, 3> const &delta,
            auto &&exchange) {
    return [rkstep = advection::runge_kutta_step(resolution, delta),
            exchange = std::move(exchange)](solver_state &state,
                                            real_t dt) mutable {
      exchange(state.data);
      rkstep(state.data1, state.data, state.data, state.u, state.v, state.w,
             dt / 3);
      exchange(state.data1);
      rkstep(state.data2, state.data1, state.data, state.u, state.v, state.w,
             dt / 2);
      exchange(state.data2);
      rkstep(state.data, state.data2, state.data, state.u, state.v, state.w,
             dt);
    };
  };
}

auto full_stepper(real_t diffusion_coeff) {
  return [diffusion_coeff](vec<std::size_t, 3> const &resolution,
                           vec<real_t, 3> const &delta, auto exchange) {
    return [hdiff = diffusion::horizontal(resolution, delta, diffusion_coeff),
            vdiff = diffusion::vertical(resolution, delta, diffusion_coeff),
            rkstep = advection::runge_kutta_step(resolution, delta),
            exchange = std::move(exchange)](solver_state &state,
                                            real_t dt) mutable {
      // VDIFF
      vdiff(state.data1, state.data, dt);
      std::swap(state.data1, state.data);

      // ADV
      exchange(state.data);
      rkstep(state.data1, state.data, state.data, state.u, state.v, state.w,
             dt / 3);
      exchange(state.data1);
      rkstep(state.data2, state.data1, state.data, state.u, state.v, state.w,
             dt / 2);
      exchange(state.data2);
      rkstep(state.data, state.data2, state.data, state.u, state.v, state.w,
             dt);

      // HDIFF
      exchange(state.data);
      hdiff(state.data1, state.data, dt);
      std::swap(state.data1, state.data);
    };
  };
}