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

struct hdiff_stepper_f {
  void operator()(solver_state &state, real_t dt) {
    exchange(state.data);
    hdiff(state.data1, state.data, dt);
    std::swap(state.data1, state.data);
  }

  diffusion::horizontal hdiff;
  std::function<void(storage_t &)> exchange;
};

auto hdiff_stepper(real_t diffusion_coeff) {
  return [diffusion_coeff](vec<std::size_t, 3> const &resolution,
                           vec<real_t, 3> const &delta, auto exchange) {
    return hdiff_stepper_f{{resolution, delta, diffusion_coeff}, {exchange}};
  };
}

struct vdiff_stepper_f {
  void operator()(solver_state &state, real_t dt) {
    vdiff(state.data1, state.data, dt);
    std::swap(state.data1, state.data);
  }

  diffusion::vertical vdiff;
};

auto vdiff_stepper(real_t diffusion_coeff) {
  return [diffusion_coeff](vec<std::size_t, 3> const &resolution,
                           vec<real_t, 3> const &delta, auto exchange) {
    return vdiff_stepper_f{{resolution, delta, diffusion_coeff}};
  };
}

struct diff_stepper_f {
  void operator()(solver_state &state, real_t dt) {
    exchange(state.data);
    hdiff(state.data1, state.data, dt);
    vdiff(state.data, state.data1, dt);
  }

  diffusion::horizontal hdiff;
  diffusion::vertical vdiff;
  std::function<void(storage_t &)> exchange;
};

auto diff_stepper(real_t diffusion_coeff) {
  return [diffusion_coeff](vec<std::size_t, 3> const &resolution,
                           vec<real_t, 3> const &delta, auto exchange) {
    return diff_stepper_f{{resolution, delta, diffusion_coeff},
                          {resolution, delta, diffusion_coeff},
                          {exchange}};
  };
}

struct hadv_stepper_f {
  void operator()(solver_state &state, real_t dt) {
    exchange(state.data);
    hadv(state.data1, state.data, state.u, state.v, dt);
    std::swap(state.data1, state.data);
  }

  advection::horizontal hadv;
  std::function<void(storage_t &)> exchange;
};

auto hadv_stepper() {
  return [](vec<std::size_t, 3> const &resolution, vec<real_t, 3> const &delta,
            auto exchange) {
    return hadv_stepper_f{{resolution, delta}, {exchange}};
  };
}

struct vadv_stepper_f {
  void operator()(solver_state &state, real_t dt) {
    vadv(state.data1, state.data, state.w, dt);
    std::swap(state.data1, state.data);
  }

  advection::vertical vadv;
};

auto vadv_stepper() {
  return [](vec<std::size_t, 3> const &resolution, vec<real_t, 3> const &delta,
            auto exchange) {
    return vadv_stepper_f{{resolution, delta}};
  };
}

struct rkadv_stepper_f {
  void operator()(solver_state &state, real_t dt) {
    exchange(state.data);
    rk_step(state.data1, state.data, state.data, state.u, state.v, state.w,
            dt / 3);
    exchange(state.data1);
    rk_step(state.data2, state.data1, state.data, state.u, state.v, state.w,
            dt / 2);
    exchange(state.data2);
    rk_step(state.data, state.data2, state.data, state.u, state.v, state.w, dt);
  }

  advection::runge_kutta_step rk_step;
  std::function<void(storage_t &)> exchange;
};

auto rkadv_stepper() {
  return [](vec<std::size_t, 3> const &resolution, vec<real_t, 3> const &delta,
            auto exchange) {
    return rkadv_stepper_f{{resolution, delta}, {exchange}};
  };
}

struct full_stepper_f {
  void operator()(solver_state &state, real_t dt) {
    // VDIFF
    vdiff(state.data1, state.data, dt);
    std::swap(state.data1, state.data);

    // ADV
    exchange(state.data);
    rk_step(state.data1, state.data, state.data, state.u, state.v, state.w,
            dt / 3);
    exchange(state.data1);
    rk_step(state.data2, state.data1, state.data, state.u, state.v, state.w,
            dt / 2);
    exchange(state.data2);
    rk_step(state.data, state.data2, state.data, state.u, state.v, state.w, dt);

    // HDIFF
    exchange(state.data);
    hdiff(state.data1, state.data, dt);
    std::swap(state.data1, state.data);
  }

  diffusion::horizontal hdiff;
  diffusion::vertical vdiff;
  advection::runge_kutta_step rk_step;
  std::function<void(storage_t &)> exchange;
};

auto full_stepper(real_t diffusion_coeff) {
  return [diffusion_coeff](vec<std::size_t, 3> const &resolution,
                           vec<real_t, 3> const &delta, auto exchange) {
    return full_stepper_f{{resolution, delta, diffusion_coeff},
                          {resolution, delta, diffusion_coeff},
                          {resolution, delta},
                          {exchange}};
  };
}