/*
 * gtbench
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "./advection.hpp"
#include "./diffusion.hpp"

namespace numerics {

struct solver_state {
  solver_state(vec<std::size_t, 3> const &resolution,
               vec<real_t, 3> const &delta)
      : resolution(resolution), delta(delta) {
    auto builder = storage_builder(resolution);
    data = builder.name("data")();
    u = builder.name("u")();
    v = builder.name("v")();
    w = builder.name("w")();
    data1 = builder.name("data1")();
    data2 = builder.name("data2")();
  }
  auto sinfo() const { return data->info(); }

  vec<std::size_t, 3> resolution;
  vec<real_t, 3> delta;

  storage_t data, u, v, w, data1, data2;
};

using exchange_t = std::function<void(storage_t &)>;

inline auto hdiff_stepper(real_t diffusion_coeff) {
  return [diffusion_coeff](solver_state const &state, exchange_t exchange) {
    return [hdiff = diffusion::horizontal(state.resolution, state.delta,
                                          diffusion_coeff),
            exchange = std::move(exchange)](solver_state &state,
                                            real_t dt) mutable {
      exchange(state.data);
      hdiff(state.data1, state.data, dt);
      std::swap(state.data1, state.data);
    };
  };
}

inline auto vdiff_stepper(real_t diffusion_coeff) {
  return [diffusion_coeff](solver_state const &state, exchange_t) {
    return [vdiff = diffusion::vertical(state.resolution, state.delta,
                                        diffusion_coeff)](solver_state &state,
                                                          real_t dt) mutable {
      vdiff(state.data1, state.data, dt);
      std::swap(state.data1, state.data);
    };
  };
}

inline auto diff_stepper(real_t diffusion_coeff) {
  return [diffusion_coeff](solver_state const &state, exchange_t exchange) {
    return [hdiff = diffusion::horizontal(state.resolution, state.delta,
                                          diffusion_coeff),
            vdiff = diffusion::vertical(state.resolution, state.delta,
                                        diffusion_coeff),
            exchange = std::move(exchange)](solver_state &state,
                                            real_t dt) mutable {
      exchange(state.data);
      hdiff(state.data1, state.data, dt);
      vdiff(state.data, state.data1, dt);
    };
  };
}

inline auto hadv_stepper() {
  return [](solver_state const &state, exchange_t exchange) {
    return [hadv = advection::horizontal(state.resolution, state.delta),
            exchange = std::move(exchange)](solver_state &state,
                                            real_t dt) mutable {
      exchange(state.data);
      hadv(state.data1, state.data, state.data, state.u, state.v, dt);
      std::swap(state.data1, state.data);
    };
  };
}

inline auto vadv_stepper() {
  return [](solver_state const &state, exchange_t) {
    return [vadv = advection::vertical(state.resolution, state.delta)](
               solver_state &state, real_t dt) mutable {
      vadv(state.data1, state.data, state.data, state.w, dt);
      std::swap(state.data1, state.data);
    };
  };
}

inline auto rkadv_stepper() {
  return [](solver_state const &state, exchange_t exchange) {
    return [hadv = advection::horizontal(state.resolution, state.delta),
            vadv = advection::vertical(state.resolution, state.delta),
            exchange = std::move(exchange)](solver_state &state,
                                            real_t dt) mutable {
      exchange(state.data);
      hadv(state.data1, state.data, state.data, state.u, state.v, dt / 3);
      vadv(state.data1, state.data, state.data1, state.w, dt / 3);
      exchange(state.data1);
      hadv(state.data2, state.data1, state.data, state.u, state.v, dt / 2);
      vadv(state.data2, state.data1, state.data2, state.w, dt / 2);
      exchange(state.data2);
      hadv(state.data1, state.data2, state.data, state.u, state.v, dt);
      vadv(state.data, state.data2, state.data1, state.w, dt);
    };
  };
}

inline auto advdiff_stepper(real_t diffusion_coeff) {
  return [diffusion_coeff](solver_state const &state, exchange_t exchange) {
    return [hdiff = diffusion::horizontal(state.resolution, state.delta,
                                          diffusion_coeff),
            vdiff = diffusion::vertical(state.resolution, state.delta,
                                        diffusion_coeff),
            hadv = advection::horizontal(state.resolution, state.delta),
            vadv = advection::vertical(state.resolution, state.delta),
            exchange = std::move(exchange)](solver_state &state,
                                            real_t dt) mutable {
      // VDIFF
      vdiff(state.data1, state.data, dt);
      std::swap(state.data1, state.data);

      // ADV
      exchange(state.data);
      hadv(state.data1, state.data, state.data, state.u, state.v, dt / 3);
      vadv(state.data1, state.data, state.data1, state.w, dt / 3);
      exchange(state.data1);
      hadv(state.data2, state.data1, state.data, state.u, state.v, dt / 2);
      vadv(state.data2, state.data1, state.data2, state.w, dt / 2);
      exchange(state.data2);
      hadv(state.data1, state.data2, state.data, state.u, state.v, dt);
      vadv(state.data, state.data2, state.data1, state.w, dt);

      // HDIFF
      exchange(state.data);
      hdiff(state.data1, state.data, dt);
      std::swap(state.data1, state.data);
    };
  };
}

} // namespace numerics
