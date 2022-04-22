/*
 * gtbench
 *
 * Copyright (c) 2014-2022, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gtbench/numerics/solver.hpp>

namespace gtbench {
namespace numerics {
stepper_t hdiff_stepper(real_t diffusion_coeff) {
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

stepper_t vdiff_stepper(real_t diffusion_coeff) {
  return [diffusion_coeff](solver_state const &state, exchange_t) {
    return [vdiff = diffusion::vertical(state.resolution, state.delta,
                                        diffusion_coeff)](solver_state &state,
                                                          real_t dt) mutable {
      vdiff(state.data1, state.data, dt);
      std::swap(state.data1, state.data);
    };
  };
}

stepper_t diff_stepper(real_t diffusion_coeff) {
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

stepper_t hadv_stepper() {
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

stepper_t vadv_stepper() {
  return [](solver_state const &state, exchange_t) {
    return [vadv = advection::vertical(state.resolution, state.delta)](
               solver_state &state, real_t dt) mutable {
      vadv(state.data1, state.data, state.data, state.w, dt);
      std::swap(state.data1, state.data);
    };
  };
}

stepper_t rkadv_stepper() {
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

stepper_t advdiff_stepper(real_t diffusion_coeff) {
  return [diffusion_coeff](solver_state const &state, exchange_t exchange) {
    return [hdiff = hdiff_stepper(diffusion_coeff)(state, exchange),
            vdiff = vdiff_stepper(diffusion_coeff)(state, exchange),
            rkadv = rkadv_stepper()(state, exchange)](solver_state &state,
                                                      real_t dt) mutable {
      hdiff(state, dt);
      rkadv(state, dt);
      vdiff(state, dt);
    };
  };
}

} // namespace numerics
} // namespace gtbench
