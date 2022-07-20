/*
 * gtbench
 *
 * Copyright (c) 2014-2022, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include "./advection.hpp"
#include "./diffusion.hpp"

namespace gtbench {
namespace numerics {

struct solver_state {
  solver_state(vec<std::size_t, 3> const &resolution,
               vec<real_t, 3> const &delta, storage_t const &u,
               storage_t const &v, storage_t const &w, storage_t const &data,
               storage_t const &data1, storage_t const &data2)
      : resolution(resolution), delta(delta), u(u), v(v), w(w), data(data),
        data1(data1), data2(data2) {}
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

  storage_t u, v, w, data, data1, data2;
};

using exchange_t = std::function<void(storage_t &)>;
using step_t = std::function<void(solver_state &state, real_t dt)>;
using stepper_t =
    std::function<step_t(solver_state const &state, exchange_t exchange)>;

stepper_t hdiff_stepper(real_t diffusion_coeff);

stepper_t vdiff_stepper(real_t diffusion_coeff);

stepper_t diff_stepper(real_t diffusion_coeff);

stepper_t hadv_stepper();

stepper_t vadv_stepper();

stepper_t rkadv_stepper();

stepper_t advdiff_stepper(real_t diffusion_coeff);

} // namespace numerics
} // namespace gtbench
