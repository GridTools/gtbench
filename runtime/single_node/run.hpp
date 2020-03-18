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

#include <chrono>

#include "../computation.hpp"
#include "../discrete_analytical.hpp"
#include "../runtime.hpp"

namespace runtime {

struct single_node {
  single_node(int, char **) {}
};

namespace single_node_impl {
numerics::exchange_t setup(vec<std::size_t, 3> const &resolution);
}

template <class Analytical, class Stepper>
result runtime_solve(single_node, Analytical analytical, Stepper stepper,
                     vec<std::size_t, 3> const &global_resolution, real_t tmax,
                     real_t dt) {
  const auto exact = discrete_analytical::discretize(
      analytical, global_resolution, global_resolution, {0, 0});

  auto state = computation::init_state(exact);
  auto exchange = single_node_impl::setup(global_resolution);

  auto step = stepper(state, exchange);

  if (tmax > 0)
    step(state, dt);

  using clock = std::chrono::high_resolution_clock;

  auto start = clock::now();

  real_t t;
  for (t = dt; t < tmax; t += dt)
    step(state, dt);

  computation::sync(state);

  auto stop = clock::now();
  double time = std::chrono::duration<double>(stop - start).count();
  double error = computation::compute_error(state, exact, t);

  return {error, time};
}
} // namespace runtime