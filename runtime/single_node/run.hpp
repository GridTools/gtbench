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

#include "../../io/io.hpp"
#include "../runtime.hpp"

namespace runtime {

struct single_node {};

namespace single_node_impl {

void runtime_register_options(single_node, options &options);

struct runtime {
  std::string m_output_filename;
};

runtime runtime_init(single_node, options_values const &options);

numerics::exchange_t exchange_func(vec<std::size_t, 3> const &resolution);

template <class Analytical, class Stepper>
result runtime_solve(runtime const &rt, Analytical analytical, Stepper stepper,
                     vec<std::size_t, 3> const &global_resolution, real_t tmax,
                     real_t dt) {
  const auto exact = discrete_analytical::discretize(
      analytical, global_resolution, global_resolution, {0, 0, 0});

  auto state = computation::init_state(exact);
  auto exchange = exchange_func(global_resolution);

  auto step = stepper(state, exchange);

  auto write = io::write_time_series(rt.m_output_filename, global_resolution,
                                     global_resolution, {0, 0, 0});
  if (write)
    write(0, state);

  if (tmax > 0)
    step(state, dt);

  using clock = std::chrono::high_resolution_clock;

  auto start = clock::now();

  real_t t;
  for (t = dt; t < tmax; t += dt)
    step(state, dt);

  computation::sync(state);
  if (write)
    write(0, state);

  auto stop = clock::now();
  double time = std::chrono::duration<double>(stop - start).count();
  double error = computation::compute_error(state, exact, t);

  return {error, time};
}

} // namespace single_node_impl

using single_node_impl::runtime_init;
using single_node_impl::runtime_register_options;
using single_node_impl::runtime_solve;

} // namespace runtime
