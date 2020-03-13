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

#include "../communication/communication.hpp"
#include "../numerics/solver.hpp"
#include "../runtime/discrete_analytical.hpp"
#include "../runtime/interface.hpp"
#include "./timer.hpp"

namespace execution {

struct result {
  double error;
  double time;
};

template <class CommGrid, class Stepper, class Analytical>
result run(CommGrid &&comm_grid, Stepper &&stepper, real_t tmax, real_t dt,
           Analytical &&exact) {
  const auto initial = runtime::discrete_analytical::discretize(
      exact, communication::global_resolution(comm_grid),
      communication::resolution(comm_grid), communication::offset(comm_grid));

  const auto n = runtime::discrete_analytical::local_resolution(initial);

  auto state = runtime::init_state(initial);

  auto exchange = communication::halo_exchanger(comm_grid, state.sinfo);

  auto step = stepper(state, exchange);

  communication::barrier(comm_grid);
  if (tmax > 0)
    step(state, dt); // do not measure execution time of inital step

  auto start = timer::now(backend_t{});
  real_t t;
  for (t = dt; t < tmax; t += dt)
    step(state, dt);
  auto stop = timer::now(backend_t{});
  double time = timer::duration(start, stop);
  communication::barrier(comm_grid);

  state.data.sync();
  auto view = gt::make_host_view(state.data);

  const auto expected = runtime::discrete_analytical::discretize(
      exact, communication::global_resolution(comm_grid),
      communication::resolution(comm_grid), communication::offset(comm_grid));
  double error = runtime::compute_error(state, expected, t);

  return {communication::global_max(comm_grid, error),
          communication::global_max(comm_grid, time)};
}

} // namespace execution
