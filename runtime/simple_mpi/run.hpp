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

#include <array>
#include <functional>
#include <memory>

#include "../computation.hpp"
#include "../discrete_analytical.hpp"
#include "../runtime.hpp"

namespace runtime {
namespace simple_mpi {

struct world {
  world(int &argc, char **&argv);
  world(world const &) = delete;
  world(world &&) = delete;
  ~world();

  world &operator=(world const &) = delete;
  world &operator=(world &&) = delete;
};

void runtime_register_options(world const &, options &options);

struct runtime {
  std::array<int, 2> cart_dims;
};

runtime runtime_init(world const &, options_values const &options);

class process_grid {
public:
  process_grid(vec<std::size_t, 3> const &global_resolution,
               std::array<int, 2> cart_dims);
  ~process_grid();

  vec<std::size_t, 3> local_resolution() const;
  vec<std::size_t, 3> local_offset() const;

  std::function<void(storage_t &)>
  exchanger(storage_info_ijk_t const &sinfo) const;

  double wtime() const;
  result collect_results(result r) const;

private:
  struct impl;
  std::unique_ptr<impl> pimpl;
};

template <class Analytical, class Stepper>
result runtime_solve(runtime &rt, Analytical analytical, Stepper stepper,
                     vec<std::size_t, 3> const &global_resolution, real_t tmax,
                     real_t dt) {
  process_grid grid(global_resolution, rt.cart_dims);

  const auto exact = discrete_analytical::discretize(
      analytical, global_resolution, grid.local_resolution(),
      grid.local_offset());

  auto state = computation::init_state(exact);
  auto exchange = grid.exchanger(state.sinfo);
  auto step = stepper(state, exchange);

  if (tmax > 0)
    step(state, dt);

  double start = grid.wtime();

  real_t t;
  for (t = dt; t < tmax; t += dt)
    step(state, dt);

  computation::sync(state);

  double time = grid.wtime() - start;
  double error = computation::compute_error(state, exact, t);

  return grid.collect_results({error, time});
}

} // namespace simple_mpi
} // namespace runtime