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

struct simple_mpi {
  simple_mpi(int &argc, char **&argv);
  simple_mpi(simple_mpi const &) = delete;
  simple_mpi(simple_mpi &&) = delete;
  ~simple_mpi();

  simple_mpi &operator=(simple_mpi const &) = delete;
  simple_mpi &operator=(simple_mpi &&) = delete;

  std::array<int, 2> cart_dims;
};

namespace simple_mpi_impl {

class grid {
public:
  grid(vec<std::size_t, 3> const &global_resolution,
       std::array<int, 2> cart_dims);
  ~grid();

  vec<std::size_t, 3> local_resolution() const;
  vec<std::size_t, 2> local_offset() const;

  std::function<void(storage_t &)>
  exchanger(storage_info_ijk_t const &sinfo) const;

  double wtime() const;

private:
  struct impl;
  std::unique_ptr<impl> pimpl;
};

} // namespace simple_mpi_impl

template <class Analytical, class Stepper>
result runtime_solve(simple_mpi &rt, Analytical analytical, Stepper stepper,
                     vec<std::size_t, 3> const &global_resolution, real_t tmax,
                     real_t dt) {
  simple_mpi_impl::grid grid(global_resolution, rt.cart_dims);

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

  return {error, time};
}

} // namespace runtime