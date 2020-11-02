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

#include "../../io/io.hpp"
#include "../function_scope.hpp"
#include "../runtime.hpp"

namespace runtime {

struct simple_mpi {};

namespace simple_mpi_impl {

void runtime_register_options(simple_mpi, options &options);

struct runtime {
  runtime(std::array<int, 2> const &cart_dims,
          std::string const &output_filename);

  function_scope m_scope;
  std::array<int, 2> m_cart_dims;
  std::string m_output_filename;
};

runtime runtime_init(simple_mpi, options_values const &options);

class process_grid {
public:
  process_grid(vec<std::size_t, 3> const &global_resolution,
               std::array<int, 2> cart_dims);
  ~process_grid();

  vec<std::size_t, 3> local_resolution() const;
  vec<std::size_t, 3> local_offset() const;

  std::function<void(storage_t &)>
  exchanger(gt::array<unsigned, 3> const &sizes,
            gt::array<unsigned, 3> const &strides) const;

  double wtime() const;
  result collect_results(result r) const;

private:
  struct impl;
  std::unique_ptr<impl> m_impl;
};

template <class Analytical, class Stepper>
result runtime_solve(runtime &rt, Analytical analytical, Stepper stepper,
                     vec<std::size_t, 3> const &global_resolution, real_t tmax,
                     real_t dt) {
  process_grid grid(global_resolution, rt.m_cart_dims);

  const auto exact = discrete_analytical::discretize(
      analytical, global_resolution, grid.local_resolution(),
      grid.local_offset());

  auto state = computation::init_state(exact);
  auto exchange =
      grid.exchanger(state.sinfo().lengths(), state.sinfo().strides());
  auto step = stepper(state, exchange);

  auto write =
      io::write_time_series(rt.m_output_filename, global_resolution,
                            grid.local_resolution(), grid.local_offset());
  if (write)
    write(0, state);

  if (tmax > 0)
    step(state, dt);

  double start = grid.wtime();

  real_t t;
  for (t = dt; t < tmax - dt / 2; t += dt)
    step(state, dt);

  computation::sync(state);
  if (write)
    write(t, state);

  double time = grid.wtime() - start;
  double error = computation::compute_error(state, exact, t);

  return grid.collect_results({error, time});
}

} // namespace simple_mpi_impl

using simple_mpi_impl::runtime_init;
using simple_mpi_impl::runtime_register_options;
using simple_mpi_impl::runtime_solve;

} // namespace runtime
