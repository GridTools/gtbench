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

#include <memory>

#include "../computation.hpp"
#include "../discrete_analytical.hpp"
#include "../runtime.hpp"

#include "./factorize.hpp"
#include <ghex/communication_object_2.hpp>
#include <ghex/glue/gridtools/field.hpp>
#include <ghex/structured/grid.hpp>
#include <ghex/structured/pattern.hpp>
#include <ghex/threads/atomic/primitives.hpp>
#include <ghex/threads/std_thread/primitives.hpp>
#include <mpi.h>
#include <numeric>

#ifdef GTBENCH_USE_GHEX_UCP
#include <ghex/transport_layer/ucx/context.hpp>
using transport = gridtools::ghex::tl::ucx_tag;
#else
#include <ghex/transport_layer/mpi/context.hpp>
using transport = gridtools::ghex::tl::mpi_tag;
#endif

#include <chrono>
#include <iostream>

namespace runtime {

namespace ghex_comm {

struct world {
  world(int &argc, char **&argv);
  world(world const &) = delete;
  world(world &&) = delete;
  ~world();

  world &operator=(world const &) = delete;
  world &operator=(world &&) = delete;
};

struct sub_grid {
  vec<std::size_t, 3> resolution;
  vec<std::size_t, 2> offset;
  std::function<void(storage_t &)> halo_exchanger;
};

class grid {
public:
  grid(vec<std::size_t, 3> const &global_resolution, int num_sub_domains);
  ~grid();

  sub_grid operator[](unsigned i);

private:
  struct impl;
  std::unique_ptr<impl> pimpl;
};

void runtime_register_options(world const &, options &options);

struct runtime {
  world const &w;
  int num_threads;
};

runtime runtime_init(world const &, options_values const &options);

template <class Analytical, class Stepper>
result runtime_solve(runtime &rt, Analytical analytical, Stepper stepper,
                     vec<std::size_t, 3> const &global_resolution, real_t tmax,
                     real_t dt) {
  grid comm_grid = {global_resolution, rt.num_threads};

  std::vector<result> results(rt.num_threads);
  auto execution_func = [&](int id = 0) {
    auto sub_grid = comm_grid[id];
    const auto exact = discrete_analytical::discretize(
        analytical, global_resolution, sub_grid.resolution, sub_grid.offset);

    auto state = computation::init_state(exact);
    auto exchange = sub_grid.halo_exchanger;
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

    results[id] = {error, time};
  };

  std::vector<std::thread> threads;
  threads.reserve(rt.num_threads);
  for (int i = 0; i < rt.num_threads; ++i)
    threads.emplace_back(execution_func, i);

  for (auto &thread : threads)
    thread.join();

  result local_result{0.0, 0.0};
  for (auto const &r : results) {
    local_result.error = std::max(local_result.error, r.error);
    local_result.time = std::max(local_result.time, r.time);
  }

  result global_result;
  MPI_Allreduce(&local_result, &global_result, 2, MPI_DOUBLE, MPI_MAX,
                MPI_COMM_WORLD);

  return global_result;
}

} // namespace ghex_comm

} // namespace runtime