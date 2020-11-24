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
#include <chrono>
#include <memory>
#include <thread>
#include <vector>

#include "../../io/io.hpp"
#include "../device/set_device.hpp"
#include "../function_scope.hpp"
#include "../runtime.hpp"

namespace runtime {

struct ghex_comm {};

namespace ghex_comm_impl {

void runtime_register_options(ghex_comm, options &options);

struct runtime {
  runtime(int num_threads, std::array<int, 2> cart_dims,
          std::array<int, 2> thread_cart_dims,
          std::vector<int> const &device_mapping,
          std::string const &output_filename);

  function_scope m_scope;
  int m_num_threads;
  std::array<int, 2> m_cart_dims;
  std::array<int, 2> m_thread_cart_dims;
  std::vector<int> m_device_mapping;
  std::string m_output_filename;
};

runtime runtime_init(ghex_comm, options_values const &options);

struct sub_grid {
  vec<std::size_t, 3> m_local_resolution, m_local_offset;
  std::function<void(storage_t &)> m_halo_exchanger;
};

class grid {
public:
  grid(vec<std::size_t, 3> const &global_resolution, int num_sub_domains,
       std::array<int, 2> cart_dims, std::array<int, 2> thread_cart_dims);
  ~grid();

  sub_grid operator[](unsigned i);

  result collect_results(result const &r) const;

private:
  struct impl;
  std::unique_ptr<impl> m_impl;
};

template <class Analytical, class Stepper>
result runtime_solve(runtime &rt, Analytical analytical, Stepper stepper,
                     vec<std::size_t, 3> const &global_resolution, real_t tmax,
                     real_t dt) {
  grid comm_grid = {global_resolution, rt.m_num_threads, rt.m_cart_dims,
                    rt.m_thread_cart_dims};

  std::vector<result> results(rt.m_num_threads);
  auto execution_func = [&](int id = 0) {
    set_device(rt.m_device_mapping[id]);
    auto sub_grid = comm_grid[id];
    const auto exact = discrete_analytical::discretize(
        analytical, global_resolution, sub_grid.m_local_resolution,
        sub_grid.m_local_offset);

    auto state = computation::init_state(exact);
    auto exchange = sub_grid.m_halo_exchanger;
    auto step = stepper(state, exchange);

    auto write = io::write_time_series(rt.m_output_filename, global_resolution,
                                       sub_grid.m_local_resolution,
                                       sub_grid.m_local_offset);
    if (write)
      write(0, state);

    if (tmax > 0)
      step(state, dt);

    using clock = std::chrono::high_resolution_clock;

    auto start = clock::now();

    real_t t;
    for (t = dt; t < tmax - dt / 2; t += dt)
      step(state, dt);

    computation::sync(state);
    if (write)
      write(t, state);

    auto stop = clock::now();
    double time = std::chrono::duration<double>(stop - start).count();
    double error = computation::compute_error(state, exact, t);

    results[id] = {error, time};
  };

  std::vector<std::thread> threads;
  threads.reserve(rt.m_num_threads - 1);
  for (int i = 1; i < rt.m_num_threads; ++i)
    threads.emplace_back(execution_func, i);
  set_device(rt.m_device_mapping[0]);
  execution_func(0);

  for (auto &thread : threads)
    thread.join();

  result local_result{0.0, 0.0};
  for (auto const &r : results) {
    local_result.error = std::max(local_result.error, r.error);
    local_result.time = std::max(local_result.time, r.time);
  }

  return comm_grid.collect_results(local_result);
}

} // namespace ghex_comm_impl

using ghex_comm_impl::runtime_init;
using ghex_comm_impl::runtime_register_options;
using ghex_comm_impl::runtime_solve;

} // namespace runtime
